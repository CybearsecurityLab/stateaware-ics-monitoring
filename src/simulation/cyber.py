"""Closed-loop, scan-based WNTR simulation.

Events are mappings with ``kind``, ``target``, ``start`` and ``end``. Sensor
events additionally specify ``mode`` (constant, offset, freeze, series or
replay), ``scope`` (controller, scada or both), and, except for freeze, ``value``.
Series values are mappings from absolute scan times to values. Threshold events
target the WNTR control name; actuator events target a link. Later events win
when priorities are equal; larger numeric priorities win otherwise.

``freeze`` is the denial-of-service mode: the recipient keeps the value that
was delivered at activation and resumes fresh delivery when the event expires.
``replay`` re-injects the recipient's own delivery history: at scan ``t`` the
recipient receives the value it was delivered at ``t - period * scan_timestep``,
optionally perturbed by uniform noise and clamped to ``min``/``max``. The source
is the recipient's delivery history, not plant truth or an attacker-supplied
series, so two recipients of the same sensor replay different values when their
histories differ. The first replayed value must come from pre-attack deliveries;
later values may come from the replay itself.

Each scan publishes the pre-command plant state, delivers sensors, evaluates
controllers, and applies resolved commands. When commands are issued, WNTR
re-solves at that same time before integrating the following interval.
``plant_truth`` is the scan publication; ``hydraulic_truth`` contains the
post-command boundary solve and every intermediate hydraulic solve. At the
final scan no subsequent interval is solved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
import wntr

from hydraulics import load_network
from st_gen.plcs_loader import load_plcs
from .runner import RunSpec, _expected_columns


#: Explicit, backward-compatible attack-class taxonomy.  Events may carry an
#: ``attack_class`` key; when absent, the class is derived from ``kind`` and
#: ``mode`` so legacy event dictionaries keep their meaning.
ATTACK_CLASS_TAXONOMY: dict[str, str] = {
    'normal': 'no injected attack window is active',
    'sensor_offset': 'additive bias on a sensor value delivered to a recipient',
    'sensor_freeze': 'recipient keeps the value delivered at attack activation',
    'sensor_replay': 'recipient receives its own earlier delivered history',
    'sensor_constant': 'sensor value replaced by an attacker constant',
    'sensor_series': 'sensor value follows an attacker-supplied series',
    'actuator_dos': 'physical actuator command overridden (denial of service)',
    'control_guard_violation': 'actuator command that contradicts the controlling guard',
    'actuator_feedback_spoof': 'observation-only actuator feedback spoof; plant truth is unchanged',
    'coordinated': 'more than one attack mechanism is active on the same scan',
    'adaptive': 'pre-registered bounded/stealthy attack chosen without detector feedback',
}

_SENSOR_MODE_CLASSES = {
    'offset': 'sensor_offset',
    'freeze': 'sensor_freeze',
    'replay': 'sensor_replay',
    'constant': 'sensor_constant',
    'series': 'sensor_series',
}


def derive_attack_class(event: dict[str, Any]) -> str:
    """Return the attack class for a legacy or explicitly tagged event."""
    kind = event.get('kind')
    if kind == 'sensor':
        mode = event.get('mode')
        try:
            return _SENSOR_MODE_CLASSES[mode]
        except KeyError as exc:
            raise ValueError(f'unsupported sensor mode for attack taxonomy: {mode!r}') from exc
    if kind == 'actuator':
        return 'actuator_dos'
    if kind == 'actuator_feedback':
        return 'actuator_feedback_spoof'
    if kind == 'threshold':
        return 'control_guard_violation'
    raise ValueError(f'unsupported attack mechanism for attack taxonomy: {kind!r}')


def _row_attack_class(active: list[dict[str, Any]]) -> str:
    """Return one per-scan label from the active events."""
    if not active:
        return 'normal'
    classes = {event['attack_class'] for event in active}
    if len(classes) == 1:
        return next(iter(classes))
    return 'coordinated'


def _noise_unit(seed: int, event_id: int, recipient: str, scan_time: int) -> float:
    """Uniform deviate in [0, 1) fixed by event, recipient and scan time."""
    material = np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        int(event_id) & 0xFFFFFFFF,
        zlib.crc32(recipient.encode('utf-8')) & 0xFFFFFFFF,
        int(scan_time) & 0xFFFFFFFF,
    ])
    return float(np.random.default_rng(material).random())


def _validate_replay(event: dict[str, Any], step: int, duration: int) -> None:
    """Validate a replay event and its pre-attack history coverage."""
    period = event.get('period')
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError('replay attack requires a positive integer period in scan steps')
    noise = event.get('noise', 0.0)
    if not isinstance(noise, (int, float)) or isinstance(noise, bool) or not math.isfinite(noise) or noise < 0:
        raise ValueError('replay noise amplitude must be a finite nonnegative number')
    for bound in ('min', 'max'):
        if bound in event:
            value = event[bound]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError('replay clamp bounds must be finite numbers')
    if 'min' in event and 'max' in event and event['min'] > event['max']:
        raise ValueError('replay clamp bounds must satisfy min <= max')
    for t in range(0, int(duration) + 1, int(step)):
        if not event['start'] <= t < event['end']:
            continue
        source = t - period * step
        if source < 0:
            raise ValueError(f'insufficient pre-attack history for replay at t={t}')


@dataclass
class CyberTraceBundle:
    plant_truth: pd.DataFrame
    hydraulic_truth: pd.DataFrame
    observed: pd.DataFrame
    controller_views: dict[str, pd.DataFrame]
    labels: pd.DataFrame
    events: list[dict[str, Any]]
    manifest: dict[str, Any]

    def validate(self) -> None:
        columns = self.manifest['expected_columns']
        times = self.manifest['solve_times']
        step = self.manifest['effective_timing']['scan_timestep']
        duration = self.manifest['timesteps']['duration']
        if times != list(range(0, int(duration) + 1, int(step))):
            raise ValueError('invalid scan grid')
        for frame in (self.plant_truth, self.observed):
            if list(frame.columns) != columns or frame.time_seconds.tolist() != times or not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError('invalid complete trace')
        if (list(self.hydraulic_truth.columns) != columns or
                self.hydraulic_truth.time_seconds.tolist() != self.manifest['hydraulic_solve_times'] or
                not np.isfinite(self.hydraulic_truth.to_numpy(dtype=float)).all() or
                not self.hydraulic_truth.time_seconds.is_monotonic_increasing or
                self.hydraulic_truth.time_seconds.duplicated().any() or
                not set(times).issubset(set(self.hydraulic_truth.time_seconds))):
            raise ValueError('invalid hydraulic trace')
        label_columns = ['time_seconds', 'attack_active', 'delivery_or_command_modified', 'physical_impact']
        optional_labels = {'attack_class', 'observable_attack'}
        if (list(self.labels.columns[:4]) != label_columns or
                set(self.labels.columns[4:]) - optional_labels or
                self.labels.time_seconds.tolist() != times):
            raise ValueError('invalid labels')
        if any(not pd.api.types.is_bool_dtype(self.labels[c]) for c in label_columns[1:]):
            raise ValueError('invalid label types')
        if 'observable_attack' in self.labels.columns and not pd.api.types.is_bool_dtype(self.labels['observable_attack']):
            raise ValueError('invalid observable_attack label type')
        if 'attack_class' in self.labels.columns:
            classes = self.labels['attack_class'].tolist()
            if any(not isinstance(value, str) or value not in ATTACK_CLASS_TAXONOMY for value in classes):
                raise ValueError('invalid attack_class label')
            if any(value != 'normal' for value in self.labels.loc[~self.labels['attack_active'], 'attack_class']):
                raise ValueError('inactive rows must carry the normal attack_class')
        if set(self.controller_views) != set(self.manifest['controller_sensors']):
            raise ValueError('missing or unexpected controller cache')
        for name, frame in self.controller_views.items():
            if list(frame.columns) != ['time_seconds'] + self.manifest['controller_sensors'][name] or frame.time_seconds.tolist() != times or not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError('invalid controller cache')


def write_cyber_bundle(bundle: CyberTraceBundle, path: str | Path) -> None:
    bundle.validate()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in [('plant_truth', bundle.plant_truth), ('hydraulic_truth', bundle.hydraulic_truth), ('observed', bundle.observed), ('labels', bundle.labels)]:
        frame.to_csv(path / (name + '.csv'), index=False, float_format='%.17g', lineterminator='\n')
    for i, name in enumerate(sorted(bundle.controller_views)):
        bundle.controller_views[name].to_csv(path / f'controller_{i}.csv', index=False, float_format='%.17g', lineterminator='\n')
    for name in ('events', 'manifest'):
        (path / (name + '.json')).write_text(json.dumps(getattr(bundle, name), sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_cyber(spec: RunSpec, attacks: list[dict[str, Any]] | None = None, *, plcs_path: str | Path | None = None) -> CyberTraceBundle:
    events = json.loads(json.dumps(attacks or [], allow_nan=False))
    result = _run(spec, events, plcs_path)
    if events:
        reference = _run(spec, [], plcs_path)
        if reference.plant_truth.time_seconds.tolist() != result.plant_truth.time_seconds.tolist():
            raise RuntimeError('baseline and cyber solve times differ')
        a = result.plant_truth.drop(columns='time_seconds').to_numpy()
        b = reference.plant_truth.drop(columns='time_seconds').to_numpy()
        result.labels['physical_impact'] = np.any(~np.isclose(a, b, rtol=1e-9, atol=1e-10), axis=1)
        observed_a = result.observed.drop(columns='time_seconds').to_numpy()
        observed_b = reference.observed.drop(columns='time_seconds').to_numpy()
        result.labels['observable_attack'] = np.any(
            ~np.isclose(observed_a, observed_b, rtol=1e-9, atol=1e-10), axis=1)
    else:
        result.labels['observable_attack'] = False
    result.validate()
    return result


def _run(spec: RunSpec, events: list[dict[str, Any]], plcs_path: str | Path | None) -> CyberTraceBundle:
    network = load_network(spec.inp_path, simulator_backend='wntr')
    wn = wntr.network.WaterNetworkModel(str(spec.inp_path))
    if hashlib.sha256(Path(spec.inp_path).read_bytes()).hexdigest() != network.provenance.sha256:
        raise ValueError('INP changed during loading')
    for node in network.junctions:
        if node.emitter_coefficient:
            raise ValueError('nonzero emitter unsupported by WNTRSimulator')
    for pump in network.pumps:
        if pump.speed != 1 or pump.speed_pattern_id:
            raise ValueError('non-unit or patterned pump speed unsupported by WNTRSimulator')
    timing = wn.options.time
    keys = ('duration', 'hydraulic_timestep', 'report_timestep', 'report_start', 'pattern_timestep', 'pattern_start', 'rule_timestep', 'quality_timestep')
    original = {k: getattr(timing, k) for k in keys}
    h, report, duration = timing.hydraulic_timestep, timing.report_timestep, timing.duration
    if not isinstance(report, (int, float)) or min(h, report, duration) <= 0:
        raise ValueError('invalid timing')
    step = min(h, report)
    if step != int(step) or max(h, report) % step or duration % step or timing.report_start < 0 or timing.report_start % step or timing.report_start > duration or timing.pattern_timestep % step or timing.pattern_start % step:
        raise ValueError('timing must align to hydraulic grid')
    controls = []
    for name, control in list(wn.controls()):
        condition = control.condition
        actions = control.actions()
        if type(control) is not wntr.network.controls.Control or len(actions) != 1 or getattr(control, '_else_actions', []):
            raise ValueError(f'unsupported rule or control: {name}')
        action = actions[0]
        target, attr = action.target()
        if isinstance(target, wntr.network.Pump) and attr != 'status':
            raise ValueError('controlled pump speed unsupported by WNTRSimulator')
        if not isinstance(condition, wntr.network.controls.ValueCondition) or target.name not in wn.link_name_list or attr != 'status' or action._value not in (0, 1):
            raise ValueError(f'unsupported time, numeric or control mechanism: {name}')
        source = condition._source_obj
        rel = condition._relation.name
        if source.name not in wn.node_name_list or condition._source_attr not in ('level', 'pressure') or rel not in ('lt', 'gt'):
            raise ValueError(f'unsupported control condition: {name}')
        controls.append(dict(id=name, link=target.name, sensor=source.name, relation=rel, threshold=float(condition._threshold), value=int(action._value)))
        wn.remove_control(name)
    if plcs_path is None:
        controllers = {'PLC': {'sensors': sorted({c['sensor'] for c in controls}), 'actuators': sorted({c['link'] for c in controls})}}
    else:
        controllers = {p.name: {'sensors': sorted(set(p.sensors)), 'actuators': sorted(set(p.actuators))} for p in load_plcs(plcs_path)}
        owned = [a for p in controllers.values() for a in p['actuators']]
        if len(owned) != len(set(owned)) or any(a not in wn.link_name_list for a in owned):
            raise ValueError('ambiguous or unknown actuator ownership')
        for c in controls:
            owners = [name for name, p in controllers.items() if c['link'] in p['actuators']]
            if len(owners) != 1:
                raise ValueError(f'ambiguous or missing PLC ownership for {c["id"]}')
            c['owner'] = owners[0]
    for c in controls:
        c.setdefault('owner', 'PLC')
    plc_config = {name: {key: list(value) for key, value in p.items()} for name, p in sorted(controllers.items())}
    dependencies = {name: {} for name in controllers}
    for c in controls:
        owner, sensor = c['owner'], c['sensor']
        suppliers = sorted(name for name, p in plc_config.items() if sensor in p['sensors'])
        if owner not in suppliers and len(suppliers) != 1:
            raise ValueError(f'ambiguous or missing sensor supplier for {c["id"]}')
        dependencies[owner][sensor] = owner if owner in suppliers else suppliers[0]
        if sensor not in controllers[owner]['sensors']:
            controllers[owner]['sensors'].append(sensor)
            controllers[owner]['sensors'].sort()
    columns = _expected_columns(network)
    if len(columns) != len(set(columns)):
        raise ValueError('ambiguous flat column names')
    for p in controllers.values():
        if any(s not in columns[1:] for s in p['sensors']):
            raise ValueError('unknown PLC sensor')
    def recipients_of(event: dict[str, Any]) -> list[str]:
        names: list[str] = []
        if event['scope'] in ('scada', 'both'):
            names.append('scada')
        if event['scope'] in ('controller', 'both'):
            names.extend(name for name, p in controllers.items()
                         if event['target'] in p['sensors'] and (not event.get('controller') or event['controller'] == name))
        return names
    history_pairs: set[tuple[str, str]] = set()
    for i, e in enumerate(events):
        if not isinstance(e, dict) or e.get('kind') not in ('sensor', 'actuator', 'actuator_feedback', 'threshold'):
            raise ValueError('unsupported attack mechanism')
        if not isinstance(e.get('start'), (int, float)) or not isinstance(e.get('end'), (int, float)) or e['start'] < 0 or e['end'] <= e['start'] or e['start'] % step or e['end'] % step or e['end'] > duration:
            raise ValueError('attack boundaries must be aligned and within duration')
        if not isinstance(e.get('priority', 0), (int, float)):
            raise ValueError('invalid priority')
        if 'attack_class' in e and (e['attack_class'] not in ATTACK_CLASS_TAXONOMY or e['attack_class'] == 'normal'):
            raise ValueError('invalid attack_class')
        kind, target = e['kind'], e.get('target')
        if kind == 'sensor':
            if target not in columns[1:] or e.get('scope') not in ('controller', 'scada', 'both') or e.get('mode') not in ('constant', 'offset', 'freeze', 'series', 'replay'):
                raise ValueError('invalid sensor attack')
            if e.get('controller') is not None and e['scope'] == 'scada':
                raise ValueError('sensor attack names a controller but only targets scada')
            if e['scope'] in ('controller', 'both') and not any(target in p['sensors'] and (not e.get('controller') or e['controller'] == name) for name, p in controllers.items()):
                raise ValueError('sensor attack has no controller recipient')
            if e['mode'] in ('constant', 'offset') and not isinstance(e.get('value'), (int, float)):
                raise ValueError('sensor value required')
            if e['mode'] == 'series' and not isinstance(e.get('value'), dict):
                raise ValueError('series mapping required')
            if e['mode'] == 'replay':
                _validate_replay(e, step, duration)
                history_pairs.update((name, target) for name in recipients_of(e))
        elif kind == 'actuator':
            if target not in wn.link_name_list or e.get('value') not in ('OPEN', 'CLOSED'):
                raise ValueError('actuator override requires OPEN or CLOSED')
        elif kind == 'actuator_feedback':
            if target not in wn.link_name_list or e.get('value') not in ('OPEN', 'CLOSED'):
                raise ValueError('actuator feedback spoof requires OPEN or CLOSED')
            if e.get('channel', 'status') != 'status':
                raise ValueError('actuator feedback spoof supports only the status channel')
            if e.get('scope') != 'scada':
                raise ValueError('observation-only actuator feedback spoof requires scope=scada')
            if e.get('controller') is not None:
                raise ValueError('observation-only actuator feedback spoof cannot name a controller')
        elif target not in [c['id'] for c in controls] or not isinstance(e.get('value'), (int, float)):
            raise ValueError('invalid threshold override')
        e.setdefault('priority', 0)
        e.setdefault('seed', spec.seed)
        e.setdefault('attack_class', derive_attack_class(e))
        e['event_id'] = i
    timing.hydraulic_timestep = step
    timing.report_timestep = 'ALL'
    timing.report_start = 0
    timing.duration = 0
    sim = wntr.sim.WNTRSimulator(wn)
    freezes: dict[tuple[int, str], float] = {}
    history: dict[tuple[str, str], dict[int, float]] = {pair: {} for pair in history_pairs}
    rows, observed, labels, hydraulic_rows = [], [], [], []
    views = {name: [] for name in controllers}
    latched = {name: int(wn.get_link(name)._user_status) for name in wn.link_name_list}
    overridden: set[str] = set()
    execution_log = []
    units = {'time_seconds': 's'}
    for link in network.links():
        units.update({f'f_{link.id}': 'm^3/s', f's_{link.id}': 'status code: 0=closed, 1=open, 2=active'})
    for node in network.junctions:
        units.update({node.id: 'm', f'demand_{node.id}': 'm^3/s', f'leak_{node.id}': 'm^3/s', f'emitter_{node.id}': 'm^3/s'})
    for tank in network.tanks:
        units.update({tank.id: 'm', f'leak_{tank.id}': 'm^3/s'})
    for reservoir in network.reservoirs:
        units[f'head_{reservoir.id}'] = 'm'
    for pump in network.pumps:
        units[f'speed_{pump.id}'] = 'dimensionless'
    for valve in network.valves:
        units[f'setting_{valve.id}'] = {'PRV': 'm', 'PSV': 'm', 'PBV': 'm', 'FCV': 'm^3/s', 'TCV': 'dimensionless'}.get(valve.valve_type.upper(), 'unknown')
        if valve.valve_type.upper() == 'GPV':
            raise ValueError('GPV setting unsupported')
    def result_row(result: Any, index: int) -> dict[str, float]:
        row: dict[str, float] = {'time_seconds': float(result.time[index])}
        for link in network.links():
            row[f'f_{link.id}'] = float(result.link['flowrate'][link.id].iloc[index])
            row[f's_{link.id}'] = float(result.link['status'][link.id].iloc[index])
        for node in network.junctions:
            row[node.id] = float(result.node['pressure'][node.id].iloc[index])
            row[f'demand_{node.id}'] = float(result.node['demand'][node.id].iloc[index])
            row[f'leak_{node.id}'] = float(result.node['leak_demand'][node.id].iloc[index])
            row[f'emitter_{node.id}'] = 0.0
        for tank in network.tanks:
            row[tank.id] = float(result.node['pressure'][tank.id].iloc[index])
            row[f'leak_{tank.id}'] = float(result.node['leak_demand'][tank.id].iloc[index])
        for reservoir in network.reservoirs:
            row[f'head_{reservoir.id}'] = float(result.node['head'][reservoir.id].iloc[index])
        for pump in network.pumps:
            row[f'speed_{pump.id}'] = 1.0
        for valve in network.valves:
            row[f'setting_{valve.id}'] = float(result.link['setting'][valve.id].iloc[index])
        if list(row) != columns or not np.isfinite(list(row.values())).all():
            raise RuntimeError('incomplete hydraulic channels')
        return row

    def solve(t: int) -> dict[str, float]:
        timing.duration = t
        result = sim.run_sim(convergence_error=True)
        if not result.time or result.time[-1] != t or any(a >= b for a, b in zip(result.time, result.time[1:])):
            raise RuntimeError(f'WNTR returned incomplete simulation at {t}')
        for i in range(len(result.time)):
            item = result_row(result, i)
            if hydraulic_rows and item['time_seconds'] == hydraulic_rows[-1]['time_seconds']:
                hydraulic_rows[-1] = item
            elif hydraulic_rows and item['time_seconds'] < hydraulic_rows[-1]['time_seconds']:
                raise RuntimeError('WNTR returned out-of-order hydraulic solve')
            else:
                hydraulic_rows.append(item)
        return result_row(result, -1)

    for t in range(0, int(duration) + 1, int(step)):
        row = solve(t)
        rows.append(row)
        active = sorted((e for e in events if e['start'] <= t < e['end']), key=lambda e: (e['priority'], e['event_id']))
        executed = {e['event_id']: {'time_seconds': t, **e, 'applied': []} for e in active}
        scada = row.copy()
        caches = {name: {sensor: row[sensor] for sensor in p['sensors']} for name, p in controllers.items()}
        modified = False
        def spoof(e: dict[str, Any], old: float, scope: str) -> tuple[float, dict[str, Any]]:
            mode = e['mode']
            if mode == 'freeze':
                return freezes.setdefault((e['event_id'], scope), old), {}
            if mode == 'series':
                if str(t) not in e['value']:
                    raise ValueError(f'missing custom series value at {t}')
                return float(e['value'][str(t)]), {}
            if mode == 'replay':
                source_time = t - e['period'] * step
                recorded = history.get((scope, e['target']), {})
                if source_time not in recorded:
                    raise ValueError(f'insufficient delivered history for replay at t={t}')
                source_value = recorded[source_time]
                amplitude = float(e.get('noise', 0.0))
                delta = amplitude * (2.0 * _noise_unit(e['seed'], e['event_id'], scope, t) - 1.0) if amplitude else 0.0
                value = source_value + delta
                clamp = 'none'
                if 'max' in e and value > e['max']:
                    value = float(e['max'])
                    clamp = 'max'
                elif 'min' in e and value < e['min']:
                    value = float(e['min'])
                    clamp = 'min'
                return value, {'source_time_seconds': float(source_time), 'source_value': float(source_value),
                               'noise': float(delta), 'clamp': clamp}
            return float(e['value']) + (old if mode == 'offset' else 0), {}
        for e in active:
            if e['kind'] != 'sensor':
                continue
            target = e['target']
            if e['scope'] in ('scada', 'both'):
                new, detail = spoof(e, row[target], 'scada')
                scada[target] = new
                executed[e['event_id']]['applied'].append({'recipient': 'scada', 'value': new, **detail})
            if e['scope'] in ('controller', 'both'):
                for name, cache in caches.items():
                    if target in cache and (not e.get('controller') or e['controller'] == name):
                        new, detail = spoof(e, row[target], name)
                        cache[target] = new
                        executed[e['event_id']]['applied'].append({'recipient': name, 'value': new, **detail})
        for e in active:
            if e['kind'] != 'actuator_feedback':
                continue
            column = f's_{e["target"]}'
            value = 1.0 if e['value'] == 'OPEN' else 0.0
            scada[column] = value
            executed[e['event_id']]['applied'].append(
                {'recipient': 'scada', 'channel': 'status', 'column': column, 'value': value})
        for recipient, sensor in sorted(history_pairs):
            history[(recipient, sensor)][t] = scada[sensor] if recipient == 'scada' else caches[recipient][sensor]
        modified |= any(scada[k] != row[k] for k in scada)
        modified |= any(value != row[sensor] for cache in caches.values() for sensor, value in cache.items())
        observed.append(scada)
        for name, cache in caches.items():
            views[name].append({'time_seconds': float(t), **cache})
        commands = {}
        nominal_commands = {}
        for c in controls:
            threshold = c['threshold']
            for e in active:
                if e['kind'] == 'threshold' and e['target'] == c['id']:
                    threshold = e['value']
                    executed[e['event_id']]['applied'].append({'control': c['id'], 'value': threshold})
            value = caches[c['owner']][c['sensor']]
            if (value < c['threshold'] if c['relation'] == 'lt' else value > c['threshold']):
                nominal_commands[c['link']] = c['value']
            if (value < threshold if c['relation'] == 'lt' else value > threshold):
                commands[c['link']] = c['value']
        for link in set(commands) | set(nominal_commands):
            modified |= commands.get(link, latched[link]) != nominal_commands.get(link, latched[link])
        latched.update(commands)
        current_overrides = {e['target'] for e in active if e['kind'] == 'actuator'}
        controlled_links = {c['link'] for c in controls}
        for link in current_overrides - overridden:
            if link not in controlled_links:
                latched[link] = int(wn.get_link(link)._user_status)
        for link in overridden - current_overrides:
            if link in latched:
                commands[link] = latched[link]
        before_overrides = {link: latched[link] for link in current_overrides}
        for e in active:
            if e['kind'] == 'actuator':
                link = e['target']
                value = int(e['value'] == 'OPEN')
                commands[link] = value
                executed[e['event_id']]['applied'].append({'link': link, 'value': value})
        modified |= any(commands[link] != before_overrides[link] for link in current_overrides)
        overridden = current_overrides
        for link, value in commands.items():
            wntr.network.controls.ControlAction(wn.get_link(link), 'status', wntr.network.LinkStatus(value)).run_control_action()
        if commands and t < duration:
            wn.sim_time = t
            solve(t)
        execution_log.extend(executed[e['event_id']] for e in active)
        labels.append({'time_seconds': float(t), 'attack_active': bool(active), 'delivery_or_command_modified': bool(modified), 'physical_impact': False, 'attack_class': _row_attack_class(active)})
    manifest = {'schema_version': 1, 'run_id': spec.run_id, 'seed': spec.seed, 'split': spec.split, 'inp_sha256': network.provenance.sha256, 'wntr_version': wntr.__version__, 'engine': 'WNTRSimulator', 'timesteps': original, 'hydraulic_options': dict(vars(wn.options.hydraulic)), 'effective_timing': {'hydraulic_timestep': step, 'report_timestep': 'ALL', 'report_start': 0, 'scan_timestep': step}, 'expected_columns': columns, 'column_units': units, 'solve_times': [r['time_seconds'] for r in rows], 'hydraulic_solve_times': [r['time_seconds'] for r in hydraulic_rows], 'controller_sensors': {n: p['sensors'] for n, p in controllers.items()}, 'plc_config': plc_config, 'plc_config_sha256': hashlib.sha256(json.dumps(plc_config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'sensor_dependencies': dependencies, 'actuator_ownership': {a: name for name, p in plc_config.items() for a in p['actuators']}, 'scheduler': 'publish pre-command scan state; deliver sensors; evaluate PLCs; apply resolved commands; re-solve at scan time for next interval; integrate to next scan. hydraulic_truth stores post-command state at scan boundaries and all intermediate solves.', 'controller_files': {name: f'controller_{i}.csv' for i, name in enumerate(sorted(controllers))}, 'controls': controls, 'priority_policy': 'ascending priority then source order; later wins', 'physical_impact_reference': 'attack-free closed-loop run, per-scan numerical comparison', 'observable_attack_policy': 'observed row differs from the attack-free observed row (rtol=1e-9, atol=1e-10); evaluation labels use attack_active OR observable_attack', 'sensor_modes': ['constant', 'offset', 'freeze', 'series', 'replay'], 'attack_class_taxonomy': ATTACK_CLASS_TAXONOMY, 'attack_class_policy': 'explicit event attack_class when supplied; otherwise derived from kind and mode; mixed active classes are labelled coordinated; normal rows carry normal', 'actuator_feedback_policy': 'SCADA-only observation status spoof; never changes PLC inputs, physical commands, plant truth or hydraulic solves', 'replay_policy': 'replay delivers the recipient\'s own pre-attack delivered history at t - period * scan_timestep; plant truth and attacker series are never used as the source; insufficient pre-attack history is an error; noise is uniform on [-noise, noise], deterministic per event, recipient and scan; clamps are applied after the addition, max before min'}
    return CyberTraceBundle(pd.DataFrame(rows), pd.DataFrame(hydraulic_rows), pd.DataFrame(observed), {n: pd.DataFrame(v) for n, v in views.items()}, pd.DataFrame(labels), execution_log, manifest)
