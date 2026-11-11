# stateaware-ics-monitoring

## Stage 4 — GFSM (`gfsm-build`)

Stage 4 ("GFSM builder") implements the paper-faithful FSM extraction
algorithm (Def 1/2: one FunctionBlock per CASE variable per PLC) plus a new
synchronous-product composer. It is derived from the Rust `fsm-extractor`
(`github.com/LaBackDoor/fsm-extractor`, pinned commit `14950d5`) but
intentionally supersedes its lead-actuator-only behavior for multi-actuator
PLCs. It reads Stage 2's `<plc>.ast.xml` files, extracts each PLC's local
FSM, and composes them into one global FSM per topology.

```bash
gfsm-build --topology anytown
# or every topology with a Stage 2 analysis manifest:
gfsm-build --all
```

For each PLC it writes into `data/generated/<topology>/gfsm/`:
`<plc>.fsm.json`, `<plc>.fsm.dot`; per topology `<topology>.gfsm.json`,
`<topology>.gfsm.dot`, `<topology>.gfsm.analysis.json`, plus
`<topology>_gfsm_manifest.json` (provenance + per-PLC status, cross-checked
against Stage 2's FSM counts).

Flags: `--max-states N` (hard cap on global states; clean error on
overflow), `--out-dir PATH`, `--jobs N` (process-pool fan-out across PLCs;
`--jobs 1` is sequential; output is byte-identical regardless of `N`).

Correctness is enforced by the structural golden tests in
`tests/gfsm/test_golden_*.py`; they verify per-CASE FSM structure and
composition against known-good fixtures.

## Stage 2 — ST analysis (`st-analyze`)

Stage 2 ("parser") runs the vendored IEC-ST-Analyzer (`src/iec_st_compiler/`,
upstream `github.com/LaBackDoor/iec_st_compiler`) over `st_gen`'s generated
`.st` files to produce the AST / PDG / invariant-template artifacts consumed by
Stage 3 (STL synthesis) and Stage 4 (GFSM extraction).

```bash
st-analyze --topology anytown
# or every topology with a manifest:
st-analyze --all
```

For each PLC it writes into `data/generated/<topology>/analysis/`:
`<plc>.ast.xml`, `<plc>.invariants.json`, `<plc>.pdg.dot`, `<plc>.pdg.json`,
plus `<topology>_analysis_manifest.json` (provenance + per-PLC status,
cross-checked against `st_gen`'s FSM counts).

The vendored analyzer is never modified; any dialect mismatch is fixed in
`st_gen`'s emitter or `st_analyze/adapter.py`. The emitter places actuation
assignments inside their CASE branches. For multi-actuator PLCs, the adapter
analyzes each CASE block separately, tags invariants with their selector, and
aggregates them; regression tests cover every actuator block.

## Stage 3 — STL monitor (`stl-monitor`) + hybrid (`ics-monitor`)

`dataio-ingest --all` normalizes the raw attack datasets (anytown 7z, **ctown =
BATADAL**) into committed `data/<topo>/dataset/` via the shared
`src/dataio/` loader (also consumed by the future invariant quantifier).
`stl-monitor --topology <t>` (or `--all`) calibrates STL thresholds from the
clean frames, synthesizes formulas, evaluates robustness with `rtamt`
(per-formula `--jobs` parallel, byte-identical regardless of worker count), and
writes `data/generated/<topo>/stl/`. `ics-monitor` runs the STL detector plus a
documented all-zeros GFSM stub and emits the logical-OR label
(`data/generated/<topo>/monitor/`); the GFSM stub is replaced when Stage 4's
runtime detector lands, with no change to fusion/CLI.

`rtamt` has no usable PyPI release on Python ≥3.13; it is pinned from git with a
`typing.io` shim — see `INSTALL.md`. BATADAL provenance (it *is* the C-Town
network) is recorded only in `data/ctown/dataset/dataset_manifest.yaml`.

## Stage "invariants" — NiaARM state-aware mining (`invariants-mine`)

Implements Algorithm 1 of the SAIN paper: per composite GFSM state `s`,
mine numerical association rules with NiaARM from baseline calibration
traces sliced to `Ds`, filter to confidence ≥ 0.7 and support ≥ 0.1, and
persist the invariant mapping `Φ(s)`.

```bash
invariants-mine --topology anytown --data-root data
# or every topology that has data/generated/<topo>/gfsm/<topo>.gfsm.json:
invariants-mine --all --data-root data
```

Writes `data/generated/<topo>/invariants/<topo>_phi.json` +
`<topo>_invariants_manifest.json` (provenance + sha256 of the upstream
gfsm and dataset manifests).

The data column for each GFSM component (one per actuator CASE) is
resolved automatically per-actuator from the CASE selector
(`<ACT>_State` → actuator `ACT`) through the dataset `column_map`
(`ACT` → `S_`+`ACT` → lowercase fallback) — no manual configuration. The runtime
monitor (`ics-monitor`) consumes `Φ` and the GFSM JSON, loaded once at
`fit()` (build-once / reuse-many; staleness-gated by sha256). Fusion
defaults to the paper's alarm-union rule `(gfsm OR invariants) OR stl`
(safe behavior is LogicValid AND PhysicsValid, so an alarm is any logical
OR physical violation); pass `--fusion intersection` for the conjunction ablation.

## Expanded full-signal evaluation

The `scripts/run_full_signal_*.py` entry points evaluate the six baselines on
Anytown, C-Town and L-Town. `scripts/export_logic_attack_predictions.py`
exports GFSM, global-Phi, STL and their three-way alarm union, and
`scripts/sweep_logic_tolerances.py` runs the nine-configuration tolerance
sweep with one Phi mining run per topology. All primary metrics use
injection-window labels; observation-aware outputs remain separate.

```bash
uv run --no-sync python scripts/export_logic_attack_predictions.py
uv run --no-sync python scripts/sweep_logic_tolerances.py
for topology in anytown ctown ltown; do
  uv run --no-sync python scripts/run_full_signal_cusum.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_sain.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_sain_local.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_sicams.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_hybrid.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_conv.py --topology "$topology" --device cpu
done
uv run --no-sync python scripts/analyze_imdisco_tractability.py
uv run --no-sync python scripts/build_expansion_pack.py
uv run --no-sync python scripts/render_expansion_results.py
```

Global Phi uses the nominal 64% training rows for NiaARM mining and the 16%
clean validation rows for its violation threshold. Its feature policy excludes
state columns, labels, timestamps and simulator-truth channels. The fused flag
is `GFSM OR Phi OR STL`; all four per-row outputs are preserved. Local SAIN
accepts `--jobs N` for process-parallel component mining with identical data
and per-component seeds.

Use `--reuse-model` with `run_full_signal_conv.py` or
`run_full_signal_hybrid.py` to re-score a saved model without retraining. Conv
recomputes the validation threshold with the same float64 residual arithmetic
used at evaluation. Hybrid prepends the final `min_scored` rows of the 64%
nominal training partition to every evaluation run, scores every evaluation
row, and preserves the existing validation predictions and independent episode
manifest provenance. On the ROCm host, set `STATEAWARE_REPO_ROOT` to the
checkout and use `--device cuda` for ConvBiLSTM training.

The IM-DISCO command is analysis-only: it records exact exponential output
bounds without materializing uncapped itemsets. Existing unavailable results
stay unscored. SICAMS writes INP-backed Laplacian fields and compressed per-row
discrimination diagnostics under `scripts/out/full_signal_sicams/<topology>/`.
SAIN keeps official QuantMiner rules separate from its empirical passive
detector. SMONITOR is not available in the official artifact and is not
claimed as reproduced.

After the final source and paper commits, create the read-only evidence archive
outside the checkout. The manifest records both commit hashes and a SHA-256 for
every selected raw or generated file.

```bash
uv run --no-sync python scripts/create_reproducibility_snapshot.py \
  --output ../dimva2027-reproducibility.tar
(cd .. && shasum -a 256 -c dimva2027-reproducibility.sha256)
```