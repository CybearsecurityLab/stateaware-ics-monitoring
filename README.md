# State-Aware ICS Monitoring

This repository implements and evaluates state-aware monitoring for industrial control systems. It combines:

- a Global Finite State Machine (GFSM) extracted from IEC 61131-3 Structured Text;
- state-indexed numerical invariants learned from nominal traces;
- topology-derived Signal Temporal Logic (STL) checks for hydraulic conservation laws;
- timing-correct action-obligation checks that connect controller inputs to subsequent actuator feedback.

The current evaluation covers Anytown, C-Town and L-Town. It includes 37 attack scenarios, GFSM/Phi/STL and obligation ablations, local/global SAIN comparisons, tolerance sensitivity, six external baselines, and runtime/memory measurements. Primary metrics use injection-window labels. Observation-aware effects and timing are reported separately.

## Current limitations

- PLC Structured Text is **surrogate code** generated from EPANET models and DHALSIM-style controls. It is not original vendor or benchmark PLC source.
- Evaluation is limited to water distribution networks. Transfer to other process domains is unevaluated.
- The official SAIN artifact does not contain an active SMONITOR runtime. SAIN detection here is passive and does not intercept or deny PLC writes.
- SICAMS operational-range discrimination needs externally sourced ranges, and leak-volume regression needs independent nonzero leakage targets.
- Uncapped IM-DISCO is intractable on the current transactions and remains unscored.
- The trusted computing base includes the PLC firmware and monitoring agent. Firmware compromise and coordinated self-consistent deception are outside the evaluated guarantees.

## Requirements and setup

The project uses `uv` for Python environments. Core tests run with:

```bash
uv run --no-sync pytest -q
```

The baseline extras add PyTorch, XGBoost, imbalanced-learn and mlxtend. On macOS, use the single-threaded OpenMP settings below to avoid native library conflicts:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync pytest -q
```

`rtamt` setup is described in `INSTALL.md`. Official QuantMiner sources, jars and its GPL notice are under `vendor/quantminer/`.

For ROCm training on the remote host, set `STATEAWARE_REPO_ROOT` to the checkout and use `--device cuda`. The same labels, splits, model configurations and validation rules are used on CPU and GPU.

## Repository layout

```text
src/st_gen/              Surrogate Structured Text generation and provenance
src/st_analyze/          AST, complete per-selector PDGs and invariants
src/gfsm/                Local FSM extraction and reachable composition
src/invariants/          State labels, rule mining and global Phi
src/monitor/             GFSM, invariants, timing and obligation detectors
src/stl/                 Topology residuals, synthesis and runtime detection
src/simulation/          Closed-loop cyber-attack generation
src/dataio/              Full-signal loading and chronological splits
baselines/               External baseline implementations and adapters
scripts/                 Experiment runners, audits and evidence generation
vendor/quantminer/       Official Java sources, adapter and license notices
scripts/out/             Generated runs, predictions, models, logs and audits
paper/                   Separate LNCS manuscript repository
```

## Surrogate PLC and PDG pipeline

`st_gen` creates surrogate IEC 61131-3 Structured Text from an EPANET INP file and a DHALSIM-style `plcs.yaml`. Generated files state their provenance and supported control subset. Rules, time-based controls and numeric/setpoint controls are not represented as supported source.

Run the pipeline with:

```bash
uv run --no-sync st-gen \
  --inp data/anytown/anytown.inp \
  --plcs data/anytown/anytown_plcs.yaml \
  --out data/generated/anytown --topology anytown
uv run --no-sync st-analyze --all
uv run --no-sync gfsm-build --all
```

Each actuator assignment appears inside its CASE branch. Multi-actuator analysis processes every CASE independently and tags invariants with the owning selector.

Non-empty `<plc>.pdg.json` files contain a `components` list with:

- `plc`, `case_var` and source-order `case_index`;
- every local `state_id` and its ordered nodes, branch conditions, edges and variables.

Legacy top-level state-ID entries retain the final CASE for compatibility. DOT output includes every component with collision-free PLC/selector/state namespaces.

## GFSM and action obligations

The original GFSM detector checks unknown states and observed transitions against source guards. Two additional ablations are available:

1. **Timing-correct GFSM:** guards use the previous observation rather than the destination observation.
2. **Action obligations:** an enabled action must be realized in the next observation, even if the actuator remains in the same state. Reasons identify the PLC, actuator, source guard, expected state, actual state and observation indices.

The original and revised outputs are preserved. Revised fusion is `GFSM_obligations OR Phi OR STL`; original fusion is `GFSM OR Phi OR STL`.

Current revised evidence:

- 11 sustained actuator-DoS true positives on each topology with zero held-out nominal alarms;
- 3/3/0 timing-only and 4/15/4 obligation-aware GFSM false positives on Anytown/C-Town/L-Town, compared with 3/256/20 for the original detector;
- unchanged fused Anytown metrics, lower fused C-Town false positives, and improved fused L-Town MCC/BA/AUPRC.

The monitor does not treat actuator feedback as an independent internal CASE-state observation. Verified internal PLC execution requires separate command or internal-state instrumentation.

### Offline scalability

`bench_gfsm_scalability.py` measures local/theoretical/reachable states, transitions, pruning ratio, construction time and peak memory without changing evaluation artifacts.

```bash
uv run --no-sync python scripts/bench_gfsm_scalability.py --topology anytown
uv run --no-sync python scripts/bench_gfsm_scalability.py --topology ctown
uv run --no-sync python scripts/bench_gfsm_scalability.py --topology ltown
```

C-Town retains all 1,024 theoretical composite states and has 613,632 transitions, so pruning does not reduce its state count. Its substantial construction memory is reported separately from online runtime.

## Global Phi and local SAIN

Global Phi partitions nominal observations by the composite GFSM state and mines NiaARM association rules with confidence 0.7 and support 0.1. The nominal 64% training partition fits rules; the disjoint 16% clean validation partition selects the violation threshold. The remaining 20% and all attacks stay held out. Unknown states and insufficient states remain explicit.

Local SAIN has two separately preserved engines:

- **NiaARM:** local per-CASE-state models on all three towns.
- **Official QuantMiner:** actual official rules parsed and evaluated per local state, with support 0.1, confidence 0.7 and the approved `max_attributes=2` adaptation.

The three-attribute QuantMiner attempt exposed upstream 32-bit factorial overflow. Exact BigInteger combinatorics fix that overflow. Where upstream arithmetic is already valid, the correction preserves combinations, support, confidence and rule semantics; it does not establish full-width equivalence or tractability. Exact state-local constant channels are excluded from QuantMiner input as static variables but retained as separately recorded point-interval invariants. Every nonconstant observable feature is retained, and all constants are named. Failed three-attribute artifacts are under `scripts/out/full_signal_sain_local_quantminer_three_attribute/`; the reduced-width runs use `scripts/out/full_signal_sain_local_quantminer/`.

Neither local engine reproduces SAIN's original execution configuration or SMONITOR.

## Expanded evaluation

Generate the main detector outputs and all nine tolerance configurations:

```bash
uv run --no-sync python scripts/export_logic_attack_predictions.py
uv run --no-sync python scripts/sweep_logic_tolerances.py
```

Each topology has one global Phi mining run per invocation. All 27 per-configuration prediction and metric directories are preserved. Default p99/scale-1 outputs are checked for parity against the corresponding sweep configuration.

Run the external baselines:

```bash
for topology in anytown ctown ltown; do
  uv run --no-sync python scripts/run_full_signal_cusum.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_sain.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_sain_local.py \
    --topology "$topology" --engine quantminer
  uv run --no-sync python scripts/run_full_signal_sicams.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_hybrid.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_imdisco.py --topology "$topology"
  uv run --no-sync python scripts/run_full_signal_conv.py \
    --topology "$topology" --device cpu
done
```

Run analysis and evidence generation:

```bash
uv run --no-sync python scripts/analyze_imdisco_tractability.py
uv run --no-sync python scripts/build_expansion_pack.py
uv run --no-sync python scripts/build_fact_pack.py
uv run --no-sync python scripts/render_expansion_results.py
```

### Baseline status

| Method | Current status | Important qualification |
|---|---|---|
| State-aware CUSUM | Runs on all three towns | Uses INP-derived signed flows and exact tank areas; no SWaT-specific threshold transfer. |
| Global source-derived SAIN | Runs on all three towns | Uses recovered PDG invariants and separate official QuantMiner artifacts; SMONITOR is unavailable. |
| Local NiaARM SAIN | Runs on all three towns | Separate per-CASE-state adaptation. |
| Local official QuantMiner | Anytown complete; C-Town/L-Town active | Two-attribute adaptation with exact overflow correction and parsed official rules. |
| SICAMS | Runs on all three towns | Per-sensor covariance models and INP-backed Laplacian fields; range classification and volume regression unavailable. |
| Hybrid Ensemble | Runs on all three towns | Independent manifested episodes, out-of-fold stacking and training-only warm-up; every evaluation row scored. |
| ConvBiLSTM-AE | Runs on all three towns | Float64 residual scoring makes every row finite; normal-validation thresholds only. |
| IM-DISCO maximal adaptation | Runs on all three towns | Thesis-inspired shared-State, entry-only TransitionGradient and retrospective completed-run semantics; exact maximal closed-pattern rules, not an official-code reproduction. High false-positive counts are reported in the maximal output summary. |
| Uncapped IM-DISCO FP-Growth/ARG | Unavailable on all three towns | Original unavailable artifacts remain separate; the corrected structural audit shows exponential frequent-itemset families and does not substitute a cap or zero score. |

## Prediction and evidence artifacts

Important paths include:

```text
scripts/out/logic_attack_predictions/   Original, timing, obligation, Phi, STL and fused predictions
scripts/out/logic_tolerance_sweep/      All 27 preserved tolerance configurations
scripts/out/full_signal_sain/           Global SAIN and official QuantMiner artifacts
scripts/out/full_signal_sain_local/     Local NiaARM artifacts
scripts/out/full_signal_sain_local_quantminer/  Local official QuantMiner artifacts
scripts/out/full_signal_sicams/         Detection, change points and compressed stage diagnostics
scripts/out/full_signal_hybrid/         Models, validation and evaluation predictions
scripts/out/full_signal_conv/           Models, finite reconstruction scores and observation-aware metrics
scripts/out/full_signal_imdisco/        Preserved original uncapped-unavailable artifacts
scripts/out/full_signal_imdisco_maximal/  Maximal predictions, models, metrics and corrected tractability audit
scripts/out/gfsm_scalability/           Offline construction time and memory
scripts/out/expansion_pack/             Recomputed expanded evidence summary
scripts/out/fact_pack/                  Fact-pack metrics and input hashes
```

Use `--reuse-model` with `run_full_signal_conv.py` or `run_full_signal_hybrid.py` to re-score a saved model without retraining. ConvBiLSTM thresholds are recomputed on normal validation rows using the same float64 residual arithmetic as evaluation. Hybrid warm-up uses only the final rows of the nominal training partition.

## Reproducibility archive

After the final source and paper commits, create a read-only archive outside the repository:

```bash
uv run --no-sync python scripts/create_reproducibility_snapshot.py \
  --output ../dimva2027-reproducibility.tar
(cd .. && shasum -a 256 -c dimva2027-reproducibility.sha256)
```

The manifest records the source and paper commits plus the size and SHA-256 of every selected raw or generated file. Verify archive members individually before pushing.

For GitHub distribution, keep large evidence out of the Git repository. Split the archive into 90 MiB release assets and upload them to a GitHub Release:

```bash
uv run --no-sync python scripts/package_reproducibility_release.py \
  --archive ../dimva2027-reproducibility.tar \
  --output-dir ../dimva2027-release
```

The package includes a manifest and `SHA256SUMS` for the archive and every chunk. Chunks are capped at 90 MiB, below the 100 MB repository-file limit.
