# Experimental HIP support-count backend

This directory contains an opt-in HIP/JNI implementation of QuantMiner's support-count loop. It does not replace the genetic algorithm, quality evaluation, sorting, random-number generation, or rule generation. The default execution path remains the original CPU Java code.

The backend is restricted to numeric configurations with one disjunction per side and matching quantitative interval geometry. Unsupported configurations use the original CPU loop. Explicit GPU enablement fails loudly when the native library cannot be loaded.

## Build on Zalman

Requirements: HIP/ROCm development packages, a JDK with JNI headers, and an AMD GPU supported by the selected architecture.

```bash
cd vendor/quantminer/gpu
make HIP_ARCH=gfx1030
```

Compile the Java sources with `--release 8`. Run with:

```bash
QUANTMINER_GPU=1 \
QUANTMINER_GPU_LIBRARY=/absolute/path/to/libquantminer_gpu.so \
java --enable-native-access=ALL-UNNAMED \
  -cp build/classes:vendor/quantminer/dist/ostermillerutils_1_07_00.jar \
  QuantMinerCli INPUT.csv OUTPUT.csv 0.1 0.7 250 100 2
```

`-Dquantminer.gpu=true` and `-Dquantminer.gpu.library=...` are equivalent to the environment variables. Do not set these flags for ordinary CPU evaluation.

## Verification and performance

On Zalman, 301 randomized fixtures across four coverage modes matched the Java CPU reference, including interval boundaries, missing-value sentinels, NaN, infinities, empty active-row sets, and population growth beyond 256 candidates.

Anytown P78 and P79 each produced 4,970 rules and byte-identical `rules.csv` files with SHA-256 `508e3390ae7b4e912b375ef03d4d9e4caad043e7d83abbe64f5662dc2ea46c4d`. GPU wall times were 183.53 and 185.38 seconds, versus 61.34 and 62.51 seconds for the existing concurrent CPU runs. Synchronous JNI transfers and frequent small kernel launches make this prototype approximately three times slower end-to-end. It is therefore retained only as negative experimental evidence and was not used for C-Town, L-Town, or manuscript metrics.
