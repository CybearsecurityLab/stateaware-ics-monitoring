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

The fixtures cover 301 randomized cases across four coverage modes, 15 row/candidate reduction geometries, coverage-cache invalidation, and the actual Java evaluator with 257 rows and 251 candidates. They include interval boundaries, missing-value sentinels, NaN, infinities, empty active-row sets, and population growth beyond 256 candidates. Anytown P78 and P79 each produce 4,970 rules and byte-identical `rules.csv` files with SHA-256 `508e3390ae7b4e912b375ef03d4d9e4caad043e7d83abbe64f5662dc2ea46c4d`.

The initial thread-per-candidate backend was approximately three times slower than CPU. The optimized block-per-candidate backend uses exact shared-memory integer reductions, packed transfers, reusable result buffers, coverage caching, and a native allocation pool. In controlled sequential benchmarks with other miners paused, original CPU P78/P79 took 64.46/60.95 seconds and optimized GPU took 32.61/32.08 seconds. Two concurrent GPU states completed in 32.25/32.01 seconds. This is approximately 1.9–2.0× per-state speedup with byte-identical output.

Profiling is opt-in and adds substantial overhead; use it only for diagnosis. Java Flight Recorder measured the original support-count loop at 79.37% of execution samples. The optimized GPU profile identifies synchronous JNI and H2D/D2H transfers as the remaining bottleneck. Exact cached factorial arithmetic and disabled unused quality summaries preserve output. A separate packed-CPU loop was slower at 94.28 seconds and remains disabled. Full C-Town/L-Town GPU evaluation is separate from these benchmark results and must complete before any corresponding manuscript update.
