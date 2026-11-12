# QuantMiner provenance and license

Upstream: https://github.com/QuantMiner/QuantMiner (Java implementation), commit 587eea87c3603a2e0ef0e4b032422ffdedf5b645.

Copyright 2007 CCLS Columbia University (USA), LIFO University of Orleans (France), BRGM (France). Authors: Cyril Nortet, Xiangrong Kong, Ansaf Salleb-Aouissi, Christel Vrain, Daniel Cassard.

QuantMiner is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

QuantMiner is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html

This package contains the upstream Java sources (`upstream/src`), upstream `build.xml` and README, the two upstream jars, and a small `QuantMinerCli.java` adapter. The adapter invokes the official `RuleTester` and `OptimizerGeneticAlgo` classes headlessly; it does not reimplement or rename the mining algorithm. Local builds compile the included upstream sources directly, so Java 11 class-file versions in the prebuilt jar are not required.

## Compatibility correction

`AprioriQuantitative.CalculerEnsemblesItems` uses exact `BigInteger` combinatorics, and `RuleTester.m_iIndiceCombinaisonItemsQuant` uses a 64-bit index. The upstream 32-bit factorial arithmetic overflowed on full-width local-state inputs and raised `ArithmeticException: / by zero`. Where upstream arithmetic is already correct, this correction preserves the same combinations, support/confidence settings, optimization method and itemset semantics. It does not establish equivalence to an upstream full-width run or make the overall search tractable. Small-input tests exercise the official Java engine, and the original failure is retained in the local QuantMiner provenance before the corrected full-width runs. The local SAIN adaptation subsequently uses `max_attributes=2` for tractability; the failed three-attribute attempt is preserved separately and is not presented as the same mining configuration. Exact state-local constant channels are excluded from QuantMiner input as static variables but remain in detection as separately recorded point-interval invariants. Every nonconstant observable feature is retained, and all constants are named.

The Python adapter can schedule independent component-state runs across multiple processes. Each process invokes the same single-threaded Java engine with identical inputs and configuration; this scheduling adaptation does not parallelize or modify the Java algorithm itself. QuantMiner has no GPU backend.
