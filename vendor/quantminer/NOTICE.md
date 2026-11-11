# QuantMiner provenance and license

Upstream: https://github.com/QuantMiner/QuantMiner (Java implementation), commit 587eea87c3603a2e0ef0e4b032422ffdedf5b645.

Copyright 2007 CCLS Columbia University (USA), LIFO University of Orleans (France), BRGM (France). Authors: Cyril Nortet, Xiangrong Kong, Ansaf Salleb-Aouissi, Christel Vrain, Daniel Cassard.

QuantMiner is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

QuantMiner is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html

This package contains the upstream Java sources (`upstream/src`), upstream `build.xml` and README, the two upstream jars, and a small `QuantMinerCli.java` adapter. The adapter invokes the official `RuleTester` and `OptimizerGeneticAlgo` classes headlessly; it does not reimplement or rename the mining algorithm. Local builds compile the included upstream sources directly, so Java 11 class-file versions in the prebuilt jar are not required.
