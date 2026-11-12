/*
 * Copyright 2007, 2011 CCLS Columbia University (USA), LIFO University of Orleans (France), BRGM (France)
 *
 * This file is part of QuantMiner.
 *
 * QuantMiner is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
 *
 * QuantMiner is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with QuantMiner. If not, see <http://www.gnu.org/licenses/>.
 */
package src.solver;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** Exact, opt-in Java bitsets for QuantMiner's single-disjunction support counts. */
public final class QuantMinerBitsetSupportCounts {
    private static final String ENABLE_PROPERTY = "quantminer.bitset";
    private static final String ENABLE_ENVIRONMENT = "QUANTMINER_BITSET";
    private static final String LOCKSTEP_PROPERTY = "quantminer.lockstep";
    private static final String LOCKSTEP_ENVIRONMENT = "QUANTMINER_LOCKSTEP";
    private static final String PROFILE_PROPERTY = "quantminer.bitset.profile";
    private static final String PROFILE_ENVIRONMENT = "QUANTMINER_BITSET_PROFILE";

    private static long lockstepEvaluations = 0;
    private static long lockstepCandidatePairs = 0;
    private static long cacheHits = 0;
    private static long cacheMisses = 0;

    private QuantMinerBitsetSupportCounts() {}

    public static boolean isRequested() {
        String property = System.getProperty(ENABLE_PROPERTY);
        if (property != null)
            return isTrue(property);
        String environment = System.getenv(ENABLE_ENVIRONMENT);
        if (environment != null)
            return isTrue(environment);
        return true;
    }

    public static boolean isProfileRequested() {
        return isTrue(System.getProperty(PROFILE_PROPERTY))
            || isTrue(System.getenv(PROFILE_ENVIRONMENT));
    }

    public static synchronized void recordCacheStatistics(long hits, long misses) {
        cacheHits += hits;
        cacheMisses += misses;
    }

    public static synchronized void dumpProfile() {
        if (!isProfileRequested())
            return;
        long requests = cacheHits + cacheMisses;
        System.err.println("QUANTMINER_BITSET_CACHE requests=" + requests
            + " hits=" + cacheHits + " misses=" + cacheMisses
            + " hit_rate=" + (requests == 0 ? 0.0 : (double)cacheHits / requests));
    }

    public static boolean isLockstepRequested() {
        return isTrue(System.getProperty(LOCKSTEP_PROPERTY))
            || isTrue(System.getenv(LOCKSTEP_ENVIRONMENT));
    }

    public static synchronized void recordLockstepEvaluation(int candidateCount) {
        lockstepEvaluations++;
        lockstepCandidatePairs += candidateCount;
    }

    public static synchronized long getLockstepEvaluationCount() {
        return lockstepEvaluations;
    }

    public static synchronized long getLockstepCandidatePairCount() {
        return lockstepCandidatePairs;
    }

    private static boolean isTrue(String value) {
        return value != null
            && (value.equalsIgnoreCase("true") || value.equals("1")
                || value.equalsIgnoreCase("yes") || value.equalsIgnoreCase("on"));
    }

    public static boolean supportsConfiguration(int leftDisjunctions, int rightDisjunctions,
                                                 int currentLeftDisjunction, int currentRightDisjunction,
                                                 int dimensions, int intervals) {
        return isRequested() && leftDisjunctions == 1 && rightDisjunctions == 1
            && currentLeftDisjunction == 0 && currentRightDisjunction == 0
            && dimensions > 0 && intervals == dimensions;
    }

    public static Context createContext(float[] dimensionMajorValues, int rowCount, int dimensions,
                                        int[] activeRows, int activeRowCount, byte[] rightEligible) {
        if (dimensionMajorValues == null || rowCount < 0 || dimensions <= 0
            || dimensionMajorValues.length != rowCount * dimensions)
            throw new IllegalArgumentException("Invalid dimension-major QuantMiner bitset value buffer");
        if (activeRows == null || activeRowCount < 0 || activeRowCount > activeRows.length)
            throw new IllegalArgumentException("Invalid QuantMiner bitset active-row buffer");
        if (rightEligible == null || rightEligible.length != rowCount)
            throw new IllegalArgumentException("Invalid QuantMiner bitset right-coverage buffer");
        return new Context(dimensionMajorValues, rowCount, dimensions,
                           activeRows, activeRowCount, rightEligible);
    }

    public static final class Context {
        private final int rowCount;
        private final int dimensions;
        private final int words;
        private final float[][] observedValues;
        private final long[][] greaterEqual;
        private final long[][] lessEqual;
        private final long[] activeMask;
        private final long[] rightEligibleMask;
        private final long[] conditionMask;
        private final long[] ruleMask;
        private final Map<RequestKey, int[]> supportCache = new HashMap<>();
        private long cacheHits = 0;
        private long cacheMisses = 0;

        private static final class RequestKey {
            private final float[] bounds;
            private final int leftDimensions;
            private final boolean evaluateLeft;
            private final boolean evaluateRight;
            private final byte[] leftCovered;
            private final byte[] rightCovered;
            private final int hashCode;

            private RequestKey(float[] packedBounds, int candidate, int dimensions,
                               int leftDimensions, boolean evaluateLeft, boolean evaluateRight,
                               byte[] leftCovered, byte[] rightCovered) {
                this.bounds = Arrays.copyOfRange(
                    packedBounds, candidate * dimensions * 2,
                    (candidate + 1) * dimensions * 2);
                this.leftDimensions = leftDimensions;
                this.evaluateLeft = evaluateLeft;
                this.evaluateRight = evaluateRight;
                this.leftCovered = leftCovered.clone();
                this.rightCovered = rightCovered.clone();
                this.hashCode = Arrays.hashCode(this.bounds)
                    + 31 * (this.leftDimensions
                        + 31 * (Boolean.hashCode(this.evaluateLeft)
                            + 31 * (Boolean.hashCode(this.evaluateRight)
                                + 31 * (Arrays.hashCode(this.leftCovered)
                                    + 31 * Arrays.hashCode(this.rightCovered)))));
            }

            @Override
            public int hashCode() {
                return hashCode;
            }

            @Override
            public boolean equals(Object other) {
                if (!(other instanceof RequestKey))
                    return false;
                RequestKey key = (RequestKey)other;
                return leftDimensions == key.leftDimensions
                    && evaluateLeft == key.evaluateLeft
                    && evaluateRight == key.evaluateRight
                    && Arrays.equals(bounds, key.bounds)
                    && Arrays.equals(leftCovered, key.leftCovered)
                    && Arrays.equals(rightCovered, key.rightCovered);
            }
        }

        private Context(float[] values, int rowCount, int dimensions,
                        int[] activeRows, int activeRowCount, byte[] rightEligible) {
            this.rowCount = rowCount;
            this.dimensions = dimensions;
            this.words = (rowCount + 63) >>> 6;
            this.observedValues = new float[dimensions][];
            this.greaterEqual = new long[dimensions][];
            this.lessEqual = new long[dimensions][];
            this.activeMask = new long[words];
            this.rightEligibleMask = new long[words];
            this.conditionMask = new long[words];
            this.ruleMask = new long[words];

            boolean[] seen = new boolean[rowCount];
            for (int index = 0; index < activeRowCount; index++) {
                int row = activeRows[index];
                if (row < 0 || row >= rowCount || seen[row])
                    throw new IllegalArgumentException("QuantMiner bitset active rows must be unique and in range");
                seen[row] = true;
                set(activeMask, row);
                if (rightEligible[row] != 0)
                    set(rightEligibleMask, row);
            }
            for (int dimension = 0; dimension < dimensions; dimension++)
                buildDimension(values, dimension);
        }

        private void buildDimension(float[] values, int dimension) {
            float[] sorted = new float[rowCount];
            System.arraycopy(values, dimension * rowCount, sorted, 0, rowCount);
            Arrays.sort(sorted);

            int uniqueCount = 0;
            for (int row = 0; row < rowCount; row++) {
                float value = sorted[row];
                if (Float.isNaN(value))
                    continue;
                if (uniqueCount == 0 || value != sorted[uniqueCount - 1])
                    sorted[uniqueCount++] = value;
            }
            observedValues[dimension] = Arrays.copyOf(sorted, uniqueCount);
            greaterEqual[dimension] = new long[uniqueCount * words];
            lessEqual[dimension] = new long[uniqueCount * words];

            for (int valueIndex = 0; valueIndex < uniqueCount; valueIndex++) {
                float bound = observedValues[dimension][valueIndex];
                int maskOffset = valueIndex * words;
                for (int row = 0; row < rowCount; row++) {
                    float value = values[dimension * rowCount + row];
                    if (value >= bound)
                        set(greaterEqual[dimension], maskOffset, row);
                    if (value <= bound)
                        set(lessEqual[dimension], maskOffset, row);
                }
            }
        }

        public synchronized void count(float[] packedBounds, int candidateCount,
                                       int intervals, int leftDimensions, boolean evaluateLeft,
                                       boolean evaluateRight, byte[] leftCovered, byte[] rightCovered,
                                       int[] result) {
            if (candidateCount < 0 || intervals != dimensions || leftDimensions < 0
                || leftDimensions > dimensions)
                throw new IllegalArgumentException("Invalid QuantMiner bitset population geometry");
            if (packedBounds == null || packedBounds.length < candidateCount * intervals * 2)
                throw new IllegalArgumentException("Invalid QuantMiner bitset packed interval buffer");
            if (leftCovered == null || rightCovered == null
                || leftCovered.length != rowCount || rightCovered.length != rowCount)
                throw new IllegalArgumentException("Invalid QuantMiner bitset coverage buffers");
            if (result == null || result.length < candidateCount * 2)
                throw new IllegalArgumentException("Invalid QuantMiner bitset result buffer");

            for (int candidate = 0; candidate < candidateCount; candidate++) {
                RequestKey key = new RequestKey(
                    packedBounds, candidate, dimensions, leftDimensions,
                    evaluateLeft, evaluateRight, leftCovered, rightCovered);
                int[] cached = supportCache.get(key);
                if (cached != null) {
                    System.arraycopy(cached, 0, result, candidate * 2, 2);
                    cacheHits++;
                    continue;
                }
                cacheMisses++;
                System.arraycopy(activeMask, 0, conditionMask, 0, words);
                if (evaluateLeft) {
                    for (int dimension = 0; dimension < leftDimensions; dimension++) {
                        int boundOffset = (candidate * dimensions + dimension) * 2;
                        intersectInterval(conditionMask, dimension,
                                          packedBounds[boundOffset], packedBounds[boundOffset + 1]);
                    }
                }
                else {
                    andCoverage(conditionMask, leftCovered);
                }
                result[candidate * 2] = cardinality(conditionMask);

                if (evaluateRight) {
                    for (int word = 0; word < words; word++)
                        ruleMask[word] = conditionMask[word] & rightEligibleMask[word];
                    for (int dimension = leftDimensions; dimension < dimensions; dimension++) {
                        int boundOffset = (candidate * dimensions + dimension) * 2;
                        intersectInterval(ruleMask, dimension,
                                          packedBounds[boundOffset], packedBounds[boundOffset + 1]);
                    }
                }
                else {
                    System.arraycopy(conditionMask, 0, ruleMask, 0, words);
                    andCoverage(ruleMask, rightCovered);
                }
                result[candidate * 2 + 1] = cardinality(ruleMask);
                supportCache.put(key, new int[] {
                    result[candidate * 2], result[candidate * 2 + 1]
                });
            }
        }

        public void close() {
            QuantMinerBitsetSupportCounts.recordCacheStatistics(cacheHits, cacheMisses);
        }

        long cacheHits() {
            return cacheHits;
        }

        long cacheMisses() {
            return cacheMisses;
        }

        private void intersectInterval(long[] target, int dimension, float lower, float upper) {
            int lowerIndex = firstGreaterEqual(observedValues[dimension], lower);
            int upperIndex = lastLessEqual(observedValues[dimension], upper);
            if (lowerIndex == observedValues[dimension].length || upperIndex < 0) {
                Arrays.fill(target, 0L);
                return;
            }
            int lowerOffset = lowerIndex * words;
            int upperOffset = upperIndex * words;
            for (int word = 0; word < words; word++) {
                target[word] &= greaterEqual[dimension][lowerOffset + word]
                    & lessEqual[dimension][upperOffset + word];
            }
        }

        private void andCoverage(long[] target, byte[] coverage) {
            for (int row = 0; row < rowCount; row++) {
                if (coverage[row] == 0)
                    target[row >>> 6] &= ~(1L << row);
            }
        }

        private int cardinality(long[] mask) {
            int result = 0;
            for (int word = 0; word < words; word++)
                result += Long.bitCount(mask[word]);
            return result;
        }

        private static int firstGreaterEqual(float[] values, float bound) {
            if (Float.isNaN(bound))
                return values.length;
            int low = 0;
            int high = values.length;
            while (low < high) {
                int middle = (low + high) >>> 1;
                if (values[middle] >= bound)
                    high = middle;
                else
                    low = middle + 1;
            }
            return low;
        }

        private static int lastLessEqual(float[] values, float bound) {
            if (Float.isNaN(bound))
                return -1;
            int low = 0;
            int high = values.length;
            while (low < high) {
                int middle = (low + high) >>> 1;
                if (values[middle] <= bound)
                    low = middle + 1;
                else
                    high = middle;
            }
            return low - 1;
        }

        private static void set(long[] mask, int row) {
            set(mask, 0, row);
        }

        private static void set(long[] mask, int offset, int row) {
            mask[offset + (row >>> 6)] |= 1L << row;
        }
    }
}
