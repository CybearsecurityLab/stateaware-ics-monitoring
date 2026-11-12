package src.solver;

import java.util.Random;

public final class QuantMinerGpuSupportCountsTest {
    private static final float MISSING = -(Float.MAX_VALUE / 2.0f);

    private QuantMinerGpuSupportCountsTest() {}

    public static void main(String[] args) {
        String mode = args.length == 0 ? "gpu" : args[0];
        if (mode.equals("default-disabled")) {
            if (QuantMinerGpuSupportCounts.isRequested())
                throw new AssertionError("GPU must remain opt-in");
            QuantMinerGpuSupportCounts.initializeIfRequested();
            System.out.println("default-disabled: PASS (native library not required)");
            return;
        }
        if (mode.equals("missing-library")) {
            try {
                QuantMinerGpuSupportCounts.initializeIfRequested();
            }
            catch (UnsatisfiedLinkError expected) {
                if (!expected.getMessage().contains("explicitly requested"))
                    throw new AssertionError("Missing-library error is not explicit", expected);
                System.out.println("missing-library: PASS (loud failure)");
                return;
            }
            throw new AssertionError("A missing explicitly configured native library must fail");
        }
        if (!mode.equals("gpu"))
            throw new IllegalArgumentException("Unknown test mode: " + mode);
        if (!QuantMinerGpuSupportCounts.isRequested())
            throw new IllegalStateException("Set QUANTMINER_GPU=1 for the GPU fixture test");

        runBoundaryFixture();
        runRandomizedFixtures();
        if (QuantMinerGpuSupportCounts.supportsConfiguration(2, 1, 0, 0, 2, 2)
            || QuantMinerGpuSupportCounts.supportsConfiguration(1, 1, 1, 0, 2, 2))
            throw new AssertionError("Unsupported disjunction configurations must use the CPU loop");
        System.out.println("gpu-fixtures: PASS (301 fixtures, four coverage modes each)");
    }

    private static void runBoundaryFixture() {
        int rows = 6;
        int dimensions = 2;
        float[] values = {
            -1.0f, 0.0f, 1.0f, MISSING, Float.NaN, 2.0f,
            5.0f, 5.0f, 4.0f, 5.0f, 5.0f, 6.0f
        };
        int[] activeRows = {0, 1, 2, 3, 4, 5};
        byte[] rightEligible = {0, 0, 0, 0, 0, 0};
        byte[] leftCovered = {1, 0, 1, 0, 1, 0};
        byte[] rightCovered = {0, 1, 0, 1, 0, 1};
        float[] lower = {-1.0f, 5.0f, 0.0f, 4.0f};
        float[] upper = {1.0f, 5.0f, 0.0f, 4.0f};

        QuantMinerGpuSupportCounts.Context context = QuantMinerGpuSupportCounts.createContext(
            values, rows, dimensions, activeRows, activeRows.length, rightEligible);
        try {
            verifyAllModes(context, values, rows, dimensions, activeRows, rightEligible,
                           lower, upper, 2, 1, leftCovered, rightCovered);
        }
        finally {
            context.close();
        }
    }

    private static void runRandomizedFixtures() {
        Random random = new Random(0x51A7E5L);
        for (int fixtureIndex = 0; fixtureIndex < 300; fixtureIndex++) {
            int rows = fixtureIndex == 0 ? 0 : 1 + random.nextInt(96);
            int dimensions = 1 + random.nextInt(5);
            int leftDimensions = random.nextInt(dimensions + 1);
            int candidates = fixtureIndex == 1 ? 257 : 1 + random.nextInt(64);
            float[] values = new float[rows * dimensions];
            for (int index = 0; index < values.length; index++) {
                switch (random.nextInt(12)) {
                    case 0: values[index] = MISSING; break;
                    case 1: values[index] = -0.0f; break;
                    case 2: values[index] = Float.NaN; break;
                    case 3: values[index] = Float.POSITIVE_INFINITY; break;
                    case 4: values[index] = Float.NEGATIVE_INFINITY; break;
                    default: values[index] = random.nextInt(9) - 4; break;
                }
            }

            int[] shuffledRows = new int[rows];
            for (int row = 0; row < rows; row++)
                shuffledRows[row] = row;
            for (int row = rows - 1; row > 0; row--) {
                int other = random.nextInt(row + 1);
                int saved = shuffledRows[row];
                shuffledRows[row] = shuffledRows[other];
                shuffledRows[other] = saved;
            }
            int activeCount = fixtureIndex % 17 == 0 ? 0 : random.nextInt(rows + 1);
            int[] activeRows = new int[activeCount];
            System.arraycopy(shuffledRows, 0, activeRows, 0, activeCount);

            byte[] rightEligible = new byte[rows];
            byte[] leftCovered = new byte[rows];
            byte[] rightCovered = new byte[rows];
            for (int row = 0; row < rows; row++) {
                rightEligible[row] = fixtureIndex % 19 == 0 ? 0 : (byte)random.nextInt(2);
                leftCovered[row] = (byte)random.nextInt(2);
                rightCovered[row] = (byte)random.nextInt(2);
            }

            float[] lower = new float[candidates * dimensions];
            float[] upper = new float[candidates * dimensions];
            for (int candidate = 0; candidate < candidates; candidate++) {
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    float first = randomBound(random, values, rows, dimension);
                    float second = randomBound(random, values, rows, dimension);
                    int offset = candidate * dimensions + dimension;
                    lower[offset] = Math.min(first, second);
                    upper[offset] = Math.max(first, second);
                }
            }

            QuantMinerGpuSupportCounts.Context context = QuantMinerGpuSupportCounts.createContext(
                values, rows, dimensions, activeRows, activeRows.length, rightEligible);
            try {
                verifyAllModes(context, values, rows, dimensions, activeRows, rightEligible,
                               lower, upper, candidates, leftDimensions, leftCovered, rightCovered);
            }
            finally {
                context.close();
            }
        }
    }

    private static float randomBound(Random random, float[] values, int rows, int dimension) {
        if (rows > 0 && random.nextInt(4) != 0) {
            for (int attempt = 0; attempt < 20; attempt++) {
                float value = values[dimension * rows + random.nextInt(rows)];
                if (value != MISSING && !Float.isNaN(value))
                    return value;
            }
        }
        return random.nextInt(13) - 6;
    }

    private static void verifyAllModes(QuantMinerGpuSupportCounts.Context context,
                                       float[] values, int rows, int dimensions, int[] activeRows,
                                       byte[] rightEligible, float[] lower, float[] upper,
                                       int candidates, int leftDimensions, byte[] leftCovered,
                                       byte[] rightCovered) {
        for (int mode = 0; mode < 4; mode++) {
            boolean evaluateLeft = (mode & 1) == 0;
            boolean evaluateRight = (mode & 2) == 0;
            int[] actual = context.count(lower, upper, candidates, dimensions, leftDimensions,
                                         evaluateLeft, evaluateRight, leftCovered, rightCovered);
            int[] expected = cpuReference(values, rows, dimensions, activeRows, rightEligible,
                                          lower, upper, candidates, leftDimensions, evaluateLeft,
                                          evaluateRight, leftCovered, rightCovered);
            for (int candidate = 0; candidate < candidates; candidate++) {
                if (actual[candidate * 2] != expected[candidate * 2]
                    || actual[candidate * 2 + 1] != expected[candidate * 2 + 1]) {
                    throw new AssertionError("Support mismatch: candidate=" + candidate
                        + ", mode=" + mode + ", expected=" + expected[candidate * 2] + "/"
                        + expected[candidate * 2 + 1] + ", actual=" + actual[candidate * 2]
                        + "/" + actual[candidate * 2 + 1]);
                }
            }
        }
    }

    private static int[] cpuReference(float[] values, int rows, int dimensions, int[] activeRows,
                                      byte[] rightEligible, float[] lower, float[] upper,
                                      int candidates, int leftDimensions, boolean evaluateLeft,
                                      boolean evaluateRight, byte[] leftCovered, byte[] rightCovered) {
        int[] supports = new int[candidates * 2];
        for (int candidate = 0; candidate < candidates; candidate++) {
            int conditionSupport = 0;
            int ruleSupport = 0;
            for (int activeIndex = 0; activeIndex < activeRows.length; activeIndex++) {
                int row = activeRows[activeIndex];
                boolean conditionCovered = true;
                if (evaluateLeft) {
                    for (int dimension = 0; dimension < leftDimensions && conditionCovered; dimension++) {
                        float value = values[dimension * rows + row];
                        conditionCovered = value >= lower[candidate * dimensions + dimension]
                            && value <= upper[candidate * dimensions + dimension];
                    }
                }
                else {
                    conditionCovered = leftCovered[row] != 0;
                }
                if (!conditionCovered)
                    continue;
                conditionSupport++;

                boolean ruleCovered;
                if (evaluateRight) {
                    ruleCovered = rightEligible[row] != 0;
                    for (int dimension = leftDimensions; dimension < dimensions && ruleCovered; dimension++) {
                        float value = values[dimension * rows + row];
                        ruleCovered = value >= lower[candidate * dimensions + dimension]
                            && value <= upper[candidate * dimensions + dimension];
                    }
                }
                else {
                    ruleCovered = rightCovered[row] != 0;
                }
                if (ruleCovered)
                    ruleSupport++;
            }
            supports[candidate * 2] = conditionSupport;
            supports[candidate * 2 + 1] = ruleSupport;
        }
        return supports;
    }
}
