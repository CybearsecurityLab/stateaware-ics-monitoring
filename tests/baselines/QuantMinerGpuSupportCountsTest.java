package src.solver;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import src.apriori.AssociationRule;
import src.apriori.AttributQuantitative;
import src.apriori.ItemQuantitative;
import src.database.DataColumn;
import src.database.DatabaseAdmin;

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
        if (mode.equals("packed-cpu")) {
            System.setProperty("quantminer.cpu.packed", "true");
            runActualEvaluatorFixture();
            System.out.println("packed-cpu-evaluator: PASS (257 rows, 251 candidates)");
            return;
        }
        if (!mode.equals("gpu"))
            throw new IllegalArgumentException("Unknown test mode: " + mode);
        if (!QuantMinerGpuSupportCounts.isRequested())
            throw new IllegalStateException("Set QUANTMINER_GPU=1 for the GPU fixture test");

        runBoundaryFixture();
        runRandomizedFixtures();
        System.out.println("gpu-fixtures: PASS (301 fixtures, four coverage modes each)");
        runReductionGeometryFixtures();
        System.out.println("reduction-geometries: PASS (15 geometries, four coverage modes each)");
        runCoverageCacheFixture();
        System.out.println("coverage-cache-invalidation: PASS");
        runActualEvaluatorFixture();
        System.out.println("actual-java-evaluator: PASS (257 rows, 251 candidates)");
        if (QuantMinerGpuSupportCounts.supportsConfiguration(2, 1, 0, 0, 2, 2)
            || QuantMinerGpuSupportCounts.supportsConfiguration(1, 1, 1, 0, 2, 2))
            throw new AssertionError("Unsupported disjunction configurations must use the CPU loop");
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
        float[] packedBounds = packBounds(lower, upper, candidates, dimensions);
        int[] actual = new int[candidates * 2];
        for (int mode = 0; mode < 4; mode++) {
            boolean evaluateLeft = (mode & 1) == 0;
            boolean evaluateRight = (mode & 2) == 0;
            context.count(packedBounds, candidates, dimensions, leftDimensions,
                          evaluateLeft, evaluateRight, leftCovered, rightCovered, actual);
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

    private static void runReductionGeometryFixtures() {
        int[] rowGeometries = {1, 255, 256, 257, 1024};
        int[] candidateGeometries = {1, 250, 251};
        for (int rows : rowGeometries) {
            for (int candidates : candidateGeometries) {
                int dimensions = 2;
                float[] values = new float[rows * dimensions];
                int[] activeRows = new int[rows];
                byte[] rightEligible = new byte[rows];
                byte[] leftCovered = new byte[rows];
                byte[] rightCovered = new byte[rows];
                for (int row = 0; row < rows; row++) {
                    activeRows[row] = row;
                    values[row] = row % 37 == 0 ? MISSING : (row % 23) - 11;
                    values[rows + row] = row % 41 == 0 ? Float.NaN : (row % 19) - 9;
                    rightEligible[row] = (byte)(row % 5 == 0 ? 0 : 1);
                    leftCovered[row] = (byte)(row % 3 == 0 ? 0 : 1);
                    rightCovered[row] = (byte)(row % 7 == 0 ? 0 : 1);
                }
                float[] lower = new float[candidates * dimensions];
                float[] upper = new float[candidates * dimensions];
                for (int candidate = 0; candidate < candidates; candidate++) {
                    lower[candidate * 2] = (candidate % 17) - 10;
                    upper[candidate * 2] = lower[candidate * 2] + candidate % 13;
                    lower[candidate * 2 + 1] = (candidate % 11) - 8;
                    upper[candidate * 2 + 1] = lower[candidate * 2 + 1] + candidate % 7;
                }
                QuantMinerGpuSupportCounts.Context context = QuantMinerGpuSupportCounts.createContext(
                    values, rows, dimensions, activeRows, rows, rightEligible);
                try {
                    verifyAllModes(context, values, rows, dimensions, activeRows, rightEligible,
                                   lower, upper, candidates, 1, leftCovered, rightCovered);
                }
                finally {
                    context.close();
                }
            }
        }
    }

    private static void runCoverageCacheFixture() {
        int rows = 257;
        int dimensions = 1;
        float[] values = new float[rows];
        int[] activeRows = new int[rows];
        byte[] rightEligible = new byte[rows];
        byte[] leftCovered = new byte[rows];
        byte[] rightCovered = new byte[rows];
        for (int row = 0; row < rows; row++) {
            values[row] = 0.0f;
            activeRows[row] = row;
            rightEligible[row] = 1;
            leftCovered[row] = 1;
            rightCovered[row] = 1;
        }
        float[] lower = {0.0f};
        float[] upper = {0.0f};
        float[] packed = {0.0f, 0.0f};
        int[] actual = new int[2];
        QuantMinerGpuSupportCounts.Context context = QuantMinerGpuSupportCounts.createContext(
            values, rows, dimensions, activeRows, rows, rightEligible);
        try {
            for (int iteration = 0; iteration < 2; iteration++) {
                context.count(packed, 1, 1, 0, false, false,
                              leftCovered, rightCovered, actual);
                assertSupports(actual, rows, rows, "unchanged coverage");
            }
            leftCovered[256] = 0;
            rightCovered[255] = 0;
            context.count(packed, 1, 1, 0, false, false,
                          leftCovered, rightCovered, actual);
            assertSupports(actual, rows - 1, rows - 2, "changed coverage");

            int[] expected = cpuReference(values, rows, dimensions, activeRows, rightEligible,
                                          lower, upper, 1, 0, false, false,
                                          leftCovered, rightCovered);
            assertSupports(actual, expected[0], expected[1], "coverage reference");
        }
        finally {
            context.close();
        }
    }

    private static void assertSupports(int[] actual, int condition, int rule, String label) {
        if (actual[0] != condition || actual[1] != rule)
            throw new AssertionError(label + ": expected=" + condition + "/" + rule
                + ", actual=" + actual[0] + "/" + actual[1]);
    }

    private static void runActualEvaluatorFixture() {
        try {
            int rows = 257;
            int candidates = 251;
            StringBuilder csv = new StringBuilder("x,y\n");
            for (int row = 0; row < rows; row++) {
                if (row % 29 != 0)
                    csv.append((row % 23) - 11);
                csv.append(',');
                if (row % 31 != 0)
                    csv.append((row % 19) - 9);
                csv.append('\n');
            }
            Path input = Files.createTempFile("quantminer-gpu-evaluator-", ".csv");
            try {
                Files.write(input, csv.toString().getBytes(StandardCharsets.UTF_8));
                DatabaseAdmin database = new DatabaseAdmin(input.toString(), "csv");
                if (!database.EstBaseDeDonneesValide())
                    throw new AssertionError("Could not load actual-evaluator fixture");
                database.PrendreEnCompteToutesLesColonnes();
                database.ChargerDonneesPrisesEnCompte();
                DataColumn x = database.ObtenirColonneBDPriseEnCompte(0);
                DataColumn y = database.ObtenirColonneBDPriseEnCompte(1);
                AssociationRule schema = new AssociationRule(1, 1, 1, 1);
                schema.AssignerItemGauche(new ItemQuantitative(
                    new AttributQuantitative("x", x), 0.0f, 0.0f), 0);
                schema.AssignerItemDroite(new ItemQuantitative(
                    new AttributQuantitative("y", y), 0.0f, 0.0f), 0);

                ActualEvaluator evaluator = new ActualEvaluator(candidates, database);
                evaluator.SpecifierSchemaRegle(schema);
                float[] lower = new float[candidates * 2];
                float[] upper = new float[candidates * 2];
                for (int candidate = 0; candidate < candidates; candidate++) {
                    lower[candidate * 2] = (candidate % 17) - 10;
                    upper[candidate * 2] = lower[candidate * 2] + candidate % 13;
                    lower[candidate * 2 + 1] = (candidate % 11) - 8;
                    upper[candidate * 2 + 1] = lower[candidate * 2 + 1] + candidate % 7;
                }
                EvaluationBaseAlgorithm.ReglePotentielle[] rules = evaluator.evaluate(lower, upper);
                float[] values = new float[rows * 2];
                System.arraycopy(x.m_tValeurReelle, 0, values, 0, rows);
                System.arraycopy(y.m_tValeurReelle, 0, values, rows, rows);
                int[] expected = cpuReference(values, rows, 2, evaluator.activeRows(),
                                              evaluator.rightEligible(), lower, upper, candidates,
                                              1, true, true, new byte[rows], new byte[rows]);
                for (int candidate = 0; candidate < candidates; candidate++) {
                    if (rules[candidate].m_iSupportCond != expected[candidate * 2]
                        || rules[candidate].m_iSupportRegle != expected[candidate * 2 + 1]) {
                        throw new AssertionError("Actual evaluator mismatch at candidate " + candidate);
                    }
                }
            }
            finally {
                Files.deleteIfExists(input);
            }
        }
        catch (java.io.IOException error) {
            throw new RuntimeException(error);
        }
    }

    private static final class ActualEvaluator extends EvaluationBaseAlgorithm {
        ActualEvaluator(int population, DatabaseAdmin database) {
            super(population, database);
        }

        @Override
        public void EvaluerQualiteReglePotentielle(ReglePotentielle rule) {
            rule.m_fQualite = rule.m_iSupportRegle;
        }

        EvaluationBaseAlgorithm.ReglePotentielle[] evaluate(float[] lower, float[] upper) {
            int candidates = lower.length / 2;
            EvaluationBaseAlgorithm.ReglePotentielle[] evaluated =
                new EvaluationBaseAlgorithm.ReglePotentielle[candidates];
            for (int candidate = 0; candidate < candidates; candidate++) {
                ReglePotentielle rule = new ReglePotentielle(2, 2);
                for (int dimension = 0; dimension < 2; dimension++) {
                    rule.m_tIntervalleMin[dimension] = lower[candidate * 2 + dimension];
                    rule.m_tIntervalleMax[dimension] = upper[candidate * 2 + dimension];
                }
                m_tReglesPotentielles[candidate] = rule;
                m_tReglesPotentiellesAEvaluer[candidate] = rule;
                evaluated[candidate] = rule;
            }
            m_meilleureReglePotentielle[0] = new ReglePotentielle(2, 2);
            m_iNombreReglesPotentiellesAEvaluer = candidates;
            EvaluerReglesPotentielles();
            return evaluated;
        }

        int[] activeRows() {
            return java.util.Arrays.copyOf(m_tLignesAPrendreEnCompte,
                                           m_iNombreLignesAPrendreEnCompte);
        }

        byte[] rightEligible() {
            byte[] result = new byte[m_iNombreTransactions];
            for (int row = 0; row < result.length; row++)
                result[row] = m_tPrendreEnCompteLigneDroite[row] ? (byte)1 : (byte)0;
            return result;
        }
    }

    private static float[] packBounds(float[] lower, float[] upper, int candidates, int dimensions) {
        float[] packed = new float[candidates * dimensions * 2];
        for (int candidate = 0; candidate < candidates; candidate++) {
            for (int dimension = 0; dimension < dimensions; dimension++) {
                int source = candidate * dimensions + dimension;
                int destination = source * 2;
                packed[destination] = lower[source];
                packed[destination + 1] = upper[source];
            }
        }
        return packed;
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
