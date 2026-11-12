package src.solver;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import src.apriori.AssociationRule;
import src.apriori.AttributQuantitative;
import src.apriori.ItemQuantitative;
import src.database.DataColumn;
import src.database.DatabaseAdmin;

public final class QuantMinerBitsetSupportCountsTest {
    private static final float MISSING = -(Float.MAX_VALUE / 2.0f);

    private QuantMinerBitsetSupportCountsTest() {}

    public static void main(String[] args) throws Exception {
        String mode = args.length == 0 ? "all" : args[0];
        if (mode.equals("default-enabled")) {
            if (!QuantMinerBitsetSupportCounts.isRequested())
                throw new AssertionError("Bitset support must be enabled by default");
            if (QuantMinerBitsetSupportCounts.isLockstepRequested())
                throw new AssertionError("Lockstep must remain opt-in");
            System.out.println("bitset-default-enabled: PASS");
            return;
        }
        if (mode.equals("opt-out")) {
            System.setProperty("quantminer.bitset", "false");
            if (QuantMinerBitsetSupportCounts.isRequested())
                throw new AssertionError("Bitset opt-out must disable the default path");
            System.clearProperty("quantminer.bitset");
            System.out.println("bitset-opt-out: PASS");
            return;
        }
        if (mode.equals("lockstep-mismatch")) {
            validateLockstepMismatchDetection();
            System.out.println("lockstep-mismatch-detection: PASS");
            return;
        }
        if (mode.equals("benchmark")) {
            if (args.length < 2 || args.length > 3)
                throw new IllegalArgumentException("benchmark INPUT.csv [REPETITIONS]");
            runActualCsvBenchmark(args[1], args.length == 3 ? Integer.parseInt(args[2]) : 20);
            return;
        }
        if (!mode.equals("all"))
            throw new IllegalArgumentException("Unknown test mode: " + mode);
        if (!QuantMinerBitsetSupportCounts.isRequested())
            throw new IllegalStateException("Set QUANTMINER_BITSET=1 or -Dquantminer.bitset=true");

        runBoundaryFixture();
        runDuplicateCacheFixture();
        System.out.println("bitset-duplicate-cache: PASS");
        runRandomizedFixtures();
        System.out.println("bitset-randomized: PASS (301 fixtures, four coverage modes each)");
        runReductionGeometryFixtures();
        System.out.println("bitset-geometries: PASS (rows 1/255/256/257/1024; candidates 1/250/251)");
        runCoverageCacheFixture();
        System.out.println("bitset-coverage-cache: PASS");
        runActualEvaluatorFixtures();
        System.out.println("bitset-actual-evaluator: PASS (257 rows, 251 candidates, four modes)");
        runActualEvaluatorLockstep();
        System.out.println("bitset-actual-lockstep: PASS (251 candidate pairs, original supports retained)");
        runUnsupportedConfigurationFallback();
        System.out.println("bitset-unsupported-fallback: PASS");
        validateLockstepMismatchDetection();
        System.out.println("lockstep-mismatch-detection: PASS");
    }

    private static void runBoundaryFixture() {
        int rows = 8;
        int dimensions = 2;
        float[] values = {
            -1.0f, -0.0f, 0.0f, 1.0f, MISSING, Float.NaN, 2.0f, 2.0f,
            5.0f, 5.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f, 4.0f
        };
        int[] activeRows = {0, 1, 2, 3, 4, 5, 6, 7};
        byte[] rightEligible = {1, 1, 1, 0, 1, 1, 1, 0};
        byte[] leftCovered = {1, 0, 1, 0, 1, 0, 1, 1};
        byte[] rightCovered = {0, 1, 0, 1, 0, 1, 1, 1};
        float[] lower = {-1.0f, -0.0f, Float.NaN, 4.0f};
        float[] upper = {0.0f, 5.0f, 5.0f, 4.0f};
        QuantMinerBitsetSupportCounts.Context context = QuantMinerBitsetSupportCounts.createContext(
            values, rows, dimensions, activeRows, activeRows.length, rightEligible);
        verifyAllModes(context, values, rows, dimensions, activeRows, rightEligible,
                       lower, upper, 2, 1, leftCovered, rightCovered);
    }

    private static void runDuplicateCacheFixture() {
        int rows = 2;
        int dimensions = 1;
        float[] values = {0.0f, 1.0f};
        int[] activeRows = {0, 1};
        byte[] rightEligible = {1, 1};
        byte[] leftCovered = {1, 1};
        byte[] rightCovered = {1, 1};
        QuantMinerBitsetSupportCounts.Context context = QuantMinerBitsetSupportCounts.createContext(
            values, rows, dimensions, activeRows, activeRows.length, rightEligible);
        for (int mode = 0; mode < 4; mode++) {
            boolean evaluateLeft = (mode & 1) == 0;
            boolean evaluateRight = (mode & 2) == 0;
            int[] first = new int[2];
            int[] second = new int[2];
            context.count(new float[] {0.0f, 1.0f}, 1, dimensions, dimensions,
                          evaluateLeft, evaluateRight, leftCovered, rightCovered, first);
            context.count(new float[] {0.0f, 1.0f}, 1, dimensions, dimensions,
                          evaluateLeft, evaluateRight, leftCovered, rightCovered, second);
            if (!Arrays.equals(first, second))
                throw new AssertionError("Duplicate cache changed support counts");
        }
        if (context.cacheHits() != 4 || context.cacheMisses() != 4)
            throw new AssertionError("Unexpected duplicate-cache counters: hits="
                + context.cacheHits() + ", misses=" + context.cacheMisses());
        context.close();
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
                    case 2: values[index] = 0.0f; break;
                    case 3: values[index] = Float.NaN; break;
                    case 4: values[index] = Float.POSITIVE_INFINITY; break;
                    case 5: values[index] = Float.NEGATIVE_INFINITY; break;
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
            int[] activeRows = Arrays.copyOf(shuffledRows, activeCount);
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
                    float first = observedBound(random, values, rows, dimension);
                    float second = observedBound(random, values, rows, dimension);
                    int offset = candidate * dimensions + dimension;
                    lower[offset] = Math.min(first, second);
                    upper[offset] = Math.max(first, second);
                }
            }
            QuantMinerBitsetSupportCounts.Context context = QuantMinerBitsetSupportCounts.createContext(
                values, rows, dimensions, activeRows, activeRows.length, rightEligible);
            verifyAllModes(context, values, rows, dimensions, activeRows, rightEligible,
                           lower, upper, candidates, leftDimensions, leftCovered, rightCovered);
        }
    }

    private static float observedBound(Random random, float[] values, int rows, int dimension) {
        return rows == 0 ? 0.0f : values[dimension * rows + random.nextInt(rows)];
    }

    private static void verifyAllModes(QuantMinerBitsetSupportCounts.Context context,
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
            assertSupports(actual, expected, candidates, "standalone mode=" + mode);
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
                    if (row % 37 == 0)
                        values[row] = MISSING;
                    else if (row % 31 == 0)
                        values[row] = -0.0f;
                    else
                        values[row] = (row % 23) - 11;
                    values[rows + row] = row % 41 == 0 ? Float.NaN : (row % 19) - 9;
                    rightEligible[row] = (byte)(row % 5 == 0 ? 0 : 1);
                    leftCovered[row] = (byte)(row % 3 == 0 ? 0 : 1);
                    rightCovered[row] = (byte)(row % 7 == 0 ? 0 : 1);
                }
                float[] lower = new float[candidates * dimensions];
                float[] upper = new float[candidates * dimensions];
                for (int candidate = 0; candidate < candidates; candidate++) {
                    for (int dimension = 0; dimension < dimensions; dimension++) {
                        float first = values[dimension * rows + (candidate * 17 + dimension) % rows];
                        float second = values[dimension * rows + (candidate * 29 + 11) % rows];
                        int offset = candidate * dimensions + dimension;
                        lower[offset] = Math.min(first, second);
                        upper[offset] = Math.max(first, second);
                    }
                }
                QuantMinerBitsetSupportCounts.Context context = QuantMinerBitsetSupportCounts.createContext(
                    values, rows, dimensions, activeRows, rows, rightEligible);
                verifyAllModes(context, values, rows, dimensions, activeRows, rightEligible,
                               lower, upper, candidates, 1, leftCovered, rightCovered);
            }
        }
    }

    private static void runCoverageCacheFixture() {
        int rows = 257;
        float[] values = new float[rows];
        int[] activeRows = new int[rows];
        byte[] rightEligible = new byte[rows];
        byte[] leftCovered = new byte[rows];
        byte[] rightCovered = new byte[rows];
        Arrays.fill(rightEligible, (byte)1);
        Arrays.fill(leftCovered, (byte)1);
        Arrays.fill(rightCovered, (byte)1);
        for (int row = 0; row < rows; row++)
            activeRows[row] = row;
        float[] packed = {-0.0f, 0.0f};
        int[] actual = new int[2];
        QuantMinerBitsetSupportCounts.Context context = QuantMinerBitsetSupportCounts.createContext(
            values, rows, 1, activeRows, rows, rightEligible);
        for (int iteration = 0; iteration < 2; iteration++) {
            context.count(packed, 1, 1, 0, false, false, leftCovered, rightCovered, actual);
            assertSupports(actual, new int[]{rows, rows}, 1, "unchanged coverage");
        }
        leftCovered[256] = 0;
        rightCovered[255] = 0;
        context.count(packed, 1, 1, 0, false, false, leftCovered, rightCovered, actual);
        assertSupports(actual, new int[]{rows - 1, rows - 2}, 1, "changed coverage");
    }

    private static void runActualEvaluatorFixtures() throws Exception {
        for (int mode = 0; mode < 4; mode++) {
            ActualFixture fixture = new ActualFixture(257, 251);
            try {
                fixture.setCoverageMode(mode);
                EvaluationBaseAlgorithm.ReglePotentielle[] rules = fixture.evaluator.evaluate(
                    fixture.lower, fixture.upper);
                int[] expected = fixture.expected(mode);
                assertRuleSupports(rules, expected, 251, "actual evaluator mode=" + mode);
            }
            finally {
                fixture.close();
            }
        }
    }

    private static void runActualEvaluatorLockstep() throws Exception {
        System.setProperty("quantminer.lockstep", "true");
        try {
            ActualFixture fixture = new ActualFixture(257, 251);
            try {
                long evaluationsBefore = QuantMinerBitsetSupportCounts.getLockstepEvaluationCount();
                long pairsBefore = QuantMinerBitsetSupportCounts.getLockstepCandidatePairCount();
                EvaluationBaseAlgorithm.ReglePotentielle[] rules = fixture.evaluator.evaluate(
                    fixture.lower, fixture.upper);
                assertRuleSupports(rules, fixture.expected(0), 251, "actual lockstep");
                if (QuantMinerBitsetSupportCounts.getLockstepEvaluationCount() != evaluationsBefore + 1
                    || QuantMinerBitsetSupportCounts.getLockstepCandidatePairCount() != pairsBefore + 251)
                    throw new AssertionError("Actual evaluator did not execute every lockstep candidate pair");
            }
            finally {
                fixture.close();
            }
        }
        finally {
            System.clearProperty("quantminer.lockstep");
        }
    }

    private static void runUnsupportedConfigurationFallback() throws Exception {
        if (QuantMinerBitsetSupportCounts.supportsConfiguration(2, 1, 0, 0, 2, 2)
            || QuantMinerBitsetSupportCounts.supportsConfiguration(1, 1, 1, 0, 2, 2))
            throw new AssertionError("Unsupported disjunctions must use the original CPU loop");
        ActualFixture fixture = new ActualFixture(32, 7);
        try {
            fixture.evaluator.forceUnsupportedDisjunctionGeometry();
            EvaluationBaseAlgorithm.ReglePotentielle[] rules = fixture.evaluator.evaluate(
                fixture.lower, fixture.upper);
            assertRuleSupports(rules, fixture.expected(0), 7, "unsupported fallback");
        }
        finally {
            fixture.close();
        }
    }

    private static void validateLockstepMismatchDetection() {
        EvaluationBaseAlgorithm.assertLockstepSupports(
            new int[]{4, 3, 8, 7}, new int[]{4, 3, 8, 7}, 2);
        try {
            EvaluationBaseAlgorithm.assertLockstepSupports(
                new int[]{4, 3}, new int[]{4, 2}, 1);
        }
        catch (AssertionError expected) {
            if (!expected.getMessage().contains("candidate 0")
                || !expected.getMessage().contains("optimized=4/3")
                || !expected.getMessage().contains("original=4/2"))
                throw new AssertionError("Lockstep mismatch lacks exact support diagnostics", expected);
            return;
        }
        throw new AssertionError("Lockstep silently accepted a support mismatch");
    }

    private static void runActualCsvBenchmark(String input, int repetitions) throws Exception {
        if (repetitions <= 0)
            throw new IllegalArgumentException("Benchmark repetitions must be positive");
        System.clearProperty("quantminer.bitset");
        System.clearProperty("quantminer.lockstep");
        ActualFixture fixture = new ActualFixture(input, 250);
        try {
            for (int iteration = 0; iteration < 3; iteration++)
                fixture.evaluator.evaluate(fixture.lower, fixture.upper);
            long originalStart = System.nanoTime();
            for (int iteration = 0; iteration < repetitions; iteration++)
                fixture.evaluator.evaluate(fixture.lower, fixture.upper);
            long originalNanos = System.nanoTime() - originalStart;

            System.setProperty("quantminer.bitset", "true");
            for (int iteration = 0; iteration < 3; iteration++)
                fixture.evaluator.evaluate(fixture.lower, fixture.upper);
            long bitsetStart = System.nanoTime();
            for (int iteration = 0; iteration < repetitions; iteration++)
                fixture.evaluator.evaluate(fixture.lower, fixture.upper);
            long bitsetNanos = System.nanoTime() - bitsetStart;

            System.setProperty("quantminer.lockstep", "true");
            long evaluationsBefore = QuantMinerBitsetSupportCounts.getLockstepEvaluationCount();
            long pairsBefore = QuantMinerBitsetSupportCounts.getLockstepCandidatePairCount();
            long lockstepStart = System.nanoTime();
            EvaluationBaseAlgorithm.ReglePotentielle[] rules = null;
            for (int iteration = 0; iteration < repetitions; iteration++)
                rules = fixture.evaluator.evaluate(fixture.lower, fixture.upper);
            long lockstepNanos = System.nanoTime() - lockstepStart;
            long evaluationDelta = QuantMinerBitsetSupportCounts.getLockstepEvaluationCount() - evaluationsBefore;
            long pairDelta = QuantMinerBitsetSupportCounts.getLockstepCandidatePairCount() - pairsBefore;
            if (evaluationDelta != repetitions || pairDelta != (long)repetitions * 250)
                throw new AssertionError("C-Town benchmark did not validate every candidate pair");
            assertRuleSupports(rules, fixture.expected(0), 250, "C-Town benchmark");

            System.out.println("ACTUAL_CSV_REDUCED_EVALUATOR input=" + input
                + " rows=" + fixture.rows + " dimensions=2 candidates=250 repetitions=" + repetitions);
            System.out.println("original_total_nanos=" + originalNanos
                + " original_mean_nanos=" + originalNanos / repetitions);
            System.out.println("bitset_total_nanos=" + bitsetNanos
                + " bitset_mean_nanos=" + bitsetNanos / repetitions);
            System.out.println("lockstep_total_nanos=" + lockstepNanos
                + " lockstep_mean_nanos=" + lockstepNanos / repetitions
                + " lockstep_evaluations=" + evaluationDelta
                + " lockstep_candidate_pairs=" + pairDelta);
        }
        finally {
            System.clearProperty("quantminer.lockstep");
            fixture.close();
        }
    }

    private static final class ActualFixture {
        final int rows;
        final int candidates;
        final Path input;
        final DatabaseAdmin database;
        final ActualEvaluator evaluator;
        final float[] values;
        final float[] lower;
        final float[] upper;
        final byte[] leftCovered;
        final byte[] rightCovered;

        ActualFixture(int rows, int candidates) throws Exception {
            this.rows = rows;
            this.candidates = candidates;
            this.input = Files.createTempFile("quantminer-bitset-evaluator-", ".csv");
            StringBuilder csv = new StringBuilder("x,y\n");
            for (int row = 0; row < rows; row++) {
                if (row % 29 == 0)
                    csv.append("NaN");
                else if (row % 31 == 0)
                    csv.append("-0.0");
                else if (row % 37 != 0)
                    csv.append((row % 23) - 11);
                csv.append(',');
                if (row % 41 == 0)
                    csv.append("NaN");
                else if (row % 31 != 0)
                    csv.append((row % 19) - 9);
                csv.append('\n');
            }
            Files.write(input, csv.toString().getBytes(StandardCharsets.UTF_8));
            database = loadDatabase(input);
            evaluator = new ActualEvaluator(candidates, database, 2);
            DataColumn x = database.ObtenirColonneBDPriseEnCompte(0);
            DataColumn y = database.ObtenirColonneBDPriseEnCompte(1);
            evaluator.SpecifierSchemaRegle(createSchema(x, y));
            values = new float[rows * 2];
            System.arraycopy(x.m_tValeurReelle, 0, values, 0, rows);
            System.arraycopy(y.m_tValeurReelle, 0, values, rows, rows);
            lower = new float[candidates * 2];
            upper = new float[candidates * 2];
            for (int candidate = 0; candidate < candidates; candidate++) {
                for (int dimension = 0; dimension < 2; dimension++) {
                    float first = values[dimension * rows + (candidate * 17 + dimension) % rows];
                    float second = values[dimension * rows + (candidate * 29 + 11) % rows];
                    lower[candidate * 2 + dimension] = Math.min(first, second);
                    upper[candidate * 2 + dimension] = Math.max(first, second);
                }
            }
            leftCovered = new byte[rows];
            rightCovered = new byte[rows];
            for (int row = 0; row < rows; row++) {
                leftCovered[row] = (byte)(row % 3 == 0 ? 0 : 1);
                rightCovered[row] = (byte)(row % 7 == 0 ? 0 : 1);
            }
        }

        ActualFixture(String source, int candidates) throws Exception {
            this.input = null;
            this.candidates = candidates;
            this.database = loadDatabase(new java.io.File(source));
            this.rows = database.ObtenirNombreLignes();
            this.evaluator = new ActualEvaluator(candidates, database, 2);
            DataColumn x = database.ObtenirColonneBDPriseEnCompte(0);
            DataColumn y = database.ObtenirColonneBDPriseEnCompte(1);
            evaluator.SpecifierSchemaRegle(createSchema(x, y));
            values = new float[rows * 2];
            System.arraycopy(x.m_tValeurReelle, 0, values, 0, rows);
            System.arraycopy(y.m_tValeurReelle, 0, values, rows, rows);
            lower = new float[candidates * 2];
            upper = new float[candidates * 2];
            Random random = new Random(0xC70B1L);
            for (int candidate = 0; candidate < candidates; candidate++) {
                for (int dimension = 0; dimension < 2; dimension++) {
                    float first = values[dimension * rows + random.nextInt(rows)];
                    float second = values[dimension * rows + random.nextInt(rows)];
                    lower[candidate * 2 + dimension] = Math.min(first, second);
                    upper[candidate * 2 + dimension] = Math.max(first, second);
                }
            }
            leftCovered = new byte[rows];
            rightCovered = new byte[rows];
        }

        private static DatabaseAdmin loadDatabase(Path path) {
            return loadDatabase(path.toFile());
        }

        private static DatabaseAdmin loadDatabase(java.io.File path) {
            DatabaseAdmin database = new DatabaseAdmin(path.toString(), "csv");
            if (!database.EstBaseDeDonneesValide())
                throw new AssertionError("Could not load actual-evaluator fixture: " + path);
            database.PrendreEnCompteToutesLesColonnes();
            database.ChargerDonneesPrisesEnCompte();
            return database;
        }

        private static AssociationRule createSchema(DataColumn x, DataColumn y) {
            AssociationRule schema = new AssociationRule(1, 1, 1, 1);
            schema.AssignerItemGauche(new ItemQuantitative(
                new AttributQuantitative("x", x), 0.0f, 0.0f), 0);
            schema.AssignerItemDroite(new ItemQuantitative(
                new AttributQuantitative("y", y), 0.0f, 0.0f), 0);
            return schema;
        }

        void setCoverageMode(int mode) {
            evaluator.setCoverageMode((mode & 1) == 0, (mode & 2) == 0,
                                      leftCovered, rightCovered);
        }

        int[] expected(int mode) {
            return cpuReference(values, rows, 2, evaluator.activeRows(), evaluator.rightEligible(),
                                lower, upper, candidates, 1, (mode & 1) == 0, (mode & 2) == 0,
                                leftCovered, rightCovered);
        }

        void close() throws Exception {
            if (input != null)
                Files.deleteIfExists(input);
        }
    }

    private static final class ActualEvaluator extends EvaluationBaseAlgorithm {
        private final int dimensions;

        ActualEvaluator(int population, DatabaseAdmin database, int dimensions) {
            super(population, database);
            this.dimensions = dimensions;
        }

        @Override
        public void EvaluerQualiteReglePotentielle(ReglePotentielle rule) {
            rule.m_fQualite = rule.m_iSupportRegle;
        }

        EvaluationBaseAlgorithm.ReglePotentielle[] evaluate(float[] lower, float[] upper) {
            int candidates = lower.length / dimensions;
            EvaluationBaseAlgorithm.ReglePotentielle[] evaluated =
                new EvaluationBaseAlgorithm.ReglePotentielle[candidates];
            for (int candidate = 0; candidate < candidates; candidate++) {
                ReglePotentielle rule = new ReglePotentielle(dimensions, dimensions);
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    rule.m_tIntervalleMin[dimension] = lower[candidate * dimensions + dimension];
                    rule.m_tIntervalleMax[dimension] = upper[candidate * dimensions + dimension];
                }
                m_tReglesPotentielles[candidate] = rule;
                m_tReglesPotentiellesAEvaluer[candidate] = rule;
                evaluated[candidate] = rule;
            }
            m_meilleureReglePotentielle[0] = new ReglePotentielle(dimensions, dimensions);
            m_iNombreReglesPotentiellesAEvaluer = candidates;
            EvaluerReglesPotentielles();
            return evaluated;
        }

        void setCoverageMode(boolean evaluateLeft, boolean evaluateRight,
                             byte[] leftCovered, byte[] rightCovered) {
            m_bPrendreEnCompteQuantitatifsGauche = evaluateLeft;
            m_bPrendreEnCompteQuantitatifsDroite = evaluateRight;
            for (int row = 0; row < m_iNombreTransactions; row++) {
                m_tLignesCouvertesGauche[row] = leftCovered[row] != 0;
                m_tLignesCouvertesDroite[row] = rightCovered[row] != 0;
            }
        }

        void forceUnsupportedDisjunctionGeometry() {
            m_schemaRegleOptimale.m_iNombreDisjonctionsGauche = 2;
        }

        int[] activeRows() {
            return Arrays.copyOf(m_tLignesAPrendreEnCompte, m_iNombreLignesAPrendreEnCompte);
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

    private static void assertSupports(int[] actual, int[] expected, int candidates, String label) {
        for (int candidate = 0; candidate < candidates; candidate++) {
            if (actual[candidate * 2] != expected[candidate * 2]
                || actual[candidate * 2 + 1] != expected[candidate * 2 + 1])
                throw new AssertionError(label + ": candidate=" + candidate
                    + ", expected=" + expected[candidate * 2] + "/" + expected[candidate * 2 + 1]
                    + ", actual=" + actual[candidate * 2] + "/" + actual[candidate * 2 + 1]);
        }
    }

    private static void assertRuleSupports(EvaluationBaseAlgorithm.ReglePotentielle[] actual,
                                           int[] expected, int candidates, String label) {
        for (int candidate = 0; candidate < candidates; candidate++) {
            if (actual[candidate].m_iSupportCond != expected[candidate * 2]
                || actual[candidate].m_iSupportRegle != expected[candidate * 2 + 1])
                throw new AssertionError(label + ": candidate=" + candidate
                    + ", expected=" + expected[candidate * 2] + "/" + expected[candidate * 2 + 1]
                    + ", actual=" + actual[candidate].m_iSupportCond + "/"
                    + actual[candidate].m_iSupportRegle);
        }
    }
}
