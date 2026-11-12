import src.apriori.AssociationRule;
import src.database.DatabaseAdmin;
import src.geneticAlgorithm.OptimizerGeneticAlgo;
import src.solver.QuantMinerBitsetSupportCounts;
import src.solver.QuantMinerGpuSupportCounts;
import src.solver.ResolutionContext;
import src.solver.RuleTester;

public final class QuantMinerCli {
    private QuantMinerCli() {}

    public static void main(String[] args) {
        if (args.length < 2 || args.length > 7) {
            System.err.println("usage: QuantMinerCli INPUT.csv OUTPUT.csv [MIN_SUPP] [MIN_CONF] [POPULATION] [GENERATIONS] [MAX_ATTRIBUTES]");
            System.exit(2);
        }
        String input = args[0];
        String output = args[1];
        float minSupport = args.length > 2 ? Float.parseFloat(args[2]) : 0.10f;
        float minConfidence = args.length > 3 ? Float.parseFloat(args[3]) : 0.60f;
        int population = args.length > 4 ? Integer.parseInt(args[4]) : 250;
        int generations = args.length > 5 ? Integer.parseInt(args[5]) : 100;
        int maxAttributes = args.length > 6 ? Integer.parseInt(args[6]) : 3;

        QuantMinerGpuSupportCounts.initializeIfRequested();

        DatabaseAdmin database = new DatabaseAdmin(input, "csv");
        if (!database.EstBaseDeDonneesValide()) {
            throw new IllegalArgumentException("invalid CSV database: " + input);
        }
        database.PrendreEnCompteToutesLesColonnes();
        database.ChargerDonneesPrisesEnCompte();

        ResolutionContext context = new ResolutionContext(null);
        context.m_gestionnaireBD = database;
        context.m_iTechniqueResolution = ResolutionContext.TECHNIQUE_ALGO_GENETIQUE;
        context.GenererStructuresDonneesSelonBDPriseEnCompte();
        context.DefinirPositionnementPourTous(ResolutionContext.PRISE_EN_COMPTE_ITEM_2_COTES, true);
        context.m_parametresReglesQuantitatives.m_fMinSupp = minSupport;
        context.m_parametresReglesQuantitatives.m_fMinConf = minConfidence;
        context.m_parametresReglesQuantitatives.m_iNombreMaxAttributsQuant = maxAttributes;
        context.m_parametresTechAlgoGenetique.m_iTaillePopulation = population;
        context.m_parametresTechAlgoGenetique.m_iNombreGenerations = generations;

        RuleTester tester = new RuleTester(context, null);
        tester.DefinirOptimiseurRegle(new OptimizerGeneticAlgo());
        tester.run();
        AssociationRule[] rules = (AssociationRule[]) tester.ObtenirListeReglesOptimales().toArray(new AssociationRule[0]);
        QuantMinerBitsetSupportCounts.dumpProfile();
        context.SauvegarderReglesCsv(output, rules);
        System.out.println("rules=" + rules.length);
        QuantMinerGpuSupportCounts.dumpProfile();
    }
}
