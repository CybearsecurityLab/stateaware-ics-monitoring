package src.apriori;

import java.math.BigInteger;
import java.util.Arrays;

public final class QuantMinerCombinationTest {
    private QuantMinerCombinationTest() {}

    public static void main(String[] args) {
        int checked = 0;
        for (int items = 0; items <= 9; items++) {
            for (int places = 0; places <= items; places++) {
                long combinations = binomial(items, places).longValue();
                for (long index = 0; index < combinations; index++) {
                    boolean[] expected = reference(index, items, places);
                    boolean[] actual = AprioriQuantitative.CalculerEnsemblesItems(index, items, places);
                    if (!Arrays.equals(expected, actual))
                        throw new AssertionError("Mismatch at n=" + items + ", p=" + places + ", index=" + index);
                    checked++;
                }
            }
        }
        for (int repeat = 0; repeat < 3; repeat++) {
            boolean[] expected = reference(123456789L, 20, 10);
            boolean[] actual = AprioriQuantitative.CalculerEnsemblesItems(123456789L, 20, 10);
            if (!Arrays.equals(expected, actual))
                throw new AssertionError("Wide repeated-index mismatch");
        }
        System.out.println("combination-parity: PASS (" + checked + " small cases plus wide repeated indices)");
    }

    private static BigInteger binomial(int n, int p) {
        BigInteger result = BigInteger.ONE;
        for (int value = 2; value <= n; value++)
            result = result.multiply(BigInteger.valueOf(value));
        BigInteger denominator = BigInteger.ONE;
        for (int value = 2; value <= p; value++)
            denominator = denominator.multiply(BigInteger.valueOf(value));
        for (int value = 2; value <= n - p; value++)
            denominator = denominator.multiply(BigInteger.valueOf(value));
        return result.divide(denominator);
    }

    private static boolean[] reference(long index, int items, int places) {
        if (items == 0)
            return null;
        boolean[] selected = new boolean[items];
        int remainingItems = items;
        int remainingPlaces = places;
        for (int item = 0; item < items && remainingPlaces > 0 && remainingItems >= remainingPlaces; item++) {
            BigInteger combinations = remainingItems > remainingPlaces
                ? binomial(remainingItems - 1, remainingPlaces - 1)
                : BigInteger.ONE;
            if (BigInteger.valueOf(index).compareTo(combinations) < 0) {
                selected[item] = true;
                remainingPlaces--;
            }
            else {
                index -= combinations.longValue();
            }
            remainingItems--;
        }
        return remainingPlaces == places ? null : selected;
    }
}
