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

/**
 * Opt-in JNI bridge for exact HIP support counting. The native library is
 * deliberately not loaded unless QUANTMINER_GPU=1 or -Dquantminer.gpu=true.
 */
public final class QuantMinerGpuSupportCounts {
    private static final String ENABLE_PROPERTY = "quantminer.gpu";
    private static final String ENABLE_ENVIRONMENT = "QUANTMINER_GPU";
    private static final String LIBRARY_PROPERTY = "quantminer.gpu.library";
    private static final String LIBRARY_ENVIRONMENT = "QUANTMINER_GPU_LIBRARY";

    private static boolean nativeLoaded = false;

    private QuantMinerGpuSupportCounts() {}

    public static boolean isRequested() {
        return isTrue(System.getProperty(ENABLE_PROPERTY))
            || isTrue(System.getenv(ENABLE_ENVIRONMENT));
    }

    private static boolean isTrue(String value) {
        return value != null
            && (value.equalsIgnoreCase("true") || value.equals("1")
                || value.equalsIgnoreCase("yes") || value.equalsIgnoreCase("on"));
    }

    public static synchronized void initializeIfRequested() {
        if (!isRequested() || nativeLoaded)
            return;

        String library = System.getProperty(LIBRARY_PROPERTY);
        if (library == null || library.length() == 0)
            library = System.getenv(LIBRARY_ENVIRONMENT);

        try {
            if (library != null && library.length() > 0) {
                if (library.indexOf('/') >= 0 || library.indexOf('\\') >= 0)
                    System.load(library);
                else
                    System.loadLibrary(library);
            }
            else {
                System.loadLibrary("quantminer_gpu");
            }
            nativeLoaded = true;
        }
        catch (UnsatisfiedLinkError error) {
            String location = library == null ? "quantminer_gpu (java.library.path)" : library;
            throw new UnsatisfiedLinkError(
                "QuantMiner GPU support was explicitly requested, but the native HIP/JNI library could not be loaded from "
                + location + ": " + error.getMessage());
        }
        catch (SecurityException error) {
            throw new IllegalStateException(
                "QuantMiner GPU support was explicitly requested, but loading its native library was denied", error);
        }
    }

    public static boolean supportsConfiguration(int leftDisjunctions, int rightDisjunctions,
                                                 int currentLeftDisjunction, int currentRightDisjunction,
                                                 int dimensions, int intervals) {
        return isRequested() && leftDisjunctions == 1 && rightDisjunctions == 1
            && currentLeftDisjunction == 0 && currentRightDisjunction == 0
            && dimensions >= 0 && intervals == dimensions;
    }

    public static Context createContext(float[] dimensionMajorValues, int rowCount, int dimensions,
                                        int[] activeRows, int activeRowCount, byte[] rightEligible) {
        initializeIfRequested();
        if (!nativeLoaded)
            throw new IllegalStateException("QuantMiner GPU context requested without GPU enablement");
        if (dimensionMajorValues == null || rowCount < 0 || dimensions <= 0
            || dimensionMajorValues.length != rowCount * dimensions)
            throw new IllegalArgumentException("Invalid dimension-major QuantMiner GPU value buffer");
        if (activeRows == null || activeRowCount < 0 || activeRowCount > activeRows.length)
            throw new IllegalArgumentException("Invalid QuantMiner GPU active-row buffer");
        if (rightEligible == null || rightEligible.length != rowCount)
            throw new IllegalArgumentException("Invalid QuantMiner GPU right-coverage buffer");

        long handle = nativeCreateContext(dimensionMajorValues, rowCount, dimensions,
                                          activeRows, activeRowCount, rightEligible);
        if (handle == 0)
            throw new IllegalStateException("HIP returned a null QuantMiner GPU context");
        return new Context(handle, rowCount, dimensions);
    }

    public static final class Context {
        private long handle;
        private final int rowCount;
        private final int dimensions;

        private Context(long handle, int rowCount, int dimensions) {
            this.handle = handle;
            this.rowCount = rowCount;
            this.dimensions = dimensions;
        }

        public synchronized int[] count(float[] lowerBounds, float[] upperBounds, int candidateCount,
                                        int intervals, int leftDimensions, boolean evaluateLeft,
                                        boolean evaluateRight, byte[] leftCovered, byte[] rightCovered) {
            if (handle == 0)
                throw new IllegalStateException("QuantMiner GPU context is closed");
            if (candidateCount < 0 || intervals != dimensions || leftDimensions < 0
                || leftDimensions > dimensions)
                throw new IllegalArgumentException("Invalid QuantMiner GPU population geometry");
            if (lowerBounds == null || upperBounds == null
                || lowerBounds.length < candidateCount * intervals
                || upperBounds.length < candidateCount * intervals)
                throw new IllegalArgumentException("Invalid QuantMiner GPU interval buffers");
            if (leftCovered == null || rightCovered == null
                || leftCovered.length != rowCount || rightCovered.length != rowCount)
                throw new IllegalArgumentException("Invalid QuantMiner GPU coverage buffers");

            int[] result = nativeCount(handle, lowerBounds, upperBounds, candidateCount, intervals,
                                       leftDimensions, evaluateLeft, evaluateRight,
                                       leftCovered, rightCovered);
            if (result == null || result.length != candidateCount * 2)
                throw new IllegalStateException("HIP returned an invalid QuantMiner support-count buffer");
            return result;
        }

        public synchronized void close() {
            if (handle != 0) {
                nativeReleaseContext(handle);
                handle = 0;
            }
        }
    }

    private static native long nativeCreateContext(float[] values, int rowCount, int dimensions,
                                                    int[] activeRows, int activeRowCount,
                                                    byte[] rightEligible);

    private static native int[] nativeCount(long handle, float[] lowerBounds, float[] upperBounds,
                                            int candidateCount, int intervals, int leftDimensions,
                                            boolean evaluateLeft, boolean evaluateRight,
                                            byte[] leftCovered, byte[] rightCovered);

    private static native void nativeReleaseContext(long handle);
}
