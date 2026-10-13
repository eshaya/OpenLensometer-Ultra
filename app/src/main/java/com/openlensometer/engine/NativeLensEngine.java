// File: app/src/main/java/com/openlensometer/engine/NativeLensEngine.java
package com.openlensometer.engine;

import java.nio.ByteBuffer;

public class NativeLensEngine {

    static {
        System.loadLibrary("lensometer_engine");
    }

    // A lightweight ping test to verify the JNI bridge is alive
    public native String pingEngine();

    public native double calculateMonitorPixelPitch(
            int screenWidthPixels, int screenHeightPixels, double screenDiagonalInches);

    public native int detectCharucoCorners(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride);

    public native float[] detectAlignmentTarget(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            float monitorDownX, float monitorDownY);

    public native int decodeAlignmentPhaseCode(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            float normalizedCenterX, float normalizedCenterY, float normalizedTargetSize,
            float monitorDownX, float monitorDownY);

    /**
     * Returns {tiltRightDeg, tiltDownDeg, distanceMm, then x,y image pixels of the corner markers
     * top-left, top-right, bottom-right, bottom-left (NaN if missing)}, or null if fewer than
     * three corner markers are seen.
     */
    public native float[] estimateMonitorTilt(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            float monitorDownX, float monitorDownY,
            float focalX, float focalY, float principalX, float principalY,
            float cornerOffsetMm);

    /**
     * Adds one exposure of the page being captured; capturePhaseFrame for the same axis/phase
     * then averages them (dropping outliers). Returns the number collected so far, or < 0.
     */
    public native int addPhaseFrame(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            int axis, int phaseIndex);

    public native int capturePhaseFrame(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            ByteBuffer depthPlane, int depthWidth, int depthHeight,
            int depthRowStride, int depthPixelStride,
            int axis, int phaseIndex, float focalX, float focalY,
            float principalX, float principalY);

    public native void resetPhaseCapture();

    /** Adds one no-lens grid frame to the single-frame method's reference. Returns the count. */
    public native int addGridReferenceFrame(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride);

    /** Finishes the grid reference. Returns {X, Y} camera pixels per period, or null. */
    public native float[] finishGridReference(float monitorDownX, float monitorDownY, String debugPath);

    /**
     * Measures the lens from one grid frame (single-frame method). Same first 8 values as
     * measureLensFromRuns (status 10 = no grid reference), then {lens type (0 single vision,
     * 1 bifocal, 2 progressive), distance sphere, cylinder, axis, near sphere, cylinder, axis,
     * add}.
     */
    public native float[] measureGridFrame(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            float monitorDownX, float monitorDownY, float lensCenterX, float lensCenterY,
            float expectedLensRadius, float screenDistanceMm, float screenOpeningPixels,
            float assumedLensDistanceMm);

    /** Forgets the stored no-lens reference run. */
    public native void clearReferenceRun();

    /**
     * Keeps the finished run as the no-lens reference and clears capture for a lens run.
     * Returns {X, Y} camera pixels per fringe period (-1 if unknown), or null if incomplete.
     */
    public native float[] saveReferenceRun(float monitorDownX, float monitorDownY);

    /**
     * Lens opening with glasses flat on the screen: {size (sqrt of area, pixels), then the
     * outline as x, y pairs in image pixels}, or null if no rim was found.
     */
    public native float[] measureLensOutline(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            float centerX, float centerY, float minSizePixels);

    /**
     * Compares the finished lens run with the reference. Returns {status, sphere, cylinder,
     * axis, lensDistanceMm, lensOpeningPx, pixelsUsed, rmsResidualPx}. Status 0 = ok,
     * 1 lens not found, 2 lens touches edge, 3 too few pixels, 4 fringes too fine,
     * 5 bad input, 10 no reference, 11 lens run incomplete.
     */
    public native float[] measureLensFromRuns(
            float monitorDownX, float monitorDownY, float lensCenterX, float lensCenterY,
            float screenDistanceMm, float screenOpeningPixels, String debugPath);

    public native float[] getCapturedCameraDepthCalibration();

        // [X mean modulation, Y mean modulation, depth p10, median, p90, valid sample count]
        public native float[] getCaptureQualitySummary();

    /**
     * Calibrates the physical geometry of the screen using a ChArUco board.
     * 
     * @param yuvBuffer Grayscale/YUV camera frame buffer
     * @param width Width of the camera frame
     * @param height Height of the camera frame
     * @param stride Row stride of the camera frame
     * @param zScreenMm Physical distance to the screen (from DEPTH16)
     * @param cameraMatrix 3x3 Intrinsic camera matrix as a flat float array (length 9)
     * @param monitorSquareSizePx The width of the ChArUco squares drawn on the monitor in pixels
     * @return The physical size of a single monitor pixel in mm, or -1.0f if detection fails.
     */
    public native float calibrateMonitor(
            ByteBuffer yuvBuffer, int width, int height, int stride,
            float zScreenMm, float[] cameraMatrix, int monitorSquareSizePx
    );

    public MonitorPitchEstimate calibrateMonitorBothWays(
            ByteBuffer yPlane, int width, int height, int rowStride,
            float zScreenMm, float[] cameraMatrix, int charucoSquareSizePx,
            int screenWidthPixels, int screenHeightPixels, double screenDiagonalInches) {
        double monitorSpecPitch = calculateMonitorPixelPitch(
                screenWidthPixels, screenHeightPixels, screenDiagonalInches);
        double charucoPitch = calibrateMonitor(
                yPlane, width, height, rowStride, zScreenMm, cameraMatrix, charucoSquareSizePx);
        return new MonitorPitchEstimate(charucoPitch, monitorSpecPitch);
    }

    public static final class MonitorPitchEstimate {
        public final double charucoPitchMm;
        public final double monitorSpecPitchMm;
        public final double selectedPitchMm;
        public final double differencePercent;
        public final boolean selectedCharucoEstimate;

        private MonitorPitchEstimate(double charucoPitchMm, double monitorSpecPitchMm) {
            this.charucoPitchMm = isValidPitch(charucoPitchMm) ? charucoPitchMm : Double.NaN;
            this.monitorSpecPitchMm = isValidPitch(monitorSpecPitchMm)
                    ? monitorSpecPitchMm : Double.NaN;
            this.selectedCharucoEstimate = Double.isFinite(this.charucoPitchMm);
            this.selectedPitchMm = selectedCharucoEstimate
                    ? this.charucoPitchMm : this.monitorSpecPitchMm;
            this.differencePercent = selectedCharucoEstimate &&
                    Double.isFinite(this.monitorSpecPitchMm)
                    ? Math.abs(this.charucoPitchMm - this.monitorSpecPitchMm) /
                            this.monitorSpecPitchMm * 100.0
                    : Double.NaN;
        }

        public boolean agreesWithin(double tolerancePercent) {
            return Double.isFinite(differencePercent) &&
                    Double.isFinite(tolerancePercent) && tolerancePercent >= 0.0 &&
                    differencePercent <= tolerancePercent;
        }

        private static boolean isValidPitch(double pitchMm) {
            return Double.isFinite(pitchMm) && pitchMm > 0.0;
        }
    }

    /**
     * [OLD METHOD] Single-pass measurement without aberration removal.
     */
    public native LensResult processPhaseShiftFrames(
            ByteBuffer yuvBuffer1, ByteBuffer yuvBuffer2, 
            ByteBuffer yuvBuffer3, ByteBuffer yuvBuffer4,
            int yuvWidth, int yuvHeight, int yuvStride,
            ByteBuffer depthBuffer, int depthWidth, int depthHeight,
            float systemLambdaMm, float pupilRadiusMm
    );

    /**
     * STEP 1: Captures the empty monitor and returns a memory pointer to the baseline phase map.
     */
    public native long captureEmptyFieldPhase(
            ByteBuffer yuvBuffer1, ByteBuffer yuvBuffer2, 
            ByteBuffer yuvBuffer3, ByteBuffer yuvBuffer4,
            int yuvWidth, int yuvHeight, int yuvStride
    );

    /**
     * STEP 2: Measures the lens and subtracts the empty field.
     */
    public native LensResult measureLens(
            long emptyFieldPtr, 
            ByteBuffer yuvBuffer1, ByteBuffer yuvBuffer2, 
            ByteBuffer yuvBuffer3, ByteBuffer yuvBuffer4,
            int yuvWidth, int yuvHeight, int yuvStride,
            ByteBuffer depthBuffer, int depthWidth, int depthHeight,
            float systemLambdaMm, float pupilRadiusMm
    );

    /**
     * STEP 3: Frees the C++ memory when the app closes or restarts calibration.
     */
    public native void releaseEmptyField(long emptyFieldPtr);
}

