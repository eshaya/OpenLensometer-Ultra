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

    public native int capturePhaseFrame(
            ByteBuffer yPlane, int width, int height, int rowStride, int pixelStride,
            ByteBuffer depthPlane, int depthWidth, int depthHeight,
            int depthRowStride, int depthPixelStride,
            int axis, int phaseIndex, float focalX, float focalY,
            float principalX, float principalY);

    public native void resetPhaseCapture();

    public native float[] getCapturedCameraDepthCalibration();

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

