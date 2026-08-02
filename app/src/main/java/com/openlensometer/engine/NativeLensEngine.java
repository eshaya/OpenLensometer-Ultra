// File: app/src/main/java/com/openlensometer/engine/NativeLensEngine.java
package com.openlensometer.engine;

import java.nio.ByteBuffer;

public class NativeLensEngine {

    static {
        System.loadLibrary("lensometer_engine");
    }

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