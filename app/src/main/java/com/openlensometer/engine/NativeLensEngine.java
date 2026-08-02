// File: app/src/main/java/com/openlensometer/engine/NativeLensEngine.java
package com.openlensometer.engine;

import java.nio.ByteBuffer;

public class NativeLensEngine {

    static {
        System.loadLibrary("lensometer_engine");
    }

    /**
     * Passes four phase-shifted hardware buffers directly to C++ (Zero-copy).
     * 
     * @param yuvBuffer1 0-degree phase shift frame
     * @param yuvBuffer2 90-degree phase shift frame
     * @param yuvBuffer3 180-degree phase shift frame
     * @param yuvBuffer4 270-degree phase shift frame
     * @param yuvWidth Width of the optical frames
     * @param yuvHeight Height of the optical frames
     * @param yuvStride Row stride (padding) of the frames
     * @param depthBuffer DEPTH16 buffer for vertex distance
     * @param depthWidth Width of the depth map
     * @param depthHeight Height of the depth map
     * @return The calculated Z_lens distance in mm.
     */
    public native float processPhaseShiftFrames(
            ByteBuffer yuvBuffer1, ByteBuffer yuvBuffer2, 
            ByteBuffer yuvBuffer3, ByteBuffer yuvBuffer4,
            int yuvWidth, int yuvHeight, int yuvStride,
            ByteBuffer depthBuffer, int depthWidth, int depthHeight
    );
}