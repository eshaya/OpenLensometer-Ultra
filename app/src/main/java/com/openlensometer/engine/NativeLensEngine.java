package com.openlensometer.engine;

import java.nio.ByteBuffer;

public class NativeLensEngine {

    static {
        System.loadLibrary("lensometer_engine");
    }

    public native float processFrame(
            ByteBuffer yuvBuffer, int yuvWidth, int yuvHeight, int yuvStride,
            ByteBuffer depthBuffer, int depthWidth, int depthHeight
    );
}
