// File: app/src/main/java/com/openlensometer/engine/LensAnalyzer.java
package com.openlensometer.engine;

import androidx.annotation.NonNull;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class LensAnalyzer implements ImageAnalysis.Analyzer {

    private final NativeLensEngine nativeEngine;
    private final float systemLambdaMm;
    private final float pupilRadiusMm;

    private final List<ByteBuffer> frameBuffers = new ArrayList<>();
    private final int requiredFrames = 4;

    public LensAnalyzer(NativeLensEngine nativeEngine, float systemLambdaMm, float pupilRadiusMm) {
        this.nativeEngine = nativeEngine;
        this.systemLambdaMm = systemLambdaMm;
        this.pupilRadiusMm = pupilRadiusMm;
    }

    @Override
    @ExperimentalGetImage
    public void analyze(@NonNull ImageProxy imageProxy) {
        try {
            // 1. Extract Y plane (grayscale intensity)
            ImageProxy.PlaneProxy yPlane = imageProxy.getPlanes()[0];
            ByteBuffer yBuffer = yPlane.getBuffer();

            // 2. Allocate direct ByteBuffer for zero-copy JNI compatibility
            ByteBuffer directCopy = ByteBuffer.allocateDirect(yBuffer.remaining());
            directCopy.put(yBuffer);
            directCopy.rewind();

            int width = imageProxy.getWidth();
            int height = imageProxy.getHeight();
            int stride = yPlane.getRowStride();

            synchronized (frameBuffers) {
                frameBuffers.add(directCopy);

                if (frameBuffers.size() == requiredFrames) {
                    ByteBuffer b1 = frameBuffers.get(0);
                    ByteBuffer b2 = frameBuffers.get(1);
                    ByteBuffer b3 = frameBuffers.get(2);
                    ByteBuffer b4 = frameBuffers.get(3);

                    // Dummy direct buffer for depth channel until depth stream is linked
                    ByteBuffer emptyDepthBuffer = ByteBuffer.allocateDirect(0);

                    // 3. Call JNI bridge
                    LensResult result = nativeEngine.processPhaseShiftFrames(
                            b1, b2, b3, b4,
                            width, height, stride,
                            emptyDepthBuffer, 0, 0,
                            systemLambdaMm, pupilRadiusMm
                    );

                    frameBuffers.clear();
                }
            }
        } finally {
            // Always close the image proxy to prevent CameraX frame pipeline starvation
            imageProxy.close();
        }
    }
}