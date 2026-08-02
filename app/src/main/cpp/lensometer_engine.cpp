// File: app/src/main/cpp/lensometer_engine.cpp
#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include "depth16_filter.h"
#include "phase_unwrapper.h" // Include our new math header

#define LOG_TAG "OpenLensometer-NDK"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jfloat JNICALL
Java_com_openlensometer_engine_NativeLensEngine_processPhaseShiftFrames(
        JNIEnv* env, jobject /* this */,
        jobject yuvBuffer1, jobject yuvBuffer2, 
        jobject yuvBuffer3, jobject yuvBuffer4,
        jint yuvWidth, jint yuvHeight, jint yuvStride,
        jobject depthBuffer, jint depthWidth, jint depthHeight) {

    // 1. Extract Direct Memory Pointers for all 4 frames
    uint8_t* yData1 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer1));
    uint8_t* yData2 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer2));
    uint8_t* yData3 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer3));
    uint8_t* yData4 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer4));
    uint16_t* depthData = static_cast<uint16_t*>(env->GetDirectBufferAddress(depthBuffer));

    if (!yData1 || !yData2 || !yData3 || !yData4 || !depthData) {
        LOGE("Failed to get direct buffer addresses. Ensure all buffers are allocateDirect().");
        return -1.0f;
    }

    // 2. Wrap the memory in OpenCV Matrices
    cv::Mat I1(yuvHeight, yuvWidth, CV_8UC1, yData1, yuvStride);
    cv::Mat I2(yuvHeight, yuvWidth, CV_8UC1, yData2, yuvStride);
    cv::Mat I3(yuvHeight, yuvWidth, CV_8UC1, yData3, yuvStride);
    cv::Mat I4(yuvHeight, yuvWidth, CV_8UC1, yData4, yuvStride);
    
    cv::Mat depthMap(depthHeight, depthWidth, CV_16U, depthData);

    // 3. Execute the Depth Extraction Filter
    float zLensMm = calculateZLensFromDepth(depthData, depthWidth, depthHeight);

    if (zLensMm < 0.0f) {
        LOGE("Depth tracking failed: Frame not detected.");
        return -1.0f; 
    }
    LOGI("Successfully locked Z_lens at: %.2f mm", zLensMm);

    // 4. Extract Wrapped Phase and Modulation Quality
    cv::Mat wrappedPhase, modulationMap;
    calculateWrappedPhase(I1, I2, I3, I4, wrappedPhase, modulationMap);
    
    // Log the mean contrast to verify the math is working on live data
    cv::Scalar meanModulation = cv::mean(modulationMap);
    LOGI("Wrapped phase extracted. Mean fringe modulation (contrast): %.2f", meanModulation[0]);

    // 5. [Next Step] Spatial Phase Unwrapping to remove 2-pi jumps...

    return zLensMm;
}