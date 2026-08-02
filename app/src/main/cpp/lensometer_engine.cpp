#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include "depth16_filter.h"

#define LOG_TAG "OpenLensometer-NDK"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jfloat JNICALL
Java_com_openlensometer_engine_NativeLensEngine_processFrame(
        JNIEnv* env, jobject /* this */,
        jobject yuvBuffer, jint yuvWidth, jint yuvHeight, jint yuvStride,
        jobject depthBuffer, jint depthWidth, jint depthHeight) {

    uint8_t* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer));
    uint16_t* depthData = static_cast<uint16_t*>(env->GetDirectBufferAddress(depthBuffer));

    if (!yData || !depthData) {
        LOGE("Failed to get direct buffer addresses.");
        return -1.0f;
    }

    cv::Mat yChannel(yuvHeight, yuvWidth, CV_8UC1, yData, yuvStride);
    cv::Mat depthMap(depthHeight, depthWidth, CV_16U, depthData);

    float zLensMm = calculateZLensFromDepth(depthData, depthWidth, depthHeight);

    if (zLensMm < 0.0f) {
        LOGI("Depth tracking failed: Frame not detected.");
        return -1.0f; 
    }

    LOGI("Successfully locked Z_lens at: %.2f mm", zLensMm);

    // TODO: Phase Demodulation & Unwrapping

    return zLensMm;
}
