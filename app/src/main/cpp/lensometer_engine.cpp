// File: app/src/main/cpp/lensometer_engine.cpp
#include <jni.h>
#include <strings>
#include <android/log.h>
#include <opencv2/opencv.hpp>

// Our modular pipeline headers
#include "depth16_filter.h"
#include "phase_unwrapper.h"
#include "zernike_fitter.h"
#include "aberration_removal.h"
#include "charuco_calibrator.h"

#define LOG_TAG "OpenLensometer-NDK"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define EXTRACT_YUV_BUFFERS() \
    uint8_t* y1 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer1)); \
    uint8_t* y2 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer2)); \
    uint8_t* y3 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer3)); \
    uint8_t* y4 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer4)); \
    if (!y1 || !y2 || !y3 || !y4) { \
        LOGE("Failed to get YUV buffer addresses."); \
        return nullptr; \
    } \
    cv::Mat I1(yuvHeight, yuvWidth, CV_8UC1, y1, yuvStride); \
    cv::Mat I2(yuvHeight, yuvWidth, CV_8UC1, y2, yuvStride); \
    cv::Mat I3(yuvHeight, yuvWidth, CV_8UC1, y3, yuvStride); \
    cv::Mat I4(yuvHeight, yuvWidth, CV_8UC1, y4, yuvStride)

jobject createJavaLensResult(JNIEnv* env, float sphere, float cylinder, float axis, float zDistance) {
    jclass resultClass = env->FindClass("com/openlensometer/engine/LensResult");
    if (resultClass == nullptr) return nullptr;

    jmethodID constructor = env->GetMethodID(resultClass, "<init>", "(FFFF)V");
    if (constructor == nullptr) return nullptr;

    return env->NewObject(resultClass, constructor, sphere, cylinder, axis, zDistance);
}


// JNI functions must be wrapped in extern "C" to prevent C++ name mangling
extern "C" {

JNIEXPORT jstring JNICALL
Java_com_openlensometer_engine_NativeLensEngine_pingEngine(
    JNIEnv* env, 
    jobject /* this */) 
{
    std::string status = "JNI Bridge Active: C++ Math Engine Ready";
    return env->NewStringUTF(status.c_str());
}

// =========================================================================
//  ChArUco Hardware Calibrator
// =========================================================================
extern "C" JNIEXPORT jfloat JNICALL
Java_com_openlensometer_engine_NativeLensEngine_calibrateMonitor(
        JNIEnv* env, jobject /* this */,
        jobject yuvBuffer, jint width, jint height, jint stride,
        jfloat zScreenMm, jfloatArray cameraMatrixArray, jint monitorSquareSizePx) {

    uint8_t* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer));
    if (!yData) {
        LOGE("Calibration failed: Null YUV buffer");
        return -1.0f;
    }

    cv::Mat frameGray(height, width, CV_8UC1, yData, stride);

    // Convert flat float array to CV_64F Mat for accurate optical geometry
    jfloat* camMatElements = env->GetFloatArrayElements(cameraMatrixArray, nullptr);
    cv::Mat cameraMatrix64F(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        cameraMatrix64F.at<double>(i / 3, i % 3) = static_cast<double>(camMatElements[i]);
    }
    env->ReleaseFloatArrayElements(cameraMatrixArray, camMatElements, JNI_ABORT);

    float pixelPitchMm = calculateMonitorPixelPitch(frameGray, zScreenMm, cameraMatrix64F, monitorSquareSizePx);
    
    if (pixelPitchMm > 0.0f) {
        LOGI("Hardware calibrated. Monitor pixel pitch: %.4f mm", pixelPitchMm);
    }
    
    return pixelPitchMm;
}


// =========================================================================
// [METHOD 1] Original Single-Pass Measurement (No Empty-Field Correction)
// =========================================================================
extern "C" JNIEXPORT jobject JNICALL
Java_com_openlensometer_engine_NativeLensEngine_processPhaseShiftFrames(
        JNIEnv* env, jobject /* this */,
        jobject yuvBuffer1, jobject yuvBuffer2, 
        jobject yuvBuffer3, jobject yuvBuffer4,
        jint yuvWidth, jint yuvHeight, jint yuvStride,
        jobject depthBuffer, jint depthWidth, jint depthHeight,
        jfloat systemLambdaMm, jfloat pupilRadiusMm) { // <-- Dynamic hardware variables added

    EXTRACT_YUV_BUFFERS();

    uint16_t* depthData = static_cast<uint16_t*>(env->GetDirectBufferAddress(depthBuffer));
    if (!depthData) return nullptr;

    float zLensMm = calculateZLensFromDepth(depthData, depthWidth, depthHeight);
    if (zLensMm < 0.0f) return nullptr;

    cv::Mat wrappedPhase, modulationMap;
    calculateWrappedPhase(I1, I2, I3, I4, wrappedPhase, modulationMap);

    cv::Mat unwrappedPhase;
    unwrapPhaseQualityGuided(wrappedPhase, modulationMap, unwrappedPhase, 10.0f);

    cv::Mat mask;
    cv::threshold(modulationMap, mask, 10.0, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    
    OpticalPrescription rx = fitZernikeAndExtractPrescription(unwrappedPhase, mask, systemLambdaMm, pupilRadiusMm);

    return createJavaLensResult(env, rx.sphere, rx.cylinder, rx.axis, zLensMm);
}


// =========================================================================
// [METHOD 2] STEP 1: Capture Empty Field Phase
// =========================================================================
extern "C" JNIEXPORT jlong JNICALL
Java_com_openlensometer_engine_NativeLensEngine_captureEmptyFieldPhase(
        JNIEnv* env, jobject /* this */,
        jobject yuvBuffer1, jobject yuvBuffer2, 
        jobject yuvBuffer3, jobject yuvBuffer4,
        jint yuvWidth, jint yuvHeight, jint yuvStride) {
    
    uint8_t* y1 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer1));
    uint8_t* y2 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer2));
    uint8_t* y3 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer3));
    uint8_t* y4 = static_cast<uint8_t*>(env->GetDirectBufferAddress(yuvBuffer4));
    if (!y1 || !y2 || !y3 || !y4) return 0;
    
    cv::Mat I1(yuvHeight, yuvWidth, CV_8UC1, y1, yuvStride);
    cv::Mat I2(yuvHeight, yuvWidth, CV_8UC1, y2, yuvStride);
    cv::Mat I3(yuvHeight, yuvWidth, CV_8UC1, y3, yuvStride);
    cv::Mat I4(yuvHeight, yuvWidth, CV_8UC1, y4, yuvStride);

    cv::Mat wrappedEmpty, modEmpty;
    calculateWrappedPhase(I1, I2, I3, I4, wrappedEmpty, modEmpty);

    cv::Mat* heapEmptyPhase = new cv::Mat();
    wrappedEmpty.copyTo(*heapEmptyPhase);

    return reinterpret_cast<jlong>(heapEmptyPhase);
}


// =========================================================================
// [METHOD 3] STEP 2: Measure Lens with Aberration Removal
// =========================================================================
extern "C" JNIEXPORT jobject JNICALL
Java_com_openlensometer_engine_NativeLensEngine_measureLens(
        JNIEnv* env, jobject /* this */,
        jlong emptyFieldPtr, 
        jobject yuvBuffer1, jobject yuvBuffer2, 
        jobject yuvBuffer3, jobject yuvBuffer4,
        jint yuvWidth, jint yuvHeight, jint yuvStride,
        jobject depthBuffer, jint depthWidth, jint depthHeight,
        jfloat systemLambdaMm, jfloat pupilRadiusMm) { // <-- Dynamic hardware variables added

    if (emptyFieldPtr == 0) return nullptr; 

    EXTRACT_YUV_BUFFERS();

    uint16_t* depthData = static_cast<uint16_t*>(env->GetDirectBufferAddress(depthBuffer));
    if (!depthData) return nullptr;

    float zLensMm = calculateZLensFromDepth(depthData, depthWidth, depthHeight);
    if (zLensMm < 0.0f) return nullptr;

    cv::Mat lensWrapped, modulationMap;
    calculateWrappedPhase(I1, I2, I3, I4, lensWrapped, modulationMap);

    cv::Mat* emptyWrappedPhase = reinterpret_cast<cv::Mat*>(emptyFieldPtr);
    cv::Mat cleanLensDeviation = subtractEmptyField(lensWrapped, *emptyWrappedPhase);

    cv::Mat unwrappedPhase;
    unwrapPhaseQualityGuided(cleanLensDeviation, modulationMap, unwrappedPhase, 10.0f);

    cv::Mat mask;
    cv::threshold(modulationMap, mask, 10.0, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    
    // Zernike Math dynamically scales using the calibrated lambda and pupil radius
    OpticalPrescription rx = fitZernikeAndExtractPrescription(unwrappedPhase, mask, systemLambdaMm, pupilRadiusMm);

    LOGI("Calibrated measurement complete. Sphere: %.2f D, Cylinder: %.2f D, Axis: %.1f deg", 
         rx.sphere, rx.cylinder, rx.axis);
    
    return createJavaLensResult(env, rx.sphere, rx.cylinder, rx.axis, zLensMm);
}


// =========================================================================
// [METHOD 4] STEP 3: Free Memory
// =========================================================================
extern "C" JNIEXPORT void JNICALL
Java_com_openlensometer_engine_NativeLensEngine_releaseEmptyField(
        JNIEnv* env, jobject /* this */, jlong emptyFieldPtr) {
    
    if (emptyFieldPtr != 0) {
        cv::Mat* mat = reinterpret_cast<cv::Mat*>(emptyFieldPtr);
        delete mat;
    }
}
}
