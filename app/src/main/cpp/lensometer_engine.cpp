// File: app/src/main/cpp/lensometer_engine.cpp
#include <jni.h>
#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>

// Our modular pipeline headers
#include "depth16_filter.h"
#include "phase_unwrapper.h"
#include "zernike_fitter.h"
#include "aberration_removal.h"
#include "charuco_calibrator.h"
#include "monitor_calibration.h"

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

namespace {
constexpr int PHASE_AXIS_COUNT = 2;
constexpr int PHASE_STEP_COUNT = 4;

struct PhaseCaptureState {
    int width = 0;
    int height = 0;
    int depthWidth = 0;
    int depthHeight = 0;
    int nextPhase[PHASE_AXIS_COUNT] = {0, 0};
    std::array<std::array<cv::Mat, PHASE_STEP_COUNT>, PHASE_AXIS_COUNT> frames;
    std::array<std::array<cv::Mat, PHASE_STEP_COUNT>, PHASE_AXIS_COUNT> depthFrames;
    std::array<float, 4> cameraIntrinsics{};
    std::array<cv::Mat, PHASE_AXIS_COUNT> wrappedPhase;
    std::array<cv::Mat, PHASE_AXIS_COUNT> modulationMap;
};

std::mutex phaseCaptureMutex;
PhaseCaptureState phaseCaptureState;
}

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

JNIEXPORT jdouble JNICALL
Java_com_openlensometer_engine_NativeLensEngine_calculateMonitorPixelPitch(
        JNIEnv* /* env */, jobject /* this */, jint screenWidthPixels,
        jint screenHeightPixels, jdouble screenDiagonalInches) {
    return calculateMonitorPixelPitchMm(
            screenWidthPixels, screenHeightPixels, screenDiagonalInches);
}

JNIEXPORT jint JNICALL
Java_com_openlensometer_engine_NativeLensEngine_detectCharucoCorners(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1) return -1;

    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return -1;

    try {
        const cv::Mat frameGray(height, width, CV_8UC1, yData,
                                static_cast<size_t>(rowStride));
        const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        const cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.030f, 0.022f, dictionary);
        const cv::aruco::CharucoDetector detector(board);
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        detector.detectBoard(frameGray, charucoCorners, charucoIds);
        return static_cast<jint>(charucoCorners.size());
    } catch (const cv::Exception& error) {
        LOGE("ChArUco frame processing failed: %s", error.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL
Java_com_openlensometer_engine_NativeLensEngine_capturePhaseFrame(
        JNIEnv* env, jobject /* this */, jobject yPlane,
    jint width, jint height, jint rowStride, jint pixelStride,
    jobject depthPlane, jint depthWidth, jint depthHeight,
    jint depthRowStride, jint depthPixelStride,
    jint axis, jint phaseIndex, jfloat focalX, jfloat focalY,
    jfloat principalX, jfloat principalY) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1 ||
        depthWidth <= 0 || depthHeight <= 0 || depthPixelStride != 2 ||
        depthRowStride < depthWidth * depthPixelStride ||
            axis < 0 || axis >= PHASE_AXIS_COUNT ||
        phaseIndex < 0 || phaseIndex >= PHASE_STEP_COUNT ||
        !std::isfinite(focalX) || !std::isfinite(focalY) ||
        !std::isfinite(principalX) || !std::isfinite(principalY) ||
        focalX <= 0.0f || focalY <= 0.0f) {
        return -1;
    }

    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    auto* depthData = static_cast<uint8_t*>(env->GetDirectBufferAddress(depthPlane));
    const jlong yCapacity = env->GetDirectBufferCapacity(yPlane);
    const jlong depthCapacity = env->GetDirectBufferCapacity(depthPlane);
    const jlong requiredYBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    const jlong requiredDepthBytes =
        static_cast<jlong>(depthRowStride) * (depthHeight - 1) +
        static_cast<jlong>(depthWidth) * depthPixelStride;
    if (!yData || yCapacity < requiredYBytes || !depthData ||
        depthCapacity < requiredDepthBytes) return -2;

    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if ((axis == 1 && phaseCaptureState.nextPhase[0] != PHASE_STEP_COUNT) ||
            phaseCaptureState.nextPhase[axis] != phaseIndex ||
            (phaseCaptureState.width != 0 &&
                         (phaseCaptureState.width != width || phaseCaptureState.height != height ||
                            phaseCaptureState.depthWidth != depthWidth ||
                            phaseCaptureState.depthHeight != depthHeight))) {
        return -3;
    }

    try {
        const cv::Mat frame(height, width, CV_8UC1, yData,
                            static_cast<size_t>(rowStride));
        frame.copyTo(phaseCaptureState.frames[axis][phaseIndex]);
        const cv::Mat depthFrame(depthHeight, depthWidth, CV_16UC1, depthData,
                     static_cast<size_t>(depthRowStride));
        depthFrame.copyTo(phaseCaptureState.depthFrames[axis][phaseIndex]);
        phaseCaptureState.cameraIntrinsics = {focalX, focalY, principalX, principalY};
        if (phaseIndex == PHASE_STEP_COUNT - 1) {
            cv::Mat wrappedPhase;
            cv::Mat modulationMap;
            calculateWrappedPhase(
                    phaseCaptureState.frames[axis][0],
                    phaseCaptureState.frames[axis][1],
                    phaseCaptureState.frames[axis][2],
                    phaseCaptureState.frames[axis][3],
                    wrappedPhase, modulationMap);
            phaseCaptureState.wrappedPhase[axis] = std::move(wrappedPhase);
            phaseCaptureState.modulationMap[axis] = std::move(modulationMap);
        }

        phaseCaptureState.width = width;
        phaseCaptureState.height = height;
        phaseCaptureState.depthWidth = depthWidth;
        phaseCaptureState.depthHeight = depthHeight;
        return ++phaseCaptureState.nextPhase[axis];
    } catch (const cv::Exception& error) {
        LOGE("Phase capture failed: %s", error.what());
        return -4;
    }
}

JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_getCapturedCameraDepthCalibration(
        JNIEnv* env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if (phaseCaptureState.nextPhase[0] != PHASE_STEP_COUNT ||
            phaseCaptureState.nextPhase[1] != PHASE_STEP_COUNT) {
        return nullptr;
    }

    std::vector<uint16_t> centerDepthSamples;
    for (int axis = 0; axis < PHASE_AXIS_COUNT; ++axis) {
        for (int phase = 0; phase < PHASE_STEP_COUNT; ++phase) {
            const cv::Mat& depth = phaseCaptureState.depthFrames[axis][phase];
            const int centerX = depth.cols / 2;
            const int centerY = depth.rows / 2;
            for (int y = std::max(0, centerY - 2); y <= std::min(depth.rows - 1, centerY + 2); ++y) {
                const uint16_t* row = depth.ptr<uint16_t>(y);
                for (int x = std::max(0, centerX - 2); x <= std::min(depth.cols - 1, centerX + 2); ++x) {
                    if (row[x] > 0) centerDepthSamples.push_back(row[x]);
                }
            }
        }
    }
    if (centerDepthSamples.empty()) return nullptr;

    std::sort(centerDepthSamples.begin(), centerDepthSamples.end());
    const float centerDepthMm = static_cast<float>(
            centerDepthSamples[centerDepthSamples.size() / 2]);
    const jfloat values[] = {
        phaseCaptureState.cameraIntrinsics[0],
        phaseCaptureState.cameraIntrinsics[1],
        phaseCaptureState.cameraIntrinsics[2],
        phaseCaptureState.cameraIntrinsics[3],
        centerDepthMm
    };
    jfloatArray result = env->NewFloatArray(5);
    if (result) env->SetFloatArrayRegion(result, 0, 5, values);
    return result;
}

JNIEXPORT void JNICALL
Java_com_openlensometer_engine_NativeLensEngine_resetPhaseCapture(
        JNIEnv* /* env */, jobject /* this */) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    phaseCaptureState = PhaseCaptureState{};
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

    if (zScreenMm <= 0.0f || monitorSquareSizePx <= 0) return -1.0f;

    cv::Vec3d rotation;
    cv::Vec3d translation;
    if (!estimate_screen_pose(frameGray, cameraMatrix64F, cv::Mat(), rotation, translation) ||
            translation[2] <= 0.0) {
        LOGE("Calibration failed: ChArUco board pose could not be estimated.");
        return -1.0f;
    }

    constexpr double charucoSquareSizeMm = 30.0;
    const double estimatedDistanceMm = translation[2] * 1000.0;
    const double scaleCorrection = zScreenMm / estimatedDistanceMm;
    const float pixelPitchMm = static_cast<float>(
            charucoSquareSizeMm * scaleCorrection / monitorSquareSizePx);
    
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
