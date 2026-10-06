// File: app/src/main/cpp/lensometer_engine.cpp
#include <jni.h>
#include <array>
#include <cmath>
#include <cstdio>
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
#include "alignment_detector.h"
#include "frame_stack.h"
#include "lens_measurement.h"
#include "grid_ftp.h"

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
    std::array<cv::Mat, PHASE_AXIS_COUNT> brightnessMap;
    // Repeated exposures of the page being captured, combined when it is committed.
    int pendingAxis = -1;
    int pendingPhase = -1;
    std::vector<cv::Mat> pendingFrames;
};

constexpr size_t MAX_FRAMES_PER_PHASE = 32;

// The no-lens run, kept while lens runs are captured and compared against it.
FringeRun referenceRun;

// Single-frame (grid) method: averaged no-lens grid frames, its carriers and phase maps.
struct GridReference {
    cv::Mat sum;
    int frames = 0;
    bool ready = false;
    GridCarriers carriers;
    FringeRun run;
};
GridReference gridReference;
constexpr int MAX_GRID_REFERENCE_FRAMES = 32;

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

JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_detectAlignmentTarget(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride,
        jfloat monitorDownX, jfloat monitorDownY) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1) return nullptr;

    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return nullptr;

    try {
        const cv::Mat frame(height, width, CV_8UC1, yData,
                            static_cast<size_t>(rowStride));
        cv::Point2f normalizedCenter;
        float normalizedSize = 0.0f;
        int phaseCode = 0;
        if (!detectAlignmentTarget(frame, normalizedCenter, normalizedSize, phaseCode,
                                   cv::Point2f(monitorDownX, monitorDownY))) {
            return nullptr;
        }
        const jfloat values[] = {
            normalizedCenter.x, normalizedCenter.y, normalizedSize,
            static_cast<jfloat>(phaseCode)
        };
        jfloatArray result = env->NewFloatArray(4);
        if (result) env->SetFloatArrayRegion(result, 0, 4, values);
        return result;
    } catch (const cv::Exception& error) {
        LOGE("Alignment target detection failed: %s", error.what());
        return nullptr;
    }
}

JNIEXPORT jint JNICALL
Java_com_openlensometer_engine_NativeLensEngine_decodeAlignmentPhaseCode(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride,
        jfloat normalizedCenterX, jfloat normalizedCenterY, jfloat normalizedTargetSize,
        jfloat monitorDownX, jfloat monitorDownY) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1 ||
            !std::isfinite(normalizedCenterX) || !std::isfinite(normalizedCenterY) ||
            !std::isfinite(normalizedTargetSize) || normalizedTargetSize <= 0.0f) {
        return -1;
    }

    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return -1;

    const cv::Mat frame(height, width, CV_8UC1, yData, static_cast<size_t>(rowStride));
    int phaseCode = -1;
    if (!decodeAlignmentPhaseCode(
            frame, cv::Point2f(normalizedCenterX, normalizedCenterY),
            normalizedTargetSize, phaseCode, cv::Point2f(monitorDownX, monitorDownY))) {
        return -1;
    }
    return phaseCode;
}

JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_estimateMonitorTilt(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride,
        jfloat monitorDownX, jfloat monitorDownY,
        jfloat focalX, jfloat focalY, jfloat principalX, jfloat principalY,
        jfloat cornerOffsetMm) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1 ||
            !std::isfinite(focalX) || !std::isfinite(focalY) || focalX <= 0.0f ||
            focalY <= 0.0f) {
        return nullptr;
    }

    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return nullptr;

    try {
        const cv::Mat frame(height, width, CV_8UC1, yData, static_cast<size_t>(rowStride));
        const cv::Matx33d cameraMatrix(
                focalX, 0.0, principalX,
                0.0, focalY, principalY,
                0.0, 0.0, 1.0);
        float tiltRightDeg = 0.0f;
        float tiltDownDeg = 0.0f;
        float distanceMm = 0.0f;
        cv::Point2f corners[4];
        if (!estimateMonitorTilt(frame, cv::Point2f(monitorDownX, monitorDownY), cameraMatrix,
                                 cornerOffsetMm, tiltRightDeg, tiltDownDeg, distanceMm,
                                 corners)) {
            return nullptr;
        }
        const jfloat values[] = {
            tiltRightDeg, tiltDownDeg, distanceMm,
            corners[0].x, corners[0].y, corners[1].x, corners[1].y,
            corners[2].x, corners[2].y, corners[3].x, corners[3].y
        };
        constexpr jsize valueCount = sizeof(values) / sizeof(values[0]);
        jfloatArray result = env->NewFloatArray(valueCount);
        if (result) env->SetFloatArrayRegion(result, 0, valueCount, values);
        return result;
    } catch (const cv::Exception& error) {
        LOGE("Monitor tilt estimation failed: %s", error.what());
        return nullptr;
    }
}

JNIEXPORT jint JNICALL
Java_com_openlensometer_engine_NativeLensEngine_addPhaseFrame(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride,
        jint axis, jint phaseIndex) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1 ||
            axis < 0 || axis >= PHASE_AXIS_COUNT ||
            phaseIndex < 0 || phaseIndex >= PHASE_STEP_COUNT) {
        return -1;
    }
    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return -2;

    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if ((axis == 1 && phaseCaptureState.nextPhase[0] != PHASE_STEP_COUNT) ||
            phaseCaptureState.nextPhase[axis] != phaseIndex) {
        return -3;
    }
    auto& pending = phaseCaptureState.pendingFrames;
    if (phaseCaptureState.pendingAxis != axis || phaseCaptureState.pendingPhase != phaseIndex ||
            (!pending.empty() && pending.front().size() != cv::Size(width, height))) {
        pending.clear();
        phaseCaptureState.pendingAxis = axis;
        phaseCaptureState.pendingPhase = phaseIndex;
    }
    if (pending.size() < MAX_FRAMES_PER_PHASE) {
        const cv::Mat frame(height, width, CV_8UC1, yData, static_cast<size_t>(rowStride));
        pending.push_back(frame.clone());
    }
    return static_cast<jint>(pending.size());
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
        if (phaseCaptureState.pendingAxis == axis && phaseCaptureState.pendingPhase == phaseIndex &&
                !phaseCaptureState.pendingFrames.empty()) {
            int keptCount = 0;
            phaseCaptureState.frames[axis][phaseIndex] =
                    combineFrames(phaseCaptureState.pendingFrames, keptCount);
            LOGI("Phase %d/%d: averaged %d of %zu frames", axis, phaseIndex, keptCount,
                 phaseCaptureState.pendingFrames.size());
        } else {
            frame.copyTo(phaseCaptureState.frames[axis][phaseIndex]);
        }
        phaseCaptureState.pendingFrames.clear();
        phaseCaptureState.pendingAxis = -1;
        phaseCaptureState.pendingPhase = -1;
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
            cv::Mat brightness = cv::Mat::zeros(phaseCaptureState.frames[axis][0].size(), CV_32F);
            for (const cv::Mat& step : phaseCaptureState.frames[axis]) {
                cv::accumulate(step, brightness);
            }
            phaseCaptureState.brightnessMap[axis] = brightness / static_cast<double>(PHASE_STEP_COUNT);
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

JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_getCaptureQualitySummary(
        JNIEnv* env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if (phaseCaptureState.nextPhase[0] != PHASE_STEP_COUNT ||
            phaseCaptureState.nextPhase[1] != PHASE_STEP_COUNT ||
            phaseCaptureState.modulationMap[0].empty() ||
            phaseCaptureState.modulationMap[1].empty()) {
        return nullptr;
    }

    std::vector<uint16_t> depthSamples;
    for (int axis = 0; axis < PHASE_AXIS_COUNT; ++axis) {
        for (int phase = 0; phase < PHASE_STEP_COUNT; ++phase) {
            const cv::Mat& depth = phaseCaptureState.depthFrames[axis][phase];
            const int centerX = depth.cols / 2;
            const int centerY = depth.rows / 2;
            for (int y = std::max(0, centerY - 2); y <= std::min(depth.rows - 1, centerY + 2); ++y) {
                const uint16_t* row = depth.ptr<uint16_t>(y);
                for (int x = std::max(0, centerX - 2); x <= std::min(depth.cols - 1, centerX + 2); ++x) {
                    if (row[x] > 0) depthSamples.push_back(row[x]);
                }
            }
        }
    }

    if (depthSamples.empty()) return nullptr;
    std::sort(depthSamples.begin(), depthSamples.end());
    const size_t lowIndex = depthSamples.size() / 10;
    const size_t medianIndex = depthSamples.size() / 2;
    const size_t highIndex = (depthSamples.size() * 9) / 10;
    const jfloat values[] = {
        static_cast<jfloat>(cv::mean(phaseCaptureState.modulationMap[0])[0]),
        static_cast<jfloat>(cv::mean(phaseCaptureState.modulationMap[1])[0]),
        static_cast<jfloat>(depthSamples[lowIndex]),
        static_cast<jfloat>(depthSamples[medianIndex]),
        static_cast<jfloat>(depthSamples[highIndex]),
        static_cast<jfloat>(depthSamples.size())
    };
    jfloatArray result = env->NewFloatArray(6);
    if (result) env->SetFloatArrayRegion(result, 0, 6, values);
    return result;
}

JNIEXPORT void JNICALL
Java_com_openlensometer_engine_NativeLensEngine_resetPhaseCapture(
        JNIEnv* /* env */, jobject /* this */) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    phaseCaptureState = PhaseCaptureState{};
}

JNIEXPORT void JNICALL
Java_com_openlensometer_engine_NativeLensEngine_clearReferenceRun(
        JNIEnv* /* env */, jobject /* this */) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    referenceRun = FringeRun{};
    gridReference = GridReference{};
}

// Keeps the just-finished run as the no-lens reference and clears capture for a lens run.
// Returns {X period, Y period} in camera pixels per fringe, or null if the run is incomplete.
JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_saveReferenceRun(
        JNIEnv* env, jobject /* this */, jfloat monitorDownX, jfloat monitorDownY) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if (phaseCaptureState.nextPhase[0] != PHASE_STEP_COUNT ||
            phaseCaptureState.nextPhase[1] != PHASE_STEP_COUNT) {
        return nullptr;
    }
    for (int axis = 0; axis < PHASE_AXIS_COUNT; ++axis) {
        referenceRun.wrapped[axis] = phaseCaptureState.wrappedPhase[axis].clone();
        referenceRun.modulation[axis] = phaseCaptureState.modulationMap[axis].clone();
        referenceRun.brightness[axis] = phaseCaptureState.brightnessMap[axis].clone();
    }
    phaseCaptureState = PhaseCaptureState{};

    double periods[2] = {-1.0, -1.0};
    fringePeriodPixels(referenceRun.wrapped, referenceRun.modulation,
                       cv::Point2f(monitorDownX, monitorDownY), periods);
    const jfloat values[] = {static_cast<jfloat>(periods[0]), static_cast<jfloat>(periods[1])};
    jfloatArray result = env->NewFloatArray(2);
    if (result) env->SetFloatArrayRegion(result, 0, 2, values);
    return result;
}

// Adds one no-lens frame of the grid to the grid reference. Returns frames collected so far.
JNIEXPORT jint JNICALL
Java_com_openlensometer_engine_NativeLensEngine_addGridReferenceFrame(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1) return -1;
    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return -2;
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if (gridReference.frames >= MAX_GRID_REFERENCE_FRAMES) return gridReference.frames;
    const cv::Mat frame(height, width, CV_8UC1, yData, static_cast<size_t>(rowStride));
    if (gridReference.sum.empty() || gridReference.sum.size() != frame.size()) {
        gridReference = GridReference{};
        gridReference.sum = cv::Mat::zeros(frame.size(), CV_32F);
    }
    cv::accumulate(frame, gridReference.sum);
    return ++gridReference.frames;
}

// Averages the grid reference frames, finds the grid's two carriers and keeps the phase
// maps. Returns {X period, Y period} in camera pixels, or null if it can't.
JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_finishGridReference(
        JNIEnv* env, jobject /* this */, jfloat monitorDownX, jfloat monitorDownY,
        jstring debugPath) {
    std::lock_guard<std::mutex> lock(phaseCaptureMutex);
    if (gridReference.frames == 0) return nullptr;
    try {
        const cv::Mat average = gridReference.sum / static_cast<double>(gridReference.frames);
        // Keep a copy for offline analysis.
        if (debugPath) {
            const char* path = env->GetStringUTFChars(debugPath, nullptr);
            cv::Mat image;
            average.convertTo(image, CV_8U);
            if (FILE* file = std::fopen(path, "wb")) {
                std::fprintf(file, "P5\n%d %d\n255\n", image.cols, image.rows);
                for (int row = 0; row < image.rows; ++row) std::fwrite(image.ptr(row), 1, image.cols, file);
                std::fclose(file);
            }
            env->ReleaseStringUTFChars(debugPath, path);
        }
        if (!findGridCarriers(average, cv::Point2f(monitorDownX, monitorDownY),
                              gridReference.carriers)) {
            LOGE("Grid reference: carriers not found");
            return nullptr;
        }
        extractGridRun(average, gridReference.carriers, gridReference.run);
        gridReference.ready = true;
    } catch (const cv::Exception& error) {
        LOGE("Grid reference failed: %s", error.what());
        return nullptr;
    }
    const jfloat values[] = {
        static_cast<jfloat>(1.0 / cv::norm(gridReference.carriers.frequency[0])),
        static_cast<jfloat>(1.0 / cv::norm(gridReference.carriers.frequency[1]))
    };
    LOGI("Grid reference from %d frames: periods %.1f / %.1f px", gridReference.frames,
         values[0], values[1]);
    jfloatArray result = env->NewFloatArray(2);
    if (result) env->SetFloatArrayRegion(result, 0, 2, values);
    return result;
}

// Measures the lens from one grid frame against the grid reference. Same result layout as
// measureLensFromRuns ({status, sphere, cylinder, axis, distance mm, opening px, pixels, rms});
// status 10 = no grid reference.
JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_measureGridFrame(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride,
        jfloat monitorDownX, jfloat monitorDownY, jfloat lensCenterX, jfloat lensCenterY,
        jfloat expectedLensRadius, jfloat screenDistanceMm, jfloat screenOpeningPixels,
        jfloat assumedLensDistanceMm) {
    LensMeasurement measurement;
    float status = 0.0f;
    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1 ||
            !yData || capacity < requiredBytes) {
        status = static_cast<float>(LensMeasurementStatus::BadInput);
    } else {
        // Copy the reference out so the slow part runs without holding the lock.
        FringeRun reference;
        GridCarriers carriers;
        bool ready = false;
        {
            std::lock_guard<std::mutex> lock(phaseCaptureMutex);
            ready = gridReference.ready;
            if (ready) {
                reference = gridReference.run;
                carriers = gridReference.carriers;
            }
        }
        if (!ready) {
            status = 10.0f;
        } else {
            try {
                const cv::Mat frame(height, width, CV_8UC1, yData, static_cast<size_t>(rowStride));
                measurement = measureLensFromGridFrame(
                        reference, carriers, frame, cv::Point2f(monitorDownX, monitorDownY),
                        cv::Point2f(lensCenterX, lensCenterY), expectedLensRadius,
                        screenDistanceMm, screenOpeningPixels, assumedLensDistanceMm);
                status = static_cast<float>(measurement.status);
            } catch (const cv::Exception& error) {
                LOGE("Grid frame measurement failed: %s", error.what());
                status = static_cast<float>(LensMeasurementStatus::BadInput);
            }
        }
    }
    const jfloat values[] = {
        status,
        static_cast<jfloat>(measurement.sphere),
        static_cast<jfloat>(measurement.cylinder),
        static_cast<jfloat>(measurement.axisDegrees),
        static_cast<jfloat>(measurement.lensDistanceMm),
        static_cast<jfloat>(measurement.lensOpeningPixels),
        static_cast<jfloat>(measurement.pixelsUsed),
        static_cast<jfloat>(measurement.rmsResidualPixels),
        static_cast<jfloat>(measurement.lensType),
        static_cast<jfloat>(measurement.distanceSphere),
        static_cast<jfloat>(measurement.distanceCylinder),
        static_cast<jfloat>(measurement.distanceAxis),
        static_cast<jfloat>(measurement.nearSphere),
        static_cast<jfloat>(measurement.nearCylinder),
        static_cast<jfloat>(measurement.nearAxis),
        static_cast<jfloat>(measurement.addPower)
    };
    constexpr jsize valueCount = sizeof(values) / sizeof(values[0]);
    jfloatArray result = env->NewFloatArray(valueCount);
    if (result) env->SetFloatArrayRegion(result, 0, valueCount, values);
    return result;
}

// Size of the lens opening (sqrt of area, image pixels) with the glasses held flat on the
// bright screen, or -1 if no rim is found around (centerX, centerY).
JNIEXPORT jfloat JNICALL
Java_com_openlensometer_engine_NativeLensEngine_measureLensOpening(
        JNIEnv* env, jobject /* this */, jobject yPlane,
        jint width, jint height, jint rowStride, jint pixelStride,
        jfloat centerX, jfloat centerY, jfloat minSizePixels) {
    if (width <= 0 || height <= 0 || rowStride < width || pixelStride != 1) return -1.0f;
    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yPlane));
    const jlong capacity = env->GetDirectBufferCapacity(yPlane);
    const jlong requiredBytes = static_cast<jlong>(rowStride) * (height - 1) + width;
    if (!yData || capacity < requiredBytes) return -1.0f;
    try {
        const cv::Mat frame(height, width, CV_8UC1, yData, static_cast<size_t>(rowStride));
        double opening = 0.0;
        if (!measureLensOpening(frame, cv::Point2f(centerX, centerY), minSizePixels, opening)) {
            return -1.0f;
        }
        return static_cast<jfloat>(opening);
    } catch (const cv::Exception& error) {
        LOGE("Lens opening measurement failed: %s", error.what());
        return -1.0f;
    }
}

// Compares the finished lens run with the reference run. Returns {status, sphere, cylinder,
// axis, lens distance mm, lens opening px, pixels used, rms residual px}; status follows
// LensMeasurementStatus, plus 10 = no reference, 11 = lens run incomplete.
JNIEXPORT jfloatArray JNICALL
Java_com_openlensometer_engine_NativeLensEngine_measureLensFromRuns(
        JNIEnv* env, jobject /* this */, jfloat monitorDownX, jfloat monitorDownY,
        jfloat lensCenterX, jfloat lensCenterY,
        jfloat screenDistanceMm, jfloat screenOpeningPixels, jstring debugPath) {
    cv::Mat brightnessRatio;
    LensMeasurement measurement;
    float status = 0.0f;
    {
        std::lock_guard<std::mutex> lock(phaseCaptureMutex);
        if (referenceRun.wrapped[0].empty() || referenceRun.wrapped[1].empty()) {
            status = 10.0f;
        } else if (phaseCaptureState.nextPhase[0] != PHASE_STEP_COUNT ||
                   phaseCaptureState.nextPhase[1] != PHASE_STEP_COUNT) {
            status = 11.0f;
        } else {
            try {
                FringeRun lensRun;
                for (int axis = 0; axis < PHASE_AXIS_COUNT; ++axis) {
                    lensRun.wrapped[axis] = phaseCaptureState.wrappedPhase[axis];
                    lensRun.modulation[axis] = phaseCaptureState.modulationMap[axis];
                    lensRun.brightness[axis] = phaseCaptureState.brightnessMap[axis];
                }
                measurement = measureLensPower(
                        referenceRun, lensRun,
                        cv::Point2f(monitorDownX, monitorDownY),
                        cv::Point2f(lensCenterX, lensCenterY),
                        screenDistanceMm, screenOpeningPixels, &brightnessRatio);
                status = static_cast<float>(measurement.status);
            } catch (const cv::Exception& error) {
                LOGE("Lens measurement failed: %s", error.what());
                status = static_cast<float>(LensMeasurementStatus::BadInput);
            }
        }
    }
    // Save lens-run/reference brightness (rim = dark) as a PGM for diagnosis.
    if (debugPath && !brightnessRatio.empty()) {
        const char* path = env->GetStringUTFChars(debugPath, nullptr);
        cv::Mat image;
        brightnessRatio.convertTo(image, CV_8U, 200.0);
        if (FILE* file = std::fopen(path, "wb")) {
            std::fprintf(file, "P5\n%d %d\n255\n", image.cols, image.rows);
            for (int row = 0; row < image.rows; ++row) std::fwrite(image.ptr(row), 1, image.cols, file);
            std::fclose(file);
        }
        env->ReleaseStringUTFChars(debugPath, path);
    }
    LOGI("Lens measurement status %.0f: sphere %.2f cyl %.2f axis %.0f d %.0f mm "
         "opening %.0f px, %d px used, rms %.3f px",
         status, measurement.sphere, measurement.cylinder, measurement.axisDegrees,
         measurement.lensDistanceMm, measurement.lensOpeningPixels, measurement.pixelsUsed,
         measurement.rmsResidualPixels);
    const jfloat values[] = {
        status,
        static_cast<jfloat>(measurement.sphere),
        static_cast<jfloat>(measurement.cylinder),
        static_cast<jfloat>(measurement.axisDegrees),
        static_cast<jfloat>(measurement.lensDistanceMm),
        static_cast<jfloat>(measurement.lensOpeningPixels),
        static_cast<jfloat>(measurement.pixelsUsed),
        static_cast<jfloat>(measurement.rmsResidualPixels)
    };
    jfloatArray result = env->NewFloatArray(8);
    if (result) env->SetFloatArrayRegion(result, 0, 8, values);
    return result;
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
