// File: app/src/main/cpp/charuco_calibrator.cpp
#include "charuco_calibrator.h"
#include <vector>
#include <cmath>
#include <android/log.h>

#define LOG_TAG "OpenLensometer-Calib"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

float calculateMonitorPixelPitch(const cv::Mat& frameGray, 
                                 float zScreenMm, 
                                 const cv::Mat& cameraMatrix, 
                                 int monitorSquareSizePx) {
    
    // 1. Initialize the ChArUco dictionary (Standard 5x5 board)
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 5, 1.0f, 0.8f, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

    // 2. Detect the raw ArUco markers
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(frameGray, dictionary, markerCorners, markerIds, params);

    if (markerIds.empty()) {
        LOGE("Calibration failed: No ArUco markers detected.");
        return -1.0f;
    }

    // 3. Interpolate the inner ChArUco corners for sub-pixel accuracy
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, frameGray, board, charucoCorners, charucoIds);

    if (charucoCorners.size() < 2) {
        LOGE("Calibration failed: Not enough ChArUco corners interpolated.");
        return -1.0f;
    }

    // 4. Calculate the average distance between adjacent corners in the camera image
    // We average multiple horizontal and vertical pairs to account for slight camera tilt
    double totalCameraPixelDist = 0.0;
    int pairCount = 0;

    for (size_t i = 0; i < charucoCorners.size() - 1; ++i) {
        // Only compare corners that are physically adjacent on the grid
        int idDiff = std::abs(charucoIds[i] - charucoIds[i+1]);
        if (idDiff == 1 || idDiff == 4) { // 4 is the grid width (5 squares = 4 inner corners)
            cv::Point2f p1 = charucoCorners[i];
            cv::Point2f p2 = charucoCorners[i+1];
            double dist = cv::norm(p1 - p2);
            totalCameraPixelDist += dist;
            pairCount++;
        }
    }

    if (pairCount == 0) return -1.0f;

    double avgCameraPixelDist = totalCameraPixelDist / pairCount;

    // 5. Apply the Optical Projection Math
    // Extract the camera focal length in pixels (average of fx and fy)
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double focalLengthPx = (fx + fy) / 2.0;

    // Project the camera pixel distance to the physical plane at the monitor
    double physicalSquareSizeMm = (avgCameraPixelDist * zScreenMm) / focalLengthPx;

    // 6. Calculate the final monitor pixel pitch
    float monitorPixelPitchMm = static_cast<float>(physicalSquareSizeMm / monitorSquareSizePx);

    return monitorPixelPitchMm;
}