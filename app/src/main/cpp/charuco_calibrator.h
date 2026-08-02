// File: app/src/main/cpp/charuco_calibrator.h
#ifndef CHARUCO_CALIBRATOR_H
#define CHARUCO_CALIBRATOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>

/**
 * Detects a ChArUco board on the monitor and calculates the physical pixel pitch.
 * 
 * @param frameGray The grayscale camera frame containing the monitor.
 * @param zScreenMm The physical distance to the screen (from the DEPTH16 sensor).
 * @param cameraMatrix The 3x3 intrinsic matrix of the smartphone camera.
 * @param monitorSquareSizePx The size of the ChArUco square in monitor pixels (e.g., 200).
 * @return The physical size of a single monitor pixel in mm, or -1.0f on failure.
 */
float calculateMonitorPixelPitch(const cv::Mat& frameGray, 
                                 float zScreenMm, 
                                 const cv::Mat& cameraMatrix, 
                                 int monitorSquareSizePx);

#endif // CHARUCO_CALIBRATOR_H