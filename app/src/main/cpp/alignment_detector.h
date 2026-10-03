#ifndef ALIGNMENT_DETECTOR_H
#define ALIGNMENT_DETECTOR_H

#include <opencv2/core.hpp>

// monitorDown is the monitor's downward direction expressed in camera-image pixels.
// The sensor image is usually rotated relative to how the phone is held, so the
// phase ticks drawn below the bullseye on the monitor may lie to the side in the image.
// Finds the central bullseye; phaseCode is the step read from the squares under it, or -1
// if they can't be read.
bool detectAlignmentTarget(
        const cv::Mat& grayscaleFrame, cv::Point2f& normalizedCenter,
        float& normalizedSize, int& phaseCode,
        const cv::Point2f& monitorDown = cv::Point2f(0.0f, 1.0f));

// Reads the step squares drawn under a bullseye at normalizedCenter (searching a little
// around it, since the size estimate is approximate). False if no squares are found there.
bool decodeAlignmentPhaseCode(
        const cv::Mat& grayscaleFrame, const cv::Point2f& normalizedCenter,
        float normalizedSize, int& phaseCode,
        const cv::Point2f& monitorDown = cv::Point2f(0.0f, 1.0f));

// Estimates how the camera is angled to the monitor from the central bullseye plus the four
// smaller corner markers drawn cornerOffsetMm right/left and up/down of it. tiltRightDeg > 0
// means the monitor's right side is farther from the camera; tiltDownDeg > 0 means its
// bottom is farther. Needs the center and at least three corners in view. cornerPoints, if
// given, receives the markers' image positions as top-left, top-right, bottom-right,
// bottom-left (in monitor terms), with NaN for any marker not found.
bool estimateMonitorTilt(
        const cv::Mat& grayscaleFrame, const cv::Point2f& monitorDown,
        const cv::Matx33d& cameraMatrix, float cornerOffsetMm,
        float& tiltRightDeg, float& tiltDownDeg, float& distanceMm,
        cv::Point2f* cornerPoints = nullptr);

#endif
