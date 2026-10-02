#ifndef CHARUCO_CALIBRATOR_H
#define CHARUCO_CALIBRATOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/charuco_detector.hpp>
#else
#include <opencv2/aruco/charuco.hpp>
#endif

// Extracts the 3D translation (distance) and rotation of the screen
bool estimate_screen_pose(
    const cv::Mat& input_image, 
    const cv::Mat& camera_matrix, 
    const cv::Mat& dist_coeffs,
    cv::Vec3d& rvec, 
    cv::Vec3d& tvec
);

#endif // CHARUCO_CALIBRATOR_H