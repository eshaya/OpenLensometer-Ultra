#include "charuco_calibrator.h"

bool estimate_screen_pose(
    const cv::Mat& input_image, 
    const cv::Mat& camera_matrix, 
    const cv::Mat& dist_coeffs,
    cv::Vec3d& rvec, 
    cv::Vec3d& tvec) 
{
    if (input_image.empty()) return false;

    // 1. Define the physical board geometry
    cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.030f, 0.022f, dict);
    
    // 2. Set up the detectors
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    
    // 3. Detect the ArUco markers first
    cv::aruco::detectMarkers(input_image, dict, marker_corners, marker_ids, params);
    
    if (marker_ids.empty()) return false;

    // 4. Refine the detection to find the inner chessboard corners
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
        marker_corners, marker_ids, input_image, board, charuco_corners, charuco_ids, camera_matrix, dist_coeffs
    );

    // 5. If we found enough corners, solve for the 3D pose (distance and tilt)
    if (charuco_corners.size() >= 4) {
        return cv::aruco::estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, rvec, tvec
        );
    }

    return false;
}