#include "charuco_calibrator.h"

bool estimate_screen_pose(
    const cv::Mat& input_image, 
    const cv::Mat& camera_matrix, 
    const cv::Mat& dist_coeffs,
    cv::Vec3d& rvec, 
    cv::Vec3d& tvec) 
{
    if (input_image.empty()) return false;

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    const cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    const cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.030f, 0.022f, dictionary);
    cv::aruco::CharucoParameters charuco_parameters;
    charuco_parameters.cameraMatrix = camera_matrix;
    charuco_parameters.distCoeffs = dist_coeffs;
    const cv::aruco::CharucoDetector detector(board, charuco_parameters);

    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    detector.detectBoard(input_image, charuco_corners, charuco_ids);

    if (charuco_corners.size() < 4 || board.checkCharucoCornersCollinear(charuco_ids)) return false;

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    board.matchImagePoints(charuco_corners, charuco_ids, object_points, image_points);

    return object_points.size() >= 4 &&
        cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
#else
    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    const auto board = cv::aruco::CharucoBoard::create(5, 7, 0.030f, 0.022f, dictionary);
    const auto detector_parameters = cv::aruco::DetectorParameters::create();

    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::detectMarkers(
        input_image, dictionary, marker_corners, marker_ids, detector_parameters);
    if (marker_ids.empty()) return false;

    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
        marker_corners, marker_ids, input_image, board, charuco_corners,
        charuco_ids, camera_matrix, dist_coeffs);
    if (charuco_corners.size() < 4) return false;

    return cv::aruco::estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, rvec, tvec);
#endif
}