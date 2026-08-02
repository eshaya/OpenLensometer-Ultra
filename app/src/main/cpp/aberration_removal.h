// File: app/src/main/cpp/aberration_removal.h
#ifndef ABERRATION_REMOVAL_H
#define ABERRATION_REMOVAL_H

#include <opencv2/opencv.hpp>

/**
 * Subtracts the empty-field phase from the lens phase in the wrapped domain.
 * This isolates the optical deviation caused purely by the eyeglass lens.
 * 
 * @param lensWrapped The wrapped phase map captured with the glasses.
 * @param emptyWrapped The wrapped phase map captured of the empty screen.
 * @return The isolated phase difference, correctly re-wrapped to [-pi, pi].
 */
cv::Mat subtractEmptyField(const cv::Mat& lensWrapped, const cv::Mat& emptyWrapped);

#endif // ABERRATION_REMOVAL_H