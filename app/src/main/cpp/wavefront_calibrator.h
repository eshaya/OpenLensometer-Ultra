#ifndef WAVEFRONT_CALIBRATOR_H
#define WAVEFRONT_CALIBRATOR_H

#include <opencv2/opencv.hpp>

// Subtracts the baseline hardware reference phase from the raw measurement
cv::Mat subtract_empty_field(const cv::Mat& raw_measurement, const cv::Mat& reference_phase);

#endif // WAVEFRONT_CALIBRATOR_H