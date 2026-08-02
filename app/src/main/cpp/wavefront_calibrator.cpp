#include "wavefront_calibrator.h"
#include <stdexcept>

cv::Mat subtract_empty_field(const cv::Mat& raw_measurement, const cv::Mat& reference_phase) {
    // Ensure both matrices are the same size and type before subtracting
    if (raw_measurement.empty() || reference_phase.empty()) {
        return cv::Mat();
    }
    if (raw_measurement.size() != reference_phase.size() || raw_measurement.type() != reference_phase.type()) {
        throw std::invalid_argument("Matrix size or type mismatch in empty field subtraction.");
    }

    return raw_measurement - reference_phase;
}