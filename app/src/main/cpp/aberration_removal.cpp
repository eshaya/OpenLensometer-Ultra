// File: app/src/main/cpp/aberration_removal.cpp
#include "aberration_removal.h"

cv::Mat subtractEmptyField(const cv::Mat& lensWrapped, const cv::Mat& emptyWrapped) {
    // 1. Direct matrix subtraction
    cv::Mat diff = lensWrapped - emptyWrapped;

    // 2. Re-wrap the differences back to [-pi, pi]
    // Because both inputs are strictly [-pi, pi], the diff is bounded [-2pi, 2pi].
    // A simple if/else check is significantly faster than using fmod() or while loops.
    for (int y = 0; y < diff.rows; ++y) {
        float* ptr = diff.ptr<float>(y);
        for (int x = 0; x < diff.cols; ++x) {
            float val = ptr[x];
            if (val > CV_PI) {
                ptr[x] = val - 2.0f * static_cast<float>(CV_PI);
            } else if (val < -CV_PI) {
                ptr[x] = val + 2.0f * static_cast<float>(CV_PI);
            }
        }
    }

    return diff;
}