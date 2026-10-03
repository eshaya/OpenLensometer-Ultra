#include "frame_stack.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace {

// Typical camera noise is a few grey levels; never reject frames differing by less than this.
constexpr double MIN_REJECTION_THRESHOLD = 2.0;
constexpr double REJECTION_FACTOR = 2.0;

cv::Mat perPixelMedian(const std::vector<cv::Mat>& frames) {
    cv::Mat median(frames.front().size(), CV_8UC1);
    std::vector<uint8_t> values(frames.size());
    std::vector<const uint8_t*> rows(frames.size());
    for (int row = 0; row < median.rows; ++row) {
        for (size_t index = 0; index < frames.size(); ++index) {
            rows[index] = frames[index].ptr<uint8_t>(row);
        }
        uint8_t* output = median.ptr<uint8_t>(row);
        for (int column = 0; column < median.cols; ++column) {
            for (size_t index = 0; index < frames.size(); ++index) {
                values[index] = rows[index][column];
            }
            const auto middle = values.begin() + values.size() / 2;
            std::nth_element(values.begin(), middle, values.end());
            output[column] = *middle;
        }
    }
    return median;
}

}  // namespace

cv::Mat combineFrames(const std::vector<cv::Mat>& frames, int& keptCount) {
    keptCount = 0;
    if (frames.empty()) return cv::Mat();
    // Work on 8-bit frames so 32 camera frames don't need 32 float copies.
    for (const cv::Mat& frame : frames) {
        CV_Assert(frame.size() == frames.front().size() && frame.type() == CV_8UC1);
    }
    cv::Mat sum = cv::Mat::zeros(frames.front().size(), CV_32F);
    if (frames.size() < 3) {
        // Too few frames to tell a bad one from a good one; average them all.
        for (const cv::Mat& frame : frames) cv::accumulate(frame, sum);
        keptCount = static_cast<int>(frames.size());
        return sum / static_cast<double>(keptCount);
    }

    const cv::Mat median = perPixelMedian(frames);
    std::vector<double> scores(frames.size());
    cv::Mat difference;
    for (size_t index = 0; index < frames.size(); ++index) {
        cv::absdiff(frames[index], median, difference);
        scores[index] = cv::mean(difference)[0];
    }
    std::vector<double> sortedScores = scores;
    std::nth_element(sortedScores.begin(), sortedScores.begin() + sortedScores.size() / 2,
                     sortedScores.end());
    const double threshold = std::max(MIN_REJECTION_THRESHOLD,
                                      REJECTION_FACTOR * sortedScores[sortedScores.size() / 2]);

    for (size_t index = 0; index < frames.size(); ++index) {
        if (scores[index] > threshold) continue;
        cv::accumulate(frames[index], sum);
        ++keptCount;
    }
    // At least half the frames score at or below the median score, so keptCount > 0.
    return sum / static_cast<double>(keptCount);
}
