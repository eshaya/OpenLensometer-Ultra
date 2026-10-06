#include <algorithm>
#include <vector>
// File: app/src/main/cpp/phase_unwrapper.cpp
#include "phase_unwrapper.h"
#include <queue>
#include <vector>

// Helper struct for the Priority Queue
struct PixelNode {
    int x, y;
    float modulation;
    
    // We want the highest modulation at the top of the queue
    bool operator<(const PixelNode& other) const {
        return modulation < other.modulation; 
    }
};

// Helper function to resolve the 2-pi jumps between adjacent pixels
inline float wrapDifference(float phase, float referencePhase) {
    float diff = phase - referencePhase;
    while (diff > CV_PI)  diff -= 2.0f * CV_PI;
    while (diff < -CV_PI) diff += 2.0f * CV_PI;
    return referencePhase + diff;
}

void calculateWrappedPhase(const cv::Mat& I1, const cv::Mat& I2,
                           const cv::Mat& I3, const cv::Mat& I4,
                           cv::Mat& wrappedPhase, cv::Mat& modulationMap) {
    CV_Assert(!I1.empty() && !I2.empty() && !I3.empty() && !I4.empty());
    CV_Assert(I1.size() == I2.size() && I1.size() == I3.size() && I1.size() == I4.size());
    CV_Assert(I1.channels() == 1 && I2.channels() == 1 && I3.channels() == 1 && I4.channels() == 1);

    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat frame3;
    cv::Mat frame4;
    I1.convertTo(frame1, CV_32F);
    I2.convertTo(frame2, CV_32F);
    I3.convertTo(frame3, CV_32F);
    I4.convertTo(frame4, CV_32F);

    const cv::Mat cosineTerm = frame1 - frame3;
    const cv::Mat sineTerm = frame4 - frame2;
    cv::phase(cosineTerm, sineTerm, wrappedPhase);

    const float twoPi = 2.0f * static_cast<float>(CV_PI);
    for (int row = 0; row < wrappedPhase.rows; ++row) {
        float* phaseValues = wrappedPhase.ptr<float>(row);
        for (int column = 0; column < wrappedPhase.cols; ++column) {
            if (phaseValues[column] > CV_PI) phaseValues[column] -= twoPi;
        }
    }

    cv::magnitude(cosineTerm, sineTerm, modulationMap);
    modulationMap *= 0.5f;
}

void unwrapPhaseQualityGuided(const cv::Mat& wrappedPhase, const cv::Mat& modulationMap,
                              cv::Mat& unwrappedPhase, float modThreshold) {
    const int rows = wrappedPhase.rows;
    const int cols = wrappedPhase.cols;
    unwrappedPhase = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat processed = cv::Mat::zeros(rows, cols, CV_8U);

    // Seeds, best quality first. Each flood covers one connected usable region; regions cut
    // off from each other (e.g. by a bifocal segment line) get their own seed, so none is
    // left un-unwrapped. Their relative 2π offsets are arbitrary, which only shifts a
    // constant that local fits absorb.
    std::vector<std::pair<float, int>> seeds;
    for (int y = 0; y < rows; ++y) {
        const float* quality = modulationMap.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (quality[x] >= modThreshold) seeds.emplace_back(quality[x], y * cols + x);
        }
    }
    std::sort(seeds.begin(), seeds.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

    const int dx[] = {0, 0, -1, 1};
    const int dy[] = {-1, 1, 0, 0};
    std::priority_queue<PixelNode> queue;
    for (const auto& seed : seeds) {
        const int seedX = seed.second % cols;
        const int seedY = seed.second / cols;
        if (processed.at<uchar>(seedY, seedX)) continue;
        unwrappedPhase.at<float>(seedY, seedX) = wrappedPhase.at<float>(seedY, seedX);
        processed.at<uchar>(seedY, seedX) = 1;
        queue.push({seedX, seedY, seed.first});

        // Flood-fill from highest quality to lowest.
        while (!queue.empty()) {
            const PixelNode current = queue.top();
            queue.pop();
            const float referencePhase = unwrappedPhase.at<float>(current.y, current.x);
            for (int i = 0; i < 4; ++i) {
                const int nx = current.x + dx[i];
                const int ny = current.y + dy[i];
                if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
                if (processed.at<uchar>(ny, nx) != 0 || modulationMap.at<float>(ny, nx) < modThreshold) continue;
                unwrappedPhase.at<float>(ny, nx) =
                        wrapDifference(wrappedPhase.at<float>(ny, nx), referencePhase);
                processed.at<uchar>(ny, nx) = 1;
                queue.push({nx, ny, modulationMap.at<float>(ny, nx)});
            }
        }
    }
}
