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
    
    int rows = wrappedPhase.rows;
    int cols = wrappedPhase.cols;
    
    // Initialize output map and a tracking mask
    unwrappedPhase = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat processed = cv::Mat::zeros(rows, cols, CV_8U);
    
    // 1. Find the best starting seed (Highest modulation)
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(modulationMap, &minVal, &maxVal, &minLoc, &maxLoc);
    
    std::priority_queue<PixelNode> pq;
    
    // 2. Initialize the seed
    unwrappedPhase.at<float>(maxLoc.y, maxLoc.x) = wrappedPhase.at<float>(maxLoc.y, maxLoc.x);
    processed.at<uchar>(maxLoc.y, maxLoc.x) = 1;
    pq.push({maxLoc.x, maxLoc.y, static_cast<float>(maxVal)});
    
    // 4-connected neighborhood offsets (Up, Down, Left, Right)
    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};
    
    // 3. Flood-fill from highest quality to lowest
    while (!pq.empty()) {
        PixelNode current = pq.top();
        pq.pop();
        
        float refPhase = unwrappedPhase.at<float>(current.y, current.x);
        
        for (int i = 0; i < 4; ++i) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];
            
            // Bounds check
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                // If not yet processed AND modulation is above our temple-masking threshold
                if (processed.at<uchar>(ny, nx) == 0 && modulationMap.at<float>(ny, nx) >= modThreshold) {
                    
                    // Read the wrapped value of the neighbor
                    float neighborWrapped = wrappedPhase.at<float>(ny, nx);
                    
                    // Unwrap it relative to the current pixel
                    unwrappedPhase.at<float>(ny, nx) = wrapDifference(neighborWrapped, refPhase);
                    
                    // Mark as processed and add to queue
                    processed.at<uchar>(ny, nx) = 1;
                    pq.push({nx, ny, modulationMap.at<float>(ny, nx)});
                }
            }
        }
    }
}