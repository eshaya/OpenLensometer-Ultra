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