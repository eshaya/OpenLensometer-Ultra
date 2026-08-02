#include "depth16_filter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

const uint16_t MIN_VALID_DEPTH_MM = 100;
const uint16_t MAX_VALID_DEPTH_MM = 600;
const float RIM_RECESS_OFFSET_MM = 1.5f;

float calculateZLensFromDepth(const uint16_t* depthData, int width, int height) {
    cv::Mat depthMap(height, width, CV_16U, (void*)depthData);
    cv::Mat validMask;
    cv::inRange(depthMap, MIN_VALID_DEPTH_MM, MAX_VALID_DEPTH_MM, validMask);

    cv::Mat depthFloat;
    depthMap.convertTo(depthFloat, CV_32F);

    cv::Mat grad;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(depthFloat, grad, cv::MORPH_GRADIENT, kernel);

    cv::Mat edgeMask;
    cv::threshold(grad, edgeMask, 4.0, 255, cv::THRESH_BINARY);
    edgeMask.convertTo(edgeMask, CV_8U);
    
    cv::bitwise_and(edgeMask, validMask, edgeMask);

    std::vector<uint16_t> edgeDepths;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edgeMask.at<uchar>(y, x) > 0) {
                edgeDepths.push_back(depthMap.at<uint16_t>(y, x));
            }
        }
    }

    if (edgeDepths.empty()) return -1.0f; 

    std::sort(edgeDepths.begin(), edgeDepths.end());

    size_t startIndex = edgeDepths.size() * 0.30;
    size_t endIndex = edgeDepths.size() * 0.80;

    double sumRimDepth = 0.0;
    int validCount = 0;
    for (size_t i = startIndex; i < endIndex; ++i) {
        sumRimDepth += edgeDepths[i];
        validCount++;
    }

    if (validCount == 0) return -1.0f;

    float avgRimDepthMm = static_cast<float>(sumRimDepth / validCount);
    return avgRimDepthMm + RIM_RECESS_OFFSET_MM; 
}
