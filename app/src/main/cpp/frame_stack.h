#ifndef FRAME_STACK_H
#define FRAME_STACK_H

#include <opencv2/core.hpp>
#include <vector>

// Combines repeated CV_8UC1 exposures of the same fringe pattern into one CV_32F frame.
// Frames that differ from the per-pixel median far more than is typical (the phone moved,
// or the monitor was mid-redraw) are dropped; the rest are averaged, which also lowers
// camera noise. keptCount receives how many frames were averaged.
cv::Mat combineFrames(const std::vector<cv::Mat>& frames, int& keptCount);

#endif
