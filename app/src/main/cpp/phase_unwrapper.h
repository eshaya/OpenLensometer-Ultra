// File: app/src/main/cpp/phase_unwrapper.h
#ifndef PHASE_UNWRAPPER_H
#define PHASE_UNWRAPPER_H

#include <opencv2/opencv.hpp>

/**
 * Extracts the wrapped phase and modulation (quality) maps from 4 phase-shifted images.
 * 
 * @param I1 Captured image at 0 phase shift
 * @param I2 Captured image at pi/2 phase shift
 * @param I3 Captured image at pi phase shift
 * @param I4 Captured image at 3pi/2 phase shift
 * @param wrappedPhase Output: The wrapped phase map in radians [-pi, pi]
 * @param modulationMap Output: The signal quality map (used to mask out bad pixels/temples)
 */
void calculateWrappedPhase(const cv::Mat& I1, const cv::Mat& I2, 
                           const cv::Mat& I3, const cv::Mat& I4, 
                           cv::Mat& wrappedPhase, cv::Mat& modulationMap);

/**
 * Unwraps the phase using a Quality-Guided Flood-Fill algorithm.
 * Routes around temples and noise by prioritizing high-modulation pixels.
 * 
 * @param wrappedPhase The input wrapped phase [-pi, pi]
 * @param modulationMap The contrast quality map
 * @param unwrappedPhase Output: The continuous unwrapped phase map
 * @param modThreshold Minimum modulation required to process a pixel (masks out temples)
 */
void unwrapPhaseQualityGuided(const cv::Mat& wrappedPhase, const cv::Mat& modulationMap, 
		cv::Mat& unwrappedPhase, float modThreshold = 5.0f);

#endif // PHASE_UNWRAPPER_H
