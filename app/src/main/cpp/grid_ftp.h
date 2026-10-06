#ifndef GRID_FTP_H
#define GRID_FTP_H

#include <opencv2/core.hpp>

#include "lens_measurement.h"

// Single-frame fringe analysis (Fourier-transform profilometry) of a crossed grid,
// I = A + B·cos(2π x / p) + B·cos(2π y / p) on the monitor. Each fringe direction shows up
// as its own peak ("carrier") in the image spectrum; isolating one peak and transforming
// back gives that direction's phase and contrast from a single exposure, so hand shake
// between frames no longer matters.

struct GridCarriers {
    // Carrier frequency (cycles per image pixel, image x and y) of the monitor-X and
    // monitor-Y fringes, signed so that each points along monitor right / down.
    cv::Point2d frequency[2];
};

// Finds the two carriers in a no-lens view of the grid. monitorDown is the monitor's
// downward direction in image pixels. False if either peak is missing.
bool findGridCarriers(const cv::Mat& grayscaleFrame, const cv::Point2f& monitorDown,
                      GridCarriers& carriers);

// Phase, contrast and fringe-free brightness of one frame (any single-channel type), in the
// same form as the four-step runs, so measureLensPower can compare it with a reference.
void extractGridRun(const cv::Mat& grayscaleFrame, const GridCarriers& carriers, FringeRun& run);

// A strong lens makes the grid noticeably finer (minus) or coarser (plus) where it is seen
// through the lens. Finds the carriers in a patch of the given radius around center, near
// the reference ones (0.4–2.5× the frequency, within 30° of the direction).
bool findLensCarriers(const cv::Mat& grayscaleFrame, const cv::Point2f& center, float radius,
                      const GridCarriers& reference, GridCarriers& lens);

// The whole single-frame measurement: carriers tuned to the lens, phase extraction, and the
// comparison with the grid reference. expectedLensRadius is the lens's expected radius in
// image pixels (it sizes the patch used to find the lens's carriers).
LensMeasurement measureLensFromGridFrame(
        const FringeRun& reference, const GridCarriers& referenceCarriers,
        const cv::Mat& grayscaleFrame, const cv::Point2f& monitorDown,
        const cv::Point2f& lensCenter, float expectedLensRadius,
        double screenDistanceMm, double screenOpeningPixels,
        double assumedLensDistanceMm = 140.0);

#endif
