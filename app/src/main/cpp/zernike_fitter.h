// File: app/src/main/cpp/zernike_fitter.h
#ifndef ZERNIKE_FITTER_H
#define ZERNIKE_FITTER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Extracts the Defocus (r^2) optical power coefficient from a raw phase map
float extract_defocus(const cv::Mat& phase_map);

// Struct to hold the final optometric results
struct OpticalPrescription {
    float sphere;
    float cylinder;
    float axis; // In degrees (0 to 180)
};

/**
 * Fits the unwrapped phase map to 2nd-order Zernike polynomials using Eigen Least Squares.
 * 
 * @param unwrappedPhase The continuous phase map from the QGPU step.
 * @param mask A binary mask (derived from modulation) to only include valid lens pixels.
 * @param lambda_mm The equivalent wavelength/sensitivity of the deflectometry setup.
 * @param pupilRadius_mm The physical radius of the analyzed lens area.
 * @return The calculated Sphere, Cylinder, and Axis in Diopters.
 */
OpticalPrescription fitZernikeAndExtractPrescription(const cv::Mat& unwrappedPhase, 
                                                     const cv::Mat& mask, 
                                                     float lambda_mm, 
                                                     float pupilRadius_mm);

#endif // ZERNIKE_FITTER_H
