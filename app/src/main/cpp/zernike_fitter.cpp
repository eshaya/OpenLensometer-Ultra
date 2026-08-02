// File: app/src/main/cpp/zernike_fitter.cpp
#include "zernike_fitter.h"
#include <Eigen/Dense>
#include <cmath>

OpticalPrescription fitZernikeAndExtractPrescription(const cv::Mat& unwrappedPhase, 
                                                     const cv::Mat& mask, 
                                                     float lambda_mm, 
                                                     float pupilRadius_mm) {
    
    int rows = unwrappedPhase.rows;
    int cols = unwrappedPhase.cols;
    
    // Find the center of our valid lens area
    cv::Moments m = cv::moments(mask, true);
    float cx = m.m10 / m.m00;
    float cy = m.m01 / m.m00;
    
    // Grid sampling step to prevent memory/CPU explosion (e.g., sample every 10th pixel)
    int step = 10; 
    std::vector<float> x_norm, y_norm, phase_vals;
    
    // Calculate max pixel radius to normalize coordinates to [-1, 1]
    float max_pixel_radius = std::min(cx, cy); 
    
    for (int y = 0; y < rows; y += step) {
        for (int x = 0; x < cols; x += step) {
            if (mask.at<uchar>(y, x) > 0) {
                float nx = (x - cx) / max_pixel_radius;
                float ny = (y - cy) / max_pixel_radius;
                
                // Only include points within the unit circle (rho <= 1)
                if (nx * nx + ny * ny <= 1.0f) {
                    x_norm.push_back(nx);
                    y_norm.push_back(ny);
                    phase_vals.push_back(unwrappedPhase.at<float>(y, x));
                }
            }
        }
    }
    
    int numPoints = x_norm.size();
    if (numPoints < 6) {
        return {0.0f, 0.0f, 0.0f}; // Fail-safe
    }
    
    // 1. Build the Eigen Design Matrix (A) and Phase Vector (b)
    Eigen::MatrixXf A(numPoints, 6);
    Eigen::VectorXf b(numPoints);
    
    for (int i = 0; i < numPoints; ++i) {
        float x = x_norm[i];
        float y = y_norm[i];
        float rho2 = x * x + y * y;
        
        // Populate OSA Standard Zernike Polynomials (up to 2nd order)
        A(i, 0) = 1.0f;                             // Z0: Piston
        A(i, 1) = 2.0f * x;                         // Z1: Tip (X tilt)
        A(i, 2) = 2.0f * y;                         // Z2: Tilt (Y tilt)
        A(i, 3) = std::sqrt(6.0f) * 2.0f * x * y;   // Z3: Oblique Astigmatism
        A(i, 4) = std::sqrt(3.0f) * (2.0f * rho2 - 1.0f); // Z4: Defocus
        A(i, 5) = std::sqrt(6.0f) * (x * x - y * y); // Z5: Vertical Astigmatism
        
        b(i) = phase_vals[i];
    }
    
    // 2. Solve the Least Squares system: A * c = b
    // HouseholderQR is exceptionally fast and numerically stable for tall matrices
    Eigen::VectorXf c = A.colPivHouseholderQr().solve(b);
    
    // Extract the coefficients
    float c_astig_obl = c(3);
    float c_defocus   = c(4);
    float c_astig_vert= c(5);
    
    // 3. Convert Coefficients to Optometric Values (Sphere, Cylinder, Axis)
    // Scale factors account for phase-to-physical-sag conversion and pupil size
    float scale = (4.0f * std::sqrt(3.0f) * lambda_mm) / (pupilRadius_mm * pupilRadius_mm);
    
    float M = c_defocus * scale; 
    float J0 = c_astig_vert * (scale * std::sqrt(2.0f) / 2.0f);
    float J45 = c_astig_obl * (scale * std::sqrt(2.0f) / 2.0f);
    
    float cylinder = -2.0f * std::sqrt(J0 * J0 + J45 * J45);
    float sphere = M - (cylinder / 2.0f);
    
    // Calculate Axis in radians, then convert to degrees [0, 180]
    float axis_rad = 0.5f * std::atan2(J45, J0);
    float axis_deg = axis_rad * (180.0f / CV_PI);
    if (axis_deg < 0) axis_deg += 180.0f;
    
    return {sphere, cylinder, axis_deg};
}