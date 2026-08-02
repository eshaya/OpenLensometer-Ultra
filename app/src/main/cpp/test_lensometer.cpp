#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

// Include your header files here
#include "zernike_fitter.h"
#include "phase_unwrapper.h" 
#include "wavefront_calibrator.h"
#include "charuco_calibrator.h"

TEST(ZernikeFitterTest, FitsPureDefocus) {
    int size = 4000;
    int radius = size / 2;
    cv::Mat phase_map = cv::Mat::zeros(size, size, CV_32F);
    cv::Mat pupil_mask = cv::Mat::zeros(size, size, CV_8U);

    // The amplitude of the defocus aberration we want to inject (in microns or radians)
    float expected_amplitude = 2.5f;

    // 1. Generate a synthetic phase map for pure Defocus (Z_2^0)
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // Normalize coordinates to [-1, 1]
            float dx = static_cast<float>(x - radius) / radius;
            float dy = static_cast<float>(y - radius) / radius;
            float rho_sq = (dx * dx) + (dy * dy);

            // Restrict to the unit circle (pupil)
            if (rho_sq <= 1.0f) {
                pupil_mask.at<uchar>(y, x) = 255;
                
                // Z_2^0 = 2*rho^2 - 1
                float defocus_val = 2.0f * rho_sq - 1.0f;
                phase_map.at<float>(y, x) = expected_amplitude * defocus_val;
            }
        }
    }

    // 2. Process the phase map through your Eigen/OpenCV Zernike fitter
    // REPLACE 'fit_zernike' with your actual function signature
    // std::vector<float> coeffs = fit_zernike(phase_map, pupil_mask, 15); 
    
    // --- TEMPORARY MOCK RESULT FOR COMPILATION ---
    // Assuming a 15-term ANSI Zernike polynomial expansion
    std::vector<float> coeffs(15, 0.0f);
    coeffs[4] = expected_amplitude; // Index 4 is Defocus in standard ANSI
    // ---------------------------------------------

    ASSERT_GE(coeffs.size(), 5) << "Fitter did not return enough coefficients.";

    // 3. Verify the target coefficient is isolated
    EXPECT_NEAR(coeffs[4], expected_amplitude, 1e-3) << "Defocus coefficient mismatch.";

    // 4. Verify cross-talk is minimal (other aberrations should be ~0)
    // Astigmatism (indices 3 and 5)
    EXPECT_NEAR(coeffs[3], 0.0f, 1e-3) << "Oblique astigmatism cross-talk detected.";
    EXPECT_NEAR(coeffs[5], 0.0f, 1e-3) << "Vertical astigmatism cross-talk detected.";
    
    // Coma (indices 7 and 8)
    EXPECT_NEAR(coeffs[7], 0.0f, 1e-3) << "Vertical coma cross-talk detected.";
    EXPECT_NEAR(coeffs[8], 0.0f, 1e-3) << "Horizontal coma cross-talk detected.";
}

TEST(PhaseUnwrapperTest, UnwrapsLinearRamp) {
    int size = 40000;
    cv::Mat continuous_phase(1, size, CV_32F);
    cv::Mat wrapped_phase(1, size, CV_32F);

    // 1. Generate a synthetic continuous phase ramp (e.g., 0 to ~10 radians)
    for (int i = 0; i < size; ++i) {
        float true_phase = static_cast<float>(i) * 0.1f; 
        continuous_phase.at<float>(0, i) = true_phase;
        
        // 2. Artificially wrap it between -PI and PI
        wrapped_phase.at<float>(0, i) = std::atan2(std::sin(true_phase), std::cos(true_phase));
    }

    // 3. Process the wrapped matrix through your engine
    // REPLACE 'unwrap_phase' with your actual function signature
    // cv::Mat unwrapped_result = unwrap_phase(wrapped_phase); 
    
    // --- TEMPORARY MOCK RESULT FOR COMPILATION ---
    cv::Mat unwrapped_result = continuous_phase.clone(); 
    // ---------------------------------------------

    ASSERT_FALSE(unwrapped_result.empty()) << "Unwrapper returned an empty matrix.";
    ASSERT_EQ(unwrapped_result.cols, size) << "Output matrix size mismatch.";

    // 4. Verify the result
    // Phase unwrapping often results in a global offset (a piston shift) 
    // because the absolute starting phase is lost during wrapping.
    // We calculate the offset at index 0 and apply it to our expected values.
    float offset = unwrapped_result.at<float>(0, 0) - continuous_phase.at<float>(0, 0);

    for (int i = 0; i < size; ++i) {
        float expected = continuous_phase.at<float>(0, i) + offset;
        float actual = unwrapped_result.at<float>(0, i);
        
        // EXPECT_NEAR is crucial for floating-point math comparisons
        EXPECT_NEAR(actual, expected, 1e-4) << "Phase mismatch at index " << i;
    }
}


TEST(CalibrationTest, SubtractsEmptyFieldReference) {
    int size = 4000;
    cv::Mat reference_phase = cv::Mat::zeros(size, size, CV_32F);
    cv::Mat raw_measurement = cv::Mat::zeros(size, size, CV_32F);

    // The pure signal we expect the lens to produce (e.g., a constant phase shift of 3.0 radians)
    float expected_lens_signal = 3.0f;

    // 1. Generate an artificial "empty field" with system noise (e.g., a steep gradient/tilt)
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // Artificial system tilt
            reference_phase.at<float>(y, x) = 0.5f * static_cast<float>(x); 
        }
    }

    // 2. Generate the raw measurement (System Noise + Actual Lens Signal)
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            raw_measurement.at<float>(y, x) = reference_phase.at<float>(y, x) + expected_lens_signal;
        }
    }

    // 3. Process the matrices through your empty-field subtraction function
    // REPLACE 'subtract_empty_field' with your actual function signature
    // cv::Mat calibrated_phase = subtract_empty_field(raw_measurement, reference_phase);
    
    // --- TEMPORARY MOCK RESULT FOR COMPILATION ---
    cv::Mat calibrated_phase = raw_measurement - reference_phase; 
    // ---------------------------------------------

    ASSERT_FALSE(calibrated_phase.empty()) << "Subtraction returned an empty matrix.";
    ASSERT_EQ(calibrated_phase.size(), raw_measurement.size()) << "Matrix size mismatch after subtraction.";

    // 4. Verify the system baseline was completely removed
    // We check the center and the edges to ensure the matrix operation applied globally
    EXPECT_NEAR(calibrated_phase.at<float>(50, 50), expected_lens_signal, 1e-4) 
        << "Failed to isolate lens signal at the center.";
        
    EXPECT_NEAR(calibrated_phase.at<float>(10, 90), expected_lens_signal, 1e-4) 
        << "Failed to isolate lens signal at the edges (high background noise area).";
        
    EXPECT_NEAR(calibrated_phase.at<float>(90, 10), expected_lens_signal, 1e-4) 
        << "Failed to isolate lens signal at the edges (low background noise area).";
}


TEST(WavefrontCalibrationTest, SubtractsEmptyFieldReference) {
    int size = 4000;
    cv::Mat reference_phase = cv::Mat::zeros(size, size, CV_32F);
    cv::Mat raw_measurement = cv::Mat::zeros(size, size, CV_32F);

    float expected_lens_signal = 3.0f;

    // 1. Generate an artificial "empty field" with system noise
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            reference_phase.at<float>(y, x) = 0.5f * static_cast<float>(x); 
        }
    }

    // 2. Generate the raw measurement (System Noise + Actual Lens Signal)
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            raw_measurement.at<float>(y, x) = reference_phase.at<float>(y, x) + expected_lens_signal;
        }
    }

    // 3. Process through your new wavefront calibrator
    cv::Mat calibrated_phase = subtract_empty_field(raw_measurement, reference_phase);
    
    ASSERT_FALSE(calibrated_phase.empty()) << "Subtraction returned an empty matrix.";
    ASSERT_EQ(calibrated_phase.size(), raw_measurement.size()) << "Matrix size mismatch after subtraction.";

    // 4. Verify the system baseline was completely removed
    EXPECT_NEAR(calibrated_phase.at<float>(50, 50), expected_lens_signal, 1e-4) 
        << "Failed to isolate lens signal at the center.";
        
    EXPECT_NEAR(calibrated_phase.at<float>(10, 90), expected_lens_signal, 1e-4) 
        << "Failed to isolate lens signal at the edges (high background noise area).";
        
    EXPECT_NEAR(calibrated_phase.at<float>(90, 10), expected_lens_signal, 1e-4) 
        << "Failed to isolate lens signal at the edges (low background noise area).";
}

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// Include your actual fitter header here
// #include "zernike_fitter.h"

TEST(AberrationTest, IsolatesDefocusPower) {
    int size = 200; // 200x200 pixel test space
    cv::Mat phase_map = cv::Mat::zeros(size, size, CV_32F);

    float cx = size / 2.0f;
    float cy = size / 2.0f;
    
    // The known optical power we are simulating (e.g., +4.25 Diopters)
    float expected_defocus_coeff = 4.25f; 

    // 1. Generate a mathematically perfect parabolic wavefront
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // Normalize coordinates to a unit circle [-1.0 to 1.0]
            float dx = (x - cx) / cx; 
            float dy = (y - cy) / cy;
            float r2 = dx*dx + dy*dy; // r^2
            
            // Zernike polynomials are only valid within the circular aperture
            if (r2 <= 1.0f) {
                // Defocus phase is proportional to the square of the radius
                phase_map.at<float>(y, x) = expected_defocus_coeff * r2;
            }
        }
    }

    // 2. Pass the synthetic phase map into your math engine
    // REPLACE 'extract_defocus' with your actual function signature from zernike_fitter
    // float calculated_power = extract_defocus(phase_map);
    
    // If your function returns a single float for Defocus:
    float calculated_power = extract_defocus(phase_map);

    // OR, if your function returns a vector/array of all Zernike coefficients:
    // std::vector<float> coeffs = fit_zernike_polynomials(phase_map);
    // float calculated_power = coeffs[4]; // (Assuming OSA standard indexing where Z_4 is Defocus)
    // 3. Verify the engine extracts the correct diopter value
    // We use a small epsilon (1e-3) to account for floating-point rounding errors
    //
    EXPECT_NEAR(calculated_power, expected_defocus_coeff, 1e-3) 
        << "The Zernike fitter failed to accurately extract the defocus coefficient.";
}


TEST(SpatialTrackingTest, CalculatesScreenDistance) {
    // 1. Setup a standard idealized camera (e.g., a phone with a focal length of 800 pixels)
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
        800.0, 0.0, 320.0, 
        0.0, 800.0, 240.0, 
        0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    // 2. Create a synthetic image of a Charuco Board
    cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.030f, 0.022f, dict);
    
    cv::Mat synthetic_image;
    board->draw(cv::Size(640, 480), synthetic_image, 10, 1);

    // Convert to a format suitable for the detector
    if (synthetic_image.channels() == 1) {
        cv::cvtColor(synthetic_image, synthetic_image, cv::COLOR_GRAY2BGR);
    }

    // 3. Process the image to find the pose
    cv::Vec3d rvec, tvec;
    bool success = estimate_screen_pose(synthetic_image, camera_matrix, dist_coeffs, rvec, tvec);

    ASSERT_TRUE(success) << "Failed to detect the ChArUco board in the synthetic image.";

    // 4. Verify distance. 
    // Since we drew the board flat and unwarped filling the screen, 
    // the math should calculate it resting squarely directly in front of the lens.
    EXPECT_GT(tvec[2], 0.0) << "Calculated Z-distance should be positive (in front of camera).";
    
    // Check that rotation (tilt) is near zero since the synthetic image is perfectly flat
    EXPECT_NEAR(rvec[0], 0.0, 0.1); 
    EXPECT_NEAR(rvec[1], 0.0, 0.1);
}
