// File: app/src/main/cpp/phase_unwrapper.cpp
#include "phase_unwrapper.h"

void calculateWrappedPhase(const cv::Mat& I1, const cv::Mat& I2, 
                           const cv::Mat& I3, const cv::Mat& I4, 
                           cv::Mat& wrappedPhase, cv::Mat& modulationMap) {
                               
    // 1. Convert 8-bit YUV channels to 32-bit floats for accurate math
    cv::Mat f1, f2, f3, f4;
    I1.convertTo(f1, CV_32F);
    I2.convertTo(f2, CV_32F);
    I3.convertTo(f3, CV_32F);
    I4.convertTo(f4, CV_32F);

    // 2. Calculate Numerator and Denominator for the arctangent
    cv::Mat num = f4 - f2;
    cv::Mat den = f1 - f3;

    // 3. Extract the Wrapped Phase
    // cv::phase computes arctan2 highly efficiently using ARM SIMD instructions.
    // The 'false' parameter ensures the output is in radians [0, 2pi].
    cv::phase(den, num, wrappedPhase, false); 
    
    // Standardize the phase range to [-pi, pi] for standard unwrapping algorithms
    wrappedPhase -= CV_PI;

    // 4. Extract the Modulation Map
    // This tells us the contrast of the fringes at every pixel. 
    // Pixels over the temples or background will have very low modulation, 
    // allowing us to easily mask them out and ignore them in the next steps.
    cv::Mat magnitude;
    cv::magnitude(den, num, magnitude);
    modulationMap = magnitude / 2.0f;
}