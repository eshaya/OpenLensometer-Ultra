#ifndef MONITOR_CALIBRATION_H
#define MONITOR_CALIBRATION_H

#include <cmath>
#include <limits>

inline double calculateMonitorPixelPitchMm(
        int screenWidthPixels, int screenHeightPixels, double screenDiagonalInches) {
    if (screenWidthPixels <= 0 || screenHeightPixels <= 0 ||
            !std::isfinite(screenDiagonalInches) || screenDiagonalInches <= 0.0) {
        return -1.0;
    }

    const double diagonalPixels = std::hypot(
            static_cast<double>(screenWidthPixels),
            static_cast<double>(screenHeightPixels));
    return 25.4 * screenDiagonalInches / diagonalPixels;
}

inline double monitorPitchDifferencePercent(double charucoPitchMm, double monitorSpecPitchMm) {
    if (!std::isfinite(charucoPitchMm) || charucoPitchMm <= 0.0 ||
            !std::isfinite(monitorSpecPitchMm) || monitorSpecPitchMm <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return std::abs(charucoPitchMm - monitorSpecPitchMm) / monitorSpecPitchMm * 100.0;
}

#endif