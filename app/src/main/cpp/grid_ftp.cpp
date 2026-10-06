#include "grid_ftp.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {

// Signed frequency (cycles per pixel) of spectrum bin `index` in a transform of length `size`.
double binFrequency(int index, int size) {
    return (index < (size + 1) / 2 ? index : index - size) / static_cast<double>(size);
}

// Wider than this many pixels per cycle is treated as background, not fringes.
constexpr double MAX_PERIOD_PIXELS = 40.0;
// Fringe directions may deviate this much from the monitor axes (tilted phone, lens).
const double MAX_CARRIER_ANGLE = 25.0 * CV_PI / 180.0;
// Filter width around each carrier, as a fraction of the carrier frequency: wide enough for
// a strong lens's change in fringe spacing, narrow enough to keep out DC and the other axis.
constexpr double FILTER_WIDTH = 0.3;

}  // namespace

bool findGridCarriers(const cv::Mat& grayscaleFrame, const cv::Point2f& monitorDown,
                      GridCarriers& carriers) {
    if (grayscaleFrame.empty() || grayscaleFrame.channels() != 1) return false;
    cv::Mat frame;
    grayscaleFrame.convertTo(frame, CV_32F);
    frame -= cv::mean(frame)[0];
    cv::Mat spectrum;
    cv::dft(frame, spectrum, cv::DFT_COMPLEX_OUTPUT);

    const double downLength = std::hypot(monitorDown.x, monitorDown.y);
    if (!(downLength > 1e-6)) return false;
    const cv::Point2d down(monitorDown.x / downLength, monitorDown.y / downLength);
    const cv::Point2d right(down.y, -down.x);
    const cv::Point2d axisDirection[2] = {right, down};

    double best[2] = {0.0, 0.0};
    bool found[2] = {false, false};
    for (int row = 0; row < spectrum.rows; ++row) {
        const cv::Vec2f* values = spectrum.ptr<cv::Vec2f>(row);
        const double fy = binFrequency(row, spectrum.rows);
        for (int column = 0; column < spectrum.cols; ++column) {
            const double fx = binFrequency(column, spectrum.cols);
            const double radius = std::hypot(fx, fy);
            if (radius < 1.0 / MAX_PERIOD_PIXELS || radius > 0.45) continue;
            const double power = values[column][0] * values[column][0] +
                                 values[column][1] * values[column][1];
            for (int axis = 0; axis < 2; ++axis) {
                const double along = (fx * axisDirection[axis].x + fy * axisDirection[axis].y) / radius;
                // Only the half of the spectrum pointing along the axis (the other is its mirror).
                if (along < std::cos(MAX_CARRIER_ANGLE)) continue;
                if (power > best[axis]) {
                    best[axis] = power;
                    carriers.frequency[axis] = cv::Point2d(fx, fy);
                    found[axis] = true;
                }
            }
        }
    }
    return found[0] && found[1];
}

void extractGridRun(const cv::Mat& grayscaleFrame, const GridCarriers& carriers, FringeRun& run) {
    cv::Mat frame;
    grayscaleFrame.convertTo(frame, CV_32F);
    const float mean = static_cast<float>(cv::mean(frame)[0]);
    cv::Mat centered = frame - mean;
    cv::Mat spectrum;
    cv::dft(centered, spectrum, cv::DFT_COMPLEX_OUTPUT);

    // The grid as found, to subtract from the frame for a fringe-free brightness image.
    cv::Mat grid = cv::Mat::zeros(frame.size(), CV_32F);
    for (int axis = 0; axis < 2; ++axis) {
        const cv::Point2d carrier = carriers.frequency[axis];
        const double sigma = FILTER_WIDTH * cv::norm(carrier);
        cv::Mat filtered(spectrum.size(), CV_32FC2);
        for (int row = 0; row < spectrum.rows; ++row) {
            const cv::Vec2f* in = spectrum.ptr<cv::Vec2f>(row);
            cv::Vec2f* out = filtered.ptr<cv::Vec2f>(row);
            const double dy = binFrequency(row, spectrum.rows) - carrier.y;
            for (int column = 0; column < spectrum.cols; ++column) {
                const double dx = binFrequency(column, spectrum.cols) - carrier.x;
                const float gain = static_cast<float>(
                        std::exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma)));
                out[column] = in[column] * gain;
            }
        }
        cv::Mat signal;
        cv::idft(filtered, signal, cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
        cv::Mat parts[2];
        cv::split(signal, parts);
        // One-sided filter: the analytic signal's angle is the fringe phase, its size half
        // the fringe amplitude (so ×2 matches the four-step modulation).
        cv::phase(parts[0], parts[1], run.wrapped[axis]);
        run.wrapped[axis].forEach<float>([](float& value, const int*) {
            if (value > CV_PI) value -= static_cast<float>(2.0 * CV_PI);
        });
        cv::magnitude(parts[0], parts[1], run.modulation[axis]);
        run.modulation[axis] *= 2.0;
        grid += 2.0 * parts[0];
    }
    // Subtracting the grid (rather than averaging it away) keeps the thin lens rim sharp,
    // whatever the grid spacing the carriers were tuned to.
    cv::Mat brightness = frame - grid;
    cv::boxFilter(brightness, brightness, -1, cv::Size(3, 3));
    run.brightness[0] = brightness;
    run.brightness[1] = brightness;
}

bool findLensCarriers(const cv::Mat& grayscaleFrame, const cv::Point2f& center, float radius,
                      const GridCarriers& reference, GridCarriers& lens) {
    const int half = std::max(16, cvRound(radius));
    const cv::Rect patchRect = cv::Rect(cvRound(center.x) - half, cvRound(center.y) - half,
                                        2 * half, 2 * half) &
                               cv::Rect(0, 0, grayscaleFrame.cols, grayscaleFrame.rows);
    if (patchRect.width < 16 || patchRect.height < 16) return false;
    cv::Mat patch;
    grayscaleFrame(patchRect).convertTo(patch, CV_32F);
    patch -= cv::mean(patch)[0];
    cv::Mat window;
    cv::createHanningWindow(window, patch.size(), CV_32F);
    patch = patch.mul(window);
    cv::Mat spectrum;
    cv::dft(patch, spectrum, cv::DFT_COMPLEX_OUTPUT);

    double best[2] = {0.0, 0.0};
    bool found[2] = {false, false};
    for (int row = 0; row < spectrum.rows; ++row) {
        const cv::Vec2f* values = spectrum.ptr<cv::Vec2f>(row);
        const double fy = binFrequency(row, spectrum.rows);
        for (int column = 0; column < spectrum.cols; ++column) {
            const double fx = binFrequency(column, spectrum.cols);
            const double radiusFrequency = std::hypot(fx, fy);
            if (radiusFrequency > 0.48) continue;
            const double power = values[column][0] * values[column][0] +
                                 values[column][1] * values[column][1];
            for (int axis = 0; axis < 2; ++axis) {
                const cv::Point2d ref = reference.frequency[axis];
                const double refFrequency = cv::norm(ref);
                if (radiusFrequency < 0.4 * refFrequency || radiusFrequency > 2.5 * refFrequency) continue;
                const double cosine = (fx * ref.x + fy * ref.y) / (radiusFrequency * refFrequency);
                if (cosine < std::cos(30.0 * CV_PI / 180.0)) continue;
                if (power > best[axis]) {
                    best[axis] = power;
                    lens.frequency[axis] = cv::Point2d(fx, fy);
                    found[axis] = true;
                }
            }
        }
    }
    return found[0] && found[1];
}

LensMeasurement measureLensFromGridFrame(
        const FringeRun& reference, const GridCarriers& referenceCarriers,
        const cv::Mat& grayscaleFrame, const cv::Point2f& monitorDown,
        const cv::Point2f& lensCenter, float expectedLensRadius,
        double screenDistanceMm, double screenOpeningPixels, double assumedLensDistanceMm) {
    // A patch well inside the lens, so the lens's grid dominates its spectrum.
    GridCarriers carriers = referenceCarriers;
    GridCarriers lensCarriers;
    if (findLensCarriers(grayscaleFrame, lensCenter, 0.7f * expectedLensRadius,
                         referenceCarriers, lensCarriers)) {
        carriers = lensCarriers;
    }
    FringeRun lens;
    extractGridRun(grayscaleFrame, carriers, lens);
    return measureLensPower(reference, lens, monitorDown, lensCenter,
                            screenDistanceMm, screenOpeningPixels, nullptr, assumedLensDistanceMm);
}
