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

// Slope (dv/du) of the longest convex-hull edge that all points lie on or below (top = true,
// v down) or on or above, spanning at least minSpan; false if there is none.
bool hullTangent(std::vector<cv::Point2f> points, bool top, double minSpan, double& slope) {
    if (points.size() < 3) return false;
    if (!top) for (auto& point : points) point.y = -point.y;
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    double bestSpan = 0.0;
    for (size_t k = 0; k < hull.size(); ++k) {
        const cv::Point2f a = hull[k], b = hull[(k + 1) % hull.size()];
        const double span = std::abs(b.x - a.x);
        if (span < minSpan || span <= bestSpan) continue;
        const double m = (b.y - a.y) / (b.x - a.x);
        const double c = a.y - m * a.x;
        bool allBelow = true;
        for (const auto& point : points) {
            if (point.y < m * point.x + c - 0.6) { allBelow = false; break; }
        }
        if (!allBelow) continue;
        bestSpan = span;
        slope = top ? m : -m;
    }
    return bestSpan > 0.0;
}

// Mirror symmetry (normalized correlation, -1..1) of `feature` about the line through
// (axisU, axisV) running at `degrees` from vertical, over a box of ±halfU × ±halfV around that
// point, sampling every `step` pixels; points whose mirror falls outside are skipped.
double mirrorScore(const cv::Mat& feature, double axisU, double axisV, double degrees,
                   int halfU, int halfV, int step) {
    const double s = std::sin(degrees * CV_PI / 180.0), c = std::cos(degrees * CV_PI / 180.0);
    auto sample = [&feature](double u, double v, float& value) {
        const int u0 = static_cast<int>(std::floor(u)), v0 = static_cast<int>(std::floor(v));
        if (u0 < 0 || v0 < 0 || u0 + 1 >= feature.cols || v0 + 1 >= feature.rows) return false;
        const float fu = static_cast<float>(u - u0), fv = static_cast<float>(v - v0);
        const float* row0 = feature.ptr<float>(v0);
        const float* row1 = feature.ptr<float>(v0 + 1);
        value = (1 - fv) * ((1 - fu) * row0[u0] + fu * row0[u0 + 1]) +
                fv * ((1 - fu) * row1[u0] + fu * row1[u0 + 1]);
        return true;
    };
    double sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
    int count = 0;
    for (int dv = -halfV; dv <= halfV; dv += step) {
        for (int du = -halfU; du <= halfU; du += step) {
            const double along = du * s + dv * c;
            float a, b;
            if (!sample(axisU + du, axisV + dv, a) ||
                !sample(axisU + 2 * along * s - du, axisV + 2 * along * c - dv, b)) continue;
            sumA += a; sumB += b; sumAA += a * a; sumBB += b * b; sumAB += a * b;
            ++count;
        }
    }
    if (count < 50) return -1.0;
    const double varA = sumAA - sumA * sumA / count, varB = sumBB - sumB * sumB / count;
    if (!(varA > 0 && varB > 0)) return -1.0;
    return (sumAB - sumA * sumB / count) / std::sqrt(varA * varB);
}

// Roll of the glasses from the mirror symmetry of the bridge, nose pads and inner lens edges,
// in a lens-centred monitor-coordinate brightness ratio (lens at (centerU, centerV)). The
// bridge is looked for on both sides of the lens; the more symmetric side wins (a hand
// holding the glasses is not symmetric). error is the tilt change that costs 2% of the
// peak symmetry, from the curvature of the peak.
bool bridgeSymmetryTilt(const cv::Mat& ratio, double centerU, double centerV, double radius,
                        double& tiltDegrees, double& errorDegrees) {
    cv::Mat gradientU, gradientV, feature;
    cv::Sobel(ratio, gradientU, CV_32F, 1, 0);
    cv::Sobel(ratio, gradientV, CV_32F, 0, 1);
    cv::magnitude(gradientU, gradientV, feature);
    const int halfU = cvRound(1.5 * radius), halfV = cvRound(1.2 * radius);
    constexpr double MAX_TILT = 10.0;
    double bestScore = -1.0, bestU = 0.0, bestTilt = 0.0;
    for (const int side : {-1, 1}) {
        for (double offset = 0.9 * radius; offset <= 2.3 * radius; offset += 3.0) {
            for (double tilt = -MAX_TILT; tilt <= MAX_TILT + 1e-9; tilt += 2.0) {
                const double score = mirrorScore(feature, centerU + side * offset, centerV, tilt,
                                                 halfU, halfV, 4);
                if (score > bestScore) {
                    bestScore = score; bestU = centerU + side * offset; bestTilt = tilt;
                }
            }
        }
    }
    if (bestScore <= 0.0) return false;
    // Bridge position and tilt trade off against each other, and the peak can be flat or
    // split: search both finely, and take the error from how far the tilt can move among the
    // nearly-as-symmetric positions.
    struct Trial { double score, u, tilt; };
    std::vector<Trial> trials;
    for (double du = -4.0; du <= 4.0 + 1e-9; du += 1.0) {
        for (double tilt = bestTilt - 3.0; tilt <= bestTilt + 3.0 + 1e-9; tilt += 0.25) {
            if (std::abs(tilt) > MAX_TILT) continue;
            trials.push_back({mirrorScore(feature, bestU + du, centerV, tilt, halfU, halfV, 2),
                              bestU + du, tilt});
        }
    }
    const Trial best = *std::max_element(trials.begin(), trials.end(),
            [](const Trial& x, const Trial& y) { return x.score < y.score; });
    if (best.score <= 0.0 || std::abs(best.tilt) >= MAX_TILT) return false;
    double lowest = best.tilt, highest = best.tilt;
    for (const auto& trial : trials) {
        if (trial.score < best.score * 0.985) continue;
        lowest = std::min(lowest, trial.tilt);
        highest = std::max(highest, trial.tilt);
    }
    tiltDegrees = best.tilt;
    errorDegrees = std::max(0.25, 0.5 * (highest - lowest));
    return true;
}

// Roll of the glasses, from the symmetry of the bridge and from lines along the tops and
// bottoms of both lenses, in the frame / reference brightness ratio (rims show dark), turned
// into monitor coordinates around the lens.
void measureGlassesTilt(const FringeRun& reference, const FringeRun& frameRun,
                        const cv::Point2f& monitorDown, const cv::Point2f& lensCenter,
                        float lensRadius, LensMeasurement& measurement) {
    const double downLength = std::hypot(monitorDown.x, monitorDown.y);
    if (!(downLength > 1e-6) || !(lensRadius > 5.0f)) return;
    const cv::Point2d down(monitorDown.x / downLength, monitorDown.y / downLength);
    const cv::Point2d right(down.y, -down.x);
    const double radius = lensRadius;
    const int halfWidth = cvRound(4.5 * radius);
    const int halfHeight = cvRound(2.4 * radius);
    // Monitor (u right, v down, centred on the lens) to image.
    const cv::Mat toImage = (cv::Mat_<double>(2, 3) <<
            right.x, down.x, lensCenter.x - right.x * halfWidth - down.x * halfHeight,
            right.y, down.y, lensCenter.y - right.y * halfWidth - down.y * halfHeight);
    const cv::Size size(2 * halfWidth + 1, 2 * halfHeight + 1);

    cv::Mat referenceBrightness, ratio, valid;
    cv::GaussianBlur(reference.brightness[0], referenceBrightness, cv::Size(), 2.0);
    cv::GaussianBlur(frameRun.brightness[0], ratio, cv::Size(), 2.0);
    ratio /= cv::max(referenceBrightness, 1.0);
    cv::warpAffine(ratio, ratio, toImage, size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
    cv::warpAffine(cv::Mat::ones(reference.brightness[0].size(), CV_8U), valid, toImage, size,
                   cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);

    const int baseRows = std::max(5, cvRound(0.5 * radius));
    const int topLimit = halfHeight + cvRound(0.8 * radius);      // top rim found above this
    const int bottomLimit = halfHeight - cvRound(0.8 * radius);   // bottom rim below this
    constexpr float DARKER = 0.12f;
    constexpr int SETTLE = 8;
    std::vector<cv::Point2f> tops, bottoms;
    std::vector<float> column(size.height), sorted;
    for (int u = 0; u < size.width; ++u) {
        bool whole = true;
        for (int v = 0; v < size.height; ++v) {
            if (!valid.at<uchar>(v, u)) { whole = false; break; }
            column[v] = ratio.at<float>(v, u);
        }
        if (!whole) continue;
        sorted.assign(column.begin(), column.begin() + baseRows);
        std::nth_element(sorted.begin(), sorted.begin() + baseRows / 2, sorted.end());
        float base = sorted[baseRows / 2];
        for (int v = 0; v < topLimit; ++v) {
            if (column[v] < base - DARKER) {
                const int end = std::min(v + SETTLE, size.height);
                tops.emplace_back(static_cast<float>(u),
                                  static_cast<float>(std::min_element(column.begin() + v, column.begin() + end) - column.begin()));
                break;
            }
        }
        sorted.assign(column.end() - baseRows, column.end());
        std::nth_element(sorted.begin(), sorted.begin() + baseRows / 2, sorted.end());
        base = sorted[baseRows / 2];
        for (int v = size.height - 1; v >= bottomLimit; --v) {
            if (column[v] < base - DARKER) {
                const int start = std::max(v - SETTLE + 1, 0);
                bottoms.emplace_back(static_cast<float>(u),
                                     static_cast<float>(std::min_element(column.begin() + start, column.begin() + v + 1) - column.begin()));
                break;
            }
        }
    }

    double edgeTilt = NAN, edgeError = NAN;
    // The other lens lies to one side, and a hand holding the glasses often to the other:
    // try the measured lens with each neighbour and keep the side where the lines agree best.
    // Lines across a hand come out steeper than glasses fitted to the outline are ever held.
    constexpr double MAX_TILT = 10.0;
    for (const int side : {-1, 1}) {
        const double near = halfWidth - side * 1.8 * radius, far = halfWidth + side * 4.5 * radius;
        const double low = std::min(near, far), high = std::max(near, far);
        double tilt[2];
        int found = 0;
        for (const bool top : {true, false}) {
            std::vector<cv::Point2f> points;
            for (const auto& point : top ? tops : bottoms) {
                if (point.x >= low && point.x <= high) points.push_back(point);
            }
            double slope = 0.0;
            if (!hullTangent(points, top, 2.4 * radius, slope)) continue;
            const double degrees = -std::atan(slope) * 180.0 / CV_PI;
            if (std::abs(degrees) < MAX_TILT) tilt[found++] = degrees;
        }
        if (found == 2) {
            const double error = 0.5 * std::abs(tilt[0] - tilt[1]);
            if (!(error >= edgeError)) {   // also replaces NaN
                edgeTilt = 0.5 * (tilt[0] + tilt[1]);
                edgeError = error;
            }
        } else if (found == 1 && std::isnan(edgeError) && std::isnan(edgeTilt)) {
            edgeTilt = tilt[0];
        }
    }

    // The bridge's symmetry is the main measure; where the edge lines were found top and bottom
    // they check it, and a disagreement counts as error. The lines alone are the fallback.
    double symmetryTilt, symmetryError;
    if (bridgeSymmetryTilt(ratio, halfWidth, halfHeight, radius, symmetryTilt, symmetryError)) {
        measurement.tiltDegrees = symmetryTilt;
        measurement.tiltErrorDegrees = std::isnan(edgeError) ? symmetryError
                : std::max(symmetryError, 0.5 * std::abs(symmetryTilt - edgeTilt));
    } else {
        measurement.tiltDegrees = edgeTilt;
        measurement.tiltErrorDegrees = edgeError;
    }
}

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
    // First pass with the reference tuning, just to find where the lens is: it may be held a
    // little off the image center.
    FringeRun firstPass;
    extractGridRun(grayscaleFrame, referenceCarriers, firstPass);
    cv::Point2f center = lensCenter;
    cv::Point2f located;
    if (locateLens(reference, firstPass, lensCenter, located)) center = located;

    // A patch well inside the lens, so the lens's grid dominates its spectrum.
    GridCarriers carriers = referenceCarriers;
    GridCarriers lensCarriers;
    if (findLensCarriers(grayscaleFrame, center, 0.7f * expectedLensRadius,
                         referenceCarriers, lensCarriers)) {
        carriers = lensCarriers;
    }
    FringeRun lens;
    extractGridRun(grayscaleFrame, carriers, lens);
    LensMeasurement measurement = measureLensPower(reference, lens, monitorDown, center,
            screenDistanceMm, screenOpeningPixels, nullptr, assumedLensDistanceMm);
    if (measurement.status == LensMeasurementStatus::Ok) {
        measureGlassesTilt(reference, firstPass, monitorDown, center, expectedLensRadius, measurement);
    }
    return measurement;
}
