#include "lens_measurement.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/imgproc.hpp>

#include "phase_unwrapper.h"

namespace {

float wrapAngle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double medianOf(std::vector<float>& values) {
    if (values.empty()) return 0.0;
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

bool validInputs(const cv::Mat wrapped[2], const cv::Mat modulation[2]) {
    for (int axis = 0; axis < 2; ++axis) {
        if (wrapped[axis].empty() || modulation[axis].empty() ||
                wrapped[axis].type() != CV_32F || modulation[axis].type() != CV_32F ||
                wrapped[axis].size() != wrapped[0].size() ||
                modulation[axis].size() != wrapped[0].size()) {
            return false;
        }
    }
    return true;
}

// Median phase gradient (radians per image pixel, image x and y) of a flat-screen reference.
bool referenceGradient(const cv::Mat& wrapped, const cv::Mat& modulation, cv::Point2d& gradient) {
    std::vector<float> modulationSamples;
    for (int row = 0; row < modulation.rows; row += 4) {
        const float* values = modulation.ptr<float>(row);
        for (int column = 0; column < modulation.cols; column += 4) {
            modulationSamples.push_back(values[column]);
        }
    }
    const float threshold = static_cast<float>(0.5 * medianOf(modulationSamples));
    if (threshold <= 0.0f) return false;

    std::vector<float> dx;
    std::vector<float> dy;
    for (int row = 0; row + 1 < wrapped.rows; row += 2) {
        const float* phase = wrapped.ptr<float>(row);
        const float* phaseBelow = wrapped.ptr<float>(row + 1);
        const float* quality = modulation.ptr<float>(row);
        const float* qualityBelow = modulation.ptr<float>(row + 1);
        for (int column = 0; column + 1 < wrapped.cols; column += 2) {
            if (quality[column] < threshold) continue;
            if (quality[column + 1] >= threshold) {
                dx.push_back(wrapAngle(phase[column + 1] - phase[column]));
            }
            if (qualityBelow[column] >= threshold) {
                dy.push_back(wrapAngle(phaseBelow[column] - phase[column]));
            }
        }
    }
    if (dx.size() < 100 || dy.size() < 100) return false;
    gradient = cv::Point2d(medianOf(dx), medianOf(dy));
    return std::hypot(gradient.x, gradient.y) > 1e-4;
}

// The lens opening in the lens run. The rim, a few centimetres from the camera, is out of
// focus, so fringes only partly fade behind it; but it still blocks light, so it shows up
// dark against the reference run. The opening is the bright region around centerGuess
// that the (gap-closed) dark rim walls off.
LensMeasurementStatus findLensRegion(const FringeRun& reference, const FringeRun& lens,
                                     const cv::Point2f& centerGuess, cv::Mat& mask,
                                     double& openingPixels, cv::Mat* brightnessRatio) {
    cv::Mat referenceBrightness = reference.brightness[0] + reference.brightness[1] + 1.0f;
    cv::Mat ratio;
    cv::divide(lens.brightness[0] + lens.brightness[1], referenceBrightness, ratio);
    if (brightnessRatio) *brightnessRatio = ratio.clone();

    const cv::Point center(cvRound(centerGuess.x), cvRound(centerGuess.y));
    if (!cv::Rect(0, 0, ratio.cols, ratio.rows).contains(center)) {
        return LensMeasurementStatus::BadInput;
    }
    constexpr float RIM_BRIGHTNESS_RATIO = 0.6f;
    cv::Mat rim = ratio < RIM_BRIGHTNESS_RATIO;
    const int closing = std::max(3, (std::min(ratio.cols, ratio.rows) / 100) | 1);
    cv::morphologyEx(rim, rim, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing, closing)));
    cv::Mat open = rim == 0;

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(open, labels, stats, centroids, 4, CV_32S);
    const int label = labels.at<int>(center);
    if (label == 0) return LensMeasurementStatus::LensNotFound;

    const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
    const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    mask = labels == label;
    openingPixels = std::sqrt(static_cast<double>(stats.at<int>(label, cv::CC_STAT_AREA)));
    if (left == 0 || top == 0 || left + width >= ratio.cols || top + height >= ratio.rows) {
        return LensMeasurementStatus::LensTouchesEdge;
    }
    return LensMeasurementStatus::Ok;
}

LensMeasurement fitLensPower(
        const FringeRun& reference, const FringeRun& lens,
        const cv::Point2f& monitorDown, const cv::Mat& lensMask, double openingPixels,
        double screenDistanceMm, double lensDistanceMm) {
    const cv::Mat* referenceWrapped = reference.wrapped;
    const cv::Mat* referenceModulation = reference.modulation;
    const cv::Mat* lensWrapped = lens.wrapped;
    const cv::Mat* lensModulation = lens.modulation;
    LensMeasurement result;
    result.lensDistanceMm = lensDistanceMm;
    result.lensOpeningPixels = openingPixels;
    if (!(lensDistanceMm > 0.0) || !(screenDistanceMm > lensDistanceMm)) return result;

    const double downLength = std::hypot(monitorDown.x, monitorDown.y);
    if (!(downLength > 1e-6)) return result;
    const cv::Point2d down(monitorDown.x / downLength, monitorDown.y / downLength);
    const cv::Point2d right(down.y, -down.x);
    const cv::Point2d axisDirection[2] = {right, down};

    // Phase change per image pixel along each monitor axis, from the flat-screen reference.
    double phasePerPixel[2];
    for (int axis = 0; axis < 2; ++axis) {
        cv::Point2d gradient;
        if (!referenceGradient(referenceWrapped[axis], referenceModulation[axis], gradient)) {
            result.status = LensMeasurementStatus::FringesTooFine;
            return result;
        }
        phasePerPixel[axis] = gradient.dot(axisDirection[axis]);
        if (std::abs(phasePerPixel[axis]) < 1e-4) {
            result.status = LensMeasurementStatus::FringesTooFine;
            return result;
        }
    }

    // Unwrap the lens-minus-reference phase inside the lens only.
    cv::Mat lensQuality = cv::min(lensModulation[0], lensModulation[1]);
    cv::Mat referenceQuality = cv::min(referenceModulation[0], referenceModulation[1]);
    cv::Mat quality = cv::min(lensQuality, referenceQuality);
    quality.setTo(0.0f, ~lensMask);
    cv::Mat shiftPixels[2];
    for (int axis = 0; axis < 2; ++axis) {
        cv::Mat difference = lensWrapped[axis] - referenceWrapped[axis];
        difference.forEach<float>([](float& value, const int*) { value = wrapAngle(value); });
        cv::Mat unwrapped;
        unwrapPhaseQualityGuided(difference, quality, unwrapped, 1e-3f);
        shiftPixels[axis] = unwrapped / phasePerPixel[axis];
    }

    // Stay clear of the blurred rim.
    cv::Mat fitMask;
    const int erosion = std::max(2, cvRound(openingPixels * 0.05));
    cv::erode(lensMask, fitMask, cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(2 * erosion + 1, 2 * erosion + 1)));
    fitMask &= quality > 1e-3f;

    struct Sample { double u; double v; double shiftX; double shiftY; };
    std::vector<Sample> samples;
    const int stride = std::max(1, cvRound(openingPixels / 120.0));
    for (int row = 0; row < fitMask.rows; row += stride) {
        const uchar* inside = fitMask.ptr<uchar>(row);
        for (int column = 0; column < fitMask.cols; column += stride) {
            if (!inside[column]) continue;
            const cv::Point2d pixel(column, row);
            samples.push_back({pixel.dot(right), pixel.dot(down),
                               shiftPixels[0].at<float>(row, column),
                               shiftPixels[1].at<float>(row, column)});
        }
    }

    // shift = intercept + K · u, per monitor axis, with two rounds of outlier rejection.
    cv::Matx22d k;
    std::vector<bool> keep(samples.size(), true);
    double rms = 0.0;
    for (int round = 0; round < 3; ++round) {
        std::vector<int> used;
        for (size_t index = 0; index < samples.size(); ++index) if (keep[index]) used.push_back(index);
        if (used.size() < 50) {
            result.status = LensMeasurementStatus::TooFewPixels;
            result.pixelsUsed = static_cast<int>(used.size());
            return result;
        }
        cv::Mat design(static_cast<int>(used.size()), 3, CV_64F);
        cv::Mat targets(static_cast<int>(used.size()), 2, CV_64F);
        for (size_t row = 0; row < used.size(); ++row) {
            const Sample& sample = samples[used[row]];
            design.at<double>(row, 0) = 1.0;
            design.at<double>(row, 1) = sample.u;
            design.at<double>(row, 2) = sample.v;
            targets.at<double>(row, 0) = sample.shiftX;
            targets.at<double>(row, 1) = sample.shiftY;
        }
        cv::Mat coefficients;
        cv::solve(design, targets, coefficients, cv::DECOMP_SVD);
        k = cv::Matx22d(coefficients.at<double>(1, 0), coefficients.at<double>(2, 0),
                        coefficients.at<double>(1, 1), coefficients.at<double>(2, 1));

        std::vector<double> residuals(samples.size());
        double sumSquares = 0.0;
        for (size_t index = 0; index < samples.size(); ++index) {
            const Sample& sample = samples[index];
            const double predictedX = coefficients.at<double>(0, 0) + k(0, 0) * sample.u + k(0, 1) * sample.v;
            const double predictedY = coefficients.at<double>(0, 1) + k(1, 0) * sample.u + k(1, 1) * sample.v;
            residuals[index] = std::hypot(sample.shiftX - predictedX, sample.shiftY - predictedY);
        }
        for (int index : used) sumSquares += residuals[index] * residuals[index];
        rms = std::sqrt(sumSquares / used.size());
        result.pixelsUsed = static_cast<int>(used.size());
        for (size_t index = 0; index < samples.size(); ++index) {
            keep[index] = residuals[index] <= std::max(3.0 * rms, 0.05);
        }
    }
    result.rmsResidualPixels = rms;

    // shift_px = -((D - d) d / D) M u  →  M = -(D / ((D - d) d)) K, with distances in metres.
    const double screenMetres = screenDistanceMm / 1000.0;
    const double lensMetres = lensDistanceMm / 1000.0;
    const double scale = -screenMetres / ((screenMetres - lensMetres) * lensMetres);
    // Symmetric part, then flip to y-up so angles run counter-clockwise from horizontal.
    const double a = scale * k(0, 0);
    const double b = -scale * 0.5 * (k(0, 1) + k(1, 0));
    const double c = scale * k(1, 1);
    const double mean = 0.5 * (a + c);
    const double radius = std::hypot(0.5 * (a - c), b);
    const double strongest = mean + radius;
    const double weakest = mean - radius;
    // Meridian of the strongest power, counter-clockwise from horizontal as the camera sees it.
    double cameraAxis = 0.5 * std::atan2(2.0 * b, a - c) * 180.0 / CV_PI;
    cameraAxis = std::fmod(cameraAxis + 180.0, 180.0);

    result.sphere = strongest;
    result.cylinder = weakest - strongest;
    // Glasses face away from the camera (arms toward the phone, as worn), so the camera sees
    // them from behind; TABO axes are read from the front, which mirrors the angle.
    double tabo = std::fmod(180.0 - cameraAxis, 180.0);
    if (tabo < 0.5) tabo = 180.0;
    result.axisDegrees = tabo;
    result.status = LensMeasurementStatus::Ok;
    return result;
}

}  // namespace

bool measureLensOpening(const cv::Mat& grayscaleFrame, const cv::Point2f& screenCenter,
                        double minSizePixels, double& openingPixels) {
    if (grayscaleFrame.empty() || grayscaleFrame.type() != CV_8UC1) return false;
    // The rim is only a few pixels wide, so no blurring and no single global threshold
    // (which the dark room drags down). A pixel is "rim" if it is clearly darker than the
    // brightest screen nearby; then close one-pixel gaps so the outline is continuous.
    const int window = std::max(9, cvRound(minSizePixels) | 1);
    cv::Mat localBright;
    cv::dilate(grayscaleFrame, localBright,
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(window, window)));
    cv::Mat scaledBright;
    localBright.convertTo(scaledBright, CV_8U, 0.7);
    cv::Mat dark = grayscaleFrame < scaledBright;
    cv::morphologyEx(dark, dark, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dark, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // Holes in dark regions that surround the screen center; the smallest big-enough one is
    // the lens opening inside the rim.
    double best = 0.0;
    for (size_t index = 0; index < contours.size(); ++index) {
        if (hierarchy[index][3] < 0) continue;
        if (cv::pointPolygonTest(contours[index], screenCenter, false) <= 0) continue;
        const double size = std::sqrt(std::abs(cv::contourArea(contours[index])));
        if (size < minSizePixels) continue;
        if (best == 0.0 || size < best) best = size;
    }
    if (best == 0.0) return false;
    openingPixels = best;
    return true;
}

bool fringePeriodPixels(const cv::Mat referenceWrapped[2], const cv::Mat referenceModulation[2],
                        const cv::Point2f& monitorDown, double periodPixels[2]) {
    if (!validInputs(referenceWrapped, referenceModulation)) return false;
    for (int axis = 0; axis < 2; ++axis) {
        cv::Point2d gradient;
        if (!referenceGradient(referenceWrapped[axis], referenceModulation[axis], gradient)) {
            return false;
        }
        periodPixels[axis] = 2.0 * CV_PI / std::hypot(gradient.x, gradient.y);
    }
    (void)monitorDown;
    return true;
}

namespace {

bool validRun(const FringeRun& run) {
    if (!validInputs(run.wrapped, run.modulation)) return false;
    for (const cv::Mat& brightness : run.brightness) {
        if (brightness.empty() || brightness.type() != CV_32F ||
                brightness.size() != run.wrapped[0].size()) {
            return false;
        }
    }
    return true;
}

}  // namespace

LensMeasurement measureLensPowerAtDistance(
        const FringeRun& reference, const FringeRun& lens,
        const cv::Point2f& monitorDown, const cv::Point2f& lensCenterGuess,
        double screenDistanceMm, double lensDistanceMm) {
    LensMeasurement result;
    if (!validRun(reference) || !validRun(lens) ||
            lens.wrapped[0].size() != reference.wrapped[0].size()) {
        return result;
    }
    cv::Mat mask;
    double openingPixels = 0.0;
    result.status = findLensRegion(reference, lens, lensCenterGuess, mask, openingPixels, nullptr);
    result.lensOpeningPixels = openingPixels;
    if (result.status != LensMeasurementStatus::Ok) return result;
    return fitLensPower(reference, lens, monitorDown, mask, openingPixels,
                        screenDistanceMm, lensDistanceMm);
}

LensMeasurement measureLensPower(
        const FringeRun& reference, const FringeRun& lens,
        const cv::Point2f& monitorDown, const cv::Point2f& lensCenterGuess,
        double screenDistanceMm, double screenOpeningPixels, cv::Mat* brightnessRatio) {
    LensMeasurement result;
    if (!validRun(reference) || !validRun(lens) ||
            lens.wrapped[0].size() != reference.wrapped[0].size() ||
            !(screenOpeningPixels > 0.0) || !(screenDistanceMm > 0.0)) {
        return result;
    }
    cv::Mat mask;
    double openingPixels = 0.0;
    result.status = findLensRegion(reference, lens, lensCenterGuess, mask, openingPixels,
                                   brightnessRatio);
    result.lensOpeningPixels = openingPixels;
    if (result.status != LensMeasurementStatus::Ok) return result;
    // Apparent size is inversely proportional to distance.
    const double lensDistanceMm = screenDistanceMm * screenOpeningPixels / openingPixels;
    return fitLensPower(reference, lens, monitorDown, mask, openingPixels,
                        screenDistanceMm, lensDistanceMm);
}
