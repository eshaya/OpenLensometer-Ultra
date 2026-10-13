#include "lens_measurement.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <opencv2/imgcodecs.hpp>
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
// focus, so it neither goes fully dark nor fully hides the fringes; but it does both partly,
// compared with the reference run. The opening is the region around centerGuess that the
// (gap-closed) rim walls off.
// A bifocal's segment line shows dark like the rim and cuts the segment out of the opening,
// and the magnified grid in the segment can punch holes in it; lens shapes are convex, so
// filling the outline's hull puts it back. Not when the opening is badly concave (merged
// with something else). The opening size (for the lens distance) stays that of the unfilled
// area, as on the screen, where the opening is measured the same way.
void fillToHull(cv::Mat& mask, double area) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return;
    std::vector<cv::Point> hull;
    cv::convexHull(*std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            }), hull);
    cv::Mat filled = cv::Mat::zeros(mask.size(), CV_8U);
    cv::fillConvexPoly(filled, hull, 255);
    if (cv::countNonZero(filled) < 1.6 * area) mask = filled;
}

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
    // A thin metal rim, blurred, only dims to ~0.6–0.8 of the reference; but it also wipes out
    // the fringes. Either sign marks rim. Inside the lens brightness stays ~0.9–1.1 and fringe
    // contrast is judged relative to the lens's own middle.
    constexpr float RIM_BRIGHTNESS_RATIO = 0.85f;
    constexpr float RIM_CONTRAST_RATIO = 0.35f;
    cv::Mat lensContrast = cv::min(lens.modulation[0], lens.modulation[1]);
    cv::Mat referenceContrast = cv::min(reference.modulation[0], reference.modulation[1]) + 1.0f;
    cv::Mat contrastRatio;
    cv::divide(lensContrast, referenceContrast, contrastRatio);
    // A strong lens lowers the fringe contrast everywhere inside it (the grid gets finer or
    // coarser than the camera reproduces well), so judge contrast against the lens's own
    // middle: only a further drop there marks the rim.
    const cv::Point middle(cvRound(centerGuess.x), cvRound(centerGuess.y));
    const int middleRadius = std::max(4, std::min(ratio.cols, ratio.rows) / 40);
    const cv::Rect middleWindow = cv::Rect(middle.x - middleRadius, middle.y - middleRadius,
                                           2 * middleRadius, 2 * middleRadius) &
                                  cv::Rect(0, 0, ratio.cols, ratio.rows);
    if (middleWindow.empty()) return LensMeasurementStatus::BadInput;
    const cv::Mat middlePatch = contrastRatio(middleWindow).clone();
    std::vector<float> middleValues(middlePatch.begin<float>(), middlePatch.end<float>());
    const float lensContrastLevel = std::min(1.0f, static_cast<float>(medianOf(middleValues)));
    cv::Mat rim = (ratio < RIM_BRIGHTNESS_RATIO) |
                  (contrastRatio < RIM_CONTRAST_RATIO * lensContrastLevel);
    if (std::getenv("LENS_RIM_BRIGHTNESS_ONLY")) rim = ratio < RIM_BRIGHTNESS_RATIO;
    if (std::getenv("LENS_DEBUG_IMAGES")) {
        cv::Mat out;
        ratio.convertTo(out, CV_8U, 200.0);
        cv::imwrite("/tmp/claude-1000/-home-eshaya-Documents-GitHub-Iglasses/ed836ff0-b5eb-47d3-a1ac-a123414347f0/scratchpad/dbg-brightness.png", out);
        contrastRatio.convertTo(out, CV_8U, 200.0 / std::max(0.01f, lensContrastLevel));
        cv::imwrite("/tmp/claude-1000/-home-eshaya-Documents-GitHub-Iglasses/ed836ff0-b5eb-47d3-a1ac-a123414347f0/scratchpad/dbg-contrast.png", out);
    }
    const int closing = std::max(3, (std::min(ratio.cols, ratio.rows) / 100) | 1);
    cv::morphologyEx(rim, rim, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing, closing)));
    cv::Mat open = rim == 0;

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(open, labels, stats, centroids, 4, CV_32S);
    const auto touchesBorder = [&](int component) {
        const int left = stats.at<int>(component, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(component, cv::CC_STAT_TOP);
        return left == 0 || top == 0 ||
               left + stats.at<int>(component, cv::CC_STAT_WIDTH) >= ratio.cols ||
               top + stats.at<int>(component, cv::CC_STAT_HEIGHT) >= ratio.rows;
    };
    int label = labels.at<int>(center);
    if (label == 0 || touchesBorder(label)) {
        // The center isn't inside a lens (glasses held a little off): use the enclosed,
        // lens-sized area nearest to it instead.
        // At least the size of a small lens near the far end of the 12–25 cm range.
        const int minimumArea = static_cast<int>(ratio.total() / 80);
        int nearest = -1;
        double nearestDistance = 0.0;
        for (int component = 1; component < count; ++component) {
            if (touchesBorder(component) ||
                    stats.at<int>(component, cv::CC_STAT_AREA) < minimumArea) continue;
            const double distance = std::hypot(centroids.at<double>(component, 0) - center.x,
                                               centroids.at<double>(component, 1) - center.y);
            if (nearest < 0 || distance < nearestDistance) {
                nearest = component;
                nearestDistance = distance;
            }
        }
        if (nearest >= 0) {
            label = nearest;
        } else if (label == 0) {
            return LensMeasurementStatus::LensNotFound;
        }
    }

    const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
    const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    mask = labels == label;
    const double area = stats.at<int>(label, cv::CC_STAT_AREA);
    fillToHull(mask, area);
    openingPixels = std::sqrt(area);
    if (left == 0 || top == 0 || left + width >= ratio.cols || top + height >= ratio.rows) {
        return LensMeasurementStatus::LensTouchesEdge;
    }
    return LensMeasurementStatus::Ok;
}

// For lenses without a full dark rim (half-rim, rimless, gold): find the lens by what it
// does. Through a lens with power the screen-point shift changes steadily from pixel to
// pixel (the grid looks magnified or shrunk); on the open screen it doesn't change at all.
LensMeasurementStatus findLensRegionByMagnification(
        const FringeRun& reference, const FringeRun& lens, const cv::Point2f& centerGuess,
        cv::Mat& mask, double& openingPixels) {
    const cv::Point center(cvRound(centerGuess.x), cvRound(centerGuess.y));
    const cv::Size size = reference.wrapped[0].size();
    if (!cv::Rect(cv::Point(1, 1), size - cv::Size(2, 2)).contains(center)) {
        return LensMeasurementStatus::BadInput;
    }
    // Rate of change of the shift, per axis, in units of the reference fringe spacing.
    cv::Mat magnification = cv::Mat::zeros(size, CV_32F);
    for (int axis = 0; axis < 2; ++axis) {
        cv::Point2d gradient;
        if (!referenceGradient(reference.wrapped[axis], reference.modulation[axis], gradient)) {
            return LensMeasurementStatus::FringesTooFine;
        }
        const float phasePerPixel = static_cast<float>(std::hypot(gradient.x, gradient.y));
        cv::Mat difference = lens.wrapped[axis] - reference.wrapped[axis];
        cv::Mat dx = cv::Mat::zeros(size, CV_32F);
        cv::Mat dy = cv::Mat::zeros(size, CV_32F);
        for (int row = 0; row + 1 < size.height; ++row) {
            const float* here = difference.ptr<float>(row);
            const float* below = difference.ptr<float>(row + 1);
            float* outX = dx.ptr<float>(row);
            float* outY = dy.ptr<float>(row);
            for (int column = 0; column + 1 < size.width; ++column) {
                outX[column] = wrapAngle(here[column + 1] - here[column]);
                outY[column] = wrapAngle(below[column] - here[column]);
            }
        }
        cv::boxFilter(dx, dx, -1, cv::Size(9, 9));
        cv::boxFilter(dy, dy, -1, cv::Size(9, 9));
        magnification += (dx.mul(dx) + dy.mul(dy)) / (phasePerPixel * phasePerPixel);
    }
    cv::sqrt(magnification, magnification);

    // About 0.5 D at the usual distances; excludes dark frame parts and fingers too.
    constexpr float MIN_MAGNIFICATION_CHANGE = 0.05f;
    cv::Mat brightness;
    cv::divide(lens.brightness[0] + lens.brightness[1],
               reference.brightness[0] + reference.brightness[1] + 1.0f, brightness);
    cv::Mat inside = (magnification > MIN_MAGNIFICATION_CHANGE) & (brightness > 0.85f);
    const cv::Mat smallKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(inside, inside, cv::MORPH_OPEN, smallKernel);
    cv::morphologyEx(inside, inside, cv::MORPH_CLOSE, smallKernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(inside, labels, stats, centroids, 4, CV_32S);
    const int label = labels.at<int>(center);
    if (label == 0) return LensMeasurementStatus::LensNotFound;
    mask = labels == label;
    openingPixels = std::sqrt(static_cast<double>(stats.at<int>(label, cv::CC_STAT_AREA)));
    fillToHull(mask, openingPixels * openingPixels);
    const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
    if (left == 0 || top == 0 || left + stats.at<int>(label, cv::CC_STAT_WIDTH) >= size.width ||
            top + stats.at<int>(label, cv::CC_STAT_HEIGHT) >= size.height) {
        return LensMeasurementStatus::LensTouchesEdge;
    }
    return LensMeasurementStatus::Ok;
}

// Goldstein branch cuts. A residue is a 2×2 loop of pixels whose wrapped phase steps don't
// add up to zero (noise, or the edge of a bifocal segment where the image jumps). Unwrapping
// around one gives a 2π error somewhere; a path-following unwrapper puts that cut wherever
// its path happens to go, sometimes right across the lens. Joining residues of opposite sign
// to each other, or to the edge of the usable area, with short straight cuts (quality set to
// zero) keeps the unwrapper from going around them.
void placeBranchCuts(const cv::Mat& wrapped, cv::Mat& quality) {
    constexpr float MIN_QUALITY = 1e-3f;
    struct Residue { cv::Point at; int charge; };
    std::vector<Residue> residues;
    for (int row = 0; row + 1 < wrapped.rows; ++row) {
        for (int column = 0; column + 1 < wrapped.cols; ++column) {
            if (quality.at<float>(row, column) < MIN_QUALITY || quality.at<float>(row, column + 1) < MIN_QUALITY ||
                quality.at<float>(row + 1, column) < MIN_QUALITY ||
                quality.at<float>(row + 1, column + 1) < MIN_QUALITY) continue;
            const float a = wrapped.at<float>(row, column);
            const float b = wrapped.at<float>(row, column + 1);
            const float c = wrapped.at<float>(row + 1, column + 1);
            const float d = wrapped.at<float>(row + 1, column);
            const float loop = wrapAngle(b - a) + wrapAngle(c - b) + wrapAngle(d - c) + wrapAngle(a - d);
            if (std::abs(loop) > CV_PI) residues.push_back({cv::Point(column, row), loop > 0 ? 1 : -1});
        }
    }
    if (residues.empty()) return;

    // Nearest unusable pixel to each residue, looked for in a box around the residues that
    // reaches the edge of the usable area.
    cv::Rect box = cv::boundingRect(quality >= MIN_QUALITY);
    box = cv::Rect(box.x - 2, box.y - 2, box.width + 4, box.height + 4) &
          cv::Rect(0, 0, quality.cols, quality.rows);
    cv::Mat outside = quality(box) >= MIN_QUALITY;
    cv::Mat distance;
    cv::Mat labels;
    cv::distanceTransform(outside, distance, labels, cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
    std::vector<cv::Point> labelPoint(box.area() + 1);
    for (int row = 0; row < outside.rows; ++row) {
        for (int column = 0; column < outside.cols; ++column) {
            if (!outside.at<uchar>(row, column)) {
                labelPoint[labels.at<int>(row, column)] = cv::Point(column + box.x, row + box.y);
            }
        }
    }

    // Shortest links first: residue to residue of opposite sign, or residue to edge.
    struct Link { double length; int first; int second; cv::Point edge; };
    std::vector<Link> links;
    for (size_t i = 0; i < residues.size(); ++i) {
        const cv::Point at = residues[i].at;
        const cv::Point inBox = at - box.tl();
        links.push_back({distance.at<float>(inBox), static_cast<int>(i), -1,
                         labelPoint[labels.at<int>(inBox)]});
        for (size_t j = i + 1; j < residues.size(); ++j) {
            if (residues[i].charge == residues[j].charge) continue;
            links.push_back({cv::norm(at - residues[j].at), static_cast<int>(i), static_cast<int>(j), {}});
        }
    }
    std::sort(links.begin(), links.end(), [](const Link& a, const Link& b) { return a.length < b.length; });
    std::vector<bool> done(residues.size(), false);
    for (const Link& link : links) {
        if (done[link.first] || (link.second >= 0 && done[link.second])) continue;
        done[link.first] = true;
        if (link.second >= 0) done[link.second] = true;
        const cv::Point end = link.second >= 0 ? residues[link.second].at : link.edge;
        // 8-connected, so a 4-connected unwrapping path can't slip through diagonally.
        cv::line(quality, residues[link.first].at, end, cv::Scalar(0.0), 1, cv::LINE_8);
    }
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
    cv::Mat usable = quality > 1e-3f;
    for (int axis = 0; axis < 2; ++axis) {
        cv::Mat difference = lensWrapped[axis] - referenceWrapped[axis];
        difference.forEach<float>([](float& value, const int*) { value = wrapAngle(value); });
        cv::Mat axisQuality = quality.clone();
        placeBranchCuts(difference, axisQuality);
        cv::Mat unwrapped;
        unwrapPhaseQualityGuided(difference, axisQuality, unwrapped, 1e-3f);
        shiftPixels[axis] = unwrapped / phasePerPixel[axis];
        usable &= axisQuality > 1e-3f;
    }

    // Stay clear of the blurred rim.
    cv::Mat fitMask;
    const int erosion = std::max(2, cvRound(openingPixels * 0.05));
    cv::erode(lensMask, fitMask, cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(2 * erosion + 1, 2 * erosion + 1)));
    fitMask &= usable;

    // Pieces of the lens the branch cuts cut off. Each gets its own intercept in the fits
    // below, since their phases are unwrapped separately (a 2π offset between them is
    // arbitrary).
    cv::Mat pieces;
    cv::connectedComponents(fitMask, pieces, 4, CV_32S);

    struct Sample { double u; double v; double shiftX; double shiftY; int piece; int row; int column; };
    std::vector<Sample> samples;
    const int stride = std::max(1, cvRound(openingPixels / 120.0));
    for (int row = 0; row < fitMask.rows; row += stride) {
        const uchar* inside = fitMask.ptr<uchar>(row);
        for (int column = 0; column < fitMask.cols; column += stride) {
            const int piece = pieces.at<int>(row, column);
            if (!inside[column] || piece == 0) continue;
            const cv::Point2d pixel(column, row);
            samples.push_back({pixel.dot(right), pixel.dot(down),
                               shiftPixels[0].at<float>(row, column),
                               shiftPixels[1].at<float>(row, column), piece, row, column});
        }
    }

    // shift = intercept + K · (u - uMiddle, v - vMiddle), per monitor axis, with one intercept
    // per piece of the lens (pieceOf: sample index → piece). Fits the samples in used;
    // residuals (NaN for a piece not in used) for every sample in evaluate.
    const auto fitGradient = [&samples](const std::vector<int>& used, double uMiddle, double vMiddle,
                                        const std::vector<int>& evaluate, const auto& pieceOf,
                                        cv::Matx22d& gradient, std::vector<double>& residuals) {
        std::map<int, int> interceptColumn;
        for (int index : used) interceptColumn.emplace(pieceOf(index), 0);
        int columns = 2;
        for (auto& entry : interceptColumn) entry.second = columns++;
        cv::Mat design = cv::Mat::zeros(static_cast<int>(used.size()), columns, CV_64F);
        cv::Mat targets(static_cast<int>(used.size()), 2, CV_64F);
        for (size_t row = 0; row < used.size(); ++row) {
            const Sample& sample = samples[used[row]];
            design.at<double>(row, 0) = sample.u - uMiddle;
            design.at<double>(row, 1) = sample.v - vMiddle;
            design.at<double>(row, interceptColumn[pieceOf(used[row])]) = 1.0;
            targets.at<double>(row, 0) = sample.shiftX;
            targets.at<double>(row, 1) = sample.shiftY;
        }
        cv::Mat coefficients;
        if (!cv::solve(design, targets, coefficients, cv::DECOMP_SVD)) return false;
        gradient = cv::Matx22d(coefficients.at<double>(0, 0), coefficients.at<double>(1, 0),
                               coefficients.at<double>(0, 1), coefficients.at<double>(1, 1));
        residuals.clear();
        for (int index : evaluate) {
            const Sample& sample = samples[index];
            const auto found = interceptColumn.find(pieceOf(index));
            if (found == interceptColumn.end()) {
                residuals.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            const double du = sample.u - uMiddle;
            const double dv = sample.v - vMiddle;
            residuals.push_back(std::hypot(
                    sample.shiftX - (coefficients.at<double>(found->second, 0) +
                                     gradient(0, 0) * du + gradient(0, 1) * dv),
                    sample.shiftY - (coefficients.at<double>(found->second, 1) +
                                     gradient(1, 0) * du + gradient(1, 1) * dv)));
        }
        return true;
    };

    // Whole lens, with two rounds of outlier rejection.
    cv::Matx22d k;
    std::vector<int> all(samples.size());
    for (size_t index = 0; index < samples.size(); ++index) all[index] = static_cast<int>(index);
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
        std::vector<double> residuals;
        const auto wholePiece = [&samples](int index) { return samples[index].piece; };
        if (!fitGradient(used, 0.0, 0.0, all, wholePiece, k, residuals)) {
            result.status = LensMeasurementStatus::TooFewPixels;
            return result;
        }
        double sumSquares = 0.0;
        for (int index : used) sumSquares += residuals[index] * residuals[index];
        rms = std::sqrt(sumSquares / used.size());
        result.pixelsUsed = static_cast<int>(used.size());
        for (size_t index = 0; index < samples.size(); ++index) {
            keep[index] = residuals[index] <= std::max(3.0 * rms, 0.05);  // false for NaN
        }
    }
    result.rmsResidualPixels = rms;

    // shift_px = -((D - d) d / D) M u  →  M = -(D / ((D - d) d)) K, with distances in metres.
    const double screenMetres = screenDistanceMm / 1000.0;
    const double lensMetres = lensDistanceMm / 1000.0;
    const double scale = -screenMetres / ((screenMetres - lensMetres) * lensMetres);

    struct Prescription { double sphere; double cylinder; double axis; double meanPower; };
    const auto prescriptionFrom = [scale](const cv::Matx22d& shiftGradient) {
        // Symmetric part, then flip to y-up so angles run counter-clockwise from horizontal.
        const double a = scale * shiftGradient(0, 0);
        const double b = -scale * 0.5 * (shiftGradient(0, 1) + shiftGradient(1, 0));
        const double c = scale * shiftGradient(1, 1);
        const double mean = 0.5 * (a + c);
        const double radius = std::hypot(0.5 * (a - c), b);
        // Meridian of the strongest power, counter-clockwise from horizontal as the camera
        // sees it. The glasses face away from the camera (arms toward the phone, as worn),
        // so it sees them from behind; TABO axes are read from the front, which mirrors it.
        double cameraAxis = 0.5 * std::atan2(2.0 * b, a - c) * 180.0 / CV_PI;
        cameraAxis = std::fmod(cameraAxis + 180.0, 180.0);
        double tabo = std::fmod(180.0 - cameraAxis, 180.0);
        if (tabo < 0.5) tabo = 180.0;
        return Prescription{mean + radius, -2.0 * radius, tabo, mean};
    };

    const Prescription whole = prescriptionFrom(k);
    result.sphere = whole.sphere;
    result.cylinder = whole.cylinder;
    result.axisDegrees = whole.axis;

    // Multifocals: power changes from the top of the lens (distance) to the bottom (near).
    // Fit small windows over all samples: the whole-lens fit discards a bifocal segment as
    // outliers, so its choice of samples can't be reused here.
    std::vector<double> us;
    std::vector<double> vs;
    for (const Sample& sample : samples) {
        us.push_back(sample.u);
        vs.push_back(sample.v);
    }
    const auto percentile = [](std::vector<double> values, double fraction) {
        std::sort(values.begin(), values.end());
        return values[std::min(values.size() - 1, static_cast<size_t>(fraction * values.size()))];
    };
    const double uCenter = percentile(us, 0.5);
    const double edgeFraction = std::getenv("LENS_EDGE_FRACTION")
            ? std::atof(std::getenv("LENS_EDGE_FRACTION")) : 0.05;
    const double vTop = percentile(vs, edgeFraction);
    const double vBottom = percentile(vs, 1.0 - edgeFraction);
    const double lensWidth = percentile(us, 0.95) - percentile(us, 0.05);
    const double windowFraction = std::getenv("LENS_WINDOW_FRACTION")
            ? std::atof(std::getenv("LENS_WINDOW_FRACTION")) : 0.18;
    const double windowRadius = windowFraction * std::min(lensWidth, vBottom - vTop);

    const auto localPrescription = [&](double uMiddle, double vMiddle, Prescription& out,
                                       double radius = 0.0) {
        if (radius <= 0.0) radius = windowRadius;
        std::vector<int> used;
        for (size_t index = 0; index < samples.size(); ++index) {
            const double du = samples[index].u - uMiddle;
            const double dv = samples[index].v - vMiddle;
            if (du * du + dv * dv <= radius * radius) used.push_back(static_cast<int>(index));
        }
        if (used.size() < 30) return false;
        // Pieces of this window: a branch cut that crosses it (from a residue to the edge of
        // the lens) carries a 2π step, so the parts on either side get their own intercepts.
        const cv::Point2d middle = uMiddle * right + vMiddle * down;
        const cv::Rect box = cv::Rect(cvFloor(middle.x - radius) - 1, cvFloor(middle.y - radius) - 1,
                                      cvCeil(2.0 * radius) + 3, cvCeil(2.0 * radius) + 3) &
                             cv::Rect(0, 0, fitMask.cols, fitMask.rows);
        cv::Mat window = cv::Mat::zeros(box.size(), CV_8U);
        cv::circle(window, cv::Point(cvRound(middle.x) - box.x, cvRound(middle.y) - box.y),
                   cvCeil(radius) + 1, cv::Scalar(255), cv::FILLED);
        window &= fitMask(box);
        cv::Mat windowPieces;
        cv::connectedComponents(window, windowPieces, 4, CV_32S);
        const auto windowPiece = [&](int index) {
            return windowPieces.at<int>(samples[index].row - box.y, samples[index].column - box.x);
        };
        // Local fit with its own outlier rejection (a window may straddle a segment line).
        cv::Matx22d gradient;
        for (int round = 0; round < 2; ++round) {
            if (used.size() < 30) return false;
            std::vector<double> residuals;
            if (!fitGradient(used, uMiddle, vMiddle, used, windowPiece, gradient, residuals)) return false;
            std::vector<double> sorted = residuals;
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double limit = std::max(0.05, 3.0 * sorted[sorted.size() / 2]);
            std::vector<int> next;
            for (size_t row = 0; row < used.size(); ++row) if (residuals[row] <= limit) next.push_back(used[row]);
            used = next;
        }
        out = prescriptionFrom(gradient);
        return true;
    };

    // Mean power down the middle of the lens, top to bottom.
    constexpr int PROFILE_POINTS = 9;
    std::vector<double> profile;
    if (windowRadius > 3.0) {
        for (int point = 0; point < PROFILE_POINTS; ++point) {
            const double v = vTop + windowRadius +
                             (vBottom - vTop - 2.0 * windowRadius) * point / (PROFILE_POINTS - 1);
            Prescription local;
            if (localPrescription(uCenter, v, local)) profile.push_back(local.meanPower);
        }
    }
    Prescription distance;
    Prescription nearZone;
    double nearU = uCenter;
    double nearV = vBottom;
    // Sphere of the distance part of the lens beside the near zone, at the same height; NaN if
    // there is none.
    double besideSphere = std::numeric_limits<double>::quiet_NaN();
    bool zonesFound = profile.size() >= 5 &&
            localPrescription(uCenter, vTop + 1.3 * windowRadius, distance);
    if (zonesFound) {
        // Near zone: the strongest clean window anywhere below the distance window. A bifocal
        // segment is often off to the nasal side, and its height is chosen per wearer, so a
        // window on the centre line or at a set height may straddle its edge. The add changes
        // only the sphere, so a window whose cylinder differs much from the distance zone's
        // straddles a line.
        zonesFound = false;
        const double uLeft = percentile(us, 0.05) + windowRadius;
        const double uRight = percentile(us, 0.95) - windowRadius;
        const double step = 0.5 * windowRadius;
        // Taking the single strongest window would favour noise that happens to read high,
        // so each candidate is averaged with its clean neighbours first.
        struct Candidate { double u; double v; Prescription prescription; bool clean; };
        std::vector<std::vector<Candidate>> grid;
        for (double v = vBottom - windowRadius; v >= vTop + 3.3 * windowRadius; v -= step) {
            grid.emplace_back();
            for (double u = uLeft; u <= uRight + 0.5; u += step) {
                Candidate candidate{u, v, {}, false};
                candidate.clean = localPrescription(u, v, candidate.prescription) &&
                        std::abs(candidate.prescription.cylinder - distance.cylinder) <= 0.5;
                grid.back().push_back(candidate);
            }
        }
        double bestPower = -1e9;
        for (int row = 0; row < static_cast<int>(grid.size()); ++row) {
            for (int column = 0; column < static_cast<int>(grid[row].size()); ++column) {
                if (!grid[row][column].clean) continue;
                int count = 0;
                double power = 0.0;
                double sphere = 0.0;
                double cylinder = 0.0;
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        const int r = row + dr;
                        const int c = column + dc;
                        if (r < 0 || r >= static_cast<int>(grid.size()) ||
                                c < 0 || c >= static_cast<int>(grid[r].size()) || !grid[r][c].clean) continue;
                        const Prescription& p = grid[r][c].prescription;
                        power += p.meanPower;
                        sphere += p.sphere;
                        cylinder += p.cylinder;
                        ++count;
                    }
                }
                if (count < 3) continue;
                power /= count;
                if (power > bestPower) {
                    bestPower = power;
                    nearZone = grid[row][column].prescription;
                    nearZone.meanPower = power;
                    nearZone.sphere = sphere / count;
                    nearZone.cylinder = cylinder / count;
                    nearU = grid[row][column].u;
                    nearV = grid[row][column].v;
                    zonesFound = true;
                }
            }
        }
        // The distance part beside the near zone: clean windows in the same rows that are
        // nowhere near the near power. Measured power drifts a little from the top of the
        // lens to the bottom, so the add is read against these rather than the top.
        if (zonesFound) {
            std::vector<double> beside;
            for (const std::vector<Candidate>& row : grid) {
                if (row.empty() || std::abs(row[0].v - nearV) > 0.75 * step) continue;
                for (const Candidate& candidate : row) {
                    if (candidate.clean && candidate.prescription.sphere <
                            distance.sphere + (nearZone.sphere - distance.sphere) / 3.0) {
                        beside.push_back(candidate.prescription.sphere);
                    }
                }
            }
            if (beside.size() >= 2) {
                std::nth_element(beside.begin(), beside.begin() + beside.size() / 2, beside.end());
                besideSphere = beside[beside.size() / 2];
            }
        }
    }
    if (std::getenv("LENS_DEBUG_MAP")) {
        // Power map: rows top to bottom, columns left to right (as the camera sees the lens).
        const double uLeft = percentile(us, 0.05);
        const double uRight = percentile(us, 0.95);
        const double step = windowRadius;
        std::fprintf(stderr, "map (mean power / cylinder @ axis), window r %.1f px\n", windowRadius);
        for (double v = vTop + windowRadius; v <= vBottom - windowRadius + 0.5; v += step) {
            std::fprintf(stderr, "v %5.0f:", v - vTop);
            for (double u = uLeft + windowRadius; u <= uRight - windowRadius + 0.5; u += step) {
                Prescription local;
                if (localPrescription(u, v, local)) {
                    std::fprintf(stderr, " %6.2f/%5.2f@%3.0f", local.meanPower, local.cylinder, local.axis);
                } else {
                    std::fprintf(stderr, "       ---      ");
                }
            }
            std::fprintf(stderr, "\n");
        }
    }
    if (std::getenv("LENS_DEBUG")) {
        std::fprintf(stderr, "zones %d r %.1f vTop %.1f vBottom %.1f width %.1f profile:", zonesFound,
                     windowRadius, vTop, vBottom, lensWidth);
        for (double p : profile) std::fprintf(stderr, " %.2f", p);
        if (zonesFound) std::fprintf(stderr, "  dist %.2f near %.2f", distance.sphere, nearZone.sphere);
        std::fprintf(stderr, "\n");
    }
    result.lensType = LensType::SingleVision;
    if (zonesFound) {
        // Add as prescribed: change in sphere from the distance zone to the near zone.
        const double add = nearZone.sphere - distance.sphere;
        // Adds start at +0.75; a single-vision lens can show up to ~0.4 D of noise here.
        constexpr double MIN_ADD = 0.75;
        if (add >= MIN_ADD) {
            // A bifocal's power jumps at the segment line; a progressive's rises along a
            // corridor a good part of the lens height long. Follow the power with small
            // windows from the distance zone down to the near zone and see how far it takes
            // to go from 20% to 80% of the way up.
            const double vDistance = vTop + 1.3 * windowRadius;
            const double smallRadius = std::max(5.0, 0.4 * windowRadius);
            std::vector<double> path;
            for (double v = vDistance; v <= nearV; v += 1.0) {
                const double fraction = (v - vDistance) / std::max(1.0, nearV - vDistance);
                Prescription local;
                if (localPrescription(uCenter + fraction * (nearU - uCenter), v, local, smallRadius)) {
                    path.push_back(local.meanPower);
                } else {
                    path.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
            const double low = distance.meanPower;
            const double high = nearZone.meanPower;
            int rise20 = -1;
            int rise80 = -1;
            for (int index = 0; index < static_cast<int>(path.size()); ++index) {
                if (!std::isfinite(path[index])) continue;
                const double level = (path[index] - low) / (high - low);
                if (level < 0.2) rise20 = -1;
                else if (rise20 < 0) rise20 = index;
                if (rise20 >= 0 && level >= 0.8) { rise80 = index; break; }
            }
            const double riseLength = rise20 >= 0 && rise80 >= 0
                    ? (rise80 - rise20) * std::hypot(nearV - vDistance, nearU - uCenter) /
                      std::max(1.0, nearV - vDistance) / (vBottom - vTop)
                    : 1.0;
            if (std::getenv("LENS_DEBUG")) {
                std::fprintf(stderr, "near at u %.0f v %.0f, rise %d-%d of %zu, length %.2f of lens\n",
                             nearU - uCenter, nearV - vTop, rise20, rise80, path.size(), riseLength);
            }
            constexpr double SEGMENT_RISE_LENGTH = 0.2;
            result.lensType = riseLength < SEGMENT_RISE_LENGTH ? LensType::Bifocal
                                                               : LensType::Progressive;
            result.distanceSphere = distance.sphere;
            result.distanceCylinder = distance.cylinder;
            result.distanceAxis = distance.axis;
            result.nearSphere = nearZone.sphere;
            result.nearCylinder = nearZone.cylinder;
            result.nearAxis = nearZone.axis;
            // A progressive's lower corners are blending zones, not distance power.
            const bool useBeside = result.lensType == LensType::Bifocal && !std::isnan(besideSphere);
            result.addPower = useBeside ? nearZone.sphere - besideSphere : add;
        }
    }
    result.status = LensMeasurementStatus::Ok;
    return result;
}

}  // namespace

bool measureLensOpening(const cv::Mat& grayscaleFrame, const cv::Point2f& screenCenter,
                        double minSizePixels, double& openingPixels,
                        std::vector<cv::Point>* outline) {
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
    size_t bestIndex = 0;
    for (size_t index = 0; index < contours.size(); ++index) {
        if (hierarchy[index][3] < 0) continue;
        if (cv::pointPolygonTest(contours[index], screenCenter, false) <= 0) continue;
        const double size = std::sqrt(std::abs(cv::contourArea(contours[index])));
        if (size < minSizePixels) continue;
        if (best == 0.0 || size < best) {
            best = size;
            bestIndex = index;
        }
    }
    if (best == 0.0) return false;
    openingPixels = best;
    if (outline) *outline = contours[bestIndex];
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

bool locateLens(const FringeRun& reference, const FringeRun& lens,
                const cv::Point2f& centerGuess, cv::Point2f& lensCenter) {
    if (!validRun(reference) || !validRun(lens)) return false;
    cv::Mat mask;
    double openingPixels = 0.0;
    if (findLensRegion(reference, lens, centerGuess, mask, openingPixels, nullptr) !=
            LensMeasurementStatus::Ok) {
        return false;
    }
    const cv::Moments moments = cv::moments(mask, true);
    if (moments.m00 <= 0.0) return false;
    lensCenter = cv::Point2f(static_cast<float>(moments.m10 / moments.m00),
                             static_cast<float>(moments.m01 / moments.m00));
    return true;
}

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
        double screenDistanceMm, double screenOpeningPixels, cv::Mat* brightnessRatio,
        double assumedLensDistanceMm) {
    LensMeasurement result;
    if (!validRun(reference) || !validRun(lens) ||
            lens.wrapped[0].size() != reference.wrapped[0].size() || !(screenDistanceMm > 0.0)) {
        return result;
    }
    cv::Mat mask;
    double openingPixels = 0.0;
    result.status = findLensRegion(reference, lens, lensCenterGuess, mask, openingPixels,
                                   brightnessRatio);
    if (result.status == LensMeasurementStatus::LensNotFound ||
            result.status == LensMeasurementStatus::LensTouchesEdge) {
        // No closed dark rim (half-rim, rimless, gold): find the lens by its effect instead.
        cv::Mat magnificationMask;
        double magnificationOpening = 0.0;
        if (findLensRegionByMagnification(reference, lens, lensCenterGuess, magnificationMask,
                                          magnificationOpening) == LensMeasurementStatus::Ok) {
            mask = magnificationMask;
            openingPixels = magnificationOpening;
            result.status = LensMeasurementStatus::Ok;
        }
    }
    result.lensOpeningPixels = openingPixels;
    if (result.status != LensMeasurementStatus::Ok) return result;
    // Apparent size is inversely proportional to distance; without a size measured on the
    // screen (no rim to measure), use the assumed distance (temple tips on the phone mount).
    const double lensDistanceMm = screenOpeningPixels > 0.0
            ? screenDistanceMm * screenOpeningPixels / openingPixels
            : assumedLensDistanceMm;
    // Glasses are held near the phone; an area implying otherwise isn't the lens.
    if (lensDistanceMm < 50.0 || lensDistanceMm > 0.6 * screenDistanceMm) {
        result.status = LensMeasurementStatus::LensNotFound;
        return result;
    }
    return fitLensPower(reference, lens, monitorDown, mask, openingPixels,
                        screenDistanceMm, lensDistanceMm);
}
