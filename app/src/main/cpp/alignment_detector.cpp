#include "alignment_detector.h"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace {

// Reads the four step squares assuming the layout of drawPhaseTicks() in
// monitor-pattern-player.html sits exactly here. Strict: every square must be evenly lit and
// clearly brighter (1) or darker (0) than a uniform mid-grey frame (gaps, row ends, and
// above and below each square), which blank screen and fringe stripes never are.
bool readSquaresAt(const cv::Mat& frame, const cv::Point2f& center, float targetSize,
                   const cv::Point2f& down, const cv::Point2f& right, int& phaseCode) {
    const float cellSize = targetSize * 0.20f;
    const float cellGap = targetSize * 0.16f;
    const float margin = targetSize * 0.12f;
    const float codeWidth = cellSize * 4.0f + cellGap * 3.0f;
    const cv::Point2f rowCenter = center + down * (targetSize * 0.72f + cellSize * 0.5f);
    const cv::Rect imageBounds(0, 0, frame.cols, frame.rows);

    // Mean of a small square around a point, or -1 if it falls off the image.
    const auto meanAt = [&](const cv::Point2f& point, float radiusPixels) {
        const int radius = std::max(0, static_cast<int>(std::round(radiusPixels)));
        const cv::Rect bounds(static_cast<int>(std::round(point.x)) - radius,
                              static_cast<int>(std::round(point.y)) - radius,
                              radius * 2 + 1, radius * 2 + 1);
        if ((bounds & imageBounds) != bounds) return -1.0;
        return cv::mean(frame(bounds))[0];
    };
    const auto alongRow = [&](float offset) { return rowCenter + right * offset; };

    constexpr double MAX_SPREAD = 30.0;
    constexpr double MIN_CONTRAST = 20.0;

    double cells[4];
    for (int bit = 0; bit < 4; ++bit) {
        const cv::Point2f cellCenter =
                alongRow(-codeWidth * 0.5f + bit * (cellSize + cellGap) + cellSize * 0.5f);
        const float quarter = cellSize * 0.2f;
        const double samples[] = {
            meanAt(cellCenter, cellSize * 0.12f),
            meanAt(cellCenter + right * quarter, 0.0f),
            meanAt(cellCenter - right * quarter, 0.0f),
            meanAt(cellCenter + down * quarter, 0.0f),
            meanAt(cellCenter - down * quarter, 0.0f),
        };
        double low = 255.0;
        double high = 0.0;
        double sum = 0.0;
        for (const double value : samples) {
            if (value < 0.0) return false;
            low = std::min(low, value);
            high = std::max(high, value);
            sum += value;
        }
        if (high - low > MAX_SPREAD) return false;
        cells[bit] = sum / 5.0;
    }

    std::vector<double> greys;
    for (int gap = 0; gap < 3; ++gap) {
        greys.push_back(meanAt(alongRow(-codeWidth * 0.5f + (gap + 1) * cellSize +
                                        gap * cellGap + cellGap * 0.5f), cellGap * 0.2f));
    }
    greys.push_back(meanAt(alongRow(-codeWidth * 0.5f - margin * 0.5f), 0.0f));
    greys.push_back(meanAt(alongRow(codeWidth * 0.5f + margin * 0.5f), 0.0f));
    // Margin above and below every square: a stripe running through a square runs through
    // these too, so stripes can't show the required square-to-grey contrast.
    for (int bit = 0; bit < 4; ++bit) {
        const cv::Point2f cellCenter =
                alongRow(-codeWidth * 0.5f + bit * (cellSize + cellGap) + cellSize * 0.5f);
        for (const float side : {-1.0f, 1.0f}) {
            greys.push_back(meanAt(cellCenter + down * (side * (cellSize * 0.5f + margin * 0.5f)), 0.0f));
        }
    }
    double low = 255.0;
    double high = 0.0;
    double sum = 0.0;
    for (const double value : greys) {
        if (value < 0.0) return false;
        low = std::min(low, value);
        high = std::max(high, value);
        sum += value;
    }
    if (high - low > MAX_SPREAD) return false;
    const double grey = sum / greys.size();

    phaseCode = 0;
    for (int bit = 0; bit < 4; ++bit) {
        if (cells[bit] > grey + MIN_CONTRAST) {
            phaseCode |= 1 << bit;
        } else if (cells[bit] >= grey - MIN_CONTRAST) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool decodeAlignmentPhaseCode(
        const cv::Mat& grayscaleFrame, const cv::Point2f& normalizedCenter,
        float normalizedSize, int& phaseCode, const cv::Point2f& monitorDown) {
    const float downLength = std::hypot(monitorDown.x, monitorDown.y);
    if (grayscaleFrame.empty() || grayscaleFrame.type() != CV_8UC1 || normalizedSize <= 0.0f ||
            !std::isfinite(downLength) || downLength < 1e-6f) {
        return false;
    }

    // Monitor right is monitor down rotated -90° (image y axis points down, no mirroring).
    const cv::Point2f down = monitorDown * (1.0f / downLength);
    const cv::Point2f right(down.y, -down.x);
    const cv::Point2f center(
            normalizedCenter.x * grayscaleFrame.cols,
            normalizedCenter.y * grayscaleFrame.rows);
    const float targetSize = normalizedSize * grayscaleFrame.rows;

    // The bullseye-based size is only good to a few percent, enough to miss squares a few
    // millimetres across, so try small shifts and size corrections, nearest first. The
    // strict checks in readSquaresAt keep anything but the real squares from matching.
    const float step = std::max(1.0f, targetSize * 0.02f);
    const int downSteps = static_cast<int>(std::ceil(targetSize * 0.16f / step));
    for (int stepIndex = 0; stepIndex <= 2 * downSteps; ++stepIndex) {
        const float alongDown = ((stepIndex + 1) / 2) * step * (stepIndex % 2 ? -1.0f : 1.0f);
        for (const float scale : {1.0f, 0.97f, 1.03f, 0.94f, 1.06f, 0.91f, 1.09f}) {
            for (const float alongRight : {0.0f, -0.03f, 0.03f, -0.06f, 0.06f}) {
                if (readSquaresAt(grayscaleFrame,
                                  center + down * alongDown + right * (alongRight * targetSize),
                                  targetSize * scale, down, right, phaseCode)) {
                    return true;
                }
            }
        }
    }
    return false;
}

namespace {

// One concentric black/yellow/black target. radius is the yellow disc's outer radius,
// which is 0.4 of the drawn bullseye diameter.
struct RingTarget {
    cv::Point2f center;
    float radius;
    float score;
};

std::vector<RingTarget> findRingTargets(const cv::Mat& grayscaleFrame) {
    const float minimumRadius = std::min(grayscaleFrame.cols, grayscaleFrame.rows) * 0.008f;
    const float maximumRadius = std::min(grayscaleFrame.cols, grayscaleFrame.rows) * 0.16f;
    std::vector<RingTarget> targets;

    for (const int thresholdType : {cv::THRESH_BINARY, cv::THRESH_BINARY_INV}) {
        cv::Mat binary;
        cv::threshold(grayscaleFrame, binary, 0.0, 255.0, thresholdType | cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        for (size_t index = 0; index < contours.size(); ++index) {
            const int childIndex = hierarchy[index][2];
            if (childIndex < 0) continue;

            const double area = std::abs(cv::contourArea(contours[index]));
            const double perimeter = cv::arcLength(contours[index], true);
            if (area <= 0.0 || perimeter <= 0.0) continue;
            const float circularity = static_cast<float>(4.0 * CV_PI * area /
                    (perimeter * perimeter));
            if (circularity < 0.82f) continue;

            cv::Point2f outerCenter;
            float outerRadius = 0.0f;
            cv::minEnclosingCircle(contours[index], outerCenter, outerRadius);
            if (outerRadius < minimumRadius || outerRadius > maximumRadius) continue;

            cv::Point2f innerCenter;
            float innerRadius = 0.0f;
            cv::minEnclosingCircle(contours[childIndex], innerCenter, innerRadius);
            const float radiusRatio = innerRadius / outerRadius;
            const float centerError = cv::norm(outerCenter - innerCenter) / outerRadius;
            if (radiusRatio < 0.30f || radiusRatio > 0.70f || centerError > 0.18f) continue;

            const float score = circularity - std::abs(radiusRatio - 0.5f) - centerError;
            if (score <= 0.0f) continue;
            const RingTarget candidate{(outerCenter + innerCenter) * 0.5f, outerRadius, score};

            // Both threshold polarities can find the same target; keep the better one.
            auto duplicate = std::find_if(targets.begin(), targets.end(),
                    [&](const RingTarget& other) {
                        return cv::norm(other.center - candidate.center) <
                               std::max(other.radius, candidate.radius);
                    });
            if (duplicate == targets.end()) {
                targets.push_back(candidate);
            } else if (candidate.score > duplicate->score) {
                *duplicate = candidate;
            }
        }
    }
    return targets;
}

// The central bullseye is drawn 5/3 the size of the corner markers. If the largest ring isn't
// clearly bigger than every other one, the center is out of sight (e.g. seen through a lens)
// and the largest is just a corner marker, so report no center rather than a wrong one.
const RingTarget* largestTarget(const std::vector<RingTarget>& targets) {
    const auto largest = std::max_element(targets.begin(), targets.end(),
            [](const RingTarget& a, const RingTarget& b) { return a.radius < b.radius; });
    if (largest == targets.end()) return nullptr;
    for (const RingTarget& other : targets) {
        if (&other != &*largest && other.radius > largest->radius * 0.8f) return nullptr;
    }
    return &*largest;
}

}  // namespace

bool detectAlignmentTarget(
    const cv::Mat& grayscaleFrame, cv::Point2f& normalizedCenter,
    float& normalizedSize, int& phaseCode, const cv::Point2f& monitorDown) {
    if (grayscaleFrame.empty() || grayscaleFrame.type() != CV_8UC1) return false;

    const std::vector<RingTarget> targets = findRingTargets(grayscaleFrame);
    const RingTarget* center = largestTarget(targets);
    if (!center) return false;
    normalizedCenter = cv::Point2f(
            center->center.x / grayscaleFrame.cols,
            center->center.y / grayscaleFrame.rows);
    normalizedSize = 2.5f * center->radius / grayscaleFrame.rows;
    // The bullseye still counts as found when its step squares can't be read (-1).
    if (!decodeAlignmentPhaseCode(
            grayscaleFrame, normalizedCenter, normalizedSize, phaseCode, monitorDown)) {
        phaseCode = -1;
    }
    return true;
}

bool estimateMonitorTilt(
        const cv::Mat& grayscaleFrame, const cv::Point2f& monitorDown,
        const cv::Matx33d& cameraMatrix, float cornerOffsetMm,
        float& tiltRightDeg, float& tiltDownDeg, float& distanceMm,
        cv::Point2f* cornerPoints) {
    const float downLength = std::hypot(monitorDown.x, monitorDown.y);
    if (grayscaleFrame.empty() || grayscaleFrame.type() != CV_8UC1 ||
            !std::isfinite(downLength) || downLength < 1e-6f || cornerOffsetMm <= 0.0f) {
        return false;
    }
    const cv::Point2f down = monitorDown * (1.0f / downLength);
    const cv::Point2f right(down.y, -down.x);

    const std::vector<RingTarget> targets = findRingTargets(grayscaleFrame);
    const RingTarget* center = largestTarget(targets);
    if (!center) return false;

    // Assign each smaller marker to a monitor quadrant by its direction from the center.
    const RingTarget* corners[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    for (const RingTarget& target : targets) {
        if (&target == center) continue;
        const float sizeRatio = target.radius / center->radius;
        if (sizeRatio < 0.35f || sizeRatio > 0.85f) continue;
        const cv::Point2f offset = target.center - center->center;
        const float alongRight = offset.dot(right);
        const float alongDown = offset.dot(down);
        if (std::abs(alongRight) < center->radius * 1.5f ||
                std::abs(alongDown) < center->radius * 1.5f) continue;
        const float diagonalRatio = std::abs(alongRight) / std::abs(alongDown);
        if (diagonalRatio < 0.5f || diagonalRatio > 2.0f) continue;
        const RingTarget*& slot = corners[alongDown > 0.0f][alongRight > 0.0f];
        if (!slot || target.score > slot->score) slot = &target;
    }

    // Monitor coordinates in mm: x right, y down, z into the screen (OpenCV convention).
    std::vector<cv::Point3f> objectPoints{cv::Point3f(0.0f, 0.0f, 0.0f)};
    std::vector<cv::Point2f> imagePoints{center->center};
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            if (!corners[row][column]) continue;
            objectPoints.emplace_back(
                    (column ? 1.0f : -1.0f) * cornerOffsetMm,
                    (row ? 1.0f : -1.0f) * cornerOffsetMm, 0.0f);
            imagePoints.push_back(corners[row][column]->center);
        }
    }
    if (cornerPoints) {
        const RingTarget* ordered[4] = {corners[0][0], corners[0][1], corners[1][1], corners[1][0]};
        for (int index = 0; index < 4; ++index) {
            cornerPoints[index] = ordered[index]
                    ? ordered[index]->center
                    : cv::Point2f(std::nanf(""), std::nanf(""));
        }
    }
    if (objectPoints.size() < 4) return false;

    cv::Mat rotationVector;
    cv::Mat translation;
    if (!cv::solvePnP(objectPoints, imagePoints, cv::Mat(cameraMatrix), cv::noArray(),
                      rotationVector, translation, false, cv::SOLVEPNP_ITERATIVE)) {
        return false;
    }
    cv::Matx33d rotation;
    cv::Rodrigues(rotationVector, rotation);
    // Columns of the rotation are the monitor's axes in camera space; their z components say
    // how fast the monitor recedes from the camera moving right / down across the screen.
    tiltRightDeg = static_cast<float>(std::asin(std::clamp(rotation(2, 0), -1.0, 1.0)) * 180.0 / CV_PI);
    tiltDownDeg = static_cast<float>(std::asin(std::clamp(rotation(2, 1), -1.0, 1.0)) * 180.0 / CV_PI);
    distanceMm = static_cast<float>(translation.at<double>(2));
    return std::isfinite(tiltRightDeg) && std::isfinite(tiltDownDeg) && distanceMm > 0.0f;
}
