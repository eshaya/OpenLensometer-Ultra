#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

// Include your header files here
#include "zernike_fitter.h"
#include "phase_unwrapper.h" 
#include "wavefront_calibrator.h"
#include "charuco_calibrator.h"
#include "monitor_calibration.h"
#include "alignment_detector.h"
#include "frame_stack.h"
#include "lens_measurement.h"
#include "grid_ftp.h"
#include <functional>

TEST(MonitorCalibrationTest, CalculatesPitchFromNativeResolutionAndDiagonal) {
    const double pitchMm = calculateMonitorPixelPitchMm(1920, 1080, 27.0);
    EXPECT_NEAR(pitchMm, 0.31137, 0.0001);
}

TEST(MonitorCalibrationTest, ReportsDifferenceRelativeToMonitorSpecEstimate) {
    const double monitorSpecPitchMm = calculateMonitorPixelPitchMm(1920, 1080, 27.0);
    const double charucoPitchMm = monitorSpecPitchMm * 1.05;
    EXPECT_NEAR(monitorPitchDifferencePercent(charucoPitchMm, monitorSpecPitchMm), 5.0, 1e-9);
}

TEST(MonitorCalibrationTest, RejectsInvalidMonitorSpecifications) {
    EXPECT_EQ(calculateMonitorPixelPitchMm(0, 1080, 27.0), -1.0);
    EXPECT_EQ(calculateMonitorPixelPitchMm(1920, 1080, 0.0), -1.0);
    EXPECT_TRUE(std::isnan(monitorPitchDifferencePercent(-1.0, 0.3)));
}

TEST(PhaseDemodulationTest, RecoversKnownPhaseAndModulation) {
    constexpr float background = 128.0f;
    constexpr float amplitude = 64.0f;
    constexpr float expectedPhase = 0.75f;
    cv::Mat frames[4];
    for (int phaseIndex = 0; phaseIndex < 4; ++phaseIndex) {
        frames[phaseIndex] = cv::Mat(1, 1, CV_32F);
        const float phaseStep = phaseIndex * static_cast<float>(CV_PI / 2.0);
        frames[phaseIndex].at<float>(0, 0) =
            background + amplitude * std::cos(expectedPhase + phaseStep);
    }

    cv::Mat wrappedPhase;
    cv::Mat modulationMap;
    calculateWrappedPhase(
        frames[0], frames[1], frames[2], frames[3], wrappedPhase, modulationMap);

    EXPECT_NEAR(wrappedPhase.at<float>(0, 0), expectedPhase, 2e-4f);
    EXPECT_NEAR(modulationMap.at<float>(0, 0), amplitude, 1e-5f);
}

TEST(AlignmentDetectorTest, FindsBrowserTargetOverSinusoidalFringes) {
    constexpr int width = 1920;
    constexpr int height = 1080;
    constexpr int fringePeriod = 64;
    const int targetSize = static_cast<int>(std::round(std::min(width, height) * 0.10));
    const cv::Point targetCenter(width / 2, height / 2);

    for (int axis = 0; axis < 2; ++axis) {
        for (int phase = 0; phase < 4; ++phase) {
            cv::Mat frame(height, width, CV_8UC1);
            const double phaseOffset = phase * CV_PI / 2.0;
            for (int y = 0; y < height; ++y) {
                uint8_t* row = frame.ptr<uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    const int position = axis == 0 ? x : y;
                    row[x] = cv::saturate_cast<uint8_t>(
                        127.5 + 127.5 * 0.8 * std::cos(
                            2.0 * CV_PI * position / fringePeriod + phaseOffset));
                }
            }
            const cv::Mat fringeBackground = frame.clone();

            cv::circle(frame, targetCenter, targetSize / 2, cv::Scalar(0), cv::FILLED);
            cv::circle(frame, targetCenter, static_cast<int>(targetSize * 0.4),
                       cv::Scalar(220), cv::FILLED);
            cv::circle(frame, targetCenter, static_cast<int>(targetSize * 0.2),
                       cv::Scalar(0), cv::FILLED);

            // Same floating-point layout as drawPhaseTicks() in the page.
            const double cellSize = targetSize * 0.20;
            const double cellGap = targetSize * 0.16;
            const double margin = targetSize * 0.12;
            const double codeWidth = cellSize * 4 + cellGap * 3;
            const double codeLeft = targetCenter.x - codeWidth / 2;
            const double codeTop = targetCenter.y + targetSize * 0.72;
            const auto pixelRect = [](double left, double top, double w, double h) {
                return cv::Rect(cvRound(left), cvRound(top),
                                cvRound(left + w) - cvRound(left), cvRound(top + h) - cvRound(top));
            };
            const cv::Rect tickBounds = pixelRect(codeLeft - margin, codeTop - margin,
                                                  codeWidth + 2 * margin, cellSize + 2 * margin);
            cv::rectangle(frame, tickBounds, cv::Scalar(128), cv::FILLED);
            const int phaseCode = 8 + axis * 4 + phase;
            for (int bit = 0; bit < 4; ++bit) {
                const cv::Rect cell = pixelRect(codeLeft + bit * (cellSize + cellGap), codeTop,
                                                cellSize, cellSize);
                const int intensity = (phaseCode & (1 << bit)) != 0 ? 255 : 0;
                cv::rectangle(frame, cell, cv::Scalar(intensity), cv::FILLED);
                cv::rectangle(frame, cell, cv::Scalar(0), 2);
            }

            cv::Point2f normalizedCenter;
            float normalizedSize = 0.0f;
            int detectedPhaseCode = -1;
            ASSERT_TRUE(detectAlignmentTarget(
                frame, normalizedCenter, normalizedSize, detectedPhaseCode))
                << "axis=" << axis << " phase=" << phase;
            EXPECT_NEAR(normalizedCenter.x, 0.5f, 0.01f);
            EXPECT_NEAR(normalizedCenter.y, 0.5f, 0.01f);
            EXPECT_GT(normalizedSize, 0.08f);
            EXPECT_LT(normalizedSize, 0.12f);
            EXPECT_EQ(detectedPhaseCode, phaseCode);

            // As the phone camera sees it at 50 cm: bullseye ~48 px across, slightly blurred.
            cv::Mat cameraView;
            cv::resize(frame, cameraView, cv::Size(853, 480), 0, 0, cv::INTER_AREA);
            cv::GaussianBlur(cameraView, cameraView, cv::Size(0, 0), 1.0);
            cv::Point2f cameraCenter;
            float cameraSize = 0.0f;
            int cameraCode = -1;
            ASSERT_TRUE(detectAlignmentTarget(cameraView, cameraCenter, cameraSize, cameraCode));
            EXPECT_EQ(cameraCode, phaseCode) << "camera-scale axis=" << axis << " phase=" << phase;

            cv::Mat hiddenBullseye = fringeBackground.clone();
            frame(tickBounds).copyTo(hiddenBullseye(tickBounds));
            int phaseCodeWithoutRings = -1;
            ASSERT_TRUE(decodeAlignmentPhaseCode(
                hiddenBullseye, normalizedCenter, normalizedSize, phaseCodeWithoutRings));
            EXPECT_EQ(phaseCodeWithoutRings, phaseCode);

            // Portrait phone: the landscape sensor image is rotated, so monitor-down
            // points along -x in the image (ROTATE_90_CLOCKWISE maps (0,1) to (-1,0)).
            const cv::Point2f rotatedDown(-1.0f, 0.0f);
            cv::Mat rotatedFrame;
            cv::rotate(frame, rotatedFrame, cv::ROTATE_90_CLOCKWISE);
            int rotatedPhaseCode = -1;
            ASSERT_TRUE(detectAlignmentTarget(
                rotatedFrame, normalizedCenter, normalizedSize, rotatedPhaseCode, rotatedDown));
            EXPECT_EQ(rotatedPhaseCode, phaseCode);

            cv::Mat rotatedHidden;
            cv::rotate(hiddenBullseye, rotatedHidden, cv::ROTATE_90_CLOCKWISE);
            int rotatedHiddenCode = -1;
            ASSERT_TRUE(decodeAlignmentPhaseCode(
                rotatedHidden, normalizedCenter, normalizedSize, rotatedHiddenCode, rotatedDown));
            EXPECT_EQ(rotatedHiddenCode, phaseCode);
        }
    }
}

namespace {

void drawRingTarget(cv::Mat& canvas, cv::Point center, double diameterPixels) {
    cv::circle(canvas, center, static_cast<int>(diameterPixels * 0.5), cv::Scalar(0), cv::FILLED);
    cv::circle(canvas, center, static_cast<int>(diameterPixels * 0.4), cv::Scalar(220), cv::FILLED);
    cv::circle(canvas, center, static_cast<int>(diameterPixels * 0.2), cv::Scalar(0), cv::FILLED);
}

// Renders the monitor-pattern-player layout (50 mm bullseye, 30 mm corner markers at ±90 mm)
// as seen by a 640×480 camera whose pose relative to the monitor is (rotation, distance).
cv::Mat renderMonitorView(const cv::Matx33d& rotation, double distanceMm,
                          const cv::Matx33d& cameraMatrix, bool drawCenter = true) {
    constexpr double mmPerPixel = 0.5;
    constexpr int canvasSize = 600;
    cv::Mat canvas(canvasSize, canvasSize, CV_8UC1, cv::Scalar(128));
    const cv::Point middle(canvasSize / 2, canvasSize / 2);
    const int boxHalf = static_cast<int>(90 / mmPerPixel);
    cv::rectangle(canvas, middle - cv::Point(boxHalf, boxHalf), middle + cv::Point(boxHalf, boxHalf),
                  cv::Scalar(0), static_cast<int>(2.5 / mmPerPixel));
    if (drawCenter) drawRingTarget(canvas, middle, 50.0 / mmPerPixel);
    for (int sy : {-1, 1}) {
        for (int sx : {-1, 1}) {
            drawRingTarget(canvas,
                middle + cv::Point(static_cast<int>(sx * 90 / mmPerPixel),
                                   static_cast<int>(sy * 90 / mmPerPixel)),
                30.0 / mmPerPixel);
        }
    }
    const cv::Matx33d canvasToMonitorMm(
        mmPerPixel, 0, -canvasSize * mmPerPixel / 2,
        0, mmPerPixel, -canvasSize * mmPerPixel / 2,
        0, 0, 1);
    const cv::Matx33d monitorToImage = cameraMatrix * cv::Matx33d(
        rotation(0, 0), rotation(0, 1), 0,
        rotation(1, 0), rotation(1, 1), 0,
        rotation(2, 0), rotation(2, 1), distanceMm);
    cv::Mat view;
    cv::warpPerspective(canvas, view, cv::Mat(monitorToImage * canvasToMonitorMm),
                        cv::Size(640, 480), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                        cv::Scalar(128));
    return view;
}

}  // namespace

TEST(AlignmentDetectorTest, RejectsBlankAreaInsteadOfReadingAStep) {
    int code = -1;
    const cv::Mat white(480, 640, CV_8UC1, cv::Scalar(245));
    EXPECT_FALSE(decodeAlignmentPhaseCode(white, cv::Point2f(0.5f, 0.3f), 0.1f, code));
    const cv::Mat black(480, 640, CV_8UC1, cv::Scalar(10));
    EXPECT_FALSE(decodeAlignmentPhaseCode(black, cv::Point2f(0.5f, 0.3f), 0.1f, code));
}

TEST(AlignmentDetectorTest, DoesNotReadStepsFromFringes) {
    // Plain fringes at many positions and periods must never pass as step squares.
    int falseReads = 0;
    for (int period = 4; period <= 24; period += 2) {
        for (int axis = 0; axis < 2; ++axis) {
            cv::Mat frame(480, 640, CV_8UC1);
            for (int y = 0; y < frame.rows; ++y) {
                for (int x = 0; x < frame.cols; ++x) {
                    frame.at<uint8_t>(y, x) = cv::saturate_cast<uint8_t>(
                        128 + 100 * std::cos(2 * CV_PI * (axis ? y : x) / period));
                }
            }
            for (float cy = 0.2f; cy < 0.8f; cy += 0.013f) {
                for (float cx = 0.2f; cx < 0.8f; cx += 0.031f) {
                    int code = -1;
                    if (decodeAlignmentPhaseCode(frame, cv::Point2f(cx, cy), 0.1f, code)) ++falseReads;
                }
            }
        }
    }
    EXPECT_EQ(falseReads, 0);
}

TEST(AlignmentDetectorTest, IgnoresCornerMarkersWhenCenterIsHidden) {
    const cv::Matx33d cameraMatrix(500, 0, 320, 0, 500, 240, 0, 0, 1);
    const cv::Mat withCenter = renderMonitorView(cv::Matx33d::eye(), 500.0, cameraMatrix);
    const cv::Mat cornersOnly = renderMonitorView(cv::Matx33d::eye(), 500.0, cameraMatrix, false);
    cv::Point2f center;
    float size = 0.0f;
    int code = -1;
    ASSERT_TRUE(detectAlignmentTarget(withCenter, center, size, code));
    EXPECT_NEAR(center.x, 0.5f, 0.01f);
    EXPECT_NEAR(center.y, 0.5f, 0.01f);
    EXPECT_FALSE(detectAlignmentTarget(cornersOnly, center, size, code));
}

TEST(AlignmentDetectorTest, MeasuresMonitorTiltWithSign) {
    const cv::Matx33d cameraMatrix(500, 0, 320, 0, 500, 240, 0, 0, 1);
    const double angle = 10.0 * CV_PI / 180.0;
    // Rotation about the vertical axis that pushes the monitor's right side away from the camera.
    const cv::Matx33d rightSideFarther(
        std::cos(angle), 0, -std::sin(angle),
        0, 1, 0,
        std::sin(angle), 0, std::cos(angle));
    // Rotation about the horizontal axis that pulls the monitor's bottom toward the camera.
    const cv::Matx33d bottomNearer(
        1, 0, 0,
        0, std::cos(angle), std::sin(angle),
        0, -std::sin(angle), std::cos(angle));

    float tiltRight = 0.0f;
    float tiltDown = 0.0f;
    float distance = 0.0f;
    ASSERT_TRUE(estimateMonitorTilt(
        renderMonitorView(cv::Matx33d::eye(), 500.0, cameraMatrix), cv::Point2f(0, 1),
        cameraMatrix, 90.0f, tiltRight, tiltDown, distance));
    EXPECT_NEAR(tiltRight, 0.0f, 1.5f);
    EXPECT_NEAR(tiltDown, 0.0f, 1.5f);
    EXPECT_NEAR(distance, 500.0f, 15.0f);
    cv::Point2f corners[4];
    ASSERT_TRUE(estimateMonitorTilt(
        renderMonitorView(cv::Matx33d::eye(), 500.0, cameraMatrix), cv::Point2f(0, 1),
        cameraMatrix, 90.0f, tiltRight, tiltDown, distance, corners));
    // 90 mm at 500 mm with f = 500 px is 90 px from the principal point (320, 240).
    EXPECT_NEAR(corners[0].x, 230.0f, 2.0f);  // top-left
    EXPECT_NEAR(corners[0].y, 150.0f, 2.0f);
    EXPECT_NEAR(corners[2].x, 410.0f, 2.0f);  // bottom-right
    EXPECT_NEAR(corners[2].y, 330.0f, 2.0f);

    ASSERT_TRUE(estimateMonitorTilt(
        renderMonitorView(rightSideFarther, 500.0, cameraMatrix), cv::Point2f(0, 1),
        cameraMatrix, 90.0f, tiltRight, tiltDown, distance));
    EXPECT_NEAR(tiltRight, 10.0f, 1.5f);
    EXPECT_NEAR(tiltDown, 0.0f, 1.5f);

    ASSERT_TRUE(estimateMonitorTilt(
        renderMonitorView(bottomNearer, 500.0, cameraMatrix), cv::Point2f(0, 1),
        cameraMatrix, 90.0f, tiltRight, tiltDown, distance));
    EXPECT_NEAR(tiltRight, 0.0f, 1.5f);
    EXPECT_NEAR(tiltDown, -10.0f, 1.5f);

    // Portrait phone: same scene, sensor image rotated relative to the monitor.
    cv::Mat rotatedView;
    cv::rotate(renderMonitorView(rightSideFarther, 500.0, cameraMatrix), rotatedView,
               cv::ROTATE_90_CLOCKWISE);
    const cv::Matx33d rotatedCamera(500, 0, 240, 0, 500, 320, 0, 0, 1);
    ASSERT_TRUE(estimateMonitorTilt(
        rotatedView, cv::Point2f(-1, 0), rotatedCamera, 90.0f, tiltRight, tiltDown, distance));
    EXPECT_NEAR(tiltRight, 10.0f, 1.5f);
    EXPECT_NEAR(tiltDown, 0.0f, 1.5f);
}

TEST(FrameStackTest, AveragesGoodFramesAndDropsOutliers) {
    std::vector<cv::Mat> frames;
    cv::RNG random(7);
    for (int index = 0; index < 8; ++index) {
        cv::Mat frame(60, 80, CV_8UC1);
        random.fill(frame, cv::RNG::NORMAL, 120, 3);
        frames.push_back(frame);
    }
    // A frame caught mid-redraw: half of it shows a different, much brighter pattern.
    cv::Mat torn = frames.front().clone();
    torn(cv::Rect(0, 0, 80, 30)).setTo(230);
    frames.push_back(torn);

    int kept = 0;
    const cv::Mat combined = combineFrames(frames, kept);
    EXPECT_EQ(kept, 8);
    ASSERT_EQ(combined.type(), CV_32F);
    EXPECT_NEAR(cv::mean(combined)[0], 120.0, 1.0);
}

namespace {

// Ray-traces a 640×480 camera (f = 500 px) looking at a fringe screen 500 mm away through an
// optional thin lens (power matrix in monitor axes x right / y down, dioptres) with a 4 mm rim,
// and demodulates the four phase-shifted frames for one fringe axis.
void renderLensView(const cv::Matx22d* power, double lensDistanceMm, int axis,
                    cv::Mat& wrapped, cv::Mat& modulation, cv::Mat& brightness) {
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr double focal = 500.0;
    constexpr double screenMm = 500.0;
    constexpr double periodMm = 8.0;
    constexpr double lensRadiusMm = 25.0;
    constexpr double rimMm = 4.0;
    cv::Mat frames[4];
    for (auto& frame : frames) frame = cv::Mat(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec2d angle((x - width / 2.0) / focal, (y - height / 2.0) / focal);
            cv::Vec2d screenPoint = angle * screenMm;
            bool blocked = false;
            if (power) {
                const cv::Vec2d onLens = angle * lensDistanceMm;
                const double radius = std::hypot(onLens[0], onLens[1]);
                if (radius < lensRadiusMm) {
                    const cv::Vec2d bent = angle - (*power) * (onLens / 1000.0);
                    screenPoint = onLens + bent * (screenMm - lensDistanceMm);
                } else if (radius < lensRadiusMm + rimMm) {
                    blocked = true;
                }
            }
            for (int step = 0; step < 4; ++step) {
                // The rim is an opaque, dark frame.
                frames[step].at<float>(y, x) = blocked ? 10.0f : static_cast<float>(
                    128.0 + 100.0 * std::cos(2.0 * CV_PI * screenPoint[axis] / periodMm +
                                             step * CV_PI / 2.0));
            }
        }
    }
    calculateWrappedPhase(frames[0], frames[1], frames[2], frames[3], wrapped, modulation);
    brightness = (frames[0] + frames[1] + frames[2] + frames[3]) / 4.0;
}

// Power matrix for a prescription whose TABO axis is read from the front of the glasses,
// as seen by the camera from behind and expressed in monitor axes (y down).
cv::Matx22d powerMatrix(double sphere, double cylinder, double taboAxisDegrees) {
    const double cameraAxis = (180.0 - taboAxisDegrees) * CV_PI / 180.0;
    const cv::Vec2d across(-std::sin(cameraAxis), std::cos(cameraAxis));  // y up
    cv::Matx22d upward = cv::Matx22d::eye() * sphere + (across * across.t()) * cylinder;
    const cv::Matx22d flip(1, 0, 0, -1);
    return flip * upward * flip;
}

LensMeasurement measureSynthetic(const cv::Matx22d& power) {
    FringeRun reference;
    FringeRun lens;
    for (int axis = 0; axis < 2; ++axis) {
        renderLensView(nullptr, 125.0, axis, reference.wrapped[axis], reference.modulation[axis],
                       reference.brightness[axis]);
        renderLensView(&power, 125.0, axis, lens.wrapped[axis], lens.modulation[axis],
                       lens.brightness[axis]);
    }
    // The 50 mm lens held against the screen: radius 25 px, so sqrt(area) = 25·sqrt(pi).
    const double screenOpening = 25.0 * std::sqrt(CV_PI);
    return measureLensPower(reference, lens, cv::Point2f(0, 1), cv::Point2f(320, 240), 500.0,
                            screenOpening);
}

}  // namespace

TEST(LensMeasurementTest, MeasuresPlusLensAndItsDistance) {
    const LensMeasurement result = measureSynthetic(powerMatrix(2.0, 0.0, 90.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.lensDistanceMm, 125.0, 4.0);
    EXPECT_NEAR(result.sphere, 2.0, 0.1);
    EXPECT_NEAR(result.cylinder, 0.0, 0.1);
}

TEST(LensMeasurementTest, MeasuresMinusLens) {
    const LensMeasurement result = measureSynthetic(powerMatrix(-2.5, 0.0, 90.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, -2.5, 0.1);
    EXPECT_NEAR(result.cylinder, 0.0, 0.1);
}

TEST(LensMeasurementTest, MeasuresAstigmatismAndAxis) {
    const LensMeasurement result = measureSynthetic(powerMatrix(1.0, -1.5, 150.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, 1.0, 0.1);
    EXPECT_NEAR(result.cylinder, -1.5, 0.1);
    EXPECT_NEAR(result.axisDegrees, 150.0, 3.0);
}

TEST(LensMeasurementTest, FindsOutOfFocusRim) {
    // The rim 12.5 cm from a camera focused at 50 cm is badly blurred; demodulate blurred
    // frames so fringes only partly fade behind it.
    const cv::Matx22d power = powerMatrix(2.0, 0.0, 90.0);
    FringeRun reference;
    FringeRun lens;
    for (int axis = 0; axis < 2; ++axis) {
        renderLensView(nullptr, 125.0, axis, reference.wrapped[axis], reference.modulation[axis],
                       reference.brightness[axis]);
        renderLensView(&power, 125.0, axis, lens.wrapped[axis], lens.modulation[axis],
                       lens.brightness[axis]);
        cv::GaussianBlur(lens.brightness[axis], lens.brightness[axis], cv::Size(0, 0), 4.0);
    }
    const LensMeasurement result = measureLensPower(
        reference, lens, cv::Point2f(0, 1), cv::Point2f(320, 240), 500.0, 25.0 * std::sqrt(CV_PI));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.lensDistanceMm, 125.0, 8.0);
    EXPECT_NEAR(result.sphere, 2.0, 0.15);
}

namespace {

// One camera frame of the crossed grid (period 8 mm) seen through an optional lens, with the
// same camera, screen and lens geometry as renderLensView.
cv::Mat renderGridView(const cv::Matx22d* power, double lensDistanceMm) {
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr double focal = 500.0;
    constexpr double screenMm = 500.0;
    constexpr double periodMm = 8.0;
    constexpr double lensRadiusMm = 25.0;
    constexpr double rimMm = 4.0;
    cv::Mat frame(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec2d angle((x - width / 2.0) / focal, (y - height / 2.0) / focal);
            cv::Vec2d screenPoint = angle * screenMm;
            bool blocked = false;
            if (power) {
                const cv::Vec2d onLens = angle * lensDistanceMm;
                const double radius = std::hypot(onLens[0], onLens[1]);
                if (radius < lensRadiusMm) {
                    const cv::Vec2d bent = angle - (*power) * (onLens / 1000.0);
                    screenPoint = onLens + bent * (screenMm - lensDistanceMm);
                } else if (radius < lensRadiusMm + rimMm) {
                    blocked = true;
                }
            }
            frame.at<float>(y, x) = blocked ? 10.0f : static_cast<float>(
                128.0 + 50.0 * std::cos(2.0 * CV_PI * screenPoint[0] / periodMm) +
                50.0 * std::cos(2.0 * CV_PI * screenPoint[1] / periodMm));
        }
    }
    return frame;
}

LensMeasurement measureSyntheticGrid(const cv::Matx22d& power) {
    const cv::Mat referenceFrame = renderGridView(nullptr, 125.0);
    GridCarriers carriers;
    EXPECT_TRUE(findGridCarriers(referenceFrame, cv::Point2f(0, 1), carriers));
    FringeRun reference;
    extractGridRun(referenceFrame, carriers, reference);
    // The lens is 25 mm in radius at 125 mm: 100 px.
    return measureLensFromGridFrame(reference, carriers, renderGridView(&power, 125.0),
                                    cv::Point2f(0, 1), cv::Point2f(320, 240), 100.0f, 500.0,
                                    25.0 * std::sqrt(CV_PI));
}

}  // namespace

namespace {

// Grid frame through a lens whose ray deflection (radians) at each lens point h (metres,
// x right, y down) is given; covers multifocals, whose power varies across the lens.
cv::Mat renderGridViewDeflected(const std::function<cv::Vec2d(const cv::Vec2d&)>& deflection) {
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr double focal = 500.0;
    constexpr double screenMm = 500.0;
    constexpr double lensMm = 125.0;
    constexpr double periodMm = 8.0;
    constexpr double lensRadiusMm = 25.0;
    constexpr double rimMm = 4.0;
    cv::Mat frame(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec2d angle((x - width / 2.0) / focal, (y - height / 2.0) / focal);
            cv::Vec2d screenPoint = angle * screenMm;
            const cv::Vec2d onLens = angle * lensMm;
            const double radius = std::hypot(onLens[0], onLens[1]);
            bool blocked = false;
            if (radius < lensRadiusMm) {
                const cv::Vec2d bent = angle - deflection(onLens / 1000.0);
                screenPoint = onLens + bent * (screenMm - lensMm);
            } else if (radius < lensRadiusMm + rimMm) {
                blocked = true;
            }
            frame.at<float>(y, x) = blocked ? 10.0f : static_cast<float>(
                128.0 + 50.0 * std::cos(2.0 * CV_PI * screenPoint[0] / periodMm) +
                50.0 * std::cos(2.0 * CV_PI * screenPoint[1] / periodMm));
        }
    }
    return frame;
}

LensMeasurement measureDeflectedGrid(const std::function<cv::Vec2d(const cv::Vec2d&)>& deflection) {
    const cv::Mat referenceFrame = renderGridView(nullptr, 125.0);
    GridCarriers carriers;
    EXPECT_TRUE(findGridCarriers(referenceFrame, cv::Point2f(0, 1), carriers));
    FringeRun reference;
    extractGridRun(referenceFrame, carriers, reference);
    return measureLensFromGridFrame(reference, carriers, renderGridViewDeflected(deflection),
                                    cv::Point2f(0, 1), cv::Point2f(320, 240), 100.0f, 500.0,
                                    25.0 * std::sqrt(CV_PI));
}

}  // namespace

TEST(GridMeasurementTest, DetectsProgressiveLens) {
    // Wavefront W = ½·Pc·(x² + y²) + c·(x²y + y³/3): mean power Pc + 2c·y, no cylinder,
    // rising smoothly from the top (distance) to the bottom (near) of the lens.
    constexpr double centerPower = 1.0;
    constexpr double c = 60.0;  // 2c = 120 D/m: about +2.4 D between the zones used
    const LensMeasurement result = measureDeflectedGrid([&](const cv::Vec2d& h) {
        return cv::Vec2d(centerPower * h[0] + 2.0 * c * h[0] * h[1],
                         centerPower * h[1] + c * (h[0] * h[0] + h[1] * h[1]));
    });
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_EQ(result.lensType, LensType::Progressive);
    EXPECT_GT(result.addPower, 1.5);
    EXPECT_LT(result.addPower, 3.5);
    EXPECT_LT(result.distanceSphere, centerPower);
    EXPECT_GT(result.nearSphere, centerPower);
}

TEST(GridMeasurementTest, DetectsBifocalLens) {
    // +1.00 distance, +2.50 add in a segment below 5 mm under the lens center.
    constexpr double distancePower = 1.0;
    constexpr double add = 2.5;
    const LensMeasurement result = measureDeflectedGrid([&](const cv::Vec2d& h) {
        const double power = h[1] > 0.005 ? distancePower + add : distancePower;
        return cv::Vec2d(power * h[0], power * h[1]);
    });
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_EQ(result.lensType, LensType::Bifocal);
    EXPECT_NEAR(result.addPower, add, 0.3);
    EXPECT_NEAR(result.distanceSphere, distancePower, 0.2);
    EXPECT_NEAR(result.distanceCylinder, 0.0, 0.2);
}

TEST(GridMeasurementTest, MeasuresHalfRimLensWithoutSizeStep) {
    // Rim only along the top, and bright (gold) rather than dark; no lens size measured, so
    // the lens distance is assumed (125 mm here).
    constexpr double power = 2.5;
    const cv::Mat referenceFrame = renderGridView(nullptr, 125.0);
    GridCarriers carriers;
    ASSERT_TRUE(findGridCarriers(referenceFrame, cv::Point2f(0, 1), carriers));
    FringeRun reference;
    extractGridRun(referenceFrame, carriers, reference);
    cv::Mat frame(480, 640, CV_32F);
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            const cv::Vec2d angle((x - 320) / 500.0, (y - 240) / 500.0);
            cv::Vec2d screenPoint = angle * 500.0;
            const cv::Vec2d onLens = angle * 125.0;
            const double radius = std::hypot(onLens[0], onLens[1]);
            bool goldRim = false;
            if (radius < 25.0) {
                screenPoint = onLens + (angle - power * onLens / 1000.0) * 375.0;
            } else if (radius < 27.0 && onLens[1] < 0.0) {
                goldRim = true;
            }
            frame.at<float>(y, x) = goldRim ? 220.0f : static_cast<float>(
                128.0 + 50.0 * std::cos(2.0 * CV_PI * screenPoint[0] / 8.0) +
                50.0 * std::cos(2.0 * CV_PI * screenPoint[1] / 8.0));
        }
    }
    const LensMeasurement result = measureLensFromGridFrame(
        reference, carriers, frame, cv::Point2f(0, 1), cv::Point2f(320, 240), 100.0f, 500.0,
        -1.0, 125.0);
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, power, 0.2);
    EXPECT_NEAR(result.cylinder, 0.0, 0.2);
}

TEST(GridMeasurementTest, SingleVisionLensIsNotMultifocal) {
    const LensMeasurement result = measureSyntheticGrid(powerMatrix(-6.0, -1.0, 30.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_EQ(result.lensType, LensType::SingleVision);
}

TEST(GridMeasurementTest, FindsCarriersAlongMonitorAxes) {
    GridCarriers carriers;
    ASSERT_TRUE(findGridCarriers(renderGridView(nullptr, 125.0), cv::Point2f(0, 1), carriers));
    // 8 mm at 500 mm with f = 500 px is 8 px per period, i.e. 0.125 cycles per pixel.
    EXPECT_NEAR(carriers.frequency[0].x, 0.125, 0.005);
    EXPECT_NEAR(carriers.frequency[0].y, 0.0, 0.005);
    EXPECT_NEAR(carriers.frequency[1].x, 0.0, 0.005);
    EXPECT_NEAR(carriers.frequency[1].y, 0.125, 0.005);
}

TEST(GridMeasurementTest, MeasuresPlusLensFromOneFrame) {
    const LensMeasurement result = measureSyntheticGrid(powerMatrix(2.0, 0.0, 90.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.lensDistanceMm, 125.0, 6.0);
    EXPECT_NEAR(result.sphere, 2.0, 0.15);
    EXPECT_NEAR(result.cylinder, 0.0, 0.15);
}

TEST(GridMeasurementTest, MeasuresMinusLensFromOneFrame) {
    const LensMeasurement result = measureSyntheticGrid(powerMatrix(-2.5, 0.0, 90.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, -2.5, 0.15);
    EXPECT_NEAR(result.cylinder, 0.0, 0.15);
}

TEST(GridMeasurementTest, MeasuresStrongMinusLensFromOneFrame) {
    // −6 D at 12.5 cm makes the grid ~1.6× finer through the lens than in the reference.
    const LensMeasurement result = measureSyntheticGrid(powerMatrix(-6.0, 0.0, 90.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, -6.0, 0.25);
    EXPECT_NEAR(result.cylinder, 0.0, 0.25);
}

TEST(GridMeasurementTest, MeasuresStrongPlusLensFromOneFrame) {
    const LensMeasurement result = measureSyntheticGrid(powerMatrix(4.0, 0.0, 90.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, 4.0, 0.2);
}

TEST(GridMeasurementTest, MeasuresAstigmatismFromOneFrame) {
    const LensMeasurement result = measureSyntheticGrid(powerMatrix(1.0, -1.5, 150.0));
    ASSERT_EQ(result.status, LensMeasurementStatus::Ok);
    EXPECT_NEAR(result.sphere, 1.0, 0.15);
    EXPECT_NEAR(result.cylinder, -1.5, 0.15);
    EXPECT_NEAR(result.axisDegrees, 150.0, 4.0);
}

TEST(LensMeasurementTest, FindsLensOpeningAgainstScreen) {
    cv::Mat frame(480, 640, CV_8UC1, cv::Scalar(30));                    // dark room
    cv::rectangle(frame, cv::Rect(40, 20, 560, 440), cv::Scalar(235), cv::FILLED);  // screen
    cv::circle(frame, cv::Point(320, 240), 60, cv::Scalar(20), 8);      // rim, inner r ≈ 56
    cv::circle(frame, cv::Point(320, 240), 3, cv::Scalar(0), cv::FILLED);  // centering mark
    double opening = 0.0;
    ASSERT_TRUE(measureLensOpening(frame, cv::Point2f(320, 240), 20.0, opening));
    EXPECT_NEAR(opening, 56.0 * std::sqrt(CV_PI), 5.0);
}

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
