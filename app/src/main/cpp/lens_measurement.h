#ifndef LENS_MEASUREMENT_H
#define LENS_MEASUREMENT_H

#include <opencv2/core.hpp>

// Two-run deflectometry: a reference run (no lens) and a lens run, each with X and Y fringe
// phase maps, give the shift of every screen point seen through the lens. For a thin lens
// at distance d from the camera, with the screen at D:
//
//     screen shift = -(D - d) * M * h,      h = point on the lens the camera ray passes through
//
// where M is the 2×2 dioptric power matrix. In camera pixels this becomes
//
//     shift_px = -((D - d) * d / D) * M * u_px     (f cancels)
//
// so a linear fit of shift against pixel position gives M, and from it sphere, cylinder
// and axis. Plus and minus lenses are handled identically; no image needs to form.

enum class LensMeasurementStatus {
    Ok = 0,
    LensNotFound = 1,
    LensTouchesEdge = 2,
    TooFewPixels = 3,
    FringesTooFine = 4,
    BadInput = 5,
};

struct LensMeasurement {
    LensMeasurementStatus status = LensMeasurementStatus::BadInput;
    double sphere = 0.0;
    double cylinder = 0.0;       // minus-cylinder form, ≤ 0
    double axisDegrees = 0.0;    // TABO: 0–180°, as seen from the front of the glasses
    double lensDistanceMm = 0.0;
    double lensOpeningPixels = 0.0;  // sqrt(area) of the lens opening in the lens run
    int pixelsUsed = 0;
    double rmsResidualPixels = 0.0;
};

// Lens opening (sqrt of its area, in pixels) in a photo of glasses held flat against a bright
// screen: the bright hole inside the dark rim that contains screenCenter. minSizePixels
// rejects small holes such as the gap inside the centering mark.
bool measureLensOpening(const cv::Mat& grayscaleFrame, const cv::Point2f& screenCenter,
                        double minSizePixels, double& openingPixels);

// Camera pixels per fringe period along each monitor axis in a reference run; small values
// mean the fringes are too fine for the camera to resolve well.
bool fringePeriodPixels(const cv::Mat referenceWrapped[2], const cv::Mat referenceModulation[2],
                        const cv::Point2f& monitorDown, double periodPixels[2]);

// One run (reference or lens): per fringe axis ([0] = X, [1] = Y) the wrapped phase and
// modulation from calculateWrappedPhase, and the mean brightness of the four frames.
struct FringeRun {
    cv::Mat wrapped[2];
    cv::Mat modulation[2];
    cv::Mat brightness[2];
};

// screenOpeningPixels comes from measureLensOpening (glasses against the screen); the ratio to
// the opening seen in the lens run gives the lens distance. lensCenterGuess is where the lens
// is expected in the image (normally the image center). brightnessRatio, if given, receives
// lens-run / reference brightness, in which the rim shows up dark (for diagnosis).
LensMeasurement measureLensPower(
        const FringeRun& reference, const FringeRun& lens,
        const cv::Point2f& monitorDown, const cv::Point2f& lensCenterGuess,
        double screenDistanceMm, double screenOpeningPixels, cv::Mat* brightnessRatio = nullptr);

// Same, with the lens distance already known (used by tests).
LensMeasurement measureLensPowerAtDistance(
        const FringeRun& reference, const FringeRun& lens,
        const cv::Point2f& monitorDown, const cv::Point2f& lensCenterGuess,
        double screenDistanceMm, double lensDistanceMm);

#endif
