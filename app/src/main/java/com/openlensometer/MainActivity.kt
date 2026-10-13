package com.openlensometer

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.Image
import android.media.ToneGenerator
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.graphics.RectF
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import com.openlensometer.engine.NativeLensEngine
import java.nio.ByteBuffer
import java.util.Locale
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sign

class MainActivity : ComponentActivity() {
    private lateinit var cameraView: ArCoreCameraView
    private lateinit var overlayView: AlignmentOverlayView
    @Volatile private var lastOverlayTarget: RectF? = null
    private lateinit var actionView: TextView
    private lateinit var statusView: TextView
    private lateinit var detectionView: TextView
    private val nativeEngine by lazy { NativeLensEngine() }
    private var arSession: Session? = null
    private var installRequested = false
    private var activityResumed = false
    @Volatile private var alignmentReady = false
    /** Reference run (no lens) → lens size (glasses on the screen) → lens run. */
    private enum class Stage { REFERENCE, LENS_SIZE, LENS }
    @Volatile private var stage = Stage.REFERENCE
    /** Lens-size readings: {opening px, then the outline as x, y pairs in image pixels}. */
    private val lensOpeningSamples = ArrayDeque<FloatArray>()
    /**
     * Lens outline as seen on the screen during the lens-size step, as offsets (image pixels)
     * from its middle; drawn enlarged to show where the lens should be at 20 cm. Null: no rim.
     */
    @Volatile private var lensOutline: FloatArray? = null
    /** When the grid page appeared; measuring waits 5 s after it for the lens to be placed. */
    @Volatile private var placementStartMs = 0L
    private var lastGridFrameMs = 0L
    private val toneGenerator: ToneGenerator? by lazy {
        try { ToneGenerator(AudioManager.STREAM_MUSIC, 100) } catch (_: RuntimeException) { null }
    }
    @Volatile private var lastDiagnosticFrameMs = 0L
    private var lastUnreadableSaveMs = 0L
    private var lastLensSizeSaveMs = 0L
    private val gridWorker = java.util.concurrent.Executors.newSingleThreadExecutor()
    private val gridWorkerBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private var gridFrameCopy: ByteBuffer? = null
    @Volatile private var lastGridStatus = -1
    @Volatile private var sawLensSizeScreen = false
    @Volatile private var lastFocalPixels = 500f
    @Volatile private var gridReferenceFrames = 0
    private val gridResults = ArrayDeque<Pair<Long, FloatArray>>()
    private var savedGridFrames = 0
    private var savedGoodFrames = 0
    private var lastGoodFrameSaveMs = 0L
    /** A grid result is on screen; it stays until a new grid appears. */
    @Volatile private var gridShowingResult = false
    private var lastResultPageCheckMs = 0L
    /** The grid that produced the result is still up; don't measure it again. */
    @Volatile private var gridWaitingForNewGrid = false
    /** Page code last seen in the lens stage, to notice the white screen appearing anew. */
    @Volatile private var lensStagePreviousCode = -1
    private val gridStatusCounts = sortedMapOf<Int, Int>()
    private var lastGridLogMs = 0L
    private var lastRemoteSpot = 0
    private var previousThumbnail: ByteArray? = null
    private var lastRawCode = Int.MIN_VALUE
    private var framesSincePageChange = 0
    @Volatile private var steadyFrames = 0
    private var lastMotion = 0f
    private val motionSamples = ArrayList<Float>()
    private var lastMotionLogMs = 0L
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0
    @Volatile private var lastMonitorDown = floatArrayOf(0f, 1f)
    /** Camera-to-screen distance during the reference run (from the bullseye size). */
    private var screenDistanceMm = 0
    /** Lens opening (sqrt of area, image pixels) with the glasses held on the screen. */
    private var screenOpeningPixels = 0f
    /** Frames of the current page handed to native code so far (averaged at commit). */
    @Volatile private var collectedFrames = 0
    private val alignmentSamples = ArrayDeque<AlignmentSample>()
    private var lastAlignmentUiMs = 0L
    private var lastPublishedAlignmentReady: Boolean? = null
    private var consecutiveAlignmentMisses = 0
    private var captureAxis = AXIS_X
    private var capturePhase = 0
    private var captureComplete = false
    private var lastAlignmentTarget: AlignmentTarget? = null
    private var candidateScreen = -1 to false
    private var candidateScreenFrames = 0
    private var stableScreen = -1 to false
    private val pendingCapture = AtomicReference<ArmedCapture?>(null)
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startArCore() else statusView.text = "Camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A measurement takes minutes with nobody touching the phone; don't let it sleep.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        actionView = TextView(this).apply {
            text = "NOW: DON'T PRESS ANY KEY. Waiting for camera"
            textSize = 16f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setTextColor(android.graphics.Color.BLACK)
            setBackgroundColor(android.graphics.Color.rgb(255, 210, 0))
            setPadding(16, 10, 16, 10)
            // Fixed heights so changing messages don't make the camera picture jump.
            setLines(3)
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        statusView = TextView(this).apply {
            text = capturePrompt()
            textSize = 13f
            setTextColor(android.graphics.Color.BLACK)
            setBackgroundColor(android.graphics.Color.rgb(255, 238, 150))
            setPadding(16, 6, 16, 2)
            setLines(1)
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        detectionView = TextView(this).apply {
            text = "PHONE: Waiting for camera"
            textSize = 12f
            setTextColor(android.graphics.Color.BLACK)
            setBackgroundColor(android.graphics.Color.rgb(255, 238, 150))
            setPadding(16, 2, 16, 6)
            setLines(2)
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        val resetButton = Button(this).apply {
            text = "Reset capture"
            setOnClickListener { resetCapture() }
        }
        overlayView = AlignmentOverlayView(this)
        cameraView = ArCoreCameraView(this).apply {
            onFrame = ::processArCoreFrame
            onFrameError = { message ->
                runOnUiThread { detectionView.text = message }
            }
        }

        fun fullWidth() = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        // Camera picture first so the text can't push it down; text and buttons below it.
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(FrameLayout(this@MainActivity).apply {
                addView(cameraView, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                ))
                addView(overlayView, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                ))
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(actionView, fullWidth())
            addView(statusView, fullWidth())
            addView(detectionView, fullWidth())
            addView(resetButton, fullWidth())
        })

        nativeEngine.resetPhaseCapture()
        discardSavedMeasurement()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startArCore()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        activityResumed = true
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            if (arSession == null) startArCore() else resumeArCoreSession()
        }
    }

    override fun onPause() {
        activityResumed = false
        arSession?.pause()
        cameraView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        toneGenerator?.release()
        arSession?.close()
        arSession = null
        super.onDestroy()
    }

    private fun startArCore() {
        if (arSession != null) {
            resumeArCoreSession()
            return
        }

        try {
            val installStatus = ArCoreApk.getInstance().requestInstall(this, !installRequested)
            if (installStatus == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true
                statusView.text = "Install or update Google Play Services for AR, then reopen"
                return
            }

            val session = Session(this)
            val config = Config(session)
            val depthSupported = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
            config.depthMode = if (depthSupported) {
                Config.DepthMode.AUTOMATIC
            } else {
                Config.DepthMode.DISABLED
            }
            session.configure(config)
            arSession = session
            cameraView.setSession(session)
            detectionView.text = "${android.os.Build.MODEL}, Android ${android.os.Build.VERSION.SDK_INT}; " +
                if (depthSupported) "ARCore depth available" else "ARCore depth unavailable"
            statusView.text = if (depthSupported) {
                capturePrompt()
            } else {
                "Depth API unsupported on this device; calibrated capture is disabled"
            }
            if (activityResumed) resumeArCoreSession()
        } catch (error: UnavailableException) {
            statusView.text = "ARCore unavailable: ${error.localizedMessage}"
        } catch (error: SecurityException) {
            statusView.text = "Camera permission is required for ARCore"
        } catch (error: Exception) {
            statusView.text = "Could not start ARCore: ${error.localizedMessage}"
        }
    }

    private fun resumeArCoreSession() {
        val session = arSession ?: return
        try {
            cameraView.onResume()
            session.resume()
        } catch (error: Exception) {
            statusView.text = "Could not resume ARCore: ${error.localizedMessage}"
        }
    }

    private fun processArCoreFrame(frame: Frame) {
        // Keep the final quality summary on screen instead of overwriting it with guidance.
        if (captureComplete) return
        // A result is on screen: only glance at the page now and then for the next key (each
        // camera image read floods the system log).
        if (stage == Stage.LENS && gridShowingResult) {
            val now = System.currentTimeMillis()
            if (now - lastResultPageCheckMs < RESULT_PAGE_CHECK_MS) return
            lastResultPageCheckMs = now
        }
        var cameraImage: Image? = null
        var depthImage: Image? = null
        try {
            cameraImage = frame.acquireCameraImage()
            val yPlane = cameraImage.planes[0]
            val monitorDown = monitorDownInImage(frame)
            imageWidth = cameraImage.width
            imageHeight = cameraImage.height
            lastMonitorDown = monitorDown
            if (stage == Stage.LENS_SIZE) {
                processLensSizeFrame(cameraImage, monitorDown)
                return
            }
            // A result is on screen: stay idle until a key on the computer changes the page
            // (G: grid again, L: white screen for another lens). Only the page is checked.
            if (stage == Stage.LENS && gridShowingResult) {
                val target = lastAlignmentTarget ?: return
                val code = decodeRemoteSquares(cameraImage, target, monitorDown)
                if (code >= 0 && code != GRID_CODE) gridWaitingForNewGrid = false
                val newGrid = code == GRID_CODE && !gridWaitingForNewGrid
                if (!newGrid && code != LENS_SIZE_SCREEN_CODE) return
            }
            // In the lens run the camera looks through the glasses, so the bullseye can't be
            // trusted; the tripod hasn't moved since the reference run, so reuse its position.
            val lensRun = stage == Stage.LENS
            val detectedTarget = if (lensRun) null else nativeEngine.detectAlignmentTarget(
                yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                yPlane.rowStride, yPlane.pixelStride, monitorDown[0], monitorDown[1]
            )
            val bullseyeVisible = detectedTarget != null && detectedTarget.size >= 4
            val target = if (bullseyeVisible) {
                AlignmentTarget(
                    detectedTarget!![0], detectedTarget[1], detectedTarget[2]
                ).also { lastAlignmentTarget = it }
                detectedTarget
            } else {
                val previousTarget = lastAlignmentTarget
                if (previousTarget == null) {
                    handleAlignmentMiss("Point the camera at the bullseye in the middle of the monitor")
                    return
                }
                if (pendingCapture.get() == null) {
                    handleAlignmentMiss(
                        "Bullseye not seen. If you just pressed a key it returns within 8 s; " +
                            "otherwise aim at the middle of the monitor"
                    )
                    return
                }
                // With the bullseye hidden, read the copy of the squares drawn above center.
                val decodedPhase = decodeRemoteSquares(cameraImage, previousTarget, monitorDown)
                if (decodedPhase < 0) {
                    if (lensRun) {
                        // The camera must not move now, so no aiming advice; save a picture
                        // (at most every few seconds) to see what hides the squares.
                        val now = System.currentTimeMillis()
                        if (now - lastUnreadableSaveMs > UNREADABLE_SAVE_INTERVAL_MS) {
                            lastUnreadableSaveMs = now
                            saveGrayscaleFrame(cameraImage, "lens-run-unreadable.pgm")
                        }
                        if (setupMovedSinceReference(cameraImage, monitorDown, previousTarget)) {
                            handleAlignmentMiss(
                                "The picture has moved since the reference run (camera moved, " +
                                    "or the page switched between window and full screen). Tap " +
                                    "“Reset capture” and redo the reference"
                            )
                        } else {
                            handleAlignmentMiss(
                                "Can't tell which page the monitor shows. PRESS R to bring " +
                                    "back the fringe pages. Don't move the phone"
                            )
                        }
                    } else {
                        handleAlignmentMiss("Can't read the step squares around the screen center")
                    }
                    return
                }
                floatArrayOf(
                    previousTarget.centerX,
                    previousTarget.centerY,
                    previousTarget.size,
                    decodedPhase.toFloat()
                )
            }
            consecutiveAlignmentMisses = 0

            depthImage = try {
                frame.acquireDepthImage16Bits()
            } catch (_: NotYetAvailableException) {
                null
            }
            if (depthImage == null) {
                handleAlignmentMiss(
                    if (lensRun) "Waiting for the camera. Don't move the phone"
                    else "Move the phone slightly to initialize depth"
                )
                return
            }

            val guide = if (lensRun) {
                alignmentReady = true
                true to "Glasses in place. Keep the phone, glasses and stand still"
            } else {
                checkAlignment(frame, cameraImage, yPlane, monitorDown, target, bullseyeVisible)
                    ?: return
            }
            val (detectedPhaseCode, stableBullseyeVisible) =
                debounceScreenState(target[3].toInt(), bullseyeVisible)
            // Once lined up, get ready for the next step without waiting for a tap.
            if (pendingCapture.get() == null && guide.first && !captureComplete) {
                pendingCapture.compareAndSet(null, ArmedCapture(CaptureRequest(captureAxis, capturePhase)))
            }
            // The white screen (L) during the lens stage: measure the lens size again, e.g. for
            // different glasses; the reference run is kept.
            // Only when the white screen newly appears: it is often still up for a few seconds
            // after the size has just been taken.
            val whiteScreenAppeared = detectedPhaseCode == LENS_SIZE_SCREEN_CODE &&
                lensStagePreviousCode != LENS_SIZE_SCREEN_CODE
            // Ignore unreadable moments (e.g. right after the lens run resets its page tracking),
            // or a still-showing white screen would look newly appeared.
            if (lensRun && detectedPhaseCode >= 0) lensStagePreviousCode = detectedPhaseCode
            if (lensRun && whiteScreenAppeared) {
                stage = Stage.LENS_SIZE
                gridShowingResult = false
                gridWaitingForNewGrid = false
                lensOpeningSamples.clear()
                return
            }
            // Size just taken while the white screen is still up: show where the lens goes.
            if (lensRun && !gridShowingResult && detectedPhaseCode == LENS_SIZE_SCREEN_CODE) {
                showPlacement(frame, cameraImage, gridShowing = false)
                return
            }
            // A grid result stays on screen until the next grid (G) starts a new measurement.
            if (lensRun && gridShowingResult && detectedPhaseCode != GRID_CODE) {
                gridWaitingForNewGrid = false
                return
            }
            if (detectedPhaseCode == GRID_CODE) {
                if (lensRun) {
                    processGridLensFrame(frame, cameraImage, monitorDown)
                } else if (stage == Stage.REFERENCE) {
                    collectGridReference(
                        cameraImage, guide,
                        target[3].toInt() == GRID_CODE && !bullseyeVisible
                    )
                }
                return
            }
            val armedCapture = pendingCapture.get()
            if (armedCapture == null) {
                val (now, keyboard) = idleGuidance(
                    guide.first, if (stableBullseyeVisible) detectedPhaseCode else -1, guide.second
                )
                publishAlignment(guide.first, guide.second, keyboard, now)
                return
            }

            val expectedPhaseCode = phaseCode(armedCapture.request)
            debugLog("armed step=${expectedPhaseCode - 7} raw code=${target[3].toInt()} " +
                "visible=$bullseyeVisible stable code=$detectedPhaseCode " +
                "visible=$stableBullseyeVisible ready=${guide.first} '${guide.second}' " +
                "center=(${target[0]}, ${target[1]}) size=${target[2]}")
            val step = expectedPhaseCode - 7
            val rawCode = target[3].toInt()
            val glassesMoved = lensRun &&
                glassesMovedSinceLastFrame(yPlane, cameraImage.width, cameraImage.height, rawCode)
            val runProgress = captureAxis != AXIS_X || capturePhase != 0 || collectedFrames > 0

            fun commitCollectedFrames() {
                if (!pendingCapture.compareAndSet(armedCapture, null)) return
                val intrinsics = frame.camera.imageIntrinsics
                val focalLength = FloatArray(2)
                val principalPoint = FloatArray(2)
                calibratedFocalLength(frame, focalLength)
                intrinsics.getPrincipalPoint(principalPoint, 0)
                val depthPlane = depthImage.planes[0]
                // Native code averages the frames collected with addPhaseFrame.
                val capturedCount = nativeEngine.capturePhaseFrame(
                    yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                    yPlane.rowStride, yPlane.pixelStride,
                    depthPlane.buffer.slice(), depthImage.width, depthImage.height,
                    depthPlane.rowStride, depthPlane.pixelStride,
                    armedCapture.request.axis, armedCapture.request.phase,
                    focalLength[0], focalLength[1], principalPoint[0], principalPoint[1]
                )
                debugLog("committed step=$step from $collectedFrames frames")
                collectedFrames = 0
                // Keep the steadiness history: on a tripod the phone hasn't moved between photos.
                consecutiveAlignmentMisses = 0
                runOnUiThread { completePhaseCapture(armedCapture.request, capturedCount) }
            }

            if (lensRun) {
                // All 8 pages must see the lens in one place: if the hands moved or a page was
                // missed, start over; the looping pages bring step 1 round again.
                val wrappedRound = detectedPhaseCode == 8 && expectedPhaseCode > 9
                val skippedAhead = detectedPhaseCode in (expectedPhaseCode + 1)..15 &&
                    collectedFrames == 0
                if (runProgress && (glassesMoved || wrappedRound || skippedAhead)) {
                    restartLensAttempt(
                        if (glassesMoved) "The glasses moved" else "A page was missed"
                    )
                    return
                }
            }

            // The monitor moved on (auto-advance) or showed the bullseye again (run finished)
            // before the target count: use the frames we have.
            if (collectedFrames > 0 &&
                (detectedPhaseCode != expectedPhaseCode || stableBullseyeVisible)
            ) {
                commitCollectedFrames()
                return
            }
            val keyStep = if (lensRun) {
                lensRunInstruction(detectedPhaseCode, expectedPhaseCode)
            } else {
                keyboardInstruction(detectedPhaseCode, expectedPhaseCode, stableBullseyeVisible)
            }
            if (keyStep != null) {
                publishAlignment(guide.first, guide.second, keyStep.second, keyStep.first)
                return
            }
            if (!guide.first) {
                publishAlignment(
                    false, guide.second,
                    "Monitor is on the right screen; no key needed. Line up the phone"
                )
                return
            }

            // The debounce already saw this page with the bullseye gone on consecutive frames;
            // also check this frame itself so a redraw in progress isn't collected.
            val steadyEnough = !lensRun || steadyFrames >= STEADY_FRAMES_NEEDED
            if (!steadyEnough) {
                publishAlignment(
                    true, "Waiting for the glasses to be still",
                    "No key. Waiting for steady glasses",
                    "HOLD THE GLASSES STEADY with both hands…"
                )
                return
            }
            if (target[3].toInt() == expectedPhaseCode && !bullseyeVisible) {
                val count = nativeEngine.addPhaseFrame(
                    yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                    yPlane.rowStride, yPlane.pixelStride,
                    armedCapture.request.axis, armedCapture.request.phase
                )
                if (count > 0) collectedFrames = count
            }
            // Static reference: many frames per page. Hand-held lens: a few, on fast pages.
            val framesWanted = if (lensRun) LENS_FRAMES_PER_PAGE else FRAMES_PER_PAGE
            if (collectedFrames < framesWanted) {
                publishAlignment(
                    true, "Hold still · photographing step $step/8 ($collectedFrames/$framesWanted)",
                    "No key. Collecting frames of step $step/8",
                    "HOLD STILL. DON'T PRESS ANY KEY. Photographing step $step/8"
                )
                return
            }
            commitCollectedFrames()
        } catch (_: NotYetAvailableException) {
            handleAlignmentMiss("ARCore camera frame warming up")
        } catch (error: Exception) {
            handleAlignmentMiss("Alignment check failed: ${error.localizedMessage}", true)
        } finally {
            depthImage?.close()
            cameraImage?.close()
        }
    }

    /**
     * True if the bullseye is in plain view but not where it was in the reference run, which
     * means the reference no longer matches the setup.
     */
    private fun setupMovedSinceReference(
        cameraImage: Image,
        monitorDown: FloatArray,
        reference: AlignmentTarget
    ): Boolean {
        val yPlane = cameraImage.planes[0]
        val seen = nativeEngine.detectAlignmentTarget(
            yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
            yPlane.rowStride, yPlane.pixelStride, monitorDown[0], monitorDown[1]
        ) ?: return false
        // Seen through the glasses the bullseye is magnified or shrunk; only trust a view
        // of about the same size.
        if (abs(seen[2] - reference.size) > 0.15f * reference.size) return false
        val shift = hypot(
            (seen[0] - reference.centerX) * cameraImage.width,
            (seen[1] - reference.centerY) * cameraImage.height
        )
        return shift > SETUP_MOVED_FRACTION * reference.size * cameraImage.height
    }

    /**
     * Reads one of the copies of the step squares drawn around the center (above, below, left,
     * right), trying the one that worked last time first: the hands holding the glasses can
     * hide some of them. Native code searches a little around each.
     */
    private fun decodeRemoteSquares(
        cameraImage: Image,
        target: AlignmentTarget,
        monitorDown: FloatArray
    ): Int {
        val yPlane = cameraImage.planes[0]
        val pixelsPerMm = target.size * cameraImage.height / BULLSEYE_DIAMETER_MM
        val rightX = monitorDown[1]
        val rightY = -monitorDown[0]
        val order = listOf(lastRemoteSpot) + REMOTE_TICK_SPOTS_MM.indices.filter { it != lastRemoteSpot }
        for (spot in order) {
            val (rightMm, downMm) = REMOTE_TICK_SPOTS_MM[spot]
            val code = nativeEngine.decodeAlignmentPhaseCode(
                yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                yPlane.rowStride, yPlane.pixelStride,
                target.centerX + (rightX * rightMm + monitorDown[0] * downMm) * pixelsPerMm / cameraImage.width,
                target.centerY + (rightY * rightMm + monitorDown[1] * downMm) * pixelsPerMm / cameraImage.height,
                target.size, monitorDown[0], monitorDown[1]
            )
            if (code >= 0) {
                lastRemoteSpot = spot
                return code
            }
        }
        return -1
    }

    /**
     * Distance, centering, tilt and steadiness from the bullseye and corner markers.
     * Returns (ready, phone message), or null after reporting a miss.
     */
    private fun checkAlignment(
        frame: Frame,
        cameraImage: Image,
        yPlane: Image.Plane,
        monitorDown: FloatArray,
        target: FloatArray,
        bullseyeVisible: Boolean
    ): Pair<Boolean, String>? {
        // ARCore depth is unreliable on a screen of repeating fringes (and on glossy
        // reflections), so distance comes from the bullseye's known physical size.
        val distanceMm = distanceFromBullseye(frame, target[2] * cameraImage.height)
        if (distanceMm <= 0) {
            handleAlignmentMiss("Bullseye size unreadable; hold the bullseye in view")
            return null
        }

        // Tilt comes from the four corner markers' perspective, not ARCore depth, which
        // gave readings of 50°+ on a striped screen. Markers are hidden with the bullseye.
        val tilt = if (bullseyeVisible) {
            val focal = FloatArray(2)
            val principal = FloatArray(2)
            calibratedFocalLength(frame, focal)
            frame.camera.imageIntrinsics.getPrincipalPoint(principal, 0)
            nativeEngine.estimateMonitorTilt(
                yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                yPlane.rowStride, yPlane.pixelStride, monitorDown[0], monitorDown[1],
                focal[0], focal[1], principal[0], principal[1], CORNER_MARKER_OFFSET_MM
            )
        } else {
            null
        }
        val pose = measurePose(
            frame, target[0] * cameraImage.width,
            target[1] * cameraImage.height, distanceMm, tilt
        )
        val sample = AlignmentSample(
            pose.viewX, pose.viewY, target[2], distanceMm,
            pose.offsetRightMm, pose.offsetDownMm, pose.tiltRightDeg, pose.tiltDownDeg
        )
        alignmentSamples.addLast(sample)
        while (alignmentSamples.size > STABLE_FRAME_COUNT) alignmentSamples.removeFirst()

        val guide = evaluateAlignment(sample)
        alignmentReady = guide.first
        val (tiltRight, tiltDown) = averagedTilts()
        fun tiltReading(label: String, value: Double?) = AlignmentOverlayView.Reading(
            label, value?.let { String.format(Locale.US, "%+.1f°", it) } ?: "–",
            value != null && abs(value) <= MAX_TILT_DEG
        )
        updateOverlay(
            frame, tilt, guide.first,
            listOf(
                tiltReading("Tilt L/R", tiltRight),
                tiltReading("Tilt F/B", tiltDown),
                AlignmentOverlayView.Reading(
                    "Distance", "${distanceMm / 10} cm",
                    distanceMm in MIN_WORKING_DISTANCE_MM..MAX_WORKING_DISTANCE_MM
                )
            )
        )
        return guide
    }

    /**
     * Orders corrections so that fixing one does not undo another: square the phone to the
     * monitor first (tilt shifts the target), then slide it to center, then set distance.
     * All directions are as the user sees the phone screen while the camera faces the monitor.
     */
    /** Tilt to the left/right and forward/back, averaged over recent frames (null = unknown). */
    private fun averagedTilts(): Pair<Double?, Double?> {
        fun average(values: List<Float>) = values.takeIf { it.size >= MIN_TILT_SAMPLES }?.average()
        return average(alignmentSamples.mapNotNull { it.tiltRightDeg }) to
            average(alignmentSamples.mapNotNull { it.tiltDownDeg })
    }

    /**
     * Whether the phone is lined up. The how-to is shown graphically (box colour, tilt and
     * distance readings, pink box), so the messages only say what state it is in.
     */
    private fun evaluateAlignment(sample: AlignmentSample): Pair<Boolean, String> {
        val (tiltRight, tiltDown) = averagedTilts()
        val tilted = listOfNotNull(tiltRight, tiltDown).any { abs(it) > MAX_TILT_DEG }
        val offCenter = sample.viewX !in CENTER_MIN..CENTER_MAX ||
            sample.viewY !in CENTER_MIN..CENTER_MAX
        val wrongDistance = sample.depthMm !in MIN_WORKING_DISTANCE_MM..MAX_WORKING_DISTANCE_MM
        if (tilted || offCenter || wrongDistance) {
            return false to "Line up until the box turns green"
        }
        if (alignmentSamples.size < STABLE_FRAME_COUNT) return false to "Hold still"

        val samples = alignmentSamples.toList()
        val centerXRange = samples.maxOf { it.viewX } - samples.minOf { it.viewX }
        val centerYRange = samples.maxOf { it.viewY } - samples.minOf { it.viewY }
        val depthRange = samples.maxOf { it.depthMm } - samples.minOf { it.depthMm }
        val sizeRange = samples.maxOf { it.targetSize } - samples.minOf { it.targetSize }
        if (centerXRange > MAX_CENTER_JITTER || centerYRange > MAX_CENTER_JITTER ||
            depthRange > MAX_DEPTH_JITTER_MM || sizeRange > MAX_SIZE_JITTER
        ) return false to "Hold still"
        return true to "Lined up. Don't move the phone"
    }

    /**
     * Returns the (phase code, bullseye visible) pair only once it has been read the same way
     * for several frames in a row, so one misread doesn't flash a wrong key instruction or
     * restart the photo countdown.
     */
    private fun debounceScreenState(code: Int, bullseyeVisible: Boolean): Pair<Int, Boolean> {
        val reading = code to bullseyeVisible
        if (reading == candidateScreen) {
            candidateScreenFrames++
        } else {
            candidateScreen = reading
            candidateScreenFrames = 1
        }
        if (candidateScreenFrames >= SCREEN_DEBOUNCE_FRAMES) stableScreen = reading
        return stableScreen
    }

    /**
     * Which key the user must press on the computer showing the fringe page, as
     * (NOW line, KEYBOARD line), or null when no key is needed.
     */
    private fun keyboardInstruction(
        detectedCode: Int,
        expectedCode: Int,
        bullseyeVisible: Boolean
    ): Pair<String, String>? {
        val step = expectedCode - 7
        val wanted = "step $step/8 (${phaseName(expectedCode - 8)})"
        val kept = if (step > 1) " (steps 1–${step - 1} are kept)" else ""
        return when {
            detectedCode == LENS_SIZE_SCREEN_CODE -> "PRESS R now (leaves the white screen)" to
                "The white lens-size screen is showing"
            detectedCode == LOOP_COUNTDOWN_CODE -> "WAIT. The pages start by themselves" to
                "Countdown running"
            detectedCode == 0 && step == 1 -> "PRESS RETURN now to start" to
                "Starts the 8 pages; auto-advance runs them all"
            detectedCode == 0 -> "PRESS RETURN now to redo step $step/8$kept" to
                "The pages restart; the phone waits until $wanted comes round"
            detectedCode !in 8..15 -> if (bullseyeVisible) {
                "DON'T PRESS ANY KEY. Can't read the squares under the bullseye: reload the " +
                    "page and make sure nothing covers them" to
                    "Can't read the step squares under the bullseye"
            } else {
                "DON'T PRESS ANY KEY. Can't tell which page the monitor shows yet" to
                    "Can't read the upper step squares"
            }
            detectedCode < expectedCode -> "WAIT. Don't press any key" to
                "Monitor shows step ${detectedCode - 7}/8; the phone needs $wanted"
            // The monitor went past this step, or the run ended, before it was captured.
            detectedCode > expectedCode || bullseyeVisible -> if (step == 1) {
                "PRESS R, then RETURN to start" to
                    "Monitor shows step ${detectedCode - 7}/8; the run starts at step 1"
            } else {
                "Redoing step $step/8: PRESS R, then RETURN$kept" to
                    "Step $step/8 was missed; the pages restart and the phone picks it up"
            }
            else -> null
        }
    }

    /** (NOW line, KEYBOARD line) for the hands-free lens run, or null to collect frames. */
    private fun lensRunInstruction(detectedCode: Int, expectedCode: Int): Pair<String, String>? {
        val step = expectedCode - 7
        return when {
            detectedCode == LENS_SIZE_SCREEN_CODE -> "Get ready: the lens run starts by itself " +
                "when the white screen ends" to "No key needed"
            detectedCode == LOOP_COUNTDOWN_CODE -> "Get ready: hold the glasses steady with " +
                "both hands. The pages start by themselves" to "No key. Countdown running"
            detectedCode == 0 -> "Time ran out. PRESS G to measure again (L redoes the lens " +
                "size)" to "G shows the grid for another 20 s"
            detectedCode !in 8..15 -> "Can't tell which page the monitor shows. PRESS R to " +
                "start the lens loop. Don't move the phone" to "Can't read the step squares"
            detectedCode == expectedCode -> null
            step == 1 -> "HOLD STEADY · starts when step 1 comes round" to
                "No key. Monitor shows step ${detectedCode - 7}/8"
            else -> "HOLD STEADY · waiting for step $step/8" to
                "No key. Monitor shows step ${detectedCode - 7}/8"
        }
    }

    /**
     * Compares a coarse sample of this frame with the previous one. Changes right after a page
     * flip are the monitor; changes while the page stays the same mean the glasses moved.
     */
    private fun glassesMovedSinceLastFrame(
        plane: Image.Plane,
        width: Int,
        height: Int,
        rawCode: Int
    ): Boolean {
        val buffer = plane.buffer
        // Only the middle of the picture, where the lens is: hands and arms moving at the
        // edges don't disturb the measurement.
        val left = (width * (1f - MOTION_REGION) / 2f).toInt()
        val top = (height * (1f - MOTION_REGION) / 2f).toInt()
        val columns = (width * MOTION_REGION).toInt() / MOTION_SAMPLE_STEP
        val rows = (height * MOTION_REGION).toInt() / MOTION_SAMPLE_STEP
        val thumbnail = ByteArray(columns * rows)
        for (row in 0 until rows) {
            val rowStart = (top + row * MOTION_SAMPLE_STEP) * plane.rowStride
            for (column in 0 until columns) {
                thumbnail[row * columns + column] = buffer.get(
                    rowStart + (left + column * MOTION_SAMPLE_STEP) * plane.pixelStride
                )
            }
        }
        val previous = previousThumbnail
        previousThumbnail = thumbnail
        if (rawCode != lastRawCode) {
            lastRawCode = rawCode
            framesSincePageChange = 0
        } else {
            framesSincePageChange++
        }
        if (previous == null || previous.size != thumbnail.size) {
            steadyFrames = 0
            return false
        }
        if (framesSincePageChange < PAGE_SETTLE_FRAMES) return false
        var total = 0L
        for (index in thumbnail.indices) {
            total += abs((thumbnail[index].toInt() and 0xff) - (previous[index].toInt() and 0xff))
        }
        val motion = total.toFloat() / thumbnail.size
        lastMotion = motion
        motionSamples += motion
        val now = System.currentTimeMillis()
        if (now - lastMotionLogMs >= 1000L && motionSamples.isNotEmpty()) {
            val sorted = motionSamples.sorted()
            debugLog("lens run motion: median ${"%.1f".format(sorted[sorted.size / 2])}, " +
                "max ${"%.1f".format(sorted.last())} over ${sorted.size} frames " +
                "(limit $MOTION_LIMIT), steady frames $steadyFrames")
            motionSamples.clear()
            lastMotionLogMs = now
        }
        return if (motion > MOTION_LIMIT) {
            steadyFrames = 0
            true
        } else {
            steadyFrames++
            false
        }
    }

    /** Throws away a disturbed lens attempt; the looping pages bring step 1 round again. */
    private fun restartLensAttempt(reason: String) {
        nativeEngine.resetPhaseCapture()
        captureAxis = AXIS_X
        capturePhase = 0
        collectedFrames = 0
        steadyFrames = 0
        pendingCapture.set(ArmedCapture(CaptureRequest(AXIS_X, 0)))
        debugLog("lens attempt restarted: $reason (motion ${"%.1f".format(lastMotion)}, " +
            "limit $MOTION_LIMIT)")
        runOnUiThread {
            actionView.text = "NOW: $reason. HOLD STEADY · starting over when step 1 comes round"
        }
    }

    /** (NOW line, KEYBOARD line) while no capture is armed. */
    private fun idleGuidance(ready: Boolean, detectedCode: Int, phoneMessage: String): Pair<String, String> {
        if (captureComplete) return "DONE. Nothing more to do" to "All 8 steps captured"
        val expectedCode = phaseCode(CaptureRequest(captureAxis, capturePhase))
        if (detectedCode in (expectedCode + 1)..15) {
            return "PRESS R now (restarts the pages)" to
                "Monitor is ahead: shows step ${detectedCode - 7}/8, need ${expectedCode - 7}/8"
        }
        return if (ready) {
            "DON'T PRESS ANY KEY. Hold steady; step ${expectedCode - 7}/8 starts by itself" to
                "No key yet"
        } else {
            dontPress(phoneMessage) to "No key yet. Line up the phone first"
        }
    }

    private fun dontPress(phoneMessage: String) = "DON'T PRESS ANY KEY. $phoneMessage"

    private fun times(count: Int): String = if (count > 1) " $count times" else ""

    private fun handleAlignmentMiss(message: String, resetImmediately: Boolean = false) {
        consecutiveAlignmentMisses++
        if (resetImmediately || consecutiveAlignmentMisses > MAX_ALIGNMENT_MISSES) {
            alignmentSamples.clear()
            alignmentReady = false
        }
        val displayedMessage = if (alignmentReady) "Bullseye briefly lost. Hold steady" else message
        lastOverlayTarget?.let {
            overlayView.update(AlignmentOverlayView.State(it, null, false, emptyList()))
        }
        // Also overwrite the KEYBOARD line so an old "press RETURN" can't linger on screen.
        publishAlignment(
            alignmentReady, displayedMessage,
            "No key right now. The phone can't see the monitor's bullseye or step squares"
        )
    }

    private fun publishAlignment(
        ready: Boolean,
        message: String,
        keyboardMessage: String? = null,
        nowMessage: String = dontPress(message)
    ) {
        val now = System.currentTimeMillis()
        if (now - lastAlignmentUiMs < ALIGNMENT_UI_INTERVAL_MS &&
            ready == lastPublishedAlignmentReady
        ) return
        lastAlignmentUiMs = now
        lastPublishedAlignmentReady = ready
        runOnUiThread {
            actionView.text = "NOW: $nowMessage"
            detectionView.text = "PHONE: $message"
            if (keyboardMessage != null) statusView.text = "KEYBOARD: $keyboardMessage"
        }
    }

    /** The monitor's downward direction in camera-image pixels, as the phone is currently held. */
    private fun monitorDownInImage(frame: Frame): FloatArray {
        val top = transformPoint(frame, Coordinates2d.VIEW_NORMALIZED, Coordinates2d.IMAGE_PIXELS, 0.5f, 0.4f)
        val bottom = transformPoint(frame, Coordinates2d.VIEW_NORMALIZED, Coordinates2d.IMAGE_PIXELS, 0.5f, 0.6f)
        val dx = bottom[0] - top[0]
        val dy = bottom[1] - top[1]
        val length = hypot(dx, dy)
        if (!length.isFinite() || length < 1e-3f) return floatArrayOf(0f, 1f)
        return floatArrayOf(dx / length, dy / length)
    }

    /**
     * Converts the target position into on-screen coordinates and physical offsets.
     * tilt is {tiltRightDeg, tiltDownDeg} from the corner markers, or null when not all seen.
     */
    private fun measurePose(
        frame: Frame,
        targetImageX: Float,
        targetImageY: Float,
        depthMm: Int,
        tilt: FloatArray?
    ): PoseEstimate {
        val intrinsics = frame.camera.imageIntrinsics
        val focal = FloatArray(2)
        val principal = FloatArray(2)
        calibratedFocalLength(frame, focal)
        intrinsics.getPrincipalPoint(principal, 0)

        fun viewToImage(viewX: Float, viewY: Float) = transformPoint(
            frame, Coordinates2d.VIEW_NORMALIZED, Coordinates2d.IMAGE_PIXELS, viewX, viewY
        )
        fun cameraPoint(image: FloatArray, z: Float) = floatArrayOf(
            (image[0] - principal[0]) * z / focal[0],
            (image[1] - principal[1]) * z / focal[1],
            z
        )
        fun lateralDistance(a: FloatArray, b: FloatArray) = hypot(b[0] - a[0], b[1] - a[1])

        val view = transformPoint(
            frame, Coordinates2d.IMAGE_PIXELS, Coordinates2d.VIEW_NORMALIZED,
            targetImageX, targetImageY
        )
        val z = depthMm.toFloat()
        val targetPoint = cameraPoint(floatArrayOf(targetImageX, targetImageY), z)
        val offsetRightMm = sign(view[0] - 0.5f) *
            lateralDistance(cameraPoint(viewToImage(0.5f, view[1]), z), targetPoint)
        val offsetDownMm = sign(view[1] - 0.5f) *
            lateralDistance(cameraPoint(viewToImage(view[0], 0.5f), z), targetPoint)

        return PoseEstimate(
            view[0], view[1], offsetRightMm, offsetDownMm,
            tilt?.getOrNull(0), tilt?.getOrNull(1)
        )
    }

    /**
     * Places the white target square where the monitor's box (corner markers 2 ×
     * CORNER_MARKER_OFFSET_MM apart) appears from TARGET_WORKING_DISTANCE_MM, centered on the
     * preview, and the pink box where the camera actually sees the corner markers.
     */
    private fun updateOverlay(
        frame: Frame,
        tilt: FloatArray?,
        aligned: Boolean,
        readings: List<AlignmentOverlayView.Reading>
    ) {
        val focal = FloatArray(2)
        val principal = FloatArray(2)
        calibratedFocalLength(frame, focal)
        frame.camera.imageIntrinsics.getPrincipalPoint(principal, 0)
        val sideImagePixels = (focal[0] + focal[1]) / 2f *
            (2f * CORNER_MARKER_OFFSET_MM) / TARGET_WORKING_DISTANCE_MM
        val start = transformPoint(
            frame, Coordinates2d.IMAGE_PIXELS, Coordinates2d.VIEW, principal[0], principal[1]
        )
        val end = transformPoint(
            frame, Coordinates2d.IMAGE_PIXELS, Coordinates2d.VIEW,
            principal[0] + sideImagePixels, principal[1]
        )
        val halfSide = hypot(end[0] - start[0], end[1] - start[1]) / 2f
        val center = transformPoint(
            frame, Coordinates2d.VIEW_NORMALIZED, Coordinates2d.VIEW, 0.5f, 0.5f
        )
        if (!halfSide.isFinite() || !center[0].isFinite() || !center[1].isFinite()) return
        val target = RectF(
            center[0] - halfSide, center[1] - halfSide,
            center[0] + halfSide, center[1] + halfSide
        )
        lastOverlayTarget = target

        val corners = if (tilt != null && tilt.size >= 11) {
            FloatArray(8) { Float.NaN }.also { viewCorners ->
                for (index in 0 until 4) {
                    val x = tilt[3 + index * 2]
                    val y = tilt[4 + index * 2]
                    if (!x.isFinite() || !y.isFinite()) continue
                    val view = transformPoint(
                        frame, Coordinates2d.IMAGE_PIXELS, Coordinates2d.VIEW, x, y
                    )
                    viewCorners[index * 2] = view[0]
                    viewCorners[index * 2 + 1] = view[1]
                }
            }
        } else {
            null
        }
        overlayView.update(AlignmentOverlayView.State(target, corners, aligned, readings))
    }

    /** Pinhole distance to the monitor: focal length × real bullseye size ÷ imaged size. */
    private fun distanceFromBullseye(frame: Frame, bullseyeImagePixels: Float): Int {
        if (!bullseyeImagePixels.isFinite() || bullseyeImagePixels <= 1f) return -1
        val focal = FloatArray(2)
        calibratedFocalLength(frame, focal)
        val focalPixels = (focal[0] + focal[1]) / 2f
        if (!focalPixels.isFinite() || focalPixels <= 0f) return -1
        val distance = focalPixels * BULLSEYE_DIAMETER_MM / bullseyeImagePixels
        return distance.roundToInt()
    }

    /**
     * ARCore's focal length, corrected by DISTANCE_SCALE. On the Pixel 7a ARCore's value is
     * ~14% low, which made every distance and so every power that much off.
     */
    private fun calibratedFocalLength(frame: Frame, out: FloatArray) {
        frame.camera.imageIntrinsics.getFocalLength(out, 0)
        out[0] *= DISTANCE_SCALE
        out[1] *= DISTANCE_SCALE
    }

    private fun transformPoint(
        frame: Frame,
        from: Coordinates2d,
        to: Coordinates2d,
        x: Float,
        y: Float
    ): FloatArray {
        val output = FloatArray(2)
        frame.transformCoordinates2d(from, floatArrayOf(x, y), to, output)
        return output
    }

    private fun completePhaseCapture(capture: CaptureRequest, capturedCount: Int) {
        if (capturedCount != capture.phase + 1) {
            alignmentReady = false
            actionView.text = "NOW: Tap “Reset capture”, then start again from step 1"
            statusView.text = when (capturedCount) {
                -1 -> "Invalid phase or camera frame dimensions"
                -2 -> "Camera Y-plane buffer is not direct or is too small"
                -3 -> "Capture order or frame dimensions do not match"
                -4 -> "Native phase capture failed"
                else -> "Unexpected native capture state"
            }
            return
        }

        if (capture.phase < PHASE_LABELS.lastIndex) {
            capturePhase++
        } else if (capture.axis == AXIS_X) {
            captureAxis = AXIS_Y
            capturePhase = 0
        } else {
            captureComplete = true
            when (stage) {
                Stage.REFERENCE -> finishReferenceRun()
                Stage.LENS -> finishLensRun()
                Stage.LENS_SIZE -> Unit
            }
            return
        }

        // Arm the next step straight away so the user is only asked to press keys.
        val next = CaptureRequest(captureAxis, capturePhase)
        pendingCapture.set(ArmedCapture(next))
        val nextStep = phaseCode(next) - 7
        actionView.text = "NOW: Step ${nextStep - 1}/8 done. WAIT for the next page"
        statusView.text = "KEYBOARD: Next is step $nextStep/8"
        detectionView.text = "PHONE: Keep the phone where it is"
    }

    private fun resetCapture() {
        pendingCapture.set(null)
        nativeEngine.resetPhaseCapture()
        alignmentSamples.clear()
        lastAlignmentTarget = null
        candidateScreen = -1 to false
        candidateScreenFrames = 0
        stableScreen = -1 to false
        consecutiveAlignmentMisses = 0
        alignmentReady = false
        captureAxis = AXIS_X
        capturePhase = 0
        captureComplete = false
        collectedFrames = 0
        stage = Stage.REFERENCE
        gridReferenceFrames = 0
        synchronized(gridResults) { gridResults.clear() }
        gridShowingResult = false
        gridWaitingForNewGrid = false
        sawLensSizeScreen = false
        screenOpeningPixels = 0f
        lensOutline = null
        nativeEngine.clearReferenceRun()
        actionView.text = "NOW: DON'T PRESS ANY KEY. Line up on the bullseye"
        statusView.text = capturePrompt()
        detectionView.text = "PHONE: Capture reset. Line up on the bullseye"
    }

    /** Every launch starts at the reference run; delete what earlier versions saved. */
    private fun discardSavedMeasurement() {
        deleteSharedPreferences("measurement")
        java.io.File(filesDir, "reference-run.bin").delete()
    }

    /** Reference run, grid page: collect no-lens grid frames while lined up and still. */
    private fun collectGridReference(
        cameraImage: Image,
        guide: Pair<Boolean, String>,
        gridShowing: Boolean
    ) {
        if (guide.first && gridShowing && gridReferenceFrames < GRID_REFERENCE_FRAMES) {
            val plane = cameraImage.planes[0]
            gridReferenceFrames = nativeEngine.addGridReferenceFrame(
                plane.buffer.slice(), cameraImage.width, cameraImage.height,
                plane.rowStride, plane.pixelStride
            )
            // The grid is the whole reference run: done once enough frames are averaged.
            if (gridReferenceFrames >= GRID_REFERENCE_FRAMES) {
                runOnUiThread { finishReferenceRun() }
                return
            }
        }
        publishAlignment(
            guide.first, guide.second,
            "No key. Grid reference ($gridReferenceFrames frames)",
            if (guide.first) "HOLD STILL. DON'T PRESS ANY KEY. Grid reference" else dontPress(guide.second)
        )
    }

    /**
     * Single-frame method: each grid frame gives a full measurement, so shake between frames
     * doesn't matter. After 5 s to put the lens inside the outline, the next good frames are
     * combined into the answer.
     */
    private fun processGridLensFrame(frame: Frame, cameraImage: Image, monitorDown: FloatArray) {
        // Same grid as the result just shown: wait for the next one (G) before measuring again.
        if (gridWaitingForNewGrid) return
        gridShowingResult = false
        val now = System.currentTimeMillis()
        // A new grid page (after the white screen, G, or a timed-out grid): the countdown for
        // placing the lens starts now, with fresh results. (Users wait for the grid before
        // moving the lens, so starting it at the size beep left too little time.)
        if (now - lastGridFrameMs > NEW_GRID_GAP_MS) {
            placementStartMs = now
            synchronized(gridResults) { gridResults.clear() }
        }
        lastGridFrameMs = now
        if (showPlacement(frame, cameraImage, gridShowing = true)) return
        // The guide is drawn on every frame so it keeps up with the live picture.
        showLensGuide(frame, cameraImage, true)
        // Measuring takes a few tenths of a second, far longer than a frame. Done here it held
        // up the preview (seconds behind); instead hand a copy of the latest frame to a
        // background worker and skip frames while it is busy.
        if (!gridWorkerBusy.compareAndSet(false, true)) return
        val width = cameraImage.width
        val height = cameraImage.height
        val copy = gridFrameCopy?.takeIf { it.capacity() == width * height }
            ?: ByteBuffer.allocateDirect(width * height).also { gridFrameCopy = it }
        val plane = cameraImage.planes[0]
        val source = plane.buffer.duplicate()
        copy.clear()
        for (row in 0 until height) {
            source.limit(min(source.capacity(), row * plane.rowStride + width * plane.pixelStride))
            source.position(row * plane.rowStride)
            if (plane.pixelStride == 1) {
                copy.put(source)
            } else {
                for (column in 0 until width) copy.put(source.get(row * plane.rowStride + column * plane.pixelStride))
            }
        }
        copy.flip()
        val down = monitorDown.copyOf()
        gridWorker.execute {
            try {
                measureGridCopy(copy, width, height, down)
            } finally {
                gridWorkerBusy.set(false)
            }
        }
    }

    /** Runs on the grid worker thread with a private copy of one frame's Y plane. */
    private fun measureGridCopy(copy: ByteBuffer, width: Int, height: Int, monitorDown: FloatArray) {
        if (gridWaitingForNewGrid) return
        val result = nativeEngine.measureGridFrame(
            copy, width, height, width, 1, monitorDown[0], monitorDown[1],
            width / 2f, height / 2f, expectedLensRadiusPixels(),
            screenDistanceMm.toFloat(), screenOpeningPixels, ASSUMED_LENS_DISTANCE_MM
        ) ?: return
        val now = System.currentTimeMillis()
        val status = result[0].toInt()
        lastGridStatus = status
        gridStatusCounts[status] = (gridStatusCounts[status] ?: 0) + 1
        // A real lens frame now and then, for offline analysis.
        if (status != 0 && now - lastDiagnosticFrameMs > DIAGNOSTIC_FRAME_INTERVAL_MS) {
            lastDiagnosticFrameMs = now
            saveGrayscaleBytes(copy, width, height, "grid-lens-$status-${savedGridFrames++ % 4}.pgm")
            debugLog("saved grid lens frame; screen opening $screenOpeningPixels px, " +
                "screen $screenDistanceMm mm, down ${monitorDown[0]}, ${monitorDown[1]}")
        }
        // A few good frames too (every 2 s, rotating), to tune multifocal analysis offline.
        if (status == 0 && now - lastGoodFrameSaveMs > 2000L) {
            lastGoodFrameSaveMs = now
            saveGrayscaleBytes(copy, width, height, "grid-good-${savedGoodFrames++ % 4}.pgm")
            debugLog(String.format(
                Locale.US, "saved good grid frame: %+.2f %+.2f x %.0f, type %.0f, screen %d mm, " +
                    "opening %.1f px, expected radius %.1f px",
                result[1], result[2], result[3], result.getOrElse(8) { 0f }, screenDistanceMm,
                screenOpeningPixels, expectedLensRadiusPixels()
            ))
        }
        val goodCount: Int
        var combinedResults: List<FloatArray>? = null
        synchronized(gridResults) {
            if (status == 0) gridResults.addLast(now to result)
            while (gridResults.isNotEmpty() && now - gridResults.first().first > GRID_RESULT_WINDOW_MS) {
                gridResults.removeFirst()
            }
            goodCount = gridResults.size
            if (goodCount >= GRID_RESULTS_NEEDED) {
                combinedResults = gridResults.map { it.second }
                gridResults.clear()
            }
        }
        if (now - lastGridLogMs >= 1000L) {
            lastGridLogMs = now
            debugLog("grid frames: status counts $gridStatusCounts, $goodCount " +
                "recent good" + if (status == 0) String.format(
                    Locale.US, "; last %+.2f %+.2f x %.0f at %.0f mm, %d px, rms %.2f",
                    result[1], result[2], result[3], result[4], result[6].toInt(), result[7]
                ) else "")
            gridStatusCounts.clear()
        }
        combinedResults?.let { results ->
            val combined = combineGridResults(results)
            gridShowingResult = true
            gridWaitingForNewGrid = true
            toneGenerator?.startTone(ToneGenerator.TONE_PROP_ACK, 400)
            runOnUiThread { showGridResult(combined) }
            return
        }
        val hint = when (status) {
            0 -> "Measuring… $goodCount/$GRID_RESULTS_NEEDED"
            1 -> "Looking for the lens: keep it inside the outline"
            2 -> "Lens runs off the picture: keep it inside the outline"
            else -> "Measuring… $goodCount/$GRID_RESULTS_NEEDED"
        }
        publishAlignment(true, hint, "No key. Grid measurement running", "HOLD THE GLASSES STEADY · $hint")
    }

    /** How much bigger the lens looks at the guide distance (20 cm) than on the screen. */
    private fun guideScale() = screenDistanceMm / LENS_GUIDE_DISTANCE_MM

    /**
     * Lens radius in image pixels at the guide distance, or for glasses without a measurable
     * rim, a typical 20 mm lens radius at that distance.
     */
    private fun expectedLensRadiusPixels() = if (screenOpeningPixels > 0f) {
        guideScale() * screenOpeningPixels / sqrt(Math.PI.toFloat())
    } else {
        lastFocalPixels * TYPICAL_LENS_RADIUS_MM / LENS_GUIDE_DISTANCE_MM
    }

    /**
     * Before measuring: the outline to keep the lens inside, and a countdown. True while the
     * 5 s are still running (or the grid isn't up yet), so nothing is measured.
     */
    private fun showPlacement(frame: Frame, cameraImage: Image, gridShowing: Boolean): Boolean {
        val remainingMs = PLACEMENT_MS - (System.currentTimeMillis() - placementStartMs)
        if (gridShowing && remainingMs <= 0) return false
        showLensGuide(frame, cameraImage, false)
        val seconds = ceil(max(0L, remainingMs) / 1000.0).toInt()
        val countdown = if (gridShowing) "Measuring in $seconds s"
            else "Measuring starts ${PLACEMENT_MS / 1000} s after the grid appears"
        val fit = if (lensOutline != null) "Place the lens inside the outline" else "Center the lens in the circle"
        publishAlignment(
            true, countdown, "No key. $countdown",
            "$fit, about ${(LENS_GUIDE_DISTANCE_MM / 10).roundToInt()} cm from the camera, arms " +
                "toward the phone. $countdown"
        )
        return true
    }

    /**
     * Where the lens should be: centered in the picture, drawn as its own outline (from the
     * lens-size step) at the size it has 20 cm from the camera, or a circle without one.
     */
    private fun showLensGuide(frame: Frame, cameraImage: Image, measuring: Boolean) {
        val focal = FloatArray(2)
        calibratedFocalLength(frame, focal)
        lastFocalPixels = (focal[0] + focal[1]) / 2f
        val centerX = cameraImage.width / 2f
        val centerY = cameraImage.height / 2f
        val radiusImage = expectedLensRadiusPixels()
        val center = transformPoint(frame, Coordinates2d.IMAGE_PIXELS, Coordinates2d.VIEW, centerX, centerY)
        val edge = transformPoint(
            frame, Coordinates2d.IMAGE_PIXELS, Coordinates2d.VIEW, centerX + radiusImage, centerY
        )
        val radius = hypot(edge[0] - center[0], edge[1] - center[1])
        if (!radius.isFinite()) return
        val outline = lensOutline?.let { offsets ->
            val scale = guideScale()
            val imagePoints = FloatArray(offsets.size) { index ->
                (if (index % 2 == 0) centerX else centerY) + scale * offsets[index]
            }
            FloatArray(offsets.size).also {
                frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, imagePoints, Coordinates2d.VIEW, it)
            }
        }
        overlayView.update(
            AlignmentOverlayView.State(
                null, null, false, emptyList(),
                AlignmentOverlayView.LensGuide(center[0], center[1], radius, measuring, outline)
            )
        )
    }

    /** Median of (sphere, cylinder, axis) triples in power-vector form, so axes average properly. */
    private fun medianPrescription(sphere: List<Float>, cylinder: List<Float>, axis: List<Float>): FloatArray {
        fun median(values: List<Double>) = values.sorted()[values.size / 2]
        val mean = sphere.indices.map { sphere[it] + cylinder[it] / 2.0 }
        val j0 = sphere.indices.map { -(cylinder[it] / 2.0) * cos(2.0 * Math.toRadians(axis[it].toDouble())) }
        val j45 = sphere.indices.map { -(cylinder[it] / 2.0) * sin(2.0 * Math.toRadians(axis[it].toDouble())) }
        val m = median(mean)
        val jc = median(j0)
        val js = median(j45)
        val combinedCylinder = -2.0 * hypot(jc, js)
        var combinedAxis = Math.toDegrees(0.5 * atan2(js, jc))
        if (combinedAxis <= 0.5) combinedAxis += 180.0
        val sortedMean = mean.sorted()
        val spread = (sortedMean[sortedMean.size * 3 / 4] - sortedMean[sortedMean.size / 4]) / 2.0
        return floatArrayOf(
            (m - combinedCylinder / 2.0).toFloat(), combinedCylinder.toFloat(), combinedAxis.toFloat(),
            spread.toFloat()
        )
    }

    /**
     * Combines the frame results. Returns {type, sphere, cylinder, axis, distance mm, spread,
     * add}: for multifocals (majority of frames) the sphere/cylinder/axis are the distance zone's.
     */
    /**
     * Median prescription of the frames: {type, sph, cyl, axis, lens distance, spread, add,
     * glasses roll (NaN if not measured), frames used}. Axes are corrected for the roll of the
     * glasses (from the lines along the tops and bottoms of both lenses), using only the frames
     * whose roll is no less certain than average.
     */
    private fun combineGridResults(results: List<FloatArray>): FloatArray {
        fun tilt(result: FloatArray) = result.getOrElse(16) { Float.NaN }
        fun tiltError(result: FloatArray) = result.getOrElse(17) { Float.NaN }
        val withError = results.filter { tilt(it).isFinite() && tiltError(it).isFinite() }
        val tilted = if (withError.isNotEmpty()) {
            val meanError = withError.map { tiltError(it) }.average()
            withError.filter { tiltError(it) <= meanError }
        } else {
            results.filter { tilt(it).isFinite() }
        }
        val used = if (tilted.isEmpty()) results else tilted.map { result ->
            result.copyOf().also { corrected ->
                for (index in intArrayOf(3, 11, 14)) {
                    if (index < corrected.size) {
                        corrected[index] = ((corrected[index] + tilt(result)) % 180f + 180f) % 180f
                    }
                }
            }
        }
        val rollDegrees = if (tilted.isEmpty()) Float.NaN else tilted.map { tilt(it) }.sorted()[tilted.size / 2]
        val multifocal = used.filter { it.size >= 16 && it[8] >= 1f }
        val useZones = multifocal.size * 2 > used.size
        val chosen = if (useZones) multifocal else used
        val (s, c, a) = if (useZones) Triple(9, 10, 11) else Triple(1, 2, 3)
        val rx = medianPrescription(chosen.map { it[s] }, chosen.map { it[c] }, chosen.map { it[a] })
        val type = if (!useZones) 0f else chosen.map { it[8] }.sorted()[chosen.size / 2]
        val add = if (useZones) chosen.map { it[15] }.sorted()[chosen.size / 2] else 0f
        return floatArrayOf(
            type, rx[0], rx[1], rx[2],
            results.map { it[4] }.sorted()[results.size / 2], rx[3], add,
            rollDegrees, chosen.size.toFloat()
        )
    }

    /** App log, silent while a result is on screen (until a key on the computer moves on). */
    private fun debugLog(message: String) {
        if (!gridShowingResult) Log.d(LOG_TAG, message)
    }

    private fun showGridResult(combined: FloatArray) {
        fun quarter(value: Float) = (value * 4f).roundToInt() / 4f
        fun tenth(value: Float) = (value * 10f).roundToInt() / 10f
        val type = combined[0].roundToInt()
        val typeName = when (type) { 1 -> "bifocal"; 2 -> "progressive"; else -> "single vision" }
        actionView.text = if (type == 0) {
            String.format(
                Locale.US, "DONE: SPH %+.2f   CYL %+.2f   AXIS %d",
                tenth(combined[1]), quarter(combined[2]), combined[3].roundToInt()
            )
        } else {
            String.format(
                Locale.US, "DONE (%s): SPH %+.2f  CYL %+.2f  AXIS %d  ADD %+.2f",
                typeName, tenth(combined[1]), quarter(combined[2]), combined[3].roundToInt(),
                tenth(combined[6])
            )
        }
        detectionView.text = String.format(
            Locale.US,
            "PHONE: %s · unrounded %+.2f %+.2f × %.0f%s from %d frames (spread ±%.2f D) · lens " +
                "%.0f mm from camera · %s · axis assumes the arms point toward the phone",
            typeName.replaceFirstChar { it.uppercase() }, combined[1], combined[2], combined[3],
            if (type == 0) "" else String.format(Locale.US, " add %+.2f", combined[6]),
            combined[8].roundToInt(), combined[5], combined[4],
            if (combined[7].isFinite()) String.format(Locale.US, "axis corrected for %+.1f° frame tilt", combined[7])
            else "frame tilt not seen, axis uncorrected"
        )
        statusView.text = "KEYBOARD: G measures again (same setup). Another lens: L. New setup: Reset"
        Log.i(LOG_TAG, "grid result: ${combined.joinToString()}")
    }

    /** Keeps the no-lens run as the reference and moves on to measuring the lens size. */
    private fun finishReferenceRun() {
        val down = lastMonitorDown
        screenDistanceMm = alignmentSamples.lastOrNull()?.depthMm ?: TARGET_WORKING_DISTANCE_MM
        val gridPeriods = nativeEngine.finishGridReference(
            down[0], down[1], java.io.File(getExternalFilesDir(null), "grid-reference.pgm").absolutePath
        )
        debugLog("grid reference saved; bullseye ${lastAlignmentTarget?.centerX}, " +
            "${lastAlignmentTarget?.centerY}, size ${lastAlignmentTarget?.size}; down " +
            "${down[0]}, ${down[1]}; screen ${screenDistanceMm} mm")
        if (gridPeriods == null) {
            gridReferenceFrames = 0
            actionView.text = "NOW: The grid couldn't be read. Reload the page, then PRESS RETURN " +
                "to redo the reference"
            return
        }
        captureComplete = false
        stage = Stage.LENS_SIZE
        overlayView.update(null)
        actionView.text = "NOW: Reference done. ${lensSizePrompt()}"
        statusView.text = "KEYBOARD: Press L for the white lens-size screen"
        detectionView.text = String.format(
            Locale.US, "PHONE: Leave the phone where it is · grid %.1f px per period · screen %d cm",
            gridPeriods.minOrNull() ?: 0f, screenDistanceMm / 10
        )
    }

    /** Runs on the camera thread when “Measure lens size” was tapped. */
    private fun processLensSizeFrame(cameraImage: Image, monitorDown: FloatArray) {
        // Measured continuously, because the user's hands are busy holding the glasses on the
        // screen; accepted once the same lens-sized opening has been seen for about a second.
        val target = lastAlignmentTarget ?: return
        // Only on the white lens-size screen, which says so with its own step squares; anything
        // else (e.g. the bullseye's ring) could pass for a lens opening.
        val screenCode = decodeRemoteSquares(cameraImage, target, monitorDown)
        // The white screen came and went without a rim being found (half-rim, rimless or
        // gold frame): carry on with an assumed distance instead of a measured size.
        if (sawLensSizeScreen && screenCode == GRID_CODE) {
            sawLensSizeScreen = false
            debugLog("lens size: no rim found; assuming $ASSUMED_LENS_DISTANCE_MM mm")
            runOnUiThread { lensSizeMeasured(-1f, 0f, null) }
            return
        }
        if (screenCode == LENS_SIZE_SCREEN_CODE) sawLensSizeScreen = true
        if (screenCode != LENS_SIZE_SCREEN_CODE) {
            lensOpeningSamples.clear()
            logLensSizeState(cameraImage, "page code $screenCode (want $LENS_SIZE_SCREEN_CODE)")
            val now = System.currentTimeMillis()
            if (now - lastAlignmentUiMs >= ALIGNMENT_UI_INTERVAL_MS) {
                lastAlignmentUiMs = now
                runOnUiThread {
                    actionView.text = "NOW: ${lensSizePrompt()}"
                    statusView.text = "KEYBOARD: PRESS L for the white lens-size screen"
                }
            }
            return
        }
        val bullseyePixels = target.size * cameraImage.height
        val yPlane = cameraImage.planes[0]
        val reading = nativeEngine.measureLensOutline(
            yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
            yPlane.rowStride, yPlane.pixelStride,
            target.centerX * cameraImage.width, target.centerY * cameraImage.height,
            bullseyePixels * MIN_LENS_OPENING_MM / BULLSEYE_DIAMETER_MM
        )
        val opening = reading?.get(0) ?: -1f
        val openingMm = opening * BULLSEYE_DIAMETER_MM / bullseyePixels
        // Without glasses the only opening around the cross is the whole screen: too big.
        val lensSized = reading != null && opening > 0f && openingMm <= MAX_LENS_OPENING_MM
        if (lensSized) lensOpeningSamples.addLast(reading!!) else lensOpeningSamples.clear()
        logLensSizeState(
            cameraImage,
            "white screen seen; opening ${"%.1f".format(opening)} px = " +
                "${"%.1f".format(openingMm)} mm, ${lensOpeningSamples.size} steady readings"
        )
        while (lensOpeningSamples.size > LENS_SIZE_STABLE_FRAMES) lensOpeningSamples.removeFirst()

        val openings = lensOpeningSamples.map { it[0] }
        val stable = openings.size == LENS_SIZE_STABLE_FRAMES &&
            openings.maxOrNull()!! - openings.minOrNull()!! <= LENS_SIZE_TOLERANCE * openings.average()
        if (stable) {
            val median = lensOpeningSamples.sortedBy { it[0] }[lensOpeningSamples.size / 2]
            val accepted = median[0]
            lensOpeningSamples.clear()
            debugLog("lens size accepted: $accepted px (${accepted * BULLSEYE_DIAMETER_MM / bullseyePixels} mm), " +
                "outline ${(median.size - 1) / 2} points")
            val outline = outlineOffsets(median)
            runOnUiThread {
                lensSizeMeasured(accepted, accepted * BULLSEYE_DIAMETER_MM / bullseyePixels, outline)
            }
            return
        }
        val now = System.currentTimeMillis()
        if (now - lastAlignmentUiMs < ALIGNMENT_UI_INTERVAL_MS) return
        lastAlignmentUiMs = now
        val progress = lensOpeningSamples.size
        runOnUiThread {
            actionView.text = if (progress == 0) {
                "NOW: ${lensSizePrompt()}"
            } else {
                "NOW: Hold the glasses still… measuring ($progress/$LENS_SIZE_STABLE_FRAMES)"
            }
        }
    }

    /** Once a second during the lens-size step: what it sees, plus a picture for diagnosis. */
    private fun logLensSizeState(cameraImage: Image, state: String) {
        val now = System.currentTimeMillis()
        if (now - lastDiagnosticFrameMs < 1000L) return
        lastDiagnosticFrameMs = now
        debugLog("lens size step: $state")
        if (now - lastLensSizeSaveMs >= 5000L) {
            lastLensSizeSaveMs = now
            saveGrayscaleFrame(cameraImage, "lens-size-step.pgm")
        }
    }

    /** Saves a tightly packed grey frame as a PGM in the app's files directory. */
    private fun saveGrayscaleBytes(pixels: ByteBuffer, width: Int, height: Int, name: String) {
        try {
            val bytes = ByteArray(width * height)
            pixels.duplicate().apply { position(0) }.get(bytes)
            val file = java.io.File(getExternalFilesDir(null), name)
            file.outputStream().use { output ->
                output.write("P5\n$width $height\n255\n".toByteArray())
                output.write(bytes)
            }
        } catch (error: Exception) {
            debugLog("could not save frame: ${error.localizedMessage}")
        }
    }

    /** Saves the camera's Y plane as a PGM in the app's files directory, for diagnosis. */
    private fun saveGrayscaleFrame(cameraImage: Image, name: String) {
        try {
            val plane = cameraImage.planes[0]
            val buffer = plane.buffer.duplicate()
            val pixels = ByteArray(cameraImage.width * cameraImage.height)
            for (row in 0 until cameraImage.height) {
                buffer.position(row * plane.rowStride)
                buffer.get(pixels, row * cameraImage.width, cameraImage.width)
            }
            val file = java.io.File(getExternalFilesDir(null), name)
            file.outputStream().use { output ->
                output.write("P5\n${cameraImage.width} ${cameraImage.height}\n255\n".toByteArray())
                output.write(pixels)
            }

        } catch (error: Exception) {
            debugLog("could not save frame: ${error.localizedMessage}")
        }
    }

    /** Lens-size instructions (L shows the white screen, which ends by itself after 8 s). */
    private fun lensSizePrompt() = "Lens size: press L on the computer and hold the glasses " +
        "flat on the white screen with the lens to measure on the cross for a few seconds"

    /** Outline points (after the size in [0]) as offsets from the outline's middle, or null. */
    private fun outlineOffsets(reading: FloatArray): FloatArray? {
        val count = (reading.size - 1) / 2
        if (count < 3) return null
        var sumX = 0.0
        var sumY = 0.0
        for (index in 0 until count) {
            sumX += reading[1 + 2 * index]
            sumY += reading[2 + 2 * index]
        }
        val middleX = (sumX / count).toFloat()
        val middleY = (sumY / count).toFloat()
        return FloatArray(2 * count) { index ->
            reading[1 + index] - if (index % 2 == 0) middleX else middleY
        }
    }

    private fun lensSizeMeasured(openingPixels: Float, openingMm: Float, outline: FloatArray?) {
        sawLensSizeScreen = false
        screenOpeningPixels = openingPixels
        lensOutline = outline
        // A beep says the size is taken; measuring starts 5 s after the grid appears.
        synchronized(gridResults) { gridResults.clear() }
        toneGenerator?.startTone(ToneGenerator.TONE_PROP_BEEP, 150)
        // The white screen is still up; don't take its continuing presence as a new request.
        lensStagePreviousCode = LENS_SIZE_SCREEN_CODE
        stage = Stage.LENS
        detectionView.text = if (openingPixels > 0f) {
            String.format(
                Locale.US, "PHONE: Lens opening ≈ %.0f mm across (seen from %d cm)",
                openingMm, screenDistanceMm / 10
            )
        } else {
            "PHONE: No full rim found, so no lens size: hold the lens in the circle " +
                "${(ASSUMED_LENS_DISTANCE_MM / 10).roundToInt()} cm from the camera"
        }
        startLensRun()
    }

    /** Clears the previous lens run (the reference is kept) and waits for the pages. */
    private fun startLensRun() {
        pendingCapture.set(null)
        nativeEngine.resetPhaseCapture()
        captureComplete = false
        captureAxis = AXIS_X
        capturePhase = 0
        collectedFrames = 0
        candidateScreen = -1 to false
        candidateScreenFrames = 0
        stableScreen = -1 to false
        // Armed straight away: the lens run never sees the bullseye, so it can't wait for one.
        pendingCapture.set(ArmedCapture(CaptureRequest(AXIS_X, 0)))
        val lensDistanceCm = (screenDistanceMm / 4f / 10f).roundToInt()
        actionView.text = "NOW: Place the lens inside the outline, about " +
            "${(LENS_GUIDE_DISTANCE_MM / 10).roundToInt()} cm from the camera, arms toward the " +
            "phone, once the grid appears. It measures by itself 5 s later"
        statusView.text = "KEYBOARD: No key needed (G restarts the grid if time runs out)"
    }

    /** Compares the lens run with the reference and shows the lens's prescription. */
    private fun finishLensRun() {
        val down = lastMonitorDown
        val result = nativeEngine.measureLensFromRuns(
            down[0], down[1], imageWidth / 2f, imageHeight / 2f,
            screenDistanceMm.toFloat(), screenOpeningPixels,
            java.io.File(getExternalFilesDir(null), "lens-brightness-ratio.pgm").absolutePath
        )
        val status = result?.getOrNull(0)?.toInt() ?: 5
        if (status != 0 || result == null) {
            // Re-arm so the looping pages give it another try without a key press.
            startLensRun()
            val lensDistanceCm = (screenDistanceMm / 4f / 10f).roundToInt()
            actionView.text = "NOW: " + when (status) {
                1 -> "Couldn't find the lens in the picture. Center it in front of the " +
                    "camera, about $lensDistanceCm cm away, and hold steady; it tries again " +
                    "when step 1 comes round"
                2 -> "The lens outline runs off the picture or the rim isn't fully visible. " +
                    "Hold the glasses a little farther from the camera; it tries again when " +
                    "step 1 comes round"
                3 -> "Too little usable area inside the lens. Raise “Fringe period” on the page, " +
                    "then PRESS R"
                4 -> "The fringes are too fine to read. Raise “Fringe period” on the page, then " +
                    "tap “Reset capture” and redo the reference"
                else -> "Measurement failed (code $status). Tap “Reset capture” and start again"
            }
            return
        }

        fun quarter(value: Float) = (value * 4f).roundToInt() / 4f
        actionView.text = String.format(
            Locale.US, "DONE: SPH %+.2f   CYL %+.2f   AXIS %d",
            quarter(result[1]), quarter(result[2]), result[3].roundToInt()
        )
        detectionView.text = String.format(
            Locale.US,
            "PHONE: Unrounded %+.2f %+.2f × %.0f · lens %.0f mm from camera · %d points · " +
                "fit error %.2f px · axis assumes the arms point toward the phone",
            result[1], result[2], result[3], result[4], result[6].toInt(), result[7]
        )
        statusView.text = "KEYBOARD: Done. For another lens, tap “Reset capture” and start " +
            "again from the beginning"
    }

    private fun showCaptureQualitySummary() {
        val quality = nativeEngine.getCaptureQualitySummary()
        val calibration = nativeEngine.getCapturedCameraDepthCalibration()
        actionView.text = "NOW: DONE. All 8 photos taken; nothing more to press"
        if (quality == null || quality.size < 6) {
            actionView.text = "NOW: Tap “Reset capture” and try again"
            statusView.text = "CAPTURE INCOMPLETE: depth/phase quality could not be checked"
            detectionView.text = "Reset capture, hold the phone steady, and try again"
            return
        }

        val xModulation = quality[0]
        val yModulation = quality[1]
        val depthLow = quality[2].toInt()
        val depthMedian = quality[3].toInt()
        val depthHigh = quality[4].toInt()
        val enoughContrast = xModulation >= MIN_MEAN_MODULATION &&
            yModulation >= MIN_MEAN_MODULATION

        statusView.text = if (enoughContrast) {
            "CAPTURE CHECK PASSED: fringe signal detected"
        } else {
            "CAPTURE CHECK WARNING: weak fringe signal"
        }
        detectionView.text = String.format(
            Locale.US,
            "Mean modulation X %.1f / Y %.1f (minimum %.0f) · center depth %d mm " +
                "(P10–P90: %d–%d mm)%s",
            xModulation,
            yModulation,
            MIN_MEAN_MODULATION,
            depthMedian,
            depthLow,
            depthHigh,
            if (calibration == null || calibration.size < 5) " · intrinsics unavailable" else ""
        )
    }

    private fun capturePrompt(): String = if (captureComplete) {
        "KEYBOARD: All 8 steps captured. Nothing more to press"
    } else {
        "KEYBOARD: No key yet. Line up the phone"
    }

    private fun phaseName(axis: Int, phase: Int): String {
        val axisName = if (axis == AXIS_X) "X" else "Y"
        return "$axisName ${PHASE_LABELS[phase]}"
    }

    private fun phaseName(stepIndex: Int): String = phaseName(stepIndex / 4, stepIndex % 4)

    /** Matches the 4-bit code drawn under the bullseye by monitor-pattern-player.html. */
    private fun phaseCode(request: CaptureRequest): Int = 8 + request.axis * 4 + request.phase

    private companion object {
        /**
         * ARCore's focal length on the Pixel 7a is low by this factor: measured once with a
         * ruler (camera to screen 19.75 in), so users don't need to calibrate.
         */
        const val DISTANCE_SCALE = 1.158749f
        const val AXIS_X = 0
        const val AXIS_Y = 1
        const val MIN_MEAN_MODULATION = 10.0f
        const val CENTER_MIN = 0.42f
        const val CENTER_MAX = 0.58f
        const val MAX_TILT_DEG = 6.0
        const val MIN_TILT_SAMPLES = 3
        /** Must match CORNER_OFFSET_MM in monitor-pattern-player.html. */
        const val CORNER_MARKER_OFFSET_MM = 90f
        /** The phone's white target box is sized for this distance. */
        const val TARGET_WORKING_DISTANCE_MM = 500
        const val MIN_WORKING_DISTANCE_MM = 460
        const val MAX_WORKING_DISTANCE_MM = 540
        /** Must match BULLSEYE_DIAMETER_MM in monitor-pattern-player.html. */
        const val BULLSEYE_DIAMETER_MM = 50f
        const val STABLE_FRAME_COUNT = 5
        const val MAX_ALIGNMENT_MISSES = 2
        const val MAX_CENTER_JITTER = 0.02f
        const val MAX_DEPTH_JITTER_MM = 60
        const val MAX_SIZE_JITTER = 0.015f
        const val ALIGNMENT_UI_INTERVAL_MS = 150L
        const val SCREEN_DEBOUNCE_FRAMES = 2
        /**
         * Hand-held glasses: grey-level change per frame that counts as moving. 5 rejected
         * normal hand tremor (measured 5.6–7.3), so allow more.
         */
        const val MOTION_LIMIT = 10f
        const val MOTION_SAMPLE_STEP = 4
        /** Fraction of the picture's width and height, around the center, watched for motion. */
        const val MOTION_REGION = 0.4f
        /** Frames after a page flip during which change is the monitor, not the hands. */
        const val PAGE_SETTLE_FRAMES = 3
        /** Still frames (about a third of a second) before frames are collected. */
        const val STEADY_FRAMES_NEEDED = 10
        /** Must match LENS_SIZE_SCREEN_CODE in monitor-pattern-player.html. */
        const val LENS_SIZE_SCREEN_CODE = 1
        /** Must match LOOP_COUNTDOWN_CODE in monitor-pattern-player.html. */
        const val LOOP_COUNTDOWN_CODE = 2
        /** Must match GRID_CODE in monitor-pattern-player.html: the crossed grid is showing. */
        const val GRID_CODE = 3
        /** Good single-frame results combined into the final answer. */
        const val GRID_RESULTS_NEEDED = 10
        /** The guide outline shows the lens as it looks this far from the camera. */
        const val LENS_GUIDE_DISTANCE_MM = 200f
        /** Lens distance used when no rim can be measured: held at the guide circle. */
        const val ASSUMED_LENS_DISTANCE_MM = LENS_GUIDE_DISTANCE_MM
        /** Time to place the lens inside the outline before measuring starts. */
        const val PLACEMENT_MS = 5000L
        const val RESULT_PAGE_CHECK_MS = 500L
        /** Grid unseen this long, then seen again: a new grid page (e.g. G). */
        const val NEW_GRID_GAP_MS = 1500L
        const val TYPICAL_LENS_RADIUS_MM = 20f
        /** No-lens grid frames averaged into the reference (matches the native maximum). */
        const val GRID_REFERENCE_FRAMES = 32
        /**
         * Only results this recent count. Long enough to cover the whole time the grid page is
         * up after the placement countdown: only 2–3 frames a second are measured, and some
         * fail, so a short window would never fill.
         */
        const val GRID_RESULT_WINDOW_MS = 15000L
        const val DIAGNOSTIC_FRAME_INTERVAL_MS = 5000L
        /** "Can't read the page" pictures: one a minute is plenty and keeps the log readable. */
        const val UNREADABLE_SAVE_INTERVAL_MS = 60000L
        /** Bullseye shift (fraction of its size, ~2.5 mm) that means the setup has moved. */
        const val SETUP_MOVED_FRACTION = 0.05f
        /** Frames averaged per fringe page; fewer if auto-advance moves on first. */
        const val FRAMES_PER_PAGE = 32
        /** Lens-loop pages flip every 0.4 s (~12 camera frames); take 5 of them. */
        const val LENS_FRAMES_PER_PAGE = 5
        /**
         * Must match REMOTE_TICK_SPOTS_MM in monitor-pattern-player.html: copies of the step
         * squares laid out as if under a bullseye at (right, down) mm from center.
         */
        val REMOTE_TICK_SPOTS_MM = listOf(0f to -145f, 0f to 63f, -150f to -41f, 150f to -41f)
        /** Smallest hole accepted as a lens opening when measuring its size on the screen. */
        const val MIN_LENS_OPENING_MM = 15f
        /** Larger "openings" are the bare screen, seen before the glasses are put on it. */
        const val MAX_LENS_OPENING_MM = 90f
        const val LENS_SIZE_STABLE_FRAMES = 25
        const val LENS_SIZE_TOLERANCE = 0.04f
        /** Below this many camera pixels per fringe the phase gets unreliable. */
        const val MIN_FRINGE_PERIOD_PIXELS = 6f
        const val LOG_TAG = "OpenLensometer"
        val PHASE_LABELS = listOf("0°", "90°", "180°", "270°")
    }

    private data class CaptureRequest(val axis: Int, val phase: Int)

    /** viewX/viewY are 0..1 across the phone screen as displayed, not the raw sensor image. */
    private data class AlignmentSample(
        val viewX: Float,
        val viewY: Float,
        val targetSize: Float,
        val depthMm: Int,
        val offsetRightMm: Float,
        val offsetDownMm: Float,
        val tiltRightDeg: Float?,
        val tiltDownDeg: Float?
    )

    private data class PoseEstimate(
        val viewX: Float,
        val viewY: Float,
        val offsetRightMm: Float,
        val offsetDownMm: Float,
        val tiltRightDeg: Float?,
        val tiltDownDeg: Float?
    )

    private data class AlignmentTarget(
        val centerX: Float,
        val centerY: Float,
        val size: Float
    )

    private data class ArmedCapture(val request: CaptureRequest)
}