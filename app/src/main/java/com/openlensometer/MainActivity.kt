package com.openlensometer

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.graphics.RectF
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import android.text.InputType
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
import com.openlensometer.engine.EyePrescription
import com.openlensometer.engine.ReferenceGlasses
import com.openlensometer.engine.ReferenceGlassesStore
import java.util.Locale
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sign

class MainActivity : ComponentActivity() {
    private lateinit var cameraView: ArCoreCameraView
    private lateinit var overlayView: AlignmentOverlayView
    @Volatile private var lastOverlayTarget: RectF? = null
    private lateinit var actionView: TextView
    private lateinit var statusView: TextView
    private lateinit var detectionView: TextView
    private lateinit var captureButton: Button
    private lateinit var cancelPhaseButton: Button
    private lateinit var referenceView: TextView
    private val referenceGlassesStore by lazy { ReferenceGlassesStore(this) }
    private val nativeEngine by lazy { NativeLensEngine() }
    private var arSession: Session? = null
    private var installRequested = false
    private var activityResumed = false
    @Volatile private var alignmentReady = false
    /** Reference run (no lens) → lens size (glasses on the screen) → lens run. */
    private enum class Stage { REFERENCE, LENS_SIZE, LENS }
    @Volatile private var stage = Stage.REFERENCE
    /** The wearer's left lens is measured first, then the right. */
    private enum class Eye(val label: String, val code: String) { LEFT("LEFT", "OS"), RIGHT("RIGHT", "OD") }
    @Volatile private var currentEye = Eye.LEFT
    /** Finished prescriptions: {sphere, cylinder, axis}. */
    private val eyeResults = mutableMapOf<Eye, FloatArray>()
    @Volatile private var lensSizeRequested = false
    private val lensOpeningSamples = ArrayDeque<Float>()
    private var lastDiagnosticFrameMs = 0L
    private var previousThumbnail: ByteArray? = null
    private var lastRawCode = Int.MIN_VALUE
    private var framesSincePageChange = 0
    @Volatile private var steadyFrames = 0
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0
    @Volatile private var lastMonitorDown = floatArrayOf(0f, 1f)
    /** Camera-to-screen distance during the reference run (from the bullseye size). */
    private var screenDistanceMm = 0
    /** Lens opening (sqrt of area, image pixels) with the glasses held on the screen. */
    private var screenOpeningPixels = 0f
    private lateinit var measureSizeButton: Button
    private lateinit var repeatLensButton: Button
    /** Frames of the current page handed to native code so far (averaged at commit). */
    @Volatile private var collectedFrames = 0
    /** Set by "Cancel" so the phone doesn't immediately re-arm itself. */
    @Volatile private var autoArmPaused = false
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
        }
        statusView = TextView(this).apply {
            text = capturePrompt()
            textSize = 13f
            setTextColor(android.graphics.Color.BLACK)
            setBackgroundColor(android.graphics.Color.rgb(255, 238, 150))
            setPadding(16, 6, 16, 2)
        }
        detectionView = TextView(this).apply {
            text = "PHONE: Waiting for camera"
            textSize = 12f
            setTextColor(android.graphics.Color.BLACK)
            setBackgroundColor(android.graphics.Color.rgb(255, 238, 150))
            setPadding(16, 2, 16, 6)
        }
        captureButton = Button(this).apply {
            setOnClickListener { requestPhaseCapture() }
        }
        cancelPhaseButton = Button(this).apply {
            text = "Cancel prepared capture"
            visibility = ViewGroup.GONE
            setOnClickListener { cancelPhaseWait() }
        }
        referenceView = TextView(this).apply {
            textSize = 12f
            setPadding(16, 8, 16, 8)
        }
        val referenceButton = Button(this).apply {
            text = "Reference glasses"
            setOnClickListener { showReferenceGlassesDialog() }
        }
        measureSizeButton = Button(this).apply {
            text = "Measure lens size now (it also measures by itself)"
            visibility = ViewGroup.GONE
            setOnClickListener {
                lensSizeRequested = true
                actionView.text = "NOW: Hold the glasses still on the screen…"
            }
        }
        repeatLensButton = Button(this).apply {
            text = "Repeat lens run"
            visibility = ViewGroup.GONE
            setOnClickListener { startLensRun() }
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
        updateCaptureButton()

        fun fullWidth() = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        fun sideBySide() = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
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
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(captureButton, sideBySide())
                addView(resetButton, sideBySide())
                addView(referenceButton, sideBySide())
            }, fullWidth())
            addView(cancelPhaseButton, fullWidth())
            addView(measureSizeButton, fullWidth())
            addView(repeatLensButton, fullWidth())
            addView(referenceView, fullWidth())
        })
        updateReferenceView()

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
                        if (now - lastDiagnosticFrameMs > DIAGNOSTIC_FRAME_INTERVAL_MS) {
                            lastDiagnosticFrameMs = now
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
                        handleAlignmentMiss("Can't read the step squares above the screen center")
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
            if (pendingCapture.get() == null && guide.first && !captureComplete &&
                !autoArmPaused
            ) {
                if (pendingCapture.compareAndSet(
                        null, ArmedCapture(CaptureRequest(captureAxis, capturePhase))
                    )
                ) runOnUiThread { updateCaptureButton() }
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
            Log.d(LOG_TAG, "armed step=${expectedPhaseCode - 7} raw code=${target[3].toInt()} " +
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
                intrinsics.getFocalLength(focalLength, 0)
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
                Log.d(LOG_TAG, "committed step=$step from $collectedFrames frames")
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
            if (collectedFrames < FRAMES_PER_PAGE) {
                publishAlignment(
                    true, "Hold still · photographing step $step/8 ($collectedFrames/$FRAMES_PER_PAGE)",
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

    /** Reads the step squares drawn ~10 cm above center (native code searches around it). */
    private fun decodeRemoteSquares(
        cameraImage: Image,
        target: AlignmentTarget,
        monitorDown: FloatArray
    ): Int {
        val yPlane = cameraImage.planes[0]
        val alongDown = REMOTE_TICKS_OFFSET_MM / BULLSEYE_DIAMETER_MM * target.size * cameraImage.height
        return nativeEngine.decodeAlignmentPhaseCode(
            yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
            yPlane.rowStride, yPlane.pixelStride,
            target.centerX + monitorDown[0] * alongDown / cameraImage.width,
            target.centerY + monitorDown[1] * alongDown / cameraImage.height,
            target.size, monitorDown[0], monitorDown[1]
        )
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
            frame.camera.imageIntrinsics.getFocalLength(focal, 0)
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
        updateOverlay(frame, tilt, guide.first)
        return guide
    }

    /**
     * Orders corrections so that fixing one does not undo another: square the phone to the
     * monitor first (tilt shifts the target), then slide it to center, then set distance.
     * All directions are as the user sees the phone screen while the camera faces the monitor.
     */
    private fun evaluateAlignment(sample: AlignmentSample): Pair<Boolean, String> {
        val tiltRight = alignmentSamples.mapNotNull { it.tiltRightDeg }
            .takeIf { it.size >= MIN_TILT_SAMPLES }?.average()
        val tiltDown = alignmentSamples.mapNotNull { it.tiltDownDeg }
            .takeIf { it.size >= MIN_TILT_SAMPLES }?.average()
        val worstTilt = listOfNotNull(
            tiltRight?.let { "RIGHT" to it }, tiltRight?.let { "LEFT" to -it },
            tiltDown?.let { "BOTTOM" to it }, tiltDown?.let { "TOP" to -it }
        ).maxByOrNull { it.second }
        if (worstTilt != null && worstTilt.second > MAX_TILT_DEG) {
            return false to "TILT: turn the phone so its ${worstTilt.first} edge moves " +
                "toward the monitor (~${worstTilt.second.roundToInt()}°). The pink box's " +
                "${worstTilt.first.lowercase()} side looks shorter"
        }

        val moves = mutableListOf<String>()
        if (sample.viewX < CENTER_MIN || sample.viewX > CENTER_MAX) {
            moves += "${if (sample.offsetRightMm < 0f) "LEFT" else "RIGHT"} ${centimeters(sample.offsetRightMm)}"
        }
        if (sample.viewY < CENTER_MIN || sample.viewY > CENTER_MAX) {
            moves += "${if (sample.offsetDownMm < 0f) "UP" else "DOWN"} ${centimeters(sample.offsetDownMm)}"
        }
        if (moves.isNotEmpty()) {
            return false to "MOVE the whole phone ${moves.joinToString(" and ")} " +
                "(slide it; don't tilt) until the pink box sits on the white box"
        }

        if (sample.depthMm > MAX_WORKING_DISTANCE_MM) {
            return false to "MOVE FORWARD: bring the phone ${centimeters(
                (sample.depthMm - TARGET_WORKING_DISTANCE_MM).toFloat()
            )} closer to the monitor (now ${sample.depthMm / 10} cm; pink box is smaller than white)"
        }
        if (sample.depthMm < MIN_WORKING_DISTANCE_MM) {
            return false to "MOVE BACK: pull the phone ${centimeters(
                (TARGET_WORKING_DISTANCE_MM - sample.depthMm).toFloat()
            )} away from the monitor (now ${sample.depthMm / 10} cm; pink box is bigger than white)"
        }
        if (alignmentSamples.size < STABLE_FRAME_COUNT) {
            return false to "Lined up at ${sample.depthMm / 10} cm. Hold still " +
                "(${alignmentSamples.size}/$STABLE_FRAME_COUNT)"
        }

        val samples = alignmentSamples.toList()
        val centerXRange = samples.maxOf { it.viewX } - samples.minOf { it.viewX }
        val centerYRange = samples.maxOf { it.viewY } - samples.minOf { it.viewY }
        val depthRange = samples.maxOf { it.depthMm } - samples.minOf { it.depthMm }
        val sizeRange = samples.maxOf { it.targetSize } - samples.minOf { it.targetSize }
        if (centerXRange > MAX_CENTER_JITTER || centerYRange > MAX_CENTER_JITTER ||
            depthRange > MAX_DEPTH_JITTER_MM || sizeRange > MAX_SIZE_JITTER
        ) return false to "Lined up. Hold the phone steadier"
        return true to "Lined up and steady at ${sample.depthMm / 10} cm. Don't move"
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
            detectedCode == LENS_SIZE_SCREEN_CODE -> "PRESS R to start the lens loop" to
                "R starts the repeating pages after a 5-second countdown"
            detectedCode == 0 -> "Get ready: hold the glasses steady with both hands. The " +
                "pages start by themselves" to "No key. Countdown running"
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
        val columns = width / MOTION_SAMPLE_STEP
        val rows = height / MOTION_SAMPLE_STEP
        val thumbnail = ByteArray(columns * rows)
        for (row in 0 until rows) {
            val rowStart = row * MOTION_SAMPLE_STEP * plane.rowStride
            for (column in 0 until columns) {
                thumbnail[row * columns + column] =
                    buffer.get(rowStart + column * MOTION_SAMPLE_STEP * plane.pixelStride)
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
        Log.d(LOG_TAG, "lens attempt restarted: $reason")
        runOnUiThread {
            actionView.text = "NOW: $reason. HOLD STEADY · starting over when step 1 comes round"
            updateCaptureButton()
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
            "TAP “Prepare step ${expectedCode - 7}/8” on the phone. Don't press any key" to
                "No key yet"
        } else {
            dontPress(phoneMessage) to "No key yet. Line up the phone first"
        }
    }

    private fun dontPress(phoneMessage: String) = "DON'T PRESS ANY KEY. $phoneMessage"

    private fun times(count: Int): String = if (count > 1) " $count times" else ""

    private fun centimeters(mm: Float): String =
        "${max(1, (abs(mm) / 10f).roundToInt())} cm"

    private fun handleAlignmentMiss(message: String, resetImmediately: Boolean = false) {
        consecutiveAlignmentMisses++
        if (resetImmediately || consecutiveAlignmentMisses > MAX_ALIGNMENT_MISSES) {
            alignmentSamples.clear()
            alignmentReady = false
        }
        val displayedMessage = if (alignmentReady) "Bullseye briefly lost. Hold steady" else message
        lastOverlayTarget?.let { overlayView.update(AlignmentOverlayView.State(it, null, false)) }
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
            captureButton.isEnabled = ready && !captureComplete && pendingCapture.get() == null
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
        intrinsics.getFocalLength(focal, 0)
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
    private fun updateOverlay(frame: Frame, tilt: FloatArray?, aligned: Boolean) {
        val focal = FloatArray(2)
        val principal = FloatArray(2)
        frame.camera.imageIntrinsics.getFocalLength(focal, 0)
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
        overlayView.update(AlignmentOverlayView.State(target, corners, aligned))
    }

    /** Pinhole distance to the monitor: focal length × real bullseye size ÷ imaged size. */
    private fun distanceFromBullseye(frame: Frame, bullseyeImagePixels: Float): Int {
        if (!bullseyeImagePixels.isFinite() || bullseyeImagePixels <= 1f) return -1
        val focal = FloatArray(2)
        frame.camera.imageIntrinsics.getFocalLength(focal, 0)
        val focalPixels = (focal[0] + focal[1]) / 2f
        if (!focalPixels.isFinite() || focalPixels <= 0f) return -1
        return (focalPixels * BULLSEYE_DIAMETER_MM / bullseyeImagePixels).roundToInt()
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

    private fun showReferenceGlassesDialog() {
        val entries = referenceGlassesStore.all()
        val builder = AlertDialog.Builder(this)
            .setTitle("Known prescription glasses")
            .setPositiveButton("Add prescription") { _, _ -> showAddReferenceDialog() }
            .setNegativeButton("Close", null)

        if (entries.isEmpty()) {
            builder.setMessage("No reference glasses saved")
        } else {
            builder.setItems(entries.map(::formatReference).toTypedArray()) { _, index ->
                referenceGlassesStore.select(entries[index].id)
                updateReferenceView()
                statusView.text = "Reference selected for later validation"
            }
            builder.setNeutralButton("Delete active") { _, _ ->
                if (referenceGlassesStore.removeActive()) updateReferenceView()
            }
        }
        builder.show()
    }

    private fun showAddReferenceDialog() {
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 8, 24, 8)
        }
        val scroll = ScrollView(this)
        val fields = linkedMapOf<String, EditText>()

        fun addField(
            key: String,
            label: String,
            value: String,
            integer: Boolean = false,
            numeric: Boolean = true
        ) {
            val editText = EditText(this).apply {
                hint = if (numeric) "Enter value" else "Enter name"
                setText(value)
                inputType = if (numeric) {
                    InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED or
                        if (integer) 0 else InputType.TYPE_NUMBER_FLAG_DECIMAL
                } else {
                    InputType.TYPE_CLASS_TEXT
                }
                setSingleLine(true)
            }
            fields[key] = editText
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(TextView(this@MainActivity).apply {
                    text = "$label:"
                    gravity = Gravity.CENTER_VERTICAL
                    contentDescription = label
                }, LinearLayout.LayoutParams(
                    (112 * resources.displayMetrics.density).toInt(),
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ))
                addView(editText, LinearLayout.LayoutParams(
                    0,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    1f
                ))
            }
            form.addView(row, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
        }

        addField(
            "name", "Glasses name", "Reference ${referenceGlassesStore.all().size + 1}",
            numeric = false
        )
        form.addView(TextView(this).apply { text = "Right eye (OD)" })
        addField("rightSphere", "Sphere", "0.00")
        addField("rightCylinder", "Cylinder", "0.00")
        addField("rightAxis", "Axis (0–180)", "0", integer = true)
        addField("rightAdd", "Add", "0.00")
        form.addView(TextView(this).apply { text = "Left eye (OS)" })
        addField("leftSphere", "Sphere", "0.00")
        addField("leftCylinder", "Cylinder", "0.00")
        addField("leftAxis", "Axis (0–180)", "0", integer = true)
        addField("leftAdd", "Add", "0.00")
        scroll.addView(form)

        val dialog = AlertDialog.Builder(this)
            .setTitle("Add known prescription")
            .setView(scroll)
            .setPositiveButton("Save", null)
            .setNegativeButton("Cancel", null)
            .create()

        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                try {
                    fun number(key: String) = fields.getValue(key).text.toString().toDouble()
                    fun axis(key: String) = fields.getValue(key).text.toString().toInt()
                    val right = EyePrescription(
                        number("rightSphere"), number("rightCylinder"),
                        axis("rightAxis"), number("rightAdd")
                    )
                    val left = EyePrescription(
                        number("leftSphere"), number("leftCylinder"),
                        axis("leftAxis"), number("leftAdd")
                    )
                    referenceGlassesStore.add(
                        fields.getValue("name").text.toString(), right, left
                    )
                    updateReferenceView()
                    statusView.text = "Reference saved; use it to validate a completed measurement"
                    dialog.dismiss()
                } catch (error: NumberFormatException) {
                    Toast.makeText(this, "Enter valid numbers for both eyes", Toast.LENGTH_SHORT).show()
                } catch (error: IllegalArgumentException) {
                    Toast.makeText(
                        this,
                        error.message ?: "Prescription values are outside supported ranges",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
        dialog.show()
    }

    private fun updateReferenceView() {
        val reference = referenceGlassesStore.active()
        referenceView.text = reference?.let { "Active reference: ${formatReference(it)}" }
            ?: "No active reference glasses"
    }

    private fun formatReference(reference: ReferenceGlasses): String {
        fun eye(value: EyePrescription): String = String.format(
            Locale.US, "%+.2f %+.2f × %03d Add %+.2f",
            value.sphere, value.cylinder, value.axis, value.add
        )
        return "${reference.name} · OD ${eye(reference.rightEye)} · OS ${eye(reference.leftEye)}"
    }

    private fun requestPhaseCapture() {
        if (!alignmentReady) {
            actionView.text = "NOW: DON'T PRESS ANY KEY. Line up and hold steady first"
            return
        }
        if (captureComplete || pendingCapture.get() != null) return
        autoArmPaused = false
        val armed = ArmedCapture(CaptureRequest(captureAxis, capturePhase))
        if (!pendingCapture.compareAndSet(null, armed)) return

        actionView.text = "NOW: DON'T PRESS ANY KEY. Reading which page the monitor shows…"
        statusView.text = "KEYBOARD: Reading which step the monitor shows…"
        detectionView.text = "PHONE: Keep the bullseye and the squares under it in view"
        updateCaptureButton()
    }

    private fun cancelPhaseWait() {
        autoArmPaused = true
        pendingCapture.set(null)
        actionView.text = "NOW: Line up the phone, then tap Prepare. Don't press any key"
        statusView.text = "KEYBOARD: No key needed. Capture paused"
        detectionView.text = "PHONE: Line up again, then tap Prepare to continue"
        updateCaptureButton()
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
            updateCaptureButton()
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
            updateCaptureButton()
            return
        }

        // Arm the next step straight away so the user is only asked to press keys.
        val next = CaptureRequest(captureAxis, capturePhase)
        pendingCapture.set(ArmedCapture(next))
        val nextStep = phaseCode(next) - 7
        actionView.text = "NOW: Step ${nextStep - 1}/8 done. WAIT for the next page"
        statusView.text = "KEYBOARD: Next is step $nextStep/8"
        detectionView.text = "PHONE: Keep the phone where it is"
        updateCaptureButton()
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
        autoArmPaused = false
        stage = Stage.REFERENCE
        currentEye = Eye.LEFT
        eyeResults.clear()
        lensSizeRequested = false
        screenOpeningPixels = 0f
        nativeEngine.clearReferenceRun()
        actionView.text = "NOW: DON'T PRESS ANY KEY. Line up on the bullseye"
        statusView.text = capturePrompt()
        detectionView.text = "PHONE: Capture reset. Line up on the bullseye"
        updateCaptureButton()
    }

    /** Every launch starts at the reference run; delete what earlier versions saved. */
    private fun discardSavedMeasurement() {
        deleteSharedPreferences("measurement")
        java.io.File(filesDir, "reference-run.bin").delete()
    }

    /** Keeps the no-lens run as the reference and moves on to measuring the lens size. */
    private fun finishReferenceRun() {
        showCaptureQualitySummary()
        val down = lastMonitorDown
        val periods = nativeEngine.saveReferenceRun(down[0], down[1])
        if (periods == null) {
            actionView.text = "NOW: The reference run couldn't be saved. Tap “Reset capture”"
            return
        }
        screenDistanceMm = alignmentSamples.lastOrNull()?.depthMm ?: TARGET_WORKING_DISTANCE_MM
        val finest = periods.filter { it > 0f }.minOrNull()
        val fringeWarning = if (finest != null && finest < MIN_FRINGE_PERIOD_PIXELS) {
            String.format(
                Locale.US,
                " · Fringes are only %.1f camera pixels apart: for lens runs raise " +
                    "“Fringe period” on the page about %.0f× and redo the reference (Reset).",
                finest, ceil(MIN_FRINGE_PERIOD_PIXELS / finest)
            )
        } else {
            ""
        }
        captureComplete = false
        captureAxis = AXIS_X
        capturePhase = 0
        collectedFrames = 0
        stage = Stage.LENS_SIZE
        overlayView.update(null)
        actionView.text = "NOW: Reference done. ${lensSizePrompt()}"
        statusView.text = "KEYBOARD: Press L for the white lens-size screen"
        detectionView.text = "PHONE: Leave the phone where it is. ${detectionView.text}$fringeWarning"
    }

    /** Runs on the camera thread when “Measure lens size” was tapped. */
    private fun processLensSizeFrame(cameraImage: Image, monitorDown: FloatArray) {
        // Measured continuously, because the user's hands are busy holding the glasses on the
        // screen; accepted once the same lens-sized opening has been seen for about a second.
        val target = lastAlignmentTarget ?: return
        // Only on the white lens-size screen, which says so with its own step squares; anything
        // else (e.g. the bullseye's ring) could pass for a lens opening.
        if (decodeRemoteSquares(cameraImage, target, monitorDown) != LENS_SIZE_SCREEN_CODE) {
            lensOpeningSamples.clear()
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
        val opening = nativeEngine.measureLensOpening(
            yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
            yPlane.rowStride, yPlane.pixelStride,
            target.centerX * cameraImage.width, target.centerY * cameraImage.height,
            bullseyePixels * MIN_LENS_OPENING_MM / BULLSEYE_DIAMETER_MM
        )
        val openingMm = opening * BULLSEYE_DIAMETER_MM / bullseyePixels
        // Without glasses the only opening around the cross is the whole screen: too big.
        val lensSized = opening > 0f && openingMm <= MAX_LENS_OPENING_MM
        if (lensSized) lensOpeningSamples.addLast(opening) else lensOpeningSamples.clear()
        while (lensOpeningSamples.size > LENS_SIZE_STABLE_FRAMES) lensOpeningSamples.removeFirst()

        val stable = lensOpeningSamples.size == LENS_SIZE_STABLE_FRAMES &&
            lensOpeningSamples.maxOrNull()!! - lensOpeningSamples.minOrNull()!! <=
            LENS_SIZE_TOLERANCE * lensOpeningSamples.average()
        if (stable || (lensSizeRequested && lensSized)) {
            lensSizeRequested = false
            val accepted = lensOpeningSamples.sorted()[lensOpeningSamples.size / 2]
            lensOpeningSamples.clear()
            Log.d(LOG_TAG, "lens size accepted: $accepted px (${accepted * BULLSEYE_DIAMETER_MM / bullseyePixels} mm)")
            runOnUiThread {
                lensSizeMeasured(accepted, accepted * BULLSEYE_DIAMETER_MM / bullseyePixels)
            }
            return
        }
        if (lensSizeRequested && !lensSized) {
            lensSizeRequested = false
            saveGrayscaleFrame(cameraImage, "lens-size-failed.pgm")
            runOnUiThread { lensSizeMeasured(-1f, 0f) }
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
            Log.d(LOG_TAG, "saved ${file.absolutePath}")
        } catch (error: Exception) {
            Log.d(LOG_TAG, "could not save frame: ${error.localizedMessage}")
        }
    }

    /** Lens-size instructions; while on the right lens they keep the left result in view. */
    private fun lensSizePrompt(): String {
        val leftDone = if (currentEye == Eye.RIGHT && eyeResults[Eye.LEFT] != null) {
            "Left lens ${formatResult(Eye.LEFT)}. "
        } else {
            ""
        }
        return "${leftDone}${currentEye.label} lens: press L on the computer for the white " +
            "screen and hold the glasses flat on it with the ${currentEye.label.lowercase()} " +
            "lens centered on the cross. It measures by itself"
    }

    private fun lensSizeMeasured(openingPixels: Float, openingMm: Float) {
        if (openingPixels <= 0f) {
            actionView.text = "NOW: Couldn't find the lens rim around the cross. Check the white " +
                "screen is showing (L) and one lens is centered on the cross, then tap " +
                "“Measure lens size” again"
            return
        }
        screenOpeningPixels = openingPixels
        stage = Stage.LENS
        detectionView.text = String.format(
            Locale.US, "PHONE: %s lens opening ≈ %.0f mm across (seen from %d cm)",
            currentEye.label.lowercase().replaceFirstChar { it.uppercase() }, openingMm,
            screenDistanceMm / 10
        )
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
        autoArmPaused = false
        // Armed straight away: the lens run never sees the bullseye, so it can't wait for one.
        pendingCapture.set(ArmedCapture(CaptureRequest(AXIS_X, 0)))
        val lensDistanceCm = (screenDistanceMm / 4f / 10f).roundToInt()
        actionView.text = "NOW: ${currentEye.label} lens: PRESS R, then hold the glasses with " +
            "both hands about $lensDistanceCm cm in front of the camera, arms toward the phone, " +
            "${currentEye.label.lowercase()} lens centered on the camera. Hold steady; it " +
            "measures by itself"
        statusView.text = "KEYBOARD: R starts the lens loop (5-second countdown)"
        updateCaptureButton()
    }

    /** Compares the lens run with the reference; left lens first, then the right. */
    private fun finishLensRun() {
        val down = lastMonitorDown
        val result = nativeEngine.measureLensFromRuns(
            down[0], down[1], imageWidth / 2f, imageHeight / 2f,
            screenDistanceMm.toFloat(), screenOpeningPixels,
            java.io.File(getExternalFilesDir(null), "lens-brightness-ratio.pgm").absolutePath
        )
        val status = result?.getOrNull(0)?.toInt() ?: 5
        val eye = currentEye.label.lowercase()
        if (status != 0 || result == null) {
            // Re-arm the same lens so the fix is just R, then RETURN on the computer.
            startLensRun()
            val lensDistanceCm = (screenDistanceMm / 4f / 10f).roundToInt()
            actionView.text = "NOW: " + when (status) {
                1 -> "Couldn't find the $eye lens in the picture. Center it in front of the " +
                    "camera, about $lensDistanceCm cm away, and hold steady; it tries again " +
                    "when step 1 comes round"
                2 -> "The $eye lens outline runs off the picture or the rim isn't fully visible. " +
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
        eyeResults[currentEye] = floatArrayOf(quarter(result[1]), quarter(result[2]), result[3])
        detectionView.text = String.format(
            Locale.US,
            "PHONE: %s unrounded %+.2f %+.2f × %.0f · lens %.0f mm from camera · %d points · " +
                "fit error %.2f px · axis assumes the arms point toward the phone",
            currentEye.code, result[1], result[2], result[3], result[4], result[6].toInt(), result[7]
        )
        if (currentEye == Eye.LEFT) {
            currentEye = Eye.RIGHT
            stage = Stage.LENS_SIZE
            screenOpeningPixels = 0f
            captureComplete = false
            lensOpeningSamples.clear()
            actionView.text = "NOW: ${lensSizePrompt()}"
            statusView.text = "KEYBOARD: Press L (stops the loop, shows the white screen)"
            return
        }
        actionView.text = "DONE: OD (right) ${formatResult(Eye.RIGHT)}  ·  " +
            "OS (left) ${formatResult(Eye.LEFT)}"
        statusView.text = "KEYBOARD: Nothing more to press. “Reset capture” starts a new pair"
    }

    private fun formatResult(eye: Eye): String {
        val result = eyeResults[eye] ?: return "not measured"
        return String.format(
            Locale.US, "SPH %+.2f  CYL %+.2f  AXIS %d",
            result[0], result[1], result[2].roundToInt()
        )
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
        "KEYBOARD: No key yet. Line up the phone, then tap Prepare"
    }

    private fun updateCaptureButton() {
        val armed = pendingCapture.get() != null
        val step = phaseCode(CaptureRequest(captureAxis, capturePhase)) - 7
        captureButton.text = when {
            captureComplete -> "Capture complete"
            armed -> "Step $step/8: follow the KEYBOARD line"
            else -> "Prepare step $step/8 (${phaseName(captureAxis, capturePhase)})"
        }
        captureButton.isEnabled = !captureComplete && !armed && alignmentReady
        cancelPhaseButton.visibility = if (armed) ViewGroup.VISIBLE else ViewGroup.GONE
        measureSizeButton.visibility =
            if (stage == Stage.LENS_SIZE) ViewGroup.VISIBLE else ViewGroup.GONE
        repeatLensButton.visibility =
            if (stage == Stage.LENS && captureComplete) ViewGroup.VISIBLE else ViewGroup.GONE
    }

    private fun phaseName(axis: Int, phase: Int): String {
        val axisName = if (axis == AXIS_X) "X" else "Y"
        return "$axisName ${PHASE_LABELS[phase]}"
    }

    private fun phaseName(stepIndex: Int): String = phaseName(stepIndex / 4, stepIndex % 4)

    /** Matches the 4-bit code drawn under the bullseye by monitor-pattern-player.html. */
    private fun phaseCode(request: CaptureRequest): Int = 8 + request.axis * 4 + request.phase

    private companion object {
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
        /** Hand-held glasses: grey-level change per frame that counts as moving. */
        const val MOTION_LIMIT = 5f
        const val MOTION_SAMPLE_STEP = 8
        /** Frames after a page flip during which change is the monitor, not the hands. */
        const val PAGE_SETTLE_FRAMES = 3
        /** Still frames (about a third of a second) before frames are collected. */
        const val STEADY_FRAMES_NEEDED = 10
        /** Must match LENS_SIZE_SCREEN_CODE in monitor-pattern-player.html. */
        const val LENS_SIZE_SCREEN_CODE = 1
        const val DIAGNOSTIC_FRAME_INTERVAL_MS = 5000L
        /** Bullseye shift (fraction of its size, ~2.5 mm) that means the setup has moved. */
        const val SETUP_MOVED_FRACTION = 0.05f
        /** Frames averaged per fringe page; fewer if auto-advance moves on first. */
        const val FRAMES_PER_PAGE = 32
        /**
         * Must match REMOTE_TICKS_OFFSET_MM in monitor-pattern-player.html: the second set of
         * step squares is laid out as if under a bullseye this far below center (negative =
         * above), putting the squares about 10 cm above the screen center.
         */
        const val REMOTE_TICKS_OFFSET_MM = -145f
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