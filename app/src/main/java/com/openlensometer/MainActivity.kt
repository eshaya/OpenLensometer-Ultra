package com.openlensometer

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.media.Image
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

class MainActivity : ComponentActivity() {
    private lateinit var cameraView: ArCoreCameraView
    private lateinit var statusView: TextView
    private lateinit var detectionView: TextView
    private lateinit var captureButton: Button
    private lateinit var referenceView: TextView
    private val referenceGlassesStore by lazy { ReferenceGlassesStore(this) }
    private val nativeEngine by lazy { NativeLensEngine() }
    private var lastStatusUpdateMs = 0L
    private var arSession: Session? = null
    private var installRequested = false
    private var activityResumed = false
    private var captureAxis = AXIS_X
    private var capturePhase = 0
    private var captureComplete = false
    private val pendingCapture = AtomicReference<CaptureRequest?>(null)
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startArCore() else statusView.text = "Camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        statusView = TextView(this).apply {
            text = capturePrompt()
            textSize = 16f
            setPadding(16, 12, 16, 12)
        }
        detectionView = TextView(this).apply {
            text = "Waiting for camera"
            textSize = 14f
            setPadding(16, 8, 16, 8)
        }
        captureButton = Button(this).apply {
            setOnClickListener { requestPhaseCapture() }
        }
        referenceView = TextView(this).apply {
            textSize = 14f
            setPadding(16, 8, 16, 8)
        }
        val referenceButton = Button(this).apply {
            text = "Reference glasses"
            setOnClickListener { showReferenceGlassesDialog() }
        }
        val resetButton = Button(this).apply {
            text = "Reset capture"
            setOnClickListener { resetCapture() }
        }
        cameraView = ArCoreCameraView(this).apply {
            onFrame = ::processArCoreFrame
            onFrameError = { message ->
                runOnUiThread { detectionView.text = message }
            }
        }
        updateCaptureButton()

        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(statusView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
            addView(referenceView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
            addView(referenceButton, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
            addView(captureButton, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
            addView(resetButton, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
            addView(detectionView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
            addView(cameraView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
            ))
        })
        updateReferenceView()

        nativeEngine.resetPhaseCapture()
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
        var cameraImage: Image? = null
        var depthImage: Image? = null
        try {
            cameraImage = frame.acquireCameraImage()
            val yPlane = cameraImage.planes[0]
            val capture = pendingCapture.get()
            if (capture != null) {
                depthImage = try {
                    frame.acquireDepthImage16Bits()
                } catch (_: NotYetAvailableException) {
                    null
                }
                if (depthImage == null) {
                    statusView.post { statusView.text = "Waiting for valid ARCore depth" }
                    return
                }

                val depthCenter = depthAtCameraCenter(frame, cameraImage, depthImage)
                if (depthCenter <= 0) {
                    statusView.post { statusView.text = "Depth unavailable at the target; hold steady" }
                    return
                }

                val camera = frame.camera
                val intrinsics = camera.imageIntrinsics
                val focalLength = FloatArray(2)
                val principalPoint = FloatArray(2)
                intrinsics.getFocalLength(focalLength, 0)
                intrinsics.getPrincipalPoint(principalPoint, 0)

                if (!pendingCapture.compareAndSet(capture, null)) return
                val depthPlane = depthImage.planes[0]
                val capturedCount = nativeEngine.capturePhaseFrame(
                    yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                    yPlane.rowStride, yPlane.pixelStride,
                    depthPlane.buffer.slice(), depthImage.width, depthImage.height,
                    depthPlane.rowStride, depthPlane.pixelStride,
                    capture.axis, capture.phase,
                    focalLength[0], focalLength[1], principalPoint[0], principalPoint[1]
                )
                runOnUiThread { completePhaseCapture(capture, capturedCount) }
            } else {
                val now = System.currentTimeMillis()
                if (now - lastStatusUpdateMs >= STATUS_UPDATE_INTERVAL_MS) {
                    lastStatusUpdateMs = now
                    val corners = nativeEngine.detectCharucoCorners(
                        yPlane.buffer.slice(), cameraImage.width, cameraImage.height,
                        yPlane.rowStride, yPlane.pixelStride
                    )
                    statusView.post {
                        detectionView.text = when {
                            corners < 0 -> "Unsupported camera Y-plane layout"
                            corners == 0 -> "No ChArUco corners detected"
                            else -> "ChArUco corners detected: $corners"
                        }
                    }
                }
            }
        } catch (_: NotYetAvailableException) {
            statusView.post { detectionView.text = "ARCore camera frame warming up" }
        } catch (error: Exception) {
            statusView.post { detectionView.text = "Frame processing failed: ${error.localizedMessage}" }
        } finally {
            depthImage?.close()
            cameraImage?.close()
        }
    }

    private fun depthAtCameraCenter(frame: Frame, cameraImage: Image, depthImage: Image): Int {
        val cameraPoint = directFloatBuffer(
            cameraImage.width / 2f,
            cameraImage.height / 2f
        )
        val normalizedPoint = ByteBuffer.allocateDirect(2 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        frame.transformCoordinates2d(
            Coordinates2d.IMAGE_PIXELS,
            cameraPoint,
            Coordinates2d.TEXTURE_NORMALIZED,
            normalizedPoint
        )
        val normalizedX = normalizedPoint.get(0)
        val normalizedY = normalizedPoint.get(1)
        if (normalizedX !in 0f..1f || normalizedY !in 0f..1f) return -1

        val centerX = (normalizedX * depthImage.width).toInt().coerceIn(0, depthImage.width - 1)
        val centerY = (normalizedY * depthImage.height).toInt().coerceIn(0, depthImage.height - 1)
        val plane = depthImage.planes[0]
        val buffer = plane.buffer.duplicate().order(ByteOrder.nativeOrder())
        val baseOffset = buffer.position()
        val samples = ArrayList<Int>(25)

        for (dy in -2..2) {
            for (dx in -2..2) {
                val x = centerX + dx
                val y = centerY + dy
                if (x !in 0 until depthImage.width || y !in 0 until depthImage.height) continue
                val offset = baseOffset + y * plane.rowStride + x * plane.pixelStride
                if (offset >= 0 && offset + Short.SIZE_BYTES <= buffer.limit()) {
                    val sample = java.lang.Short.toUnsignedInt(buffer.getShort(offset))
                    if (sample > 0) samples.add(sample)
                }
            }
        }

        if (samples.size < 5) return -1
        samples.sort()
        return samples[samples.size / 2]
    }

    private fun directFloatBuffer(x: Float, y: Float): FloatBuffer {
        return ByteBuffer.allocateDirect(2 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(floatArrayOf(x, y))
            .apply { position(0) }
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
                hint = label
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
            form.addView(editText)
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
        if (captureComplete || !pendingCapture.compareAndSet(
                null, CaptureRequest(captureAxis, capturePhase)
            )
        ) return

        captureButton.isEnabled = false
        statusView.text = "Waiting for ${phaseName(captureAxis, capturePhase)} frame"
    }

    private fun completePhaseCapture(capture: CaptureRequest, capturedCount: Int) {
        if (capturedCount != capture.phase + 1) {
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
            statusView.text = "Captured ${phaseName(capture.axis, capture.phase)}. " +
                "Set the monitor to ${phaseName(captureAxis, capturePhase)}"
        } else if (capture.axis == AXIS_X) {
            captureAxis = AXIS_Y
            capturePhase = 0
            statusView.text = "X phase set processed. Switch the monitor to Y fringes."
        } else {
            captureComplete = true
            statusView.text = "X and Y phase sets captured and demodulated"
            val calibration = nativeEngine.getCapturedCameraDepthCalibration()
            detectionView.text = if (calibration != null && calibration.size >= 5) {
                "ARCore calibration: depth ${calibration[4].toInt()} mm; " +
                    "fx ${calibration[0].toInt()} px; fy ${calibration[1].toInt()} px"
            } else {
                "Captured phase maps, but camera/depth diagnostics are incomplete"
            }
        }

        updateCaptureButton()
    }

    private fun resetCapture() {
        pendingCapture.set(null)
        nativeEngine.resetPhaseCapture()
        captureAxis = AXIS_X
        capturePhase = 0
        captureComplete = false
        statusView.text = capturePrompt()
        updateCaptureButton()
    }

    private fun capturePrompt(): String = if (captureComplete) {
        "Phase capture complete"
    } else {
        "Set the monitor to ${phaseName(captureAxis, capturePhase)}"
    }

    private fun updateCaptureButton() {
        captureButton.text = if (captureComplete) {
            "Capture complete"
        } else {
            "Capture ${phaseName(captureAxis, capturePhase)}"
        }
        captureButton.isEnabled = !captureComplete && pendingCapture.get() == null
    }

    private fun phaseName(axis: Int, phase: Int): String {
        val axisName = if (axis == AXIS_X) "X" else "Y"
        return "$axisName phase ${PHASE_LABELS[phase]} (${phase + 1}/4)"
    }

    private companion object {
        const val AXIS_X = 0
        const val AXIS_Y = 1
        const val STATUS_UPDATE_INTERVAL_MS = 500L
        val PHASE_LABELS = listOf("0°", "90°", "180°", "270°")
    }

    private data class CaptureRequest(val axis: Int, val phase: Int)
}