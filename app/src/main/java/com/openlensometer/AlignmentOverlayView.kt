package com.openlensometer

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View

/**
 * Drawn over the camera preview: a fixed target square showing where the monitor's alignment
 * box should appear at the working distance (red, green once everything is aligned), the box
 * as the camera actually sees it (pink), and tilt/distance readings along the top, each red
 * or green. The user moves the phone until it all turns green.
 */
class AlignmentOverlayView(context: Context) : View(context) {

    class Reading(val label: String, val value: String, val inRange: Boolean)

    /** Lens step: where the lens should appear, and its size at the right distance (view px). */
    class LensGuide(val centerX: Float, val centerY: Float, val radius: Float, val measuring: Boolean)

    class State(
        /** Where the monitor's box should appear, in view pixels (null: not lining up). */
        val target: RectF?,
        /** Seen corners in view pixels: TL, TR, BR, BL as x,y pairs (NaN if missing), or null. */
        val monitorCorners: FloatArray?,
        val aligned: Boolean,
        val readings: List<Reading>,
        val lensGuide: LensGuide? = null
    )

    @Volatile private var state: State? = null

    private val density = resources.displayMetrics.density
    private val targetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
    }
    private val monitorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        color = Color.MAGENTA
    }
    private val backgroundPaint = Paint().apply { color = Color.argb(200, 0, 0, 0) }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 15f * density
        setShadowLayer(3f * density, 0f, 0f, Color.BLACK)
    }

    fun update(newState: State?) {
        state = newState
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val current = state ?: return
        current.lensGuide?.let { drawLensGuide(canvas, it) }
        val target = current.target ?: return
        val targetColor = if (current.aligned) Color.GREEN else Color.RED
        targetPaint.color = targetColor
        canvas.drawRect(target, targetPaint)
        val crossSize = 10f * density
        canvas.drawLine(
            target.centerX() - crossSize, target.centerY(),
            target.centerX() + crossSize, target.centerY(), targetPaint
        )
        canvas.drawLine(
            target.centerX(), target.centerY() - crossSize,
            target.centerX(), target.centerY() + crossSize, targetPaint
        )

        current.monitorCorners?.let { drawMonitorBox(canvas, it) }

        // Readings along the top of the picture (the wall above the monitor), on a dark band so
        // the fringes don't swamp them; each red (out of range) or green.
        if (current.readings.isNotEmpty()) {
            val lineHeight = labelPaint.textSize * 1.4f
            canvas.drawRect(0f, 0f, width.toFloat(), lineHeight * 1.5f, backgroundPaint)
            val slot = width.toFloat() / current.readings.size
            labelPaint.textAlign = Paint.Align.CENTER
            current.readings.forEachIndexed { index, reading ->
                labelPaint.color = if (reading.inRange) Color.GREEN else Color.RED
                canvas.drawText(
                    "${reading.label} ${reading.value}", slot * (index + 0.5f), lineHeight * 1.05f, labelPaint
                )
            }
            labelPaint.textAlign = Paint.Align.LEFT
        }
        labelPaint.color = Color.MAGENTA
        canvas.drawText(
            "Pink = monitor box as the camera sees it; put it on the box",
            8f * density, height - labelPaint.textSize * 1.3f, labelPaint
        )
    }

    private fun drawLensGuide(canvas: Canvas, guide: LensGuide) {
        val color = if (guide.measuring) Color.GREEN else Color.CYAN
        targetPaint.color = color
        canvas.drawCircle(guide.centerX, guide.centerY, guide.radius, targetPaint)
        labelPaint.color = color
        labelPaint.textAlign = Paint.Align.CENTER
        canvas.drawText(
            if (guide.measuring) "Measuring: hold still" else "Fill this circle with the lens",
            guide.centerX, guide.centerY + guide.radius + labelPaint.textSize * 1.4f, labelPaint
        )
        labelPaint.textAlign = Paint.Align.LEFT
    }

    private fun drawMonitorBox(canvas: Canvas, corners: FloatArray) {
        // Connect neighbouring corners that were both found; a missing marker leaves a gap.
        for (index in 0 until 4) {
            val next = (index + 1) % 4
            val x0 = corners[index * 2]
            val y0 = corners[index * 2 + 1]
            val x1 = corners[next * 2]
            val y1 = corners[next * 2 + 1]
            if (x0.isFinite() && y0.isFinite() && x1.isFinite() && y1.isFinite()) {
                canvas.drawLine(x0, y0, x1, y1, monitorPaint)
            }
        }
        val dots = Path()
        for (index in 0 until 4) {
            val x = corners[index * 2]
            val y = corners[index * 2 + 1]
            if (x.isFinite() && y.isFinite()) dots.addCircle(x, y, 5f * density, Path.Direction.CW)
        }
        canvas.drawPath(dots, monitorPaint)
    }
}
