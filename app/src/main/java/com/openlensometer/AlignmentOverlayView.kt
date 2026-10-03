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
 * box should appear at the working distance, and the box as the camera actually sees it.
 * The user moves the phone until the two match.
 */
class AlignmentOverlayView(context: Context) : View(context) {

    class State(
        /** Where the monitor's box should appear, in view pixels. */
        val target: RectF,
        /** Seen corners in view pixels: TL, TR, BR, BL as x,y pairs (NaN if missing), or null. */
        val monitorCorners: FloatArray?,
        val aligned: Boolean
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
        val targetColor = if (current.aligned) Color.GREEN else Color.WHITE
        targetPaint.color = targetColor
        canvas.drawRect(current.target, targetPaint)
        val crossSize = 10f * density
        canvas.drawLine(
            current.target.centerX() - crossSize, current.target.centerY(),
            current.target.centerX() + crossSize, current.target.centerY(), targetPaint
        )
        canvas.drawLine(
            current.target.centerX(), current.target.centerY() - crossSize,
            current.target.centerX(), current.target.centerY() + crossSize, targetPaint
        )

        current.monitorCorners?.let { drawMonitorBox(canvas, it) }

        val lineHeight = labelPaint.textSize * 1.3f
        labelPaint.color = targetColor
        canvas.drawText(
            if (current.aligned) "Boxes match. Hold still" else "Goal: same size and place as the white box",
            8f * density, height - lineHeight, labelPaint
        )
        labelPaint.color = Color.MAGENTA
        canvas.drawText(
            "Pink = monitor box as the camera sees it",
            8f * density, height - 2 * lineHeight, labelPaint
        )
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
