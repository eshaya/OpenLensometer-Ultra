package com.openlensometer

import android.content.Context
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.Coordinates2d
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class ArCoreCameraView(context: Context, attrs: AttributeSet? = null) :
    GLSurfaceView(context, attrs), GLSurfaceView.Renderer {

    var onFrame: ((Frame) -> Unit)? = null
    var onFrameError: ((String) -> Unit)? = null

    private var session: Session? = null
    private var cameraTextureId = 0
    private var program = 0
    private var positionLocation = 0
    private var textureCoordinateLocation = 0
    private var textureLocation = 0
    private var surfaceWidth = 0
    private var surfaceHeight = 0

    init {
        setEGLContextClientVersion(2)
        setRenderer(this)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setSession(session: Session) {
        this.session = session
        queueEvent {
            if (cameraTextureId != 0) session.setCameraTextureName(cameraTextureId)
        }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        cameraTextureId = createCameraTexture()
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionLocation = GLES20.glGetAttribLocation(program, "a_Position")
        textureCoordinateLocation = GLES20.glGetAttribLocation(program, "a_TexCoord")
        textureLocation = GLES20.glGetUniformLocation(program, "sTexture")
        session?.setCameraTextureName(cameraTextureId)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceWidth = width
        surfaceHeight = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        val activeSession = session ?: return
        try {
            activeSession.setDisplayGeometry(display?.rotation ?: 0, surfaceWidth, surfaceHeight)
            val frame = activeSession.update()
            onFrame?.invoke(frame)
            drawCameraBackground(frame)
        } catch (error: Exception) {
            onFrameError?.invoke(error.localizedMessage ?: "ARCore camera frame failed")
        }
    }

    private fun drawCameraBackground(frame: Frame) {
        val vertices = floatBufferOf(
            -1f, -1f, 1f, -1f,
            -1f, 1f, 1f, 1f
        )
        val textureCoordinates = FloatBuffer.allocate(8)
        frame.transformCoordinates2d(
            Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
            vertices.duplicate(),
            Coordinates2d.TEXTURE_NORMALIZED,
            textureCoordinates
        )

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES20.glUniform1i(textureLocation, 0)

        vertices.position(0)
        GLES20.glVertexAttribPointer(positionLocation, 2, GLES20.GL_FLOAT, false, 0, vertices)
        GLES20.glEnableVertexAttribArray(positionLocation)
        textureCoordinates.position(0)
        GLES20.glVertexAttribPointer(
            textureCoordinateLocation, 2, GLES20.GL_FLOAT, false, 0, textureCoordinates)
        GLES20.glEnableVertexAttribArray(textureCoordinateLocation)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(positionLocation)
        GLES20.glDisableVertexAttribArray(textureCoordinateLocation)
    }

    private fun createCameraTexture(): Int {
        val texture = IntArray(1)
        GLES20.glGenTextures(1, texture, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texture[0])
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_WRAP_S,
            GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_WRAP_T,
            GLES20.GL_CLAMP_TO_EDGE
        )
        return texture[0]
    }

    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertex = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource)
        val fragment = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
        return GLES20.glCreateProgram().also { result ->
            GLES20.glAttachShader(result, vertex)
            GLES20.glAttachShader(result, fragment)
            GLES20.glLinkProgram(result)
            val status = IntArray(1)
            GLES20.glGetProgramiv(result, GLES20.GL_LINK_STATUS, status, 0)
            check(status[0] == GLES20.GL_TRUE) {
                GLES20.glGetProgramInfoLog(result)
            }
            GLES20.glDeleteShader(vertex)
            GLES20.glDeleteShader(fragment)
        }
    }

    private fun compileShader(type: Int, source: String): Int {
        return GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, source)
            GLES20.glCompileShader(shader)
            val status = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
            check(status[0] == GLES20.GL_TRUE) {
                GLES20.glGetShaderInfoLog(shader)
            }
        }
    }

    private fun floatBufferOf(vararg values: Float): FloatBuffer {
        return ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(values)
            .apply { position(0) }
    }

    private companion object {
        const val VERTEX_SHADER =
            "attribute vec4 a_Position;" +
            "attribute vec2 a_TexCoord;" +
            "varying vec2 v_TexCoord;" +
            "void main(){ gl_Position=a_Position; v_TexCoord=a_TexCoord; }"

        const val FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float;" +
            "uniform samplerExternalOES sTexture;" +
            "varying vec2 v_TexCoord;" +
            "void main(){ gl_FragColor=texture2D(sTexture,v_TexCoord); }"
    }
}