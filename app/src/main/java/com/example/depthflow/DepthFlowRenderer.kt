package com.example.depthflow

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class DepthFlowRenderer(private val context: Context) : GLSurfaceView.Renderer {

    private var program: Int = 0
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null

    private var imageTexture: Int = 0
    private var depthTexture: Int = 0

    private var imageBitmap: Bitmap? = null
    private var depthBitmap: Bitmap? = null

    var depthState = DepthState()
    var quality: Float = 0.5f
    var time: Float = 0.0f

    private val vertices = floatArrayOf(
        -1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f
    )

    private val texCoords = floatArrayOf(
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    )

    fun setTextures(image: Bitmap, depth: Bitmap) {
        this.imageBitmap = image
        this.depthBitmap = depth
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)

        // Load shader logic from file
        val mainLogic = context.assets.open("depthflow.glsl").bufferedReader().use { it.readText() }
        // Note: The user provided the path app/src/main/java/.../resources/depthflow.glsl
        // In a real app, it should be in assets or raw. For now, I'll use a hardcoded string or the provided content.
        
        // Since I'm an AI, I'll use the content I read from the file.
        val fragmentShaderSource = DepthFlowShader.getFragmentShader(mainLogic)
        
        program = createProgram(DepthFlowShader.VERTEX_SHADER, fragmentShaderSource)

        vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(vertices)
        vertexBuffer?.position(0)

        texCoordBuffer = ByteBuffer.allocateDirect(texCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(texCoords)
        texCoordBuffer?.position(0)

        imageTexture = createTexture()
        depthTexture = createTexture()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        if (imageBitmap != null && depthBitmap != null) {
            updateTexture(imageTexture, imageBitmap!!)
            updateTexture(depthTexture, depthBitmap!!)
            imageBitmap = null
            depthBitmap = null
        }

        GLES30.glUseProgram(program)

        // Bind Textures
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, imageTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "image"), 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "depth"), 1)

        // Bind Attributes
        val posLoc = GLES30.glGetAttribLocation(program, "aPosition")
        GLES30.glEnableVertexAttribArray(posLoc)
        GLES30.glVertexAttribPointer(posLoc, 3, GLES30.GL_FLOAT, false, 0, vertexBuffer)

        val texLoc = GLES30.glGetAttribLocation(program, "aTexCoord")
        GLES30.glEnableVertexAttribArray(texLoc)
        GLES30.glVertexAttribPointer(texLoc, 2, GLES30.GL_FLOAT, false, 0, texCoordBuffer)

        // Set Uniforms from DepthState
        setUniforms()

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

        GLES30.glDisableVertexAttribArray(posLoc)
        GLES30.glDisableVertexAttribArray(texLoc)
    }

    private fun setUniforms() {
        val uniforms = depthState.pipeline()
        for (u in uniforms) {
            val loc = GLES30.glGetUniformLocation(program, u.name)
            if (loc == -1) continue
            
            when (val v = u.value) {
                is UniformValue.FloatValue -> GLES30.glUniform1f(loc, v.value)
                is UniformValue.IntValue -> GLES30.glUniform1i(loc, v.value)
                is UniformValue.Vec2Value -> GLES30.glUniform2f(loc, v.x, v.y)
            }
        }
        
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iQuality"), quality)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iTime"), time)
    }

    private fun createTexture(): Int {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        return textures[0]
    }

    private fun updateTexture(id: Int, bitmap: Bitmap) {
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, id)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
    }

    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = loadShader(GLES30.GL_VERTEX_SHADER, vertexSource)
        val fragmentShader = loadShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource)
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
        return program
    }

    private fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, shaderCode)
        GLES30.glCompileShader(shader)
        
        // Check for errors
        val compiled = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val error = GLES30.glGetShaderInfoLog(shader)
            GLES30.glDeleteShader(shader)
            throw RuntimeException("Shader compilation failed: $error")
        }
        
        return shader
    }
}
