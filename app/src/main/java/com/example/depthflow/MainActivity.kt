package com.example.depthflow

import android.content.Context
import android.os.Bundle
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : AppCompatActivity() {
    private var realEstimator: com.example.depthflow.estimators.OnnxDepthEstimator? = null
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: ParallaxRenderer
    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null

    private val sensorListener = object : SensorEventListener {
        private var lastLogTime = 0L
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR || 
                event.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR) {
                val rotationMatrix = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                val orientation = FloatArray(3)
                SensorManager.getOrientation(rotationMatrix, orientation)
                
                // orientation[1] is pitch (tilt front/back), orientation[2] is roll (tilt left/right)
                val pitch = orientation[1]
                val roll = orientation[2]
                
                // Very high sensitivity for testing
                renderer.setOffset(-roll * 1.2f, -pitch * 1.2f)

                if (System.currentTimeMillis() - lastLogTime > 1000) {
                    android.util.Log.d("Parallax", "Sensor: roll=$roll, pitch=$pitch")
                    lastLogTime = System.currentTimeMillis()
                }
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            findViewById<android.view.View>(R.id.progressBar).visibility = android.view.View.VISIBLE
            lifecycleScope.launch(Dispatchers.IO) {
                val bitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                    val source = android.graphics.ImageDecoder.createSource(contentResolver, it)
                    android.graphics.ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        decoder.isMutableRequired = true
                    }
                } else {
                    android.provider.MediaStore.Images.Media.getBitmap(contentResolver, it)
                }

                if (realEstimator == null) {
                    realEstimator = com.example.depthflow.estimators.OnnxDepthEstimator(this@MainActivity, "depth_anything_v2_small.onnx")
                }

                val depthMap = realEstimator?.estimate(bitmap) ?: bitmap
                
                withContext(Dispatchers.Main) {
                    findViewById<android.view.View>(R.id.progressBar).visibility = android.view.View.GONE
                    renderer.setBitmaps(bitmap, depthMap)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        glSurfaceView = findViewById(R.id.glSurfaceView)
        glSurfaceView.setEGLContextClientVersion(3)
        renderer = ParallaxRenderer(this)
        glSurfaceView.setRenderer(renderer)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

        findViewById<com.google.android.material.floatingactionbutton.FloatingActionButton>(R.id.fabPickImage).setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    override fun onResume() {
        super.onResume()
        glSurfaceView.onResume()
        rotationSensor?.let {
            sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        glSurfaceView.onPause()
        sensorManager.unregisterListener(sensorListener)
    }

    inner class ParallaxRenderer(private val context: Context) : GLSurfaceView.Renderer {
        private var program: Int = 0
        private var textureId: Int = 0
        private var depthTextureId: Int = 0
        private var vertexBuffer: FloatBuffer
        private var imageBitmap: Bitmap? = null
        private var depthBitmap: Bitmap? = null
        private var offset = PointF(0f, 0f)
        private var imageAspectRatio: Float = 1f
        private var viewportAspectRatio: Float = 1f
        private var viewportWidth: Int = 1080
        private var viewportHeight: Int = 1920

        private val vertices = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)

        init {
            vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(vertices)
            vertexBuffer.position(0)
        }

        fun setBitmaps(image: Bitmap, depth: Bitmap) {
            imageBitmap = image
            depthBitmap = depth
            imageAspectRatio = image.width.toFloat() / image.height.toFloat()
        }

        fun setOffset(x: Float, y: Float) {
            offset.set(x, y)
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES30.glClearColor(0f, 0f, 0f, 1f)
            setupShaders()
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            GLES30.glViewport(0, 0, width, height)
            viewportWidth = width
            viewportHeight = height
            viewportAspectRatio = width.toFloat() / height.toFloat()
        }

        override fun onDrawFrame(gl: GL10?) {
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            imageBitmap?.let { updateTextures(); imageBitmap = null }
            if (program == 0 || textureId == 0 || depthTextureId == 0) return
            GLES30.glUseProgram(program)

            // Add a tiny auto-oscillation for debugging
            val time = System.currentTimeMillis() % 10000 / 1000f
            val autoX = Math.sin(time.toDouble()).toFloat() * 0.05f
            val autoY = Math.cos(time.toDouble()).toFloat() * 0.05f

            GLES30.glUniform2f(GLES30.glGetUniformLocation(program, "iResolution"), viewportWidth.toFloat(), viewportHeight.toFloat())
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iAspectRatio"), imageAspectRatio)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iViewportAspect"), viewportAspectRatio)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iWantAspect"), viewportAspectRatio)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iQuality"), 0.8f)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "iCameraMode"), 1)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "iCameraProjection"), 0)
            GLES30.glUniform3f(GLES30.glGetUniformLocation(program, "iCameraPosition"), 0f, 0f, 0f)
            GLES30.glUniform3f(GLES30.glGetUniformLocation(program, "iCameraForward"), 0f, 0f, 1f)
            GLES30.glUniform3f(GLES30.glGetUniformLocation(program, "iCameraUpward"), 0f, 1f, 0f)
            GLES30.glUniform3f(GLES30.glGetUniformLocation(program, "iCameraRight"), 1f, 0f, 0f)
            GLES30.glUniform3f(GLES30.glGetUniformLocation(program, "iCameraZenith"), 0f, 0f, 1f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iCameraFocalLength"), 1.0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iCameraZoom"), 1.0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iCameraOrbital"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iCameraDolly"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iCameraSeparation"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iCameraIsometric"), 0f)

            GLES30.glUniform2f(GLES30.glGetUniformLocation(program, "iDepthOffset"), offset.x + autoX, offset.y + autoY)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iDepthHeight"), 0.4f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iDepthSteady"), 0.5f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iDepthZoom"), 1.0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iDepthIsometric"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iDepthDolly"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iDepthFocus"), 0.5f)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(program, "iDepthCenter"), 0f, 0f)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(program, "iDepthOrigin"), 0f, 0f)

            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iTau"), 0.0f)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "iRealtime"), 1)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iFramerate"), 60.0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iInpaint"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iLensIntensity"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iLensQuality"), 1f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iBlurIntensity"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iVigIntensity"), 0f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iColorsSaturation"), 1f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iColorsContrast"), 1f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iColorsBrightness"), 1f)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "iColorsSepia"), 0f)

            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "image"), 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uDepthMap"), 1)

            val posHandle = GLES30.glGetAttribLocation(program, "a_Position")
            GLES30.glEnableVertexAttribArray(posHandle)
            GLES30.glVertexAttribPointer(posHandle, 2, GLES30.GL_FLOAT, false, 0, vertexBuffer)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
            GLES30.glDisableVertexAttribArray(posHandle)
        }

        private fun updateTextures() {
            if (textureId != 0) GLES30.glDeleteTextures(1, intArrayOf(textureId), 0)
            textureId = loadTexture(imageBitmap!!)
            if (depthTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(depthTextureId), 0)
            depthTextureId = loadTexture(depthBitmap!!)
        }

        private fun loadTexture(bitmap: Bitmap): Int {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
            return textures[0]
        }

        private fun setupShaders() {
            val vShaderCode = "#version 300 es\n" +
                "layout(location = 0) in vec2 a_Position;\n" +
                "out vec2 v_gluv; out vec2 v_agluv; out vec2 v_stuv; out vec2 v_astuv;\n" +
                "uniform float iViewportAspect;\n" +
                "void main() {\n" +
                "    gl_Position = vec4(a_Position, 0.0, 1.0);\n" +
                "    v_agluv = a_Position * vec2(1.0, -1.0);\n" + // Flip Y for Android Bitmaps
                "    v_gluv = v_agluv * vec2(iViewportAspect, 1.0);\n" +
                "    v_astuv = (v_agluv + 1.0) / 2.0;\n" +
                "    v_stuv = (v_gluv + 1.0) / 2.0;\n" +
                "}\n"

            fun loadAsset(path: String): String {
                return context.assets.open(path).bufferedReader().use { it.readText() }
                    .replace(Regex("#version.*"), "")
                    .replace("#ifndef", "//#ifndef")
                    .replace("#endif", "//#endif")
                    .trim() + "\n"
            }

            val shaderFlow = loadAsset("shaders/include/shaderflow.glsl")
                .replace(Regex("#define iAspectRatio.*"), "//")
            val cameraFlow = loadAsset("shaders/include/camera.glsl")
            val depthFlowLogic = loadAsset("depthflow.glsl")
                .replace("void main() {", "void main() { agluv = v_agluv; gluv = v_gluv; stuv = v_stuv; astuv = v_astuv; ")
                .replace("DepthMake(iCamera, iDepth, depth)", "DepthMake(iCamera, iDepth, uDepthMap)")
                .replace("for (int it=0; it<1000; it++)", "for (int it=0; it<100; it++)")

            val fShaderCode = "#version 300 es\n" +
                "precision highp float;\n" +
                "uniform vec2 iResolution; uniform float iAspectRatio; uniform float iWantAspect; uniform float iViewportAspect;\n" +
                "uniform sampler2D image; uniform sampler2D uDepthMap; uniform float iQuality;\n" +
                "uniform int iCameraMode; uniform int iCameraProjection; uniform vec3 iCameraPosition;\n" +
                "uniform float iCameraOrbital; uniform float iCameraDolly; uniform vec3 iCameraZenith;\n" +
                "uniform vec3 iCameraUpward; uniform vec3 iCameraRight; uniform vec3 iCameraForward;\n" +
                "uniform float iCameraFocalLength; uniform float iCameraZoom; uniform float iCameraSeparation; uniform float iCameraIsometric;\n" +
                "uniform float iDepthIsometric; uniform float iDepthDolly; uniform float iDepthZoom;\n" +
                "uniform vec2 iDepthOffset; uniform float iDepthHeight; uniform float iDepthFocus;\n" +
                "uniform vec2 iDepthCenter; uniform float iDepthSteady; uniform vec2 iDepthOrigin;\n" +
                "uniform float iInpaint; uniform float iLensIntensity; uniform float iLensDecay; uniform float iLensQuality;\n" +
                "uniform float iBlurIntensity; uniform float iBlurStart; uniform float iBlurEnd; uniform float iBlurExponent;\n" +
                "uniform float iBlurDirections; uniform float iBlurQuality; uniform float iVigIntensity; uniform float iVigDecay;\n" +
                "uniform float iColorsSaturation; uniform float iColorsContrast; uniform float iColorsBrightness; uniform float iColorsSepia;\n" +
                "uniform bool iRealtime; uniform float iTau; uniform float iFramerate;\n" +
                "in vec2 v_gluv; in vec2 v_agluv; in vec2 v_stuv; in vec2 v_astuv;\n" +
                "out vec4 fragColor;\n" +
                "vec2 agluv; vec2 gluv; vec2 stuv; vec2 astuv;\n" +
                shaderFlow + cameraFlow + depthFlowLogic

            program = createProgram(vShaderCode, fShaderCode)
        }

        private fun createProgram(vSource: String, fSource: String): Int {
            val vShader = loadShader(GLES30.GL_VERTEX_SHADER, vSource)
            val fShader = loadShader(GLES30.GL_FRAGMENT_SHADER, fSource)
            val prog = GLES30.glCreateProgram()
            GLES30.glAttachShader(prog, vShader)
            GLES30.glAttachShader(prog, fShader)
            GLES30.glLinkProgram(prog)
            val linkStatus = IntArray(1)
            GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, linkStatus, 0)
            if (linkStatus[0] == 0) {
                android.util.Log.e("Parallax", "Link failed: " + GLES30.glGetProgramInfoLog(prog))
                GLES30.glDeleteProgram(prog)
                return 0
            }
            return prog
        }

        private fun loadShader(type: Int, source: String): Int {
            val shader = GLES30.glCreateShader(type)
            GLES30.glShaderSource(shader, source)
            GLES30.glCompileShader(shader)
            val compiled = IntArray(1)
            GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
            if (compiled[0] == 0) {
                android.util.Log.e("Parallax", "Compile failed: " + GLES30.glGetShaderInfoLog(shader))
                GLES30.glDeleteShader(shader)
                return 0
            }
            return shader
        }
    }
}
