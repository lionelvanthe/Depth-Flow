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

    // Throttle sensor → only re-render when offset changes meaningfully
    private val sensorListener = object : SensorEventListener {
        private var lastLogTime = 0L
        private var lastRoll = 0f
        private var lastPitch = 0f
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR ||
                event.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR) {
                val rotationMatrix = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                val orientation = FloatArray(3)
                SensorManager.getOrientation(rotationMatrix, orientation)

                val pitch = orientation[1]
                val roll  = orientation[2]

                // Only request a new frame if the offset changed by at least 0.002
                val dx = kotlin.math.abs(roll  - lastRoll)
                val dy = kotlin.math.abs(pitch - lastPitch)
                if (dx > 0.002f || dy > 0.002f) {
                    renderer.setOffset(-roll * 1.2f, -pitch * 1.2f)
                    glSurfaceView.requestRender()
                    lastRoll  = roll
                    lastPitch = pitch

                    if (System.currentTimeMillis() - lastLogTime > 1000) {
                        android.util.Log.d("Parallax", "Sensor: roll=$roll, pitch=$pitch")
                        lastLogTime = System.currentTimeMillis()
                    }
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
        // RENDERMODE_WHEN_DIRTY: only draw when we explicitly call requestRender()
        // This is the single biggest power/heat optimization — idle = 0 GPU work
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY

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
            // SENSOR_DELAY_UI (~60ms) is sufficient for parallax; GAME (~20ms) is overkill
            sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_UI)
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

        // ── Cached uniform locations (populated once after shader link) ──────────
        private var uResolution     = -1; private var uAspectRatio    = -1
        private var uImageAspect    = -1  // separate: image w/h for center-crop sampling
        private var uViewportAspect = -1; private var uWantAspect     = -1
        private var uQuality        = -1; private var uCameraMode     = -1
        private var uCameraProj     = -1; private var uCameraPos      = -1
        private var uCameraFwd      = -1; private var uCameraUp       = -1
        private var uCameraRight    = -1; private var uCameraZenith   = -1
        private var uCameraFocal    = -1; private var uCameraZoom     = -1
        private var uCameraOrbit    = -1; private var uCameraDolly    = -1
        private var uCameraSep      = -1; private var uCameraIso      = -1
        private var uDepthOffset    = -1; private var uDepthHeight    = -1
        private var uDepthSteady    = -1; private var uDepthZoom      = -1
        private var uDepthIso       = -1; private var uDepthDolly     = -1
        private var uDepthFocus     = -1; private var uDepthCenter    = -1
        private var uDepthOrigin    = -1; private var uTau            = -1
        private var uRealtime       = -1; private var uFramerate      = -1
        private var uInpaint        = -1; private var uLensIntensity  = -1
        private var uLensQuality    = -1; private var uBlurIntensity  = -1
        private var uVigIntensity   = -1; private var uSaturation     = -1
        private var uContrast       = -1; private var uBrightness     = -1
        private var uSepia          = -1
        private var uImage          = -1; private var uDepthMap       = -1
        private var uTime           = -1  // For GPU-side auto-oscillation
        // ─────────────────────────────────────────────────────────────────────────

        private val vertices = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val startTime = System.currentTimeMillis()

        init {
            vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(vertices)
            vertexBuffer.position(0)
        }

        fun setBitmaps(image: Bitmap, depth: Bitmap) {
            imageBitmap = image
            depthBitmap = depth
            imageAspectRatio = image.width.toFloat() / image.height.toFloat()
            // Wake up the renderer once to upload new textures
            glSurfaceView.requestRender()
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

            // ── Use cached uniform locations (no string lookup per frame) ────────
            // Auto-oscillation time passed as a float uniform → GPU does sin/cos
            val timeSec = (System.currentTimeMillis() - startTime) / 1000f

            GLES30.glUniform2f(uResolution,     viewportWidth.toFloat(), viewportHeight.toFloat())
            // iAspectRatio = viewport aspect → camera.glsl uses it for coordinate space
            // iImageAspect = image aspect    → used by our center-crop gtexture override
            GLES30.glUniform1f(uAspectRatio,    viewportAspectRatio)
            GLES30.glUniform1f(uImageAspect,    imageAspectRatio)
            GLES30.glUniform1f(uViewportAspect, viewportAspectRatio)
            GLES30.glUniform1f(uWantAspect,     viewportAspectRatio)
            GLES30.glUniform1f(uQuality,        0.5f)  // Balanced: lower heat, still looks good
            GLES30.glUniform1i(uCameraMode,     1)
            GLES30.glUniform1i(uCameraProj,     0)
            GLES30.glUniform3f(uCameraPos,      0f, 0f, 0f)
            GLES30.glUniform3f(uCameraFwd,      0f, 0f, 1f)
            GLES30.glUniform3f(uCameraUp,       0f, 1f, 0f)
            GLES30.glUniform3f(uCameraRight,    1f, 0f, 0f)
            GLES30.glUniform3f(uCameraZenith,   0f, 0f, 1f)
            GLES30.glUniform1f(uCameraFocal,    1.0f)
            GLES30.glUniform1f(uCameraZoom,     1.0f)
            GLES30.glUniform1f(uCameraOrbit,    0f)
            GLES30.glUniform1f(uCameraDolly,    0f)
            GLES30.glUniform1f(uCameraSep,      0f)
            GLES30.glUniform1f(uCameraIso,      0f)

            // Pass sensor offset; GPU uniform iTime handles the idle oscillation
            GLES30.glUniform2f(uDepthOffset,    offset.x, offset.y)
            GLES30.glUniform1f(uDepthHeight,    0.2f)
            GLES30.glUniform1f(uDepthSteady,    0.5f)
            GLES30.glUniform1f(uDepthZoom,      1.0f)
            GLES30.glUniform1f(uDepthIso,       0f)
            GLES30.glUniform1f(uDepthDolly,     0f)
            GLES30.glUniform1f(uDepthFocus,     0.5f)
            GLES30.glUniform2f(uDepthCenter,    0f, 0f)
            GLES30.glUniform2f(uDepthOrigin,    0f, 0f)

            GLES30.glUniform1f(uTau,            0.0f)
            GLES30.glUniform1i(uRealtime,       1)
            GLES30.glUniform1f(uFramerate,      60.0f)
            GLES30.glUniform1f(uInpaint,        0f)
            GLES30.glUniform1f(uLensIntensity,  0f)
            GLES30.glUniform1f(uLensQuality,    1f)
            GLES30.glUniform1f(uBlurIntensity,  0f)
            GLES30.glUniform1f(uVigIntensity,   0f)
            GLES30.glUniform1f(uSaturation,     1f)
            GLES30.glUniform1f(uContrast,       1f)
            GLES30.glUniform1f(uBrightness,     1f)
            GLES30.glUniform1f(uSepia,          0f)
            GLES30.glUniform1f(uTime,           timeSec)

            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLES30.glUniform1i(uImage, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
            GLES30.glUniform1i(uDepthMap, 1)

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

            // Patch the gtexture function in shaderflow to do proper center-crop:
            // Replace it in-place so GLSL sees only one definition (no 'already has a body').
            val centerCropFunc = """
                // Center-crop: fills viewport while maintaining image aspect ratio
                vec4 gtexture(sampler2D image, vec2 gluvIn) {
                    float scale = iAspectRatio / iImageAspect;
                    vec2 uv;
                    if (scale >= 1.0) {
                        uv.x = (gluvIn.x / iAspectRatio) * 0.5 + 0.5;
                        uv.y = (gluvIn.y * scale) * 0.5 + 0.5;
                    } else {
                        uv.x = (gluvIn.x / iImageAspect) * 0.5 + 0.5;
                        uv.y = gluvIn.y * 0.5 + 0.5;
                    }
                    return texture(image, uv);
                }
            """.trimIndent()

            val shaderFlow = loadAsset("shaders/include/shaderflow.glsl")
                .replace(Regex("#define iAspectRatio.*"), "//")
                .replace(
                    // Replace the original gtexture(image, gluv) body with center-crop version
                    Regex("""vec4 gtexture\(sampler2D image, vec2 gluv\) \{[\s\S]*?\}"""),
                    centerCropFunc
                )
            val cameraFlow = loadAsset("shaders/include/camera.glsl")
            val depthFlowLogic = loadAsset("depthflow.glsl")
                .replace("void main() {", "void main() { agluv = v_agluv; gluv = v_gluv; stuv = v_stuv; astuv = v_astuv; ")
                .replace("DepthMake(iCamera, iDepth, depth)", "DepthMake(iCamera, iDepth, uDepthMap)")
                .replace("for (int it=0; it<1000; it++)", "for (int it=0; it<60; it++)")

            val fShaderCode = "#version 300 es\n" +
                "precision highp float;\n" +
                "uniform vec2 iResolution; uniform float iAspectRatio; uniform float iImageAspect; uniform float iWantAspect; uniform float iViewportAspect;\n" +
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
                "uniform bool iRealtime; uniform float iTau; uniform float iFramerate; uniform float iTime;\n" +
                "in vec2 v_gluv; in vec2 v_agluv; in vec2 v_stuv; in vec2 v_astuv;\n" +
                "out vec4 fragColor;\n" +
                "vec2 agluv; vec2 gluv; vec2 stuv; vec2 astuv;\n" +
                shaderFlow + cameraFlow + depthFlowLogic

            program = createProgram(vShaderCode, fShaderCode)
            cacheUniformLocations()  // Cache all locations once after link
        }

        /** Look up all uniform locations exactly once after the program is linked. */
        private fun cacheUniformLocations() {
            fun loc(name: String) = GLES30.glGetUniformLocation(program, name)
            uResolution     = loc("iResolution");     uAspectRatio    = loc("iAspectRatio")
            uImageAspect    = loc("iImageAspect")  // image w/h for center-crop
            uViewportAspect = loc("iViewportAspect"); uWantAspect     = loc("iWantAspect")
            uQuality        = loc("iQuality");         uCameraMode     = loc("iCameraMode")
            uCameraProj     = loc("iCameraProjection");uCameraPos      = loc("iCameraPosition")
            uCameraFwd      = loc("iCameraForward");   uCameraUp       = loc("iCameraUpward")
            uCameraRight    = loc("iCameraRight");     uCameraZenith   = loc("iCameraZenith")
            uCameraFocal    = loc("iCameraFocalLength");uCameraZoom    = loc("iCameraZoom")
            uCameraOrbit    = loc("iCameraOrbital");   uCameraDolly    = loc("iCameraDolly")
            uCameraSep      = loc("iCameraSeparation");uCameraIso      = loc("iCameraIsometric")
            uDepthOffset    = loc("iDepthOffset");     uDepthHeight    = loc("iDepthHeight")
            uDepthSteady    = loc("iDepthSteady");     uDepthZoom      = loc("iDepthZoom")
            uDepthIso       = loc("iDepthIsometric");  uDepthDolly     = loc("iDepthDolly")
            uDepthFocus     = loc("iDepthFocus");      uDepthCenter    = loc("iDepthCenter")
            uDepthOrigin    = loc("iDepthOrigin");     uTau            = loc("iTau")
            uRealtime       = loc("iRealtime");        uFramerate      = loc("iFramerate")
            uInpaint        = loc("iInpaint");         uLensIntensity  = loc("iLensIntensity")
            uLensQuality    = loc("iLensQuality");     uBlurIntensity  = loc("iBlurIntensity")
            uVigIntensity   = loc("iVigIntensity");    uSaturation     = loc("iColorsSaturation")
            uContrast       = loc("iColorsContrast");  uBrightness     = loc("iColorsBrightness")
            uSepia          = loc("iColorsSepia");     uImage          = loc("image")
            uDepthMap       = loc("uDepthMap");        uTime           = loc("iTime")
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
