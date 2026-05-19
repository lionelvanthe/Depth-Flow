package com.example.depthflow

import android.graphics.Bitmap
import android.graphics.PointF
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.cos
import kotlin.math.sin
import androidx.core.graphics.scale

class MainActivity : AppCompatActivity() {
    private var realEstimator: com.example.depthflow.estimators.OnnxDepthEstimator? = null
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: ParallaxRenderer
    private lateinit var parallax: Parallax

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            findViewById<android.view.View>(R.id.progressBar).visibility = android.view.View.VISIBLE
            lifecycleScope.launch(Dispatchers.IO) {
                val bitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                    val source = android.graphics.ImageDecoder.createSource(contentResolver, it)
                    android.graphics.ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                        decoder.isMutableRequired = true
                        // Optimization: Downscale if larger than 2048px to save RAM and processing time
                        val maxDim = 2048
                        if (info.size.width > maxDim || info.size.height > maxDim) {
                            val ratio = info.size.width.toFloat() / info.size.height.toFloat()
                            val targetW = if (ratio > 1f) maxDim else (maxDim * ratio).toInt()
                            val targetH = if (ratio > 1f) (maxDim / ratio).toInt() else maxDim
                            decoder.setTargetSize(targetW, targetH)
                        }
                    }
                } else {
                    val original = android.provider.MediaStore.Images.Media.getBitmap(contentResolver, it)
                    val maxDim = 2048
                    if (original.width > maxDim || original.height > maxDim) {
                        val ratio = original.width.toFloat() / original.height.toFloat()
                        val targetW = if (ratio > 1f) maxDim else (maxDim * ratio).toInt()
                        val targetH = if (ratio > 1f) (maxDim / ratio).toInt() else maxDim
                        val scaled = original.scale(targetW, targetH)
                        if (scaled != original) original.recycle()
                        scaled
                    } else original
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
        renderer = ParallaxRenderer()
        glSurfaceView.setRenderer(renderer)
        // RENDERMODE_WHEN_DIRTY: only draw when we explicitly call requestRender()
        // This is the single biggest power/heat optimization — idle = 0 GPU work
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY

        parallax = Parallax(this).apply {
            setSensitivity(0.2) // Tương đương với filter factor cũ
            setFallback(0.01)   // Tốc độ hồi về tâm
            onUpdate = { degX, degY ->
                val rollRad = Math.toRadians(degX).toFloat()
                val pitchRad = Math.toRadians(degY).toFloat()
                renderer.setOffset(-rollRad * 1f, -pitchRad * 1f)
                glSurfaceView.requestRender()
            }
        }

        // Pre-load estimator to avoid delay on first run
        lifecycleScope.launch(Dispatchers.IO) {
            if (realEstimator == null) {
                realEstimator = com.example.depthflow.estimators.OnnxDepthEstimator(this@MainActivity, "depth_anything_v2_small.onnx")
            }
        }

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
        parallax.start()
    }

    override fun onPause() {
        super.onPause()
        glSurfaceView.onPause()
        parallax.stop()
    }

    inner class ParallaxRenderer : GLSurfaceView.Renderer {
        private var program: Int = 0
        private var textureId: Int = 0
        private var depthTextureId: Int = 0
        
        // Use a simple quad. Spatial Parallax is now handled in the Fragment Shader.
        private val GRID_SIZE = 2
        private lateinit var vertexBuffer: FloatBuffer
        private lateinit var indexBuffer: java.nio.IntBuffer
        private var indexCount: Int = 0

        private var imageBitmap: Bitmap? = null
        private var depthBitmap: Bitmap? = null
        private var needsUpdateTextures = false
        private var offset = PointF(0f, 0f)
        private var imageAspectRatio: Float = 1f
        private var viewportAspectRatio: Float = 1f

        // Cached uniform locations
        private var uAspectRatio    = -1
        private var uImageAspect    = -1
        private var uOffset         = -1
        private var uDepthHeight    = -1
        private var uTime           = -1
        private var uImage          = -1
        private var uDepthMap       = -1

        private val startTime = System.currentTimeMillis()

        init {
            generateGrid()
        }

        private fun generateGrid() {
            // Create a grid of vertices covering [-1.1, 1.1] to allow for displacement without edges showing
            val vertices = FloatArray(GRID_SIZE * GRID_SIZE * 2)
            var vIdx = 0
            val scale = 1.0f // Sửa thành 1.0f để vừa khít hoàn toàn ảnh gốc, không bị zoom
            for (y in 0 until GRID_SIZE) {
                val yPos = ((y.toFloat() / (GRID_SIZE - 1)) * 2f - 1f) * scale
                for (x in 0 until GRID_SIZE) {
                    val xPos = ((x.toFloat() / (GRID_SIZE - 1)) * 2f - 1f) * scale
                    vertices[vIdx++] = xPos
                    vertices[vIdx++] = yPos
                }
            }
            vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer().put(vertices)
            vertexBuffer.position(0)

            val indices = IntArray((GRID_SIZE - 1) * (GRID_SIZE - 1) * 6)
            var iIdx = 0
            for (y in 0 until GRID_SIZE - 1) {
                for (x in 0 until GRID_SIZE - 1) {
                    val topLeft = (y * GRID_SIZE + x)
                    val topRight = (topLeft + 1)
                    val bottomLeft = ((y + 1) * GRID_SIZE + x)
                    val bottomRight = (bottomLeft + 1)

                    indices[iIdx++] = topLeft
                    indices[iIdx++] = bottomLeft
                    indices[iIdx++] = topRight
                    indices[iIdx++] = topRight
                    indices[iIdx++] = bottomLeft
                    indices[iIdx++] = bottomRight
                }
            }
            indexCount = indices.size
            indexBuffer = ByteBuffer.allocateDirect(indices.size * 4)
                .order(ByteOrder.nativeOrder()).asIntBuffer().put(indices)
            indexBuffer.position(0)
        }

        fun setBitmaps(image: Bitmap, depth: Bitmap) {
            imageBitmap = image
            depthBitmap = depth
            imageAspectRatio = image.width.toFloat() / image.height.toFloat()
            needsUpdateTextures = true
            glSurfaceView.requestRender()
        }

        fun setOffset(x: Float, y: Float) {
            offset.set(x, y)
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES30.glClearColor(0f, 0f, 0f, 1f)
            setupShaders()
            // Reset texture IDs because the EGL context is new (previous textures are lost)
            textureId = 0
            depthTextureId = 0
            needsUpdateTextures = true
            // Request a render call to ensure the screen updates after restoration
            glSurfaceView.requestRender()
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            GLES30.glViewport(0, 0, width, height)
            viewportAspectRatio = width.toFloat() / height.toFloat()
        }

        override fun onDrawFrame(gl: GL10?) {
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            
            if (needsUpdateTextures && imageBitmap != null && depthBitmap != null) {
                updateTextures()
                needsUpdateTextures = false
            }

            if (program == 0 || textureId == 0 || depthTextureId == 0) return
            
            GLES30.glUseProgram(program)

            val timeSec = (System.currentTimeMillis() - startTime) / 1000f
            
            // Chuyển tính toán idle (gợn sóng) lên CPU để tiết kiệm hàng triệu phép tính sin/cos mỗi frame cho GPU
            val idleX = (sin(timeSec * 0.8) * 0.004).toFloat()
            val idleY = (cos(timeSec * 0.7) * 0.004).toFloat()

            GLES30.glUniform1f(uAspectRatio,    viewportAspectRatio)
            GLES30.glUniform1f(uImageAspect,    imageAspectRatio)
            GLES30.glUniform2f(uOffset,         offset.x + idleX, offset.y + idleY)
            GLES30.glUniform1f(uDepthHeight,    0.10f)

            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLES30.glUniform1i(uImage, 0)
            
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
            GLES30.glUniform1i(uDepthMap, 1)

            val posHandle = GLES30.glGetAttribLocation(program, "a_Position")
            GLES30.glEnableVertexAttribArray(posHandle)
            GLES30.glVertexAttribPointer(posHandle, 2, GLES30.GL_FLOAT, false, 0, vertexBuffer)
            
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, indexCount, GLES30.GL_UNSIGNED_INT, indexBuffer)
            
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
            val vShaderCode = """
                #version 300 es
                layout(location = 0) in vec2 a_Position;
                out vec2 v_TexCoord;

                void main() {
                    // Map -1.0..1.0 range back to 0..1 for UV sampling
                    vec2 uv = (a_Position + 1.0) * 0.5;
                    uv.y = 1.0 - uv.y; // Flip Y for Android Bitmaps
                    v_TexCoord = uv;
                    
                    // No vertex displacement. Spatial Parallax is handled in Fragment Shader.
                    gl_Position = vec4(a_Position, 0.0, 1.0);
                }
            """.trimIndent()

            val fShaderCode = """
                #version 300 es
                precision highp float;
                uniform sampler2D image;
                uniform sampler2D uDepthMap;
                uniform vec2 uOffset;
                uniform float uDepthHeight;
                uniform float iAspectRatio;
                uniform float iImageAspect;
                in vec2 v_TexCoord;
                out vec4 fragColor;

                void main() {
                    float ratio = iAspectRatio / iImageAspect;
                    vec2 uv = v_TexCoord;
                    if (ratio > 1.0) {
                        uv.y = (uv.y - 0.5) / ratio + 0.5;
                    } else {
                        uv.x = (uv.x - 0.5) * ratio + 0.5;
                    }
                    
                    // Tính toán idle đã được chuyển lên CPU (giảm tải GPU)
                    vec2 parallax = uOffset * uDepthHeight;
                    
                    // Relief Mapping: Linear Search + Binary Search
                    // [TỐI ƯU HÓA GPU]
                    // Do ảnh depth đã được làm mờ siêu mượt từ Kotlin, chúng ta không cần dò tia quá nhiều.
                    // Việc giảm số bước lặp (từ 38 xuống 14) giúp giảm 65% tải GPU, máy hết nóng và đỡ tốn pin.
                    
                    const float STEPS = 10.0;
                    vec2 delta = parallax / STEPS;
                    float stepDepth = 1.0 / STEPS;
                    
                    // Start at the front-most layer (depth = 1.0)
                    vec2 currentUV = uv - parallax * 0.5; 
                    float currentLayerDepth = 1.0;
                    float currentDepthMapValue = texture(uDepthMap, currentUV).r;
                    
                    // 1. Linear Search: Find the first crossing (Max 10 steps)
                    for (float i = 0.0; i < STEPS; i++) {
                        if (currentLayerDepth <= currentDepthMapValue) {
                            break; // Hit the surface!
                        }
                        currentUV += delta;
                        currentLayerDepth -= stepDepth;
                        currentDepthMapValue = texture(uDepthMap, currentUV).r;
                    }
                    
                    // 2. Binary Search: Refine intersection (Max 4 steps)
                    vec2 minUV = currentUV - delta;
                    vec2 maxUV = currentUV;
                    float minLayerDepth = currentLayerDepth + stepDepth;
                    float maxLayerDepth = currentLayerDepth;
                    
                    for (int j = 0; j < 4; j++) {
                        vec2 midUV = (minUV + maxUV) * 0.5;
                        float midLayerDepth = (minLayerDepth + maxLayerDepth) * 0.5;
                        float midDepthMapValue = texture(uDepthMap, midUV).r;
                        
                        if (midLayerDepth > midDepthMapValue) {
                            minUV = midUV;
                            minLayerDepth = midLayerDepth;
                        } else {
                            maxUV = midUV;
                            maxLayerDepth = midLayerDepth;
                        }
                    }
                    
                    // Use the perfectly refined coordinate
                    fragColor = texture(image, maxUV);
                }
            """.trimIndent()

            program = createProgram(vShaderCode, fShaderCode)
            
            uAspectRatio = GLES30.glGetUniformLocation(program, "iAspectRatio")
            uImageAspect = GLES30.glGetUniformLocation(program, "iImageAspect")
            uOffset      = GLES30.glGetUniformLocation(program, "uOffset")
            uDepthHeight = GLES30.glGetUniformLocation(program, "uDepthHeight")
            uTime        = GLES30.glGetUniformLocation(program, "iTime")
            uImage       = GLES30.glGetUniformLocation(program, "image")
            uDepthMap    = GLES30.glGetUniformLocation(program, "uDepthMap")
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
