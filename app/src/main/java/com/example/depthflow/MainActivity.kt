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
        renderer = ParallaxRenderer { glSurfaceView.requestRender() }
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


}
