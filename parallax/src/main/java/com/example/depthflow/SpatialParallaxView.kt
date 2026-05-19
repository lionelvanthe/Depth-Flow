package com.example.depthflow

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLSurfaceView
import android.util.AttributeSet

class SpatialParallaxView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    private val parallaxRenderer = ParallaxRenderer { requestRender() }
    private val parallaxSensor = Parallax(context)

    init {
        setEGLContextClientVersion(3)
        setRenderer(parallaxRenderer)
        renderMode = RENDERMODE_WHEN_DIRTY

        parallaxSensor.apply {
            setSensitivity(0.2)
            setFallback(0.01)
            onUpdate = { degX, degY ->
                val rollRad = Math.toRadians(degX).toFloat()
                val pitchRad = Math.toRadians(degY).toFloat()
                parallaxRenderer.setOffset(-rollRad * 1f, -pitchRad * 1f)
                requestRender()
            }
        }
    }

    fun setImages(image: Bitmap, depth: Bitmap) {
        parallaxRenderer.setBitmaps(image, depth)
    }

    override fun onResume() {
        super.onResume()
        parallaxSensor.start()
    }

    override fun onPause() {
        super.onPause()
        parallaxSensor.stop()
    }
}
