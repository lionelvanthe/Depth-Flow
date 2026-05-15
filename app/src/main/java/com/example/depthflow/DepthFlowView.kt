package com.example.depthflow

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLSurfaceView
import android.util.AttributeSet

class DepthFlowView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    val renderer: DepthFlowRenderer

    init {
        setEGLContextClientVersion(3)
        renderer = DepthFlowRenderer(context)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setImages(image: Bitmap, depth: Bitmap) {
        queueEvent {
            renderer.setTextures(image, depth)
        }
    }

    fun setState(state: DepthState) {
        queueEvent {
            renderer.depthState = state
        }
    }
}
