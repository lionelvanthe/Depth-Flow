package com.example.depthflow

import android.graphics.Bitmap

open class ShaderScene {
    var resolution: Pair<Int, Int> = Pair(0, 0)
    var runtime: Float = 0.0f
    var tau: Float = 0.0f // Current time normalized?

    open fun build() {}
    open fun setup() {}
    open fun update() {}
    open fun handle(message: Any) {}
    open fun pipeline(): List<Uniform> = emptyList()
}

class ShaderTexture(val name: String) {
    var size: Pair<Int, Int> = Pair(0, 0)
    fun repeat(value: Boolean): ShaderTexture = this
    fun fromBitmap(bitmap: Bitmap) {
        size = Pair(bitmap.width, bitmap.height)
    }
    fun isEmpty(): Boolean = size.first == 0
}

class Shader {
    var fragment: String = ""
}
