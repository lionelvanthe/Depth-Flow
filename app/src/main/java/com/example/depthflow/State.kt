package com.example.depthflow

data class VignetteState(
    var intensity: Float = 0.00f,
    var decay: Float = 20.00f
) {
    fun pipeline(): List<Uniform> = listOf(
        Uniform("float", "iVigIntensity", intensity),
        Uniform("float", "iVigDecay", decay)
    )
}

data class LensState(
    var intensity: Float = 0.00f,
    var decay: Float = 0.4f,
    var quality: Int = 30
) {
    fun pipeline(): List<Uniform> = listOf(
        Uniform("float", "iLensIntensity", intensity),
        Uniform("float", "iLensDecay", decay),
        Uniform("int", "iLensQuality", quality)
    )
}

data class InpaintState(
    var limit: Float = 0.0f
) {
    fun pipeline(): List<Uniform> = listOf(
        Uniform("float", "iInpaint", limit)
    )
}

data class BlurState(
    var intensity: Float = 0.00f,
    var start: Float = 0.60f,
    var end: Float = 1.00f,
    var exponent: Float = 2.00f,
    var quality: Int = 4,
    var directions: Int = 16
) {
    fun pipeline(): List<Uniform> = listOf(
        Uniform("float", "iBlurIntensity", intensity / 100f),
        Uniform("float", "iBlurStart", start),
        Uniform("float", "iBlurEnd", end),
        Uniform("float", "iBlurExponent", exponent),
        Uniform("int", "iBlurQuality", quality),
        Uniform("int", "iBlurDirections", directions)
    )
}

data class ColorState(
    var saturation: Float = 100.0f,
    var contrast: Float = 100.0f,
    var brightness: Float = 100.0f,
    var sepia: Float = 0.0f
) {
    fun pipeline(): List<Uniform> = listOf(
        Uniform("float", "iColorsSaturation", saturation / 100f),
        Uniform("float", "iColorsContrast", contrast / 100f),
        Uniform("float", "iColorsBrightness", brightness / 100f),
        Uniform("float", "iColorsSepia", sepia / 100f)
    )
}

data class DepthState(
    var height: Float = 0.20f,
    var steady: Float = 0.15f,
    var focus: Float = 0.00f,
    var zoom: Float = 1.00f,
    var isometric: Float = 0.00f,
    var dolly: Float = 0.00f,
    var offset: Pair<Float, Float> = Pair(0.00f, 0.00f),
    var center: Pair<Float, Float> = Pair(0.00f, 0.00f),
    var origin: Pair<Float, Float> = Pair(0.00f, 0.00f),
    var vignette: VignetteState = VignetteState(),
    var lens: LensState = LensState(),
    var inpaint: InpaintState = InpaintState(),
    var color: ColorState = ColorState(),
    var blur: BlurState = BlurState()
) {
    fun pipeline(): List<Uniform> {
        val uniforms = mutableListOf<Uniform>()
        uniforms.add(Uniform("float", "iDepthHeight", height))
        uniforms.add(Uniform("float", "iDepthSteady", steady))
        uniforms.add(Uniform("float", "iDepthFocus", focus))
        uniforms.add(Uniform("float", "iDepthZoom", zoom))
        uniforms.add(Uniform("float", "iDepthIsometric", isometric))
        uniforms.add(Uniform("float", "iDepthDolly", dolly))
        uniforms.add(Uniform("vec2", "iDepthOffset", offset))
        uniforms.add(Uniform("vec2", "iDepthCenter", center))
        uniforms.add(Uniform("vec2", "iDepthOrigin", origin))
        uniforms.addAll(vignette.pipeline())
        uniforms.addAll(lens.pipeline())
        uniforms.addAll(inpaint.pipeline())
        uniforms.addAll(color.pipeline())
        uniforms.addAll(blur.pipeline())
        return uniforms
    }
}
