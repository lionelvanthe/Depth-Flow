package com.example.depthflow.estimators

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import java.util.EnumSet

class OnnxDepthEstimator(private val context: Context, private val modelPath: String) : DepthEstimator {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    init {
        loadModel()
    }

    private fun loadModel() {
        val modelBytes = context.assets.open(modelPath).readBytes()
        val options = OrtSession.SessionOptions()
        try {
            // Use NNAPI for hardware acceleration on Android
            options.addNnapi()
        } catch (e: Exception) {
            // Fallback to CPU if NNAPI is not available
        }
        ortSession = ortEnv.createSession(modelBytes, options)
    }

    override fun estimate(image: Bitmap): Bitmap {
        val session = ortSession ?: return image

        // 1. Preprocessing: Resize to model input size (518x518 for DepthAnythingV2)
        // Optimization: Depth map doesn't need to be high-res for parallax
        val inputSize = 518 
        val resizedBitmap = if (image.width == inputSize && image.height == inputSize) image else image.scale(inputSize, inputSize)
        
        // 2. Convert Bitmap to FloatBuffer (Normalization)
        // Using a single pass over pixels is much faster than nested loops
        val floatBuffer = FloatBuffer.allocate(1 * 3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val channelSize = inputSize * inputSize
        for (i in 0 until channelSize) {
            val p = pixels[i]
            // Fast bitwise color extraction
            val r = (p shr 16 and 0xFF) / 255.0f
            val g = (p shr 8 and 0xFF) / 255.0f
            val b = (p and 0xFF) / 255.0f
            
            // CHW format
            floatBuffer.put(i, r)
            floatBuffer.put(i + channelSize, g)
            floatBuffer.put(i + 2 * channelSize, b)
        }
        floatBuffer.rewind()

        // 3. Run Inference
        val inputName = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
        
        val results = session.run(mapOf(inputName to inputTensor))
        val outputTensor = results[0] as OnnxTensor
        val outputData = FloatArray(channelSize)
        outputTensor.floatBuffer.get(outputData)

        // 4. Postprocessing: Convert output to Grayscale Bitmap
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        for (v in outputData) {
            if (v < min) min = v
            if (v > max) max = v
        }

        val depthBitmap = createBitmap(inputSize, inputSize)
        val depthPixels = IntArray(channelSize)
        val range = max - min
        val scale = if (range != 0f) 255f / range else 0f
        
        for (i in 0 until channelSize) {
            val normalized = ((outputData[i] - min) * scale).toInt().coerceIn(0, 255)
            // Manual ARGB pixel construction for speed
            depthPixels[i] = -0x1000000 or (normalized shl 16) or (normalized shl 8) or normalized
        }
        depthBitmap.setPixels(depthPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Optimization: Return the 518x518 depth map directly.
        // Scaling it back to 4K/Original is slow and unnecessary for the GLSL sampler.
        return depthBitmap
    }

    fun close() {
        ortSession?.close()
        ortEnv.close()
    }
}
