package com.example.depthflow.estimators

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class OnnxDepthEstimator(private val context: Context, private val modelPath: String) : DepthEstimator {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    init {
        loadModel()
    }

    private fun loadModel() {
        val modelBytes = context.assets.open(modelPath).readBytes()
        ortSession = ortEnv.createSession(modelBytes)
    }

    override fun estimate(image: Bitmap): Bitmap {
        val session = ortSession ?: return image

        // 1. Preprocessing: Resize image to model input size (e.g., 518x518 for DepthAnythingV2)
        val inputSize = 518 
        val resizedBitmap = Bitmap.createScaledBitmap(image, inputSize, inputSize, true)
        
        // 2. Convert Bitmap to FloatBuffer (Normalization)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // CHW format (Channel, Height, Width)
        for (c in 0 until 3) {
            for (p in pixels) {
                val color = when (c) {
                    0 -> Color.red(p)
                    1 -> Color.green(p)
                    else -> Color.blue(p)
                }
                // Normalize to [0, 1] - adjust based on your specific model requirements
                floatBuffer.put(color / 255.0f)
            }
        }
        floatBuffer.rewind()

        // 3. Run Inference
        val inputName = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
        
        val results = session.run(mapOf(inputName to inputTensor))
        val outputTensor = results[0] as OnnxTensor
        val outputData = outputTensor.floatBuffer.array() // Assuming output is Float

        // 4. Postprocessing: Convert output to Grayscale Bitmap
        // Need to find min/max for normalization to [0, 255]
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        for (v in outputData) {
            if (v < min) min = v
            if (v > max) max = v
        }

        val depthBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val depthPixels = IntArray(inputSize * inputSize)
        for (i in outputData.indices) {
            val normalized = ((outputData[i] - min) / (max - min) * 255).toInt()
            depthPixels[i] = Color.rgb(normalized, normalized, normalized)
        }
        depthBitmap.setPixels(depthPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Resize back to original image size
        return Bitmap.createScaledBitmap(depthBitmap, image.width, image.height, true)
    }

    fun close() {
        ortSession?.close()
        ortEnv.close()
    }
}
