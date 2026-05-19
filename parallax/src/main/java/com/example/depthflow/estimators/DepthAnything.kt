package com.example.depthflow.estimators

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtException
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale

class DepthAnything(context: Context, modelName: String) {

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val inputName: String

    private val inputDim: Int
    private val outputDim: Int

    init {
        when {
            modelName.contains("_256") -> {
                inputDim = 256
                outputDim = 252
            }
            modelName.contains("_512") -> {
                inputDim = 512
                outputDim = 504
            }
            else -> throw IllegalArgumentException("Unsupported model size")
        }

        // 1. Đảm bảo model được copy ra cache một cách an toàn (atomic)
        val modelFile = File(context.cacheDir, modelName)

        fun copyModel() {
            val tempFile = File(context.cacheDir, "$modelName.tmp")
            try {
                context.assets.open(modelName).use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                if (!tempFile.renameTo(modelFile)) {
                    tempFile.copyTo(modelFile, overwrite = true)
                    tempFile.delete()
                }
            } catch (e: Exception) {
                tempFile.delete()
                throw e
            }
        }

        if (!modelFile.exists() || modelFile.length() == 0L) {
            copyModel()
        }

        val options = OrtSession.SessionOptions()
        try {
            // 2. Tối ưu hóa: Cấu hình đa luồng và sử dụng NNAPI để tăng tốc NPU/GPU
            options.setIntraOpNumThreads(2)
            options.addNnapi()
        } catch (e: Exception) {
            // Fallback về CPU nếu NNAPI không khả dụng
        }

        ortSession = try {
            ortEnvironment.createSession(modelFile.absolutePath, options)
        } catch (e: OrtException) {
            // 3. Xử lý lỗi file bị hỏng (ORT_INVALID_PROTOBUF)
            if (e.message?.contains("ORT_INVALID_PROTOBUF") == true || 
                e.message?.contains("Protobuf parsing failed") == true) {
                Log.e("DepthAnything", "Model file corrupted, retrying copy: ${e.message}")
                modelFile.delete()
                copyModel()
                ortEnvironment.createSession(modelFile.absolutePath, options)
            } else {
                throw e
            }
        }
        inputName = ortSession.inputNames.iterator().next()
    }

    suspend fun predict(inputImage: Bitmap): Bitmap =
        withContext(Dispatchers.Default) {
            // 1. Tiền xử lý: Resize về kích thước đầu vào của model
            val resizedImage = if (inputImage.width == inputDim && inputImage.height == inputDim) {
                inputImage
            } else {
                inputImage.scale(inputDim, inputDim)
            }

            // 2. Chuyển đổi Bitmap sang NHWC UINT8 tối ưu
            val imagePixels = convert(resizedImage)
            val inputTensor = OnnxTensor.createTensor(
                ortEnvironment,
                imagePixels,
                longArrayOf(1, inputDim.toLong(), inputDim.toLong(), 3),
                OnnxJavaType.UINT8
            )

            val outputs = ortSession.run(mapOf(inputName to inputTensor))

            val outputTensor = outputs[0] as OnnxTensor
            val outputBuffer = outputTensor.byteBuffer
            val channelSize = outputDim * outputDim

            // 3. Hậu xử lý: Chuyển dữ liệu sang FloatArray để thực hiện Blur và Normalization
            val outputData = FloatArray(channelSize)
            outputBuffer.rewind()
            if (outputTensor.info.type == OnnxJavaType.FLOAT) {
                outputTensor.floatBuffer.get(outputData)
            } else {
                for (i in 0 until channelSize) {
                    outputData[i] = (outputBuffer.get().toInt() and 0xFF).toFloat()
                }
            }

            // 4. Hậu xử lý: Sliding Window Box Blur (O(1))
            // Bước này biến các vách đá độ sâu sắc nhọn thành sườn dốc mịn, giảm răng cưa khi parallax.
            val blurRadius = 20
            val blurredData = FloatArray(channelSize)

            // Quét ngang
            for (y in 0 until outputDim) {
                var sum = 0f
                var count = 0
                val rowOffset = y * outputDim
                for (kx in -blurRadius..blurRadius) {
                    if (kx in 0..<outputDim) {
                        sum += outputData[rowOffset + kx]
                        count++
                    }
                }
                blurredData[rowOffset] = sum / count
                for (x in 1 until outputDim) {
                    val left = x - blurRadius - 1
                    val right = x + blurRadius
                    if (left >= 0) {
                        sum -= outputData[rowOffset + left]
                        count--
                    }
                    if (right < outputDim) {
                        sum += outputData[rowOffset + right]
                        count++
                    }
                    blurredData[rowOffset + x] = sum / count
                }
            }

            // Quét dọc
            for (x in 0 until outputDim) {
                var sum = 0f
                var count = 0
                for (ky in -blurRadius..blurRadius) {
                    if (ky in 0..<outputDim) {
                        sum += blurredData[ky * outputDim + x]
                        count++
                    }
                }
                outputData[x] = sum / count
                for (y in 1 until outputDim) {
                    val top = y - blurRadius - 1
                    val bottom = y + blurRadius
                    if (top >= 0) {
                        sum -= blurredData[top * outputDim + x]
                        count--
                    }
                    if (bottom < outputDim) {
                        sum += blurredData[bottom * outputDim + x]
                        count++
                    }
                    outputData[y * outputDim + x] = sum / count
                }
            }

            // 5. Chuẩn hóa Min-Max để tận dụng tối đa dải động 0-255
            var min = Float.MAX_VALUE
            var max = Float.MIN_VALUE
            for (v in outputData) {
                if (v < min) min = v
                if (v > max) max = v
            }
            val range = max - min
            val scale = if (range != 0f) 255f / range else 0f

            // 6. Chuyển đổi sang Grayscale Bitmap (ARGB_8888 để sampler GLSL dễ đọc kênh .r)
            val depthBitmap = createBitmap(outputDim, outputDim)
            val depthPixels = IntArray(channelSize)
            for (i in 0 until channelSize) {
                val normalized = ((outputData[i] - min) * scale).toInt().coerceIn(0, 255)
                depthPixels[i] = -0x1000000 or (normalized shl 16) or (normalized shl 8) or normalized
            }
            depthBitmap.setPixels(depthPixels, 0, outputDim, 0, 0, outputDim, outputDim)

            // 7. Scale lại về độ phân giải gốc của ảnh đầu vào
            return@withContext depthBitmap.scale(inputImage.width, inputImage.height)
        }

    private fun convert(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val imgData = ByteBuffer.allocateDirect(1 * width * height * 3)
        imgData.order(ByteOrder.nativeOrder())
        imgData.rewind()

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            // Trích xuất màu bằng bitwise (R, G, B) nhanh hơn Color.red/green/blue
            imgData.put((pixel shr 16 and 0xFF).toByte())
            imgData.put((pixel shr 8 and 0xFF).toByte())
            imgData.put((pixel and 0xFF).toByte())
        }
        imgData.rewind()
        return imgData
    }
}