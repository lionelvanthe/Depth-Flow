package com.example.depthflow

import android.os.Bundle
import android.graphics.Bitmap
import android.graphics.Color
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

class MainActivity : AppCompatActivity() {
    private var realEstimator: com.example.depthflow.estimators.OnnxDepthEstimator? = null

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

                // Khởi tạo bộ ước tính thực tế từ file ONNX (Chạy trong IO Thread)
                if (realEstimator == null) {
                    realEstimator = com.example.depthflow.estimators.OnnxDepthEstimator(this@MainActivity, "depth_anything_v2_small.onnx")
                }

                // Sử dụng nó để xử lý ảnh từ Image Picker
                val depthMap = realEstimator?.estimate(bitmap) ?: bitmap
                
                withContext(Dispatchers.Main) {
                    findViewById<ImageView>(R.id.depthMap).setImageBitmap(depthMap)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        findViewById<com.google.android.material.floatingactionbutton.FloatingActionButton>(R.id.fabPickImage).setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

}