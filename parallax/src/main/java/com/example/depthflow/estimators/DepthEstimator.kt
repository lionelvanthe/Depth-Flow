package com.example.depthflow.estimators

import android.graphics.Bitmap

interface DepthEstimator {
    fun estimate(image: Bitmap): Bitmap
}
