package com.example.depthflow.dsp

class RollingAverageFilter(private val width: Int, private val size: Int) {
    private val total: FloatArray = FloatArray(width)
    private var index = 0
    private val samples: Array<FloatArray> = Array(size) { FloatArray(width) }

    init {

        for (i in 0..<size) for (j in 0..<width) samples[i][j] = 0f
    }

    fun add(x: FloatArray) {
        for (j in 0..<width) total[j] -= samples[index][j]

        samples[index] = x.clone()

        for (j in 0..<width) total[j] += x[j]

        if (++index == size) index = 0
    }

    val average: FloatArray
        get() {
            val output = FloatArray(width)

            for (j in 0..<width) output[j] = total[j] / size

            return output
        }
}