package com.example.depthflow.dsp

class LowPassFilter(private val width: Int) {
    private var factor = 0.0
    private var last: DoubleArray

    init {
        last = DoubleArray(width)
    }

    fun setFactor(factor: Double) {
        this.factor = factor
    }

    fun setLast(last: DoubleArray) {
        this.last = last.clone()
    }

    fun filter(input: DoubleArray): DoubleArray {
        val output = DoubleArray(width)

        for (i in 0..<width) {
            output[i] = factor * input[i] + (1 - factor) * last[i]
        }

        last = output.clone()

        return output
    }
}
