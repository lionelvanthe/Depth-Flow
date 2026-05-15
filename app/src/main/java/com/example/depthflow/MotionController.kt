package com.example.depthflow

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.max
import kotlin.math.min

class MotionController(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    var sensitivityX = 1.0f
    var sensitivityY = 1.0f
    var maxOffset = 0.5f

    private var onOffsetChanged: ((Float, Float) -> Unit)? = null

    fun setOnOffsetChangedListener(listener: (Float, Float) -> Unit) {
        onOffsetChanged = listener
    }

    fun start() {
        rotationSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    private val rotationMatrix = FloatArray(9)
    private val orientation = FloatArray(3)

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientation)

            val pitch = orientation[1]
            val roll = orientation[2]

            val offsetX = clamp(roll * sensitivityX, -maxOffset, maxOffset)
            val offsetY = clamp(pitch * sensitivityY, -maxOffset, maxOffset)

            onOffsetChanged?.invoke(offsetX, offsetY)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun clamp(value: Float, min: Float, max: Float): Float {
        return max(min, min(max, value))
    }
}
