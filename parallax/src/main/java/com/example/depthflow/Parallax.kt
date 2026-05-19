package com.example.depthflow

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.example.depthflow.dsp.LowPassFilter
import com.example.depthflow.sensors.AccelerationParser
import com.example.depthflow.sensors.GenericParser
import com.example.depthflow.sensors.GravityParser
import com.example.depthflow.sensors.RotationParser

internal class Parallax(private val context: Context) : SensorEventListener {
    private val TAG: String = javaClass.simpleName

    // Filters
    private val sensitivityFilter = LowPassFilter(2)
    private val fallbackFilter = LowPassFilter(2)
    private var resetDeg = DoubleArray(2)
    private var filtersInit: Boolean

    // Outputs
    var degX: Double = 0.0
        private set
    var degY: Double = 0.0
        private set

    var onUpdate: ((degX: Double, degY: Double) -> Unit)? = null

    private val sensorManager: SensorManager? =
        context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager?
    private val parser: GenericParser?

    init {

        parser = getParser()

        if (parser == null) {
            Log.e(TAG, "No valid sensor available!")
        }

        filtersInit = false
    }

    fun setFallback(fallback: Double) {
        fallbackFilter.setFactor(fallback)
    }

    fun setSensitivity(sensitivity: Double) {
        sensitivityFilter.setFactor(sensitivity)
    }

    fun start() {
        if (parser != null) {
            for (sensor in parser.sensors) {
                sensorManager?.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
            }
        }

        Log.d(TAG, "Sensor listener started!")
    }

    fun stop() {
        sensorManager?.unregisterListener(this)

        Log.d(TAG, "Sensor listener stopped!")
    }

    // SensorEventListenerMethods
    override fun onSensorChanged(event: SensorEvent?) {
        var newDeg = parser?.parse(event) ?: return

        // Set the initial value of the filters to current val
        if (!filtersInit) {
            sensitivityFilter.setLast(newDeg)
            fallbackFilter.setLast(newDeg)
            filtersInit = true
        }

        // Apply filter
        newDeg = sensitivityFilter.filter(newDeg)

        degY = newDeg[0] - resetDeg[0]
        degX = newDeg[1] - resetDeg[1]

        resetDeg = fallbackFilter.filter(newDeg)

        if (degX > 180) {
            resetDeg[1] += degX - 180
            degX = 180.0
        }

        if (degX < -180) {
            resetDeg[1] += degX + 180
            degX = -180.0
        }

        onUpdate?.invoke(degX, degY)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }

    // Return the best sensor available
    private fun getParser(): GenericParser? {
        if (sensorManager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) != null) {
            Log.d(TAG, "Using rotation vector")
            return RotationParser(context)
        }

        if (sensorManager?.getDefaultSensor(Sensor.TYPE_GRAVITY) != null) {
            Log.d(TAG, "Using gravity")
            return GravityParser(context)
        }

        if (sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null && sensorManager.getDefaultSensor(
                Sensor.TYPE_MAGNETIC_FIELD
            ) != null
        ) {
            Log.d(TAG, "Using accelerometer+magnetometer")
            return AccelerationParser(context)
        }

        return null
    }
}
