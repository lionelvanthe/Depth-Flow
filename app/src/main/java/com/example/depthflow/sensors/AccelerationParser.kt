package com.example.depthflow.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorManager
import com.example.depthflow.dsp.RollingAverageFilter

class AccelerationParser(context: Context) : RotationParser(context) {
    private var accValues: FloatArray? = null
    private var magValues: FloatArray? = null
    private var degHolder: DoubleArray? = doubleArrayOf(0.0, 0.0)
    private val accFilt = RollingAverageFilter(3, 5)
    private val magFilt = RollingAverageFilter(3, 5)

    override val sensors: Array<Sensor?> = arrayOf(
        sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
        sensorManager?.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    )

    override fun parse(event: SensorEvent?): DoubleArray? {
        event ?: return null
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            accValues = FloatArray(3)
            fixOrientation(event.values, accValues!!)
            accFilt.add(accValues!!)
        }

        if (event.sensor.type == Sensor.TYPE_MAGNETIC_FIELD) {
            magValues = FloatArray(3)
            fixOrientation(event.values, magValues!!)
            magFilt.add(magValues!!)
        }

        if (magValues != null && accValues != null) {
            val rotationMatrix = FloatArray(9)

            if (SensorManager.getRotationMatrix(
                    rotationMatrix,
                    null,
                    accFilt.average,
                    magFilt.average
                )
            ) {
                degHolder = parseRoatationMatrix(rotationMatrix)
            }
        }

        return degHolder
    }
}
