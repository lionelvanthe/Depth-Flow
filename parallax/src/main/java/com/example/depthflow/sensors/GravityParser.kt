package com.example.depthflow.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import kotlin.math.atan2
import kotlin.math.sqrt

class GravityParser(context: Context) : GenericParser(context) {

    override val sensors: Array<Sensor?> =
        arrayOf(sensorManager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR))

    override fun parse(event: SensorEvent?): DoubleArray? {
        val sensorValues = event?.values ?: return null
        val fixedValues = FloatArray(3)

        // Remap axis according to orientation
        fixOrientation(sensorValues, fixedValues)

        // Compute the gravity vector module
        val module =
            sqrt((fixedValues[0] * fixedValues[0] + fixedValues[1] * fixedValues[1] + fixedValues[2] * fixedValues[2]).toDouble())

        if (module != 0.0) {
            // Normalize
            fixedValues[0] /= module.toFloat()
            fixedValues[1] /= module.toFloat()
            fixedValues[2] /= module.toFloat()
        }

        var pitch = 0.0
        var roll = 0.0

        // Compute roll and pitch
        if (fixedValues[2] != 0f) {
            roll = Math.toDegrees(
                atan2(
                    fixedValues[0].toDouble(),
                    sqrt(fixedValues[2] * fixedValues[2] + 0.01 * fixedValues[1] * fixedValues[1])
                )
            )

            if (fixedValues[0] != 0f) {
                pitch = Math.toDegrees(
                    atan2(
                        fixedValues[1].toDouble(),
                        sqrt((fixedValues[0] * fixedValues[0] + fixedValues[2] * fixedValues[2]).toDouble())
                    )
                )
            }
        }

        return doubleArrayOf(pitch, roll)
    }

    public override fun reset() {
    }
}
