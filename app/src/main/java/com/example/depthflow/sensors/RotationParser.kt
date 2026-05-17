package com.example.depthflow.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorManager
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2

open class RotationParser(context: Context) : GenericParser(context) {
    private var oldRoll = 0.0
    private var oldPitch = 0.0
    private var deltaCross = 0.0

    // Rotation direction fixer
    private var oldAngle = 0.0
    private var baseRoll = 0.0

    override fun parse(event: SensorEvent?): DoubleArray? {
        val sensorValues = event?.values ?: return null
        val fixedValues = FloatArray(4)

        // Remap axis according to orientation
        fixOrientation(sensorValues, fixedValues)

        val rotationMatrix = FloatArray(9)

        // Compute rotation matrix
        SensorManager.getRotationMatrixFromVector(rotationMatrix, fixedValues)

        return parseRoatationMatrix(rotationMatrix)
    }

    override val sensors: Array<Sensor?> =
        arrayOf(sensorManager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR))

    public override fun reset() {
        oldRoll = 0.0
        oldPitch = 0.0
        deltaCross = 0.0
        baseRoll = 0.0
    }

    fun parseRoatationMatrix(rotationMatrix: FloatArray): DoubleArray? {
        // Remap for pitch
        SensorManager.remapCoordinateSystem(
            rotationMatrix,
            SensorManager.AXIS_X,
            SensorManager.AXIS_Y,
            rotationMatrix
        )

        // Grab pitch
        val pitch = Math.toDegrees(asin(rotationMatrix[7].toDouble()))

        // Roll can be extracted without problems when pitch is smaller than 70 degrees
        val orientationValues = FloatArray(3)
        var roll: Double
        if (pitch < 70) {
            roll = Math.toDegrees(
                atan2(
                    -rotationMatrix[6].toDouble(),
                    abs(rotationMatrix[8]).toDouble()
                )
            )
        } else {
            // Remap axis to extract roll
            SensorManager.remapCoordinateSystem(
                rotationMatrix,
                SensorManager.AXIS_Y,
                SensorManager.AXIS_Z,
                rotationMatrix
            )
            SensorManager.getOrientation(rotationMatrix, orientationValues)
            roll = -Math.toDegrees(orientationValues[0].toDouble())
        }

        // Normalize roll (0, 90, 0, -90)
        /*if (roll > 90) {
            roll = 180 - roll;
        } else if (roll < -90) {
            roll = -180 - roll;
        }*/
        // TODO Move elsewhere
        if (oldAngle > 150 && roll < -150) {
            baseRoll += 360.0
        } else if (oldAngle < -150 && roll > 150) {
            baseRoll -= 360.0
        }
        oldAngle = roll
        roll += baseRoll

        // Fix cross panic
        if ((oldPitch < 70 && pitch >= 70) || (pitch < 70 && oldPitch >= 70)) {
            deltaCross = roll - oldRoll
        }
        roll = roll - deltaCross

        // Update old values
        oldRoll = roll
        oldPitch = pitch

        return doubleArrayOf(pitch, roll)
    }
}
