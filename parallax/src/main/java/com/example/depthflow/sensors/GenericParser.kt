package com.example.depthflow.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorManager
import android.hardware.display.DisplayManager
import android.view.Display
import android.view.Surface

abstract class GenericParser internal constructor(context: Context) {
    private val display: Display =
        (context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager).getDisplay(Display.DEFAULT_DISPLAY)
    val sensorManager: SensorManager? =
        context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager?

    init {
        reset()
    }

    abstract val sensors: Array<Sensor?>

    // Parse a sensor event and returns rotation
    abstract fun parse(event: SensorEvent?): DoubleArray?

    // resets any internal data
    protected abstract fun reset()

    fun fixOrientation(input: FloatArray, fixed: FloatArray) {
        when (display.rotation) {
            Surface.ROTATION_0 -> {
                fixed[0] = input[0]
                fixed[1] = input[1]
            }

            Surface.ROTATION_90 -> {
                fixed[0] = -input[1]
                fixed[1] = input[0]
            }

            Surface.ROTATION_180 -> {
                fixed[0] = -input[0]
                fixed[1] = -input[1]
            }

            Surface.ROTATION_270 -> {
                fixed[0] = input[1]
                fixed[1] = -input[0]
            }
        }

        fixed[2] = input[2]

        if (input.size > 3) {
            fixed[3] = input[3]
        }
    }
}
