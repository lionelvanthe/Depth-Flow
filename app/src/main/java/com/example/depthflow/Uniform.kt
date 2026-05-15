package com.example.depthflow

sealed class UniformValue {
    data class FloatValue(val value: Float) : UniformValue()
    data class IntValue(val value: Int) : UniformValue()
    data class Vec2Value(val x: Float, val y: Float) : UniformValue()
}

data class Uniform(
    val type: String,
    val name: String,
    val value: UniformValue
) {
    constructor(type: String, name: String, value: Float) : this(type, name, UniformValue.FloatValue(value))
    constructor(type: String, name: String, value: Int) : this(type, name, UniformValue.IntValue(value))
    constructor(type: String, name: String, value: Pair<Float, Float>) : this(type, name, UniformValue.Vec2Value(value.first, value.second))
}
