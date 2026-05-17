package com.sign2speak.app

data class LandmarkPoint(val x: Float, val y: Float)

/** One detected hand: 21 normalised MediaPipe points. isLeft = user's left hand. */
data class DetectedHand(val points: List<LandmarkPoint>, val isLeft: Boolean)
