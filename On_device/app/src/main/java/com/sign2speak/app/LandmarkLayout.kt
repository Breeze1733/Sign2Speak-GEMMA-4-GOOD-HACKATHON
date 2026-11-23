package com.sign2speak.app

import org.json.JSONObject

/**
 * Canonical landmark layout shared by the local recognizer, the P2 encoder and
 * the server-side preprocessing. Values are copied from SignRecognizer.kt and
 * must not be "improved": they define parity with the trained model.
 */
object LandmarkLayout {
    const val ROWS_PER_FRAME = 543
    const val HAND_NUM_LANDMARKS = 21
    const val LEFT_HAND_OFFSET = 468
    const val RIGHT_HAND_OFFSET = 522

    val NOSE = intArrayOf(1, 2, 98, 327)
    val LIP = intArrayOf(
        0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    )
    val REYE = intArrayOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
    val LEYE = intArrayOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
    val LHAND = IntArray(21) { 468 + it }
    val RHAND = IntArray(21) { 522 + it }

    val POINT_LANDMARKS: IntArray = LIP + LHAND + RHAND + NOSE + REYE + LEYE

    val FACE_OVERLAY_IDX: List<Int> = listOf(
        // LIP (40)
        0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        // NOSE (4)
        1, 2, 98, 327,
        // RIGHT EYE (16)
        33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
        // LEFT EYE (16)
        263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
    )

    const val NUM_NODES = 118
    const val CHANNELS = 708
    const val MAX_LEN = 64
    const val PAD = -100f
    const val NOSE_HOLISTIC_IDX = 17
    const val NOSE_NODE_IDX = 15

    const val PREDICTION_HISTORY_SIZE = 8
    const val DEFAULT_THRESHOLD = 0.2f
    const val INFER_EVERY_N = 3
    const val MIN_FRAMES_FOR_INFER = 10

    fun modelVersion(meta: JSONObject, fallback: String): String =
        meta.optString("model_version", fallback).ifBlank { fallback }
}
