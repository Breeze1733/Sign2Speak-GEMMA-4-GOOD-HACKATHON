package com.sign2speak.app

import java.util.ArrayDeque

data class WindowSnapshot(
    val frameCount: Int,
    val firstFrameId: Long,
    val lastFrameId: Long,
    val frameIds: LongArray,
    val timestampsMs: LongArray,
    /** frameCount * pointCount * coordCount, point-major. Zero for absent points. */
    val values: FloatArray,
    /** frameCount * maskBytes, LSB-first bits. */
    val masks: ByteArray,
    val hasHandCurrentFrame: Boolean,
)

/** P2 phone-side sliding window; mirrors SignRecognizer's 64-frame deque. */
class FrameWindow(private val maxLen: Int = LandmarkLayout.MAX_LEN) {

    private data class Entry(
        val frame: FloatArray,
        val frameId: Long,
        val timestampMs: Long,
        val hasHand: Boolean,
    )

    private val frames = ArrayDeque<Entry>(maxLen + 1)
    private var fallbackFrameId = 1L

    val size: Int get() = frames.size

    fun add(holistic: FloatArray, frameId: Long, timestampMs: Long, hasHand: Boolean) {
        val id = if (frameId >= 0L) frameId else fallbackFrameId++
        frames.addLast(Entry(holistic, id, timestampMs, hasHand))
        if (frames.size > maxLen) frames.removeFirst()
    }

    fun clear() {
        frames.clear()
        fallbackFrameId = 1L
    }

    fun snapshot(
        pointIndices: IntArray = LandmarkLayout.POINT_LANDMARKS,
        coordCount: Int = P2Protocol.COORD_COUNT,
    ): WindowSnapshot {
        val n = frames.size
        val pointCount = pointIndices.size
        val maskBytes = P2Protocol.maskBytes(pointCount)
        val values = FloatArray(n * pointCount * coordCount)
        val masks = ByteArray(n * maskBytes)
        val frameIds = LongArray(n)
        val timestamps = LongArray(n)

        var frame = 0
        var valueIndex = 0
        var maskIndex = 0
        for (entry in frames) {
            frameIds[frame] = entry.frameId
            timestamps[frame] = entry.timestampMs
            for (pi in 0 until pointCount) {
                val global = pointIndices[pi]
                val x = entry.frame[global * 3]
                if (!x.isNaN()) {
                    val byteIndex = maskIndex + (pi ushr 3)
                    masks[byteIndex] =
                        (masks[byteIndex].toInt() or (1 shl (pi and 7))).toByte()
                    values[valueIndex] = x
                    if (coordCount > 1) values[valueIndex + 1] = entry.frame[global * 3 + 1]
                    if (coordCount > 2) values[valueIndex + 2] = entry.frame[global * 3 + 2]
                }
                valueIndex += coordCount
            }
            frame++
            maskIndex += maskBytes
        }

        return WindowSnapshot(
            frameCount = n,
            firstFrameId = if (n > 0) frameIds[0] else 0L,
            lastFrameId = if (n > 0) frameIds[n - 1] else 0L,
            frameIds = frameIds,
            timestampsMs = timestamps,
            values = values,
            masks = masks,
            hasHandCurrentFrame = frames.peekLast()?.hasHand ?: false,
        )
    }
}
