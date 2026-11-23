package com.sign2speak.app

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** S2LW v1 encoder. See server/docs/P2_PROTOCOL.md for the authoritative spec. */
object P2Protocol {
    const val MAGIC = "S2LW"
    const val VERSION = 1
    const val HEADER_BYTES = 40
    const val POINT_COUNT = LandmarkLayout.NUM_NODES
    const val COORD_COUNT = 3

    const val FLAG_HAS_HAND = 1 shl 0
    const val FLAG_RESET_VOTE = 1 shl 1
    const val FLAG_FULL_543 = 1 shl 2

    fun maskBytes(pointCount: Int = POINT_COUNT): Int =
        ((pointCount + 7) / 8 + 3) / 4 * 4

    fun frameStride(
        pointCount: Int = POINT_COUNT,
        coordCount: Int = COORD_COUNT,
    ): Int = 16 + maskBytes(pointCount) + pointCount * coordCount * 4

    fun bodyBytes(
        frameCount: Int,
        pointCount: Int = POINT_COUNT,
        coordCount: Int = COORD_COUNT,
    ): Int = HEADER_BYTES + frameCount * frameStride(pointCount, coordCount)

    fun encode(
        snapshot: WindowSnapshot,
        windowId: Long,
        episodeId: Long,
        resetVote: Boolean,
        pointCount: Int = POINT_COUNT,
        coordCount: Int = COORD_COUNT,
    ): ByteArray {
        val expected = bodyBytes(snapshot.frameCount, pointCount, coordCount)
        val buffer = ByteBuffer.allocate(expected).order(ByteOrder.LITTLE_ENDIAN)

        buffer.put(MAGIC.toByteArray(Charsets.US_ASCII))
        buffer.put(VERSION.toByte())
        buffer.put(HEADER_BYTES.toByte())
        buffer.putShort(pointCount.toShort())
        buffer.putShort(coordCount.toShort())
        buffer.putShort(snapshot.frameCount.toShort())

        var flags = 0
        if (snapshot.hasHandCurrentFrame) flags = flags or FLAG_HAS_HAND
        if (resetVote) flags = flags or FLAG_RESET_VOTE
        if (pointCount == 543) flags = flags or FLAG_FULL_543
        buffer.putShort(flags.toShort())
        buffer.putShort(0)

        buffer.putLong(windowId)
        buffer.putLong(episodeId)
        buffer.putLong(snapshot.firstFrameId)

        val maskBytes = maskBytes(pointCount)
        val valuesPerFrame = pointCount * coordCount
        for (i in 0 until snapshot.frameCount) {
            buffer.putLong(snapshot.frameIds[i])
            buffer.putLong(snapshot.timestampsMs[i])
            buffer.put(snapshot.masks, i * maskBytes, maskBytes)
            val base = i * valuesPerFrame
            for (j in 0 until valuesPerFrame) {
                buffer.putFloat(snapshot.values[base + j])
            }
        }

        check(buffer.position() == expected) {
            "S2LW encoded ${buffer.position()} bytes, expected $expected"
        }
        return buffer.array()
    }
}
