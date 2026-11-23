package com.sign2speak.app

import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/** Phone-side M2 assembly and overlay extraction, shared by P0/P1/P2. */
object HolisticAssembler {

    fun hasHand(handResult: HandLandmarkerResult?): Boolean =
        handResult != null && handResult.landmarks().isNotEmpty()

    fun assemble(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
    ): FloatArray {
        val holistic = FloatArray(LandmarkLayout.ROWS_PER_FRAME * 3) { Float.NaN }

        if (faceResult != null && faceResult.faceLandmarks().isNotEmpty()) {
            val lms = faceResult.faceLandmarks()[0]
            val count = minOf(lms.size, 468)
            for (i in 0 until count) {
                val lm = lms[i]
                holistic[i * 3 + 0] = lm.x()
                holistic[i * 3 + 1] = lm.y()
                holistic[i * 3 + 2] = lm.z()
            }
        }

        if (handResult != null && handResult.landmarks().isNotEmpty()) {
            for (hIdx in 0 until minOf(handResult.landmarks().size, 2)) {
                if (hIdx >= handResult.handedness().size) break
                val label = handResult.handedness()[hIdx][0].categoryName()
                val lms = handResult.landmarks()[hIdx]
                val offset = if (label == "Left") {
                    LandmarkLayout.LEFT_HAND_OFFSET
                } else {
                    LandmarkLayout.RIGHT_HAND_OFFSET
                }
                for (i in 0 until minOf(lms.size, LandmarkLayout.HAND_NUM_LANDMARKS)) {
                    val lm = lms[i]
                    holistic[(offset + i) * 3 + 0] = lm.x()
                    holistic[(offset + i) * 3 + 1] = lm.y()
                    holistic[(offset + i) * 3 + 2] = lm.z()
                }
            }
        }

        return holistic
    }

    fun extractHandOverlay(r: HandLandmarkerResult?): List<DetectedHand> {
        if (r == null) return emptyList()
        val lmsList = r.landmarks()
        if (lmsList.isEmpty()) return emptyList()
        return lmsList.mapIndexed { i, lms ->
            val isLeft = r.handedness().getOrNull(i)?.firstOrNull()?.categoryName() == "Left"
            DetectedHand(lms.map { lm -> LandmarkPoint(lm.x(), lm.y()) }, isLeft)
        }
    }

    fun extractFaceOverlay(r: FaceLandmarkerResult?): List<LandmarkPoint> {
        if (r == null) return emptyList()
        val faces = r.faceLandmarks()
        if (faces.isEmpty()) return emptyList()
        val lms = faces[0]
        return LandmarkLayout.FACE_OVERLAY_IDX.mapNotNull { idx ->
            if (idx < lms.size) LandmarkPoint(lms[idx].x(), lms[idx].y()) else null
        }
    }
}
