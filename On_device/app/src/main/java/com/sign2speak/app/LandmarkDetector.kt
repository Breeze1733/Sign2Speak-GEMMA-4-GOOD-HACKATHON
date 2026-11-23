package com.sign2speak.app

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

private const val TAG = "LandmarkDetector"

// Face landmarks every N frames (CPU optimization)
private const val FACE_EVERY_N = 3

data class LandmarkDetectionResult(
    val handResult: HandLandmarkerResult?,
    val faceResult: FaceLandmarkerResult?,
)

/**
 * Shared MediaPipe Tasks detector (HandLandmarker + FaceLandmarker) used by both
 * live camera input ([CameraManager]) and imported `.bin` frame packs ([FramePackPlayer]).
 *
 * Not thread-safe: invoke [initialize], [detect], and [close] from a single worker thread
 * or under external serialization.
 */
class LandmarkDetector(private val context: Context) {

    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null

    private var faceResultCache: FaceLandmarkerResult? = null
    private var faceFrameCounter = 0

    fun initialize() {
        close()
        faceResultCache = null
        faceFrameCounter = 0
        // Try GPU first, fallback to CPU if unavailable
        handLandmarker = createHandLandmarker(Delegate.GPU)
            ?: createHandLandmarker(Delegate.CPU)
        faceLandmarker = createFaceLandmarker(Delegate.GPU)
            ?: createFaceLandmarker(Delegate.CPU)
    }

    fun detect(bitmap: Bitmap, timestampMs: Long): LandmarkDetectionResult {
        val mpImage = BitmapImageBuilder(bitmap).build()

        val handResult = try {
            handLandmarker?.detectForVideo(mpImage, timestampMs)
        } catch (e: Exception) {
            Log.w(TAG, "HandLandmarker: ${e.message}")
            null
        }

        val shouldRunFace = faceFrameCounter == 0
        faceFrameCounter = (faceFrameCounter + 1) % FACE_EVERY_N
        if (shouldRunFace) {
            faceResultCache = try {
                faceLandmarker?.detectForVideo(mpImage, timestampMs)
            } catch (e: Exception) {
                Log.w(TAG, "FaceLandmarker: ${e.message}")
                faceResultCache
            }
        }

        return LandmarkDetectionResult(
            handResult = handResult,
            faceResult = faceResultCache,
        )
    }

    fun close() {
        try { handLandmarker?.close() } catch (_: Exception) {}
        try { faceLandmarker?.close() } catch (_: Exception) {}
        handLandmarker = null
        faceLandmarker = null
        faceResultCache = null
        faceFrameCounter = 0
    }

    private fun createHandLandmarker(delegate: Delegate): HandLandmarker? = try {
        val options = HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .setDelegate(delegate)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.2f)
            .setMinHandPresenceConfidence(0.2f)
            .setMinTrackingConfidence(0.2f)
            .build()
        HandLandmarker.createFromOptions(context, options).also {
            Log.i(TAG, "HandLandmarker initialisé ($delegate)")
        }
    } catch (e: Exception) {
        Log.w(TAG, "HandLandmarker $delegate ÉCHEC: ${e.message}")
        null
    }

    private fun createFaceLandmarker(delegate: Delegate): FaceLandmarker? = try {
        val options = FaceLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath("face_landmarker.task")
                    .setDelegate(delegate)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.3f)
            .setMinFacePresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .build()
        FaceLandmarker.createFromOptions(context, options).also {
            Log.i(TAG, "FaceLandmarker initialisé ($delegate)")
        }
    } catch (e: Exception) {
        Log.w(TAG, "FaceLandmarker $delegate ÉCHEC: ${e.message}")
        null
    }
}
