package com.sign2speak.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors

private const val TAG = "CameraManager"

// Face landmarks toutes les N frames (optimisation CPU)
private const val FACE_EVERY_N = 3

class CameraManager(
    private val context: Context,
    private val onFrame: (handResult: HandLandmarkerResult?, faceResult: FaceLandmarkerResult?, timestampMs: Long, imgW: Int, imgH: Int) -> Unit,
) {

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null

    private var faceResultCache: FaceLandmarkerResult? = null
    private var faceFrameCounter = 0

    private var frameIndex = 0L
    private var cameraProvider: ProcessCameraProvider? = null

    fun initialize() {
        try {
            val handOptions = HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.2f)
                .setMinHandPresenceConfidence(0.2f)
                .setMinTrackingConfidence(0.2f)
                .build()
            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.i(TAG, "HandLandmarker initialisé")
        } catch (e: Exception) {
            Log.e(TAG, "HandLandmarker ÉCHEC: ${e.message}")
        }

        try {
            val faceOptions = FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.3f)
                .setMinFacePresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()
            faceLandmarker = FaceLandmarker.createFromOptions(context, faceOptions)
            Log.i(TAG, "FaceLandmarker initialisé")
        } catch (e: Exception) {
            Log.e(TAG, "FaceLandmarker ÉCHEC: ${e.message}")
        }
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // RGBA_8888 : copie directe sans JPEG → ~10× plus rapide que YUV→NV21→JPEG
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
                Log.i(TAG, "CameraX démarré (caméra avant, RGBA_8888)")
            } catch (e: Exception) {
                Log.e(TAG, "Erreur démarrage caméra : ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val timestampMs = frameIndex * 33L

        try {
            val bitmap = imageProxy.toRgbaBitmap() ?: return

            // Log dimensions 1 seule fois pour diagnostiquer l'orientation
            if (frameIndex == 0L) {
                Log.i(TAG, "Bitmap dims: ${bitmap.width}×${bitmap.height}, rotation=${imageProxy.imageInfo.rotationDegrees}°")
            }

            val mpImage = BitmapImageBuilder(bitmap).build()

            val handResult = try {
                handLandmarker?.detectForVideo(mpImage, timestampMs)
            } catch (e: Exception) {
                Log.w(TAG, "HandLandmarker: ${e.message}"); null
            }

            faceFrameCounter = (faceFrameCounter + 1) % FACE_EVERY_N
            if (faceFrameCounter == 0) {
                faceResultCache = try {
                    faceLandmarker?.detectForVideo(mpImage, timestampMs)
                } catch (e: Exception) {
                    Log.w(TAG, "FaceLandmarker: ${e.message}"); faceResultCache
                }
            }

            onFrame(handResult, faceResultCache, timestampMs, bitmap.width, bitmap.height)
            frameIndex++

        } finally {
            imageProxy.close()
        }
    }

    fun stopCamera() {
        cameraProvider?.unbindAll()
    }

    fun close() {
        stopCamera()
        try { handLandmarker?.close() } catch (_: Exception) {}
        try { faceLandmarker?.close() } catch (_: Exception) {}
        analysisExecutor.shutdown()
    }
}

// ─── RGBA_8888 → Bitmap (direct pixel copy, pas de JPEG) ─────────────────────

private fun ImageProxy.toRgbaBitmap(): Bitmap? {
    return try {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        planes[0].buffer.rewind()
        bitmap.copyPixelsFromBuffer(planes[0].buffer)

        val rotation = imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            bitmap.recycle()
            rotated
        } else {
            bitmap
        }
    } catch (e: Exception) {
        Log.e(TAG, "toRgbaBitmap: ${e.message}")
        null
    }
}
