package com.sign2speak.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
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
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

private const val TAG = "CameraManager"

// Face landmarks toutes les N frames (optimisation CPU)
private const val FACE_EVERY_N = 2

/**
 * CameraManager — gère le pipeline CameraX + MediaPipe Tasks.
 *
 * Modèles requis dans app/src/main/assets/ :
 *   - hand_landmarker.task  (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
 *   - face_landmarker.task  (https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
 *
 * IMPORTANT : MediaPipe VIDEO mode doit être appelé depuis un seul thread en
 * séquence avec des timestamps strictement croissants. On utilise un
 * single-threaded ExecutorService et des timestamps basés sur frameIndex * 33L.
 */
class CameraManager(
    private val context: Context,
    private val onFrame: (handResult: HandLandmarkerResult?, faceResult: FaceLandmarkerResult?, timestampMs: Long) -> Unit,
) {

    // Single-threaded executor pour garantir la séquentialité MediaPipe VIDEO mode
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
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath("hand_landmarker.task")
                        .build()
                )
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.2f)
                .setMinHandPresenceConfidence(0.2f)
                .setMinTrackingConfidence(0.2f)
                .build()
            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.i(TAG, "HandLandmarker initialisé")
        } catch (e: Exception) {
            Log.e(TAG, "HandLandmarker ÉCHEC — hand_landmarker.task absent des assets ? ${e.message}")
        }

        try {
            val faceOptions = FaceLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath("face_landmarker.task")
                        .build()
                )
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.3f)
                .setMinFacePresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()
            faceLandmarker = FaceLandmarker.createFromOptions(context, faceOptions)
            Log.i(TAG, "FaceLandmarker initialisé")
        } catch (e: Exception) {
            Log.e(TAG, "FaceLandmarker ÉCHEC — face_landmarker.task absent des assets ? ${e.message}")
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
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
                Log.i(TAG, "CameraX démarré (caméra avant)")
            } catch (e: Exception) {
                Log.e(TAG, "Erreur démarrage caméra : ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Traite une frame caméra.
     * Appelé depuis analysisExecutor (single-threaded) → thread-safe pour MediaPipe.
     * TOUJOURS fermer imageProxy à la fin.
     */
    private fun processFrame(imageProxy: ImageProxy) {
        // Timestamp monotone basé sur l'index de frame (33 ms ≈ 30 fps)
        // Ne pas utiliser imageProxy.imageInfo.timestamp car il peut ne pas être monotone.
        val timestampMs = frameIndex * 33L

        try {
            val bitmap = imageProxy.toBitmap() ?: return
            val mpImage = BitmapImageBuilder(bitmap).build()

            // Hand : chaque frame
            val handResult = try {
                handLandmarker?.detectForVideo(mpImage, timestampMs)
            } catch (e: Exception) {
                Log.w(TAG, "HandLandmarker erreur : ${e.message}")
                null
            }

            // Face : toutes les FACE_EVERY_N frames (cache entre-temps)
            // En VIDEO mode, detectForVideo doit recevoir des timestamps strictement croissants.
            // On passe le même timestampMs que pour la main (déjà monotone).
            faceFrameCounter = (faceFrameCounter + 1) % FACE_EVERY_N
            if (faceFrameCounter == 0) {
                faceResultCache = try {
                    faceLandmarker?.detectForVideo(mpImage, timestampMs)
                } catch (e: Exception) {
                    Log.w(TAG, "FaceLandmarker erreur : ${e.message}")
                    faceResultCache
                }
            }
            val faceResult = faceResultCache

            onFrame(handResult, faceResult, timestampMs)
            frameIndex++

        } finally {
            // CRITIQUE : toujours fermer l'ImageProxy pour libérer le buffer caméra
            imageProxy.close()
        }
    }

    fun stopCamera() {
        cameraProvider?.unbindAll()
        Log.i(TAG, "CameraX arrêté")
    }

    fun close() {
        stopCamera()
        try { handLandmarker?.close() } catch (_: Exception) {}
        try { faceLandmarker?.close() } catch (_: Exception) {}
        analysisExecutor.shutdown()
    }
}

// ─── Extension : YUV_420_888 → Bitmap ────────────────────────────────────────

/**
 * Convertit un ImageProxy (YUV_420_888) en Bitmap ARGB_8888.
 * On passe par NV21 → JPEG → Bitmap pour la compatibilité maximale.
 */
private fun ImageProxy.toBitmap(): Bitmap? {
    return try {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 85, out)
        val jpegBytes = out.toByteArray()

        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

        // Rotation pour compenser l'orientation de la caméra avant
        val rotation = imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    } catch (e: Exception) {
        Log.e(TAG, "Erreur conversion ImageProxy → Bitmap : ${e.message}")
        null
    }
}
