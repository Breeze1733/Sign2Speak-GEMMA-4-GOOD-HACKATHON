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
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors

private const val TAG = "CameraManager"

class CameraManager(
    private val context: Context,
    private val onFrame: (handResult: HandLandmarkerResult?, faceResult: FaceLandmarkerResult?, timestampMs: Long, imgW: Int, imgH: Int, perf: PerfFrame) -> Unit,
) {

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val landmarkDetector = LandmarkDetector(context)

    @Volatile private var isDetectorReady = false
    @Volatile private var isRunning = false

    private var frameIndex = 0L
    private var cameraProvider: ProcessCameraProvider? = null

    fun initialize() {
        analysisExecutor.execute {
            landmarkDetector.initialize()
            isDetectorReady = true
        }
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        isRunning = true
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            if (!isRunning) return@addListener
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
        val perfTrial = PerfTrace.current
        val observedNs = PerfTrace.now()

        try {
            if (!isRunning || !isDetectorReady) return

            val timestampMs = frameIndex * 33L
            val bitmap = imageProxy.toRgbaBitmap() ?: return

            if (frameIndex == 0L) {
                Log.i(TAG, "Bitmap dims: ${bitmap.width}×${bitmap.height}, rotation=${imageProxy.imageInfo.rotationDegrees}°")
            }

            val detection = landmarkDetector.detect(bitmap, timestampMs)
            val perf = if (BuildConfig.TIMING_ENABLED) {
                val handFinishedNs = PerfTrace.now()
                PerfTrace.observeHand(
                    PerfFrame(perfTrial, observedNs, handFinishedNs, frameIndex),
                    detection.handResult?.landmarks()?.isNotEmpty() == true,
                )
            } else {
                PerfFrame.Disabled
            }
            onFrame(detection.handResult, detection.faceResult, timestampMs, bitmap.width, bitmap.height, perf)
            frameIndex++
        } finally {
            imageProxy.close()
        }
    }

    fun stopCamera() {
        isRunning = false
        cameraProvider?.unbindAll()
    }

    fun close() {
        stopCamera()
        isDetectorReady = false
        analysisExecutor.execute {
            landmarkDetector.close()
        }
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
