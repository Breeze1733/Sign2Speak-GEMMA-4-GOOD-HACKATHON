package com.sign2speak.app

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import kotlin.math.sqrt

private const val TAG = "SignRecognizer"

// ─── Constantes du preprocessing (identiques à live_winner.py) ─────────────

/** Nombre total de rows dans l'assemblage Holistic. */
private const val ROWS_PER_FRAME = 543

/** holistic[0..467] = face landmarks (478 points en tout, 0-based index 0..477). */
/** Pour les mains, on utilise la convention MediaPipe Tasks vs Holistic :
 *  "Right" (caméra POV) → offset 468 (left_hand dans l'index Holistic gagnant)
 *  "Left"  (caméra POV) → offset 522 (right_hand dans l'index Holistic gagnant)
 */
private const val LEFT_HAND_OFFSET  = 468   // holistic index for left hand
private const val RIGHT_HAND_OFFSET = 522   // holistic index for right hand
private const val HAND_NUM_LANDMARKS = 21

private val NOSE  = intArrayOf(1, 2, 98, 327)
private val LIP   = intArrayOf(
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308
)
private val REYE  = intArrayOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
private val LEYE  = intArrayOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
private val LHAND = IntArray(21) { 468 + it }  // 21 pts
private val RHAND = IntArray(21) { 522 + it }  // 21 pts

/** 118 landmarks utilisés pour la classification. */
private val POINT_LANDMARKS: IntArray = LIP + LHAND + RHAND + NOSE + REYE + LEYE

private const val NUM_NODES = 118       // LIP(40) + LHAND(21) + RHAND(21) + NOSE(4) + REYE(16) + LEYE(16)
private const val CHANNELS  = 708       // 6 * 118
private const val MAX_LEN   = 64
private const val PAD       = -100f

/** Index du nez pour le centrage (holistic[17] = LIP[15]). */
private const val NOSE_HOLISTIC_IDX = 17

/** Nombre de prédictions conservées pour le vote majoritaire. */
private const val PREDICTION_HISTORY_SIZE = 8

/** Seuil minimal de confiance pour valider une prédiction. */
private const val DEFAULT_THRESHOLD = 0.2f

/** On soumet une inférence toutes les N frames. */
private const val INFER_EVERY_N = 3

/** Nb minimum de frames dans le buffer avant d'inférer. */
private const val MIN_FRAMES_FOR_INFER = 10

private val FACE_OVERLAY_IDX = listOf(
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

data class RecognitionResult(
    val topLabel: String,
    val topConf: Float,
    val top3: List<Pair<String, Float>>,
    val hasHand: Boolean,
    val inferenceSerial: Long = 0,
    val detectedHands: List<DetectedHand> = emptyList(),
    val faceDots: List<LandmarkPoint> = emptyList(),
)

/**
 * SignRecognizer — cœur du pipeline de reconnaissance.
 */
class SignRecognizer(private val context: Context) {

    private var idxToLabel: Map<Int, String> = emptyMap()
    private var numClasses: Int = 0

    private var interpreter: Interpreter? = null

    private val frameBuffer = ArrayDeque<FloatArray>(MAX_LEN + 1)
    private val predictionHistory = ArrayDeque<Int>(PREDICTION_HISTORY_SIZE + 1)

    private var inferenceSerial = 0L
    private var frameIndex = 0
    private var currentLabel = ""
    private var currentConf  = 0f
    private var currentTop3  = emptyList<Pair<String, Float>>()

    init {
        try {
            val metaJson = context.assets.open("model_meta.json").bufferedReader().readText()
            val metaObj  = JSONObject(metaJson)
            numClasses   = metaObj.getInt("num_classes")
            val idxObj   = metaObj.getJSONObject("idx_to_label")
            idxToLabel   = buildMap {
                idxObj.keys().forEach { k -> put(k.toInt(), idxObj.getString(k)) }
            }
            Log.i(TAG, "Métadonnées : $numClasses classes")
        } catch (e: Exception) {
            Log.e(TAG, "model_meta.json absent : ${e.message}")
        }

        try {
            val modelBuffer = loadModelFile("classifier.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Modèle LiteRT chargé ($numClasses classes, CPU)")
        } catch (e: Exception) {
            Log.e(TAG, "classifier.tflite erreur : ${e.message}")
        }
    }

    fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long
    ): RecognitionResult {

        val holistic = assembleHolistic(handResult, faceResult)
        frameBuffer.addLast(holistic)
        if (frameBuffer.size > MAX_LEN) frameBuffer.removeFirst()

        val hasHand = handResult != null && handResult.landmarks().isNotEmpty()

        if (interpreter != null && frameIndex % INFER_EVERY_N == 0 && frameBuffer.size >= MIN_FRAMES_FOR_INFER) {
            runInference(hasHand)
        }

        frameIndex++

        return RecognitionResult(
            topLabel     = currentLabel,
            topConf      = currentConf,
            top3         = currentTop3,
            hasHand      = hasHand,
            inferenceSerial = inferenceSerial,
            detectedHands = extractHandOverlay(handResult),
            faceDots      = extractFaceOverlay(faceResult),
        )
    }

    private fun extractHandOverlay(r: HandLandmarkerResult?): List<DetectedHand> {
        if (r == null) return emptyList()
        val lmsList = r.landmarks()
        if (lmsList.isEmpty()) return emptyList()
        return lmsList.mapIndexed { i, lms ->
            // Front camera on this device: MediaPipe "Left" = user's left hand
            val isLeft = r.handedness().getOrNull(i)?.firstOrNull()?.categoryName() == "Left"
            DetectedHand(lms.map { lm -> LandmarkPoint(lm.x(), lm.y()) }, isLeft)
        }
    }

    private fun extractFaceOverlay(r: FaceLandmarkerResult?): List<LandmarkPoint> {
        if (r == null) return emptyList()
        val faces = r.faceLandmarks()
        if (faces.isEmpty()) return emptyList()
        val lms = faces[0]
        return FACE_OVERLAY_IDX.mapNotNull { idx ->
            if (idx < lms.size) LandmarkPoint(lms[idx].x(), lms[idx].y()) else null
        }
    }

    private fun assembleHolistic(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
    ): FloatArray {
        val holistic = FloatArray(ROWS_PER_FRAME * 3) { Float.NaN }

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
            // Match live_winner.py: no score filter, take all detected hands (max 2)
            for (hIdx in 0 until minOf(handResult.landmarks().size, 2)) {
                if (hIdx >= handResult.handedness().size) break
                val label = handResult.handedness()[hIdx][0].categoryName()
                val lms   = handResult.landmarks()[hIdx]
                // "Left" = user left hand → LHAND offset 468 (matches live_winner.py LHAND)
                // "Right" = user right hand → RHAND offset 522 (matches live_winner.py RHAND)
                val offset = if (label == "Left") LEFT_HAND_OFFSET else RIGHT_HAND_OFFSET
                for (i in 0 until minOf(lms.size, HAND_NUM_LANDMARKS)) {
                    val lm = lms[i]
                    holistic[(offset + i) * 3 + 0] = lm.x()
                    holistic[(offset + i) * 3 + 1] = lm.y()
                    holistic[(offset + i) * 3 + 2] = lm.z()
                }
            }
        }

        return holistic
    }

    private fun runInference(hasHand: Boolean) {
        val seqList = frameBuffer.toList()
        val T = seqList.size

        val pts = Array(T) { t ->
            val frame = seqList[t]
            Array(NUM_NODES) { ni ->
                val gi = POINT_LANDMARKS[ni]
                floatArrayOf(
                    frame[gi * 3 + 0],
                    frame[gi * 3 + 1],
                    frame[gi * 3 + 2],
                )
            }
        }

        val validFrameIndices = (0 until T).filter { t ->
            pts[t].any { lm -> !lm[0].isNaN() }
        }
        if (validFrameIndices.size < 3) return

        val validPts = validFrameIndices.map { t -> pts[t] }.toTypedArray()
        val Tv = validPts.size

        val noseNodeIdx = indexOfInArray(POINT_LANDMARKS, NOSE_HOLISTIC_IDX).takeIf { it >= 0 } ?: 15

        val noseValues = Array(Tv) { t -> validPts[t][noseNodeIdx] }
        val meanNose = FloatArray(3) { c ->
            val vals = noseValues.mapNotNull { it[c].takeUnless { v -> v.isNaN() } }
            if (vals.isEmpty()) 0.5f else vals.average().toFloat()
        }

        val stdPerCoord = FloatArray(3) { c ->
            var sumSq = 0f
            var cnt = 0
            for (t in 0 until Tv) {
                for (ni in 0 until NUM_NODES) {
                    val v = validPts[t][ni][c]
                    if (!v.isNaN()) {
                        val d = v - meanNose[c]
                        sumSq += d * d
                        cnt++
                    }
                }
            }
            if (cnt > 0) {
                val s = sqrt(sumSq / cnt)
                if (s < 1e-6f) 1f else s
            } else 1f
        }

        val xNorm = Array(Tv) { t ->
            Array(NUM_NODES) { ni ->
                FloatArray(2) { c ->
                    val v = validPts[t][ni][c]
                    if (v.isNaN()) Float.NaN else (v - meanNose[c]) / stdPerCoord[c]
                }
            }
        }

        val dx = Array(Tv) { Array(NUM_NODES) { FloatArray(2) } }
        for (t in 0 until Tv - 1) {
            for (ni in 0 until NUM_NODES) {
                for (c in 0 until 2) {
                    val cur  = xNorm[t][ni][c]
                    val next = xNorm[t + 1][ni][c]
                    dx[t][ni][c] = if (cur.isNaN() || next.isNaN()) Float.NaN else next - cur
                }
            }
        }

        val dx2 = Array(Tv) { Array(NUM_NODES) { FloatArray(2) } }
        for (t in 0 until Tv - 2) {
            for (ni in 0 until NUM_NODES) {
                for (c in 0 until 2) {
                    val cur  = xNorm[t][ni][c]
                    val far  = xNorm[t + 2][ni][c]
                    dx2[t][ni][c] = if (cur.isNaN() || far.isNaN()) Float.NaN else far - cur
                }
            }
        }

        val features = Array(Tv) { t ->
            FloatArray(CHANNELS) { ch ->
                val block = ch / (NUM_NODES * 2)
                val rest  = ch % (NUM_NODES * 2)
                val ni    = rest / 2
                val c     = rest % 2
                val v = when (block) {
                    0    -> xNorm[t][ni][c]
                    1    -> dx[t][ni][c]
                    else -> dx2[t][ni][c]
                }
                if (v.isNaN()) 0f else v
            }
        }

        val padded = Array(MAX_LEN) { t ->
            if (t < Tv) features[t]
            else FloatArray(CHANNELS) { PAD }
        }

        val interp = interpreter ?: return
        val inputArray = arrayOf(padded)
        val outputArray = Array(1) { FloatArray(numClasses) }

        try {
            interp.run(inputArray, outputArray)
        } catch (e: Exception) {
            Log.e(TAG, "Erreur inférence LiteRT : ${e.message}")
            return
        }

        if (BuildConfig.TIMING_ENABLED) inferenceSerial++
        val logits = outputArray[0]
        val probs  = softmax(logits)

        val topIdx  = probs.indices.maxByOrNull { probs[it] } ?: 0
        val topConf = probs[topIdx]

        val sortedIdx = probs.indices.sortedByDescending { probs[it] }.take(3)
        val top3 = sortedIdx.map { i ->
            Pair(idxToLabel.getOrDefault(i, "?"), probs[i])
        }

        if (topConf >= DEFAULT_THRESHOLD && hasHand) {
            predictionHistory.addLast(topIdx)
            if (predictionHistory.size > PREDICTION_HISTORY_SIZE) predictionHistory.removeFirst()

            val votes = predictionHistory.groupingBy { it }.eachCount()
            val bestIdx = votes.maxByOrNull { it.value }?.key ?: topIdx
            currentLabel = idxToLabel.getOrDefault(bestIdx, "?")
            currentConf  = topConf
        } else {
            if (!hasHand) predictionHistory.clear()
            currentLabel = ""
            currentConf  = 0f
        }
        currentTop3 = top3
    }

    fun clearBuffer() {
        frameBuffer.clear()
        predictionHistory.clear()
        currentLabel = ""
        currentConf  = 0f
        currentTop3  = emptyList()
        frameIndex   = 0
    }

    fun close() {
        try { interpreter?.close() } catch (_: Exception) {}
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fd = context.assets.openFd(fileName)
        return FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exp = FloatArray(logits.size) { i -> Math.exp((logits[i] - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return FloatArray(exp.size) { i -> exp[i] / sum }
    }

    private fun indexOfInArray(arr: IntArray, value: Int): Int {
        for (i in arr.indices) if (arr[i] == value) return i
        return -1
    }
}
