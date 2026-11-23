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

// ─── Layout constants (canonical definitions live in LandmarkLayout.kt) ─────

private const val NUM_NODES = LandmarkLayout.NUM_NODES
private const val CHANNELS = LandmarkLayout.CHANNELS
private const val MAX_LEN = LandmarkLayout.MAX_LEN
private const val PAD = LandmarkLayout.PAD
private const val NOSE_HOLISTIC_IDX = LandmarkLayout.NOSE_HOLISTIC_IDX
private const val PREDICTION_HISTORY_SIZE = LandmarkLayout.PREDICTION_HISTORY_SIZE
private const val DEFAULT_THRESHOLD = LandmarkLayout.DEFAULT_THRESHOLD
private const val INFER_EVERY_N = LandmarkLayout.INFER_EVERY_N
private const val MIN_FRAMES_FOR_INFER = LandmarkLayout.MIN_FRAMES_FOR_INFER
private val POINT_LANDMARKS: IntArray = LandmarkLayout.POINT_LANDMARKS

data class RecognitionResult(
    val topLabel: String,
    val topConf: Float,
    val top3: List<Pair<String, Float>>,
    val hasHand: Boolean,
    val inferenceSerial: Long = 0,
    val inferenceCount: Long = 0,
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
    private var inferenceCount = 0L
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

        val holistic = HolisticAssembler.assemble(handResult, faceResult)
        frameBuffer.addLast(holistic)
        if (frameBuffer.size > MAX_LEN) frameBuffer.removeFirst()

        val hasHand = HolisticAssembler.hasHand(handResult)

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
            inferenceCount = inferenceCount,
            detectedHands = HolisticAssembler.extractHandOverlay(handResult),
            faceDots      = HolisticAssembler.extractFaceOverlay(faceResult),
        )
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
        inferenceCount++
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
        inferenceCount = 0L
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
