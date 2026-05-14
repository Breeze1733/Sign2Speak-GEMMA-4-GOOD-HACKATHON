package com.sign2speak.app

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Interpreter
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
private const val LEFT_HAND_OFFSET  = 468   // caméra voit "Right" → main gauche utilisateur
private const val RIGHT_HAND_OFFSET = 522   // caméra voit "Left"  → main droite utilisateur
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
private val LHAND = (468..488).toIntArray()  // 21 pts
private val RHAND = (522..542).toIntArray()  // 21 pts

/** 118 landmarks utilisés pour la classification. */
private val POINT_LANDMARKS: IntArray = LIP + LHAND + RHAND + NOSE + REYE + LEYE

private const val NUM_NODES = 118       // LIP(40) + LHAND(21) + RHAND(21) + NOSE(4) + REYE(16) + LEYE(16)
private const val CHANNELS  = 708       // 6 * 118
private const val MAX_LEN   = 64
private const val PAD       = -100f

/** Longueur brute d'un frame Holistic : 543 points × 3 coords = 1629 floats. */
private const val FRAME_FLAT_SIZE = ROWS_PER_FRAME * 3

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

data class RecognitionResult(
    val topLabel: String,
    val topConf: Float,
    val top3: List<Pair<String, Float>>,
    val hasHand: Boolean,
)

/**
 * SignRecognizer — cœur du pipeline de reconnaissance.
 *
 * IMPORTANT : appelez [processFrame] depuis un seul thread (ou protégez avec un Mutex).
 * CameraManager garantit cette contrainte via un ExecutorService single-threaded.
 *
 * Placez dans app/src/main/assets/ :
 *   - classifier.tflite   (généré par convert_to_tflite.py)
 *   - model_meta.json     (copie depuis model_winner/)
 */
class SignRecognizer(private val context: Context) {

    // ─── Métadonnées ─────────────────────────────────────────────────────────
    private var idxToLabel: Map<Int, String> = emptyMap()
    private var numClasses: Int = 0

    // ─── LiteRT Interpreter (nullable — null si les assets sont absents) ─────
    private var interpreter: Interpreter? = null

    // ─── Buffers ─────────────────────────────────────────────────────────────
    /** Buffer circulaire de frames brutes Holistic (MAX_LEN frames max). */
    private val frameBuffer = ArrayDeque<FloatArray>(MAX_LEN + 1)

    /** Historique des prédictions pour le vote majoritaire. */
    private val predictionHistory = ArrayDeque<Int>(PREDICTION_HISTORY_SIZE + 1)

    private var frameIndex = 0
    private var currentLabel = ""
    private var currentConf  = 0f
    private var currentTop3  = emptyList<Pair<String, Float>>()

    val isReady: Boolean get() = interpreter != null

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
            Log.e(TAG, "model_meta.json absent des assets : ${e.message}")
        }

        try {
            val modelBuffer = loadModelFile("classifier.tflite")
            val options = Interpreter.Options().apply { numThreads = 4 }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Modèle LiteRT chargé ($numClasses classes)")
        } catch (e: Exception) {
            Log.e(TAG, "classifier.tflite absent des assets — inférence désactivée : ${e.message}")
        }
    }

    /**
     * Traite une frame MediaPipe Tasks et retourne le résultat courant.
     * Doit être appelé depuis un thread unique (thread d'analyse caméra).
     */
    fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long
    ): RecognitionResult {

        // 1. Assemblage Holistic 543×3
        val holistic = assembleHolistic(handResult, faceResult)
        frameBuffer.addLast(holistic)
        if (frameBuffer.size > MAX_LEN) frameBuffer.removeFirst()

        val hasHand = handResult != null && handResult.handLandmarks().isNotEmpty()

        // 2. Inférence toutes les INFER_EVERY_N frames (si le modèle est chargé)
        if (interpreter != null && frameIndex % INFER_EVERY_N == 0 && frameBuffer.size >= MIN_FRAMES_FOR_INFER) {
            runInference(hasHand)
        }

        frameIndex++

        return RecognitionResult(
            topLabel = currentLabel,
            topConf  = currentConf,
            top3     = currentTop3,
            hasHand  = hasHand,
        )
    }

    /** Assemble les résultats MediaPipe Tasks en tableau Holistic 543×3. */
    private fun assembleHolistic(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
    ): FloatArray {
        val holistic = FloatArray(ROWS_PER_FRAME * 3) { Float.NaN }

        // Face landmarks (0..467)
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

        // Hand landmarks
        // Convention MediaPipe Tasks : handedness label est depuis le POV de la caméra
        //   "Right" (caméra) → c'est la main droite de la caméra = main gauche de l'utilisateur → offset 468
        //   "Left"  (caméra) → c'est la main gauche de la caméra = main droite de l'utilisateur → offset 522
        if (handResult != null && handResult.handLandmarks().isNotEmpty()) {
            for (hIdx in handResult.handLandmarks().indices) {
                if (hIdx >= handResult.handedness().size) break
                val label = handResult.handedness()[hIdx][0].categoryName()
                val offset = if (label == "Right") LEFT_HAND_OFFSET else RIGHT_HAND_OFFSET
                val lms = handResult.handLandmarks()[hIdx]
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

    /** Lance l'inférence sur le buffer courant (synchrone, dans le thread appelant). */
    private fun runInference(hasHand: Boolean) {
        // Snapshot du buffer
        val seqList = frameBuffer.toList()
        val T = seqList.size

        // Extraction des POINT_LANDMARKS et filtrage des frames toutes-NaN
        // shape : (T, 543, 3) → extrait (T, 118, 3)
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

        // Garde uniquement les frames qui ont au moins un point non-NaN
        val validFrameIndices = (0 until T).filter { t ->
            pts[t].any { lm -> !lm[0].isNaN() }
        }
        if (validFrameIndices.size < 3) return

        val validPts = validFrameIndices.map { t -> pts[t] }.toTypedArray()
        val Tv = validPts.size

        // ── Preprocessing (identique à preprocess_sequence dans live_winner.py) ──

        // Nez de référence : holistic[17] (= POINT_LANDMARKS[15] puisque NOSE_HOLISTIC_IDX=17 est LIP[15])
        // On prend la moyenne du nez sur toutes les frames valides
        val noseNodeIdx = indexOfInArray(POINT_LANDMARKS, NOSE_HOLISTIC_IDX)
            .takeIf { it >= 0 } ?: 15   // fallback : LIP[15] = holistic[17]

        // mean_nose = nanmean sur (T, 118, 3) → (1, 1, 3) : moyenne temporelle du nez
        val noseValues = Array(Tv) { t -> validPts[t][noseNodeIdx] }
        val meanNose = FloatArray(3) { c ->
            val vals = noseValues.mapNotNull { it[c].takeUnless { v -> v.isNaN() } }
            if (vals.isEmpty()) 0.5f else vals.average().toFloat()
        }

        // std PER COORDINATE (identique à numpy axis=(1,2) sur (1,T,118,3))
        // std_c = sqrt(nanmean((val_c - mean_nose_c)^2)) sur tous les T et les 118 nodes
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

        // x normalisé, keepdims sur x,y uniquement → (Tv, 118, 2)
        // Chaque coordonnée est divisée par son propre std (identique à Python)
        val xNorm = Array(Tv) { t ->
            Array(NUM_NODES) { ni ->
                FloatArray(2) { c ->
                    val v = validPts[t][ni][c]
                    if (v.isNaN()) Float.NaN else (v - meanNose[c]) / stdPerCoord[c]
                }
            }
        }

        // dx[i] = x[i+1] - x[i]   (velocity)
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

        // dx2[i] = x[i+2] - x[i]  (acceleration)
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

        // features = concat(x, dx, dx2) reshape → (Tv, 708)
        // NaN → 0.0
        val features = Array(Tv) { t ->
            FloatArray(CHANNELS) { ch ->
                val block = ch / (NUM_NODES * 2)   // 0=x, 1=dx, 2=dx2
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

        // Padding / troncature à MAX_LEN
        val padded = Array(MAX_LEN) { t ->
            if (t < Tv) features[t]
            else FloatArray(CHANNELS) { PAD }
        }

        // ── Inférence LiteRT ──────────────────────────────────────────────────
        val interp = interpreter ?: return
        val inputArray = arrayOf(padded)
        val outputArray = Array(1) { FloatArray(numClasses) }

        try {
            interp.run(inputArray, outputArray)
        } catch (e: Exception) {
            Log.e(TAG, "Erreur inférence LiteRT : ${e.message}")
            return
        }

        val logits = outputArray[0]
        val probs  = softmax(logits)

        val topIdx  = probs.indices.maxByOrNull { probs[it] } ?: 0
        val topConf = probs[topIdx]

        // Top 3
        val sortedIdx = probs.indices.sortedByDescending { probs[it] }.take(3)
        val top3 = sortedIdx.map { i ->
            Pair(idxToLabel.getOrDefault(i, "?"), probs[i])
        }

        // Vote majoritaire sur les 8 dernières prédictions
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

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fd = context.assets.openFd(fileName)
        return FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exp = FloatArray(logits.size) { i -> Math.exp((logits[i] - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return FloatArray(exp.size) { i -> exp[i] / sum }
    }

    /** Cherche l'index d'une valeur dans un IntArray. */
    private fun indexOfInArray(arr: IntArray, value: Int): Int {
        for (i in arr.indices) if (arr[i] == value) return i
        return -1
    }
}

// Note : IntArray.plus() est fourni par la stdlib Kotlin — pas besoin de redéfinir.
