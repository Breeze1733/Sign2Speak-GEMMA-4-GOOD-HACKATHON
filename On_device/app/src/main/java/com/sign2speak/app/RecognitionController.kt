package com.sign2speak.app

import android.content.Context
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/** Per-frame context captured when a window is submitted. */
data class RecognitionContext(
    val perf: PerfFrame,
    val trial: PerfTrial,
    val sentenceId: Long,
    val wordIndex: Int,
    val submittedNs: Long,
    val windowEndObservedNs: Long,
    val recognitionEpoch: Long = 0L,
)

/** Immediate per-frame output: hand presence and overlays. */
data class RecognitionFrame(
    val hasHand: Boolean,
    val detectedHands: List<DetectedHand>,
    val faceDots: List<LandmarkPoint>,
)

/** RecognitionEvent (M3 -> M4), from local inference or the P2 service. */
data class RecognitionEvent(
    val topLabel: String,
    val topConf: Float,
    val top3: List<Pair<String, Float>>,
    val hasHand: Boolean,
    val inferred: Boolean,
    val source: String,
    val windowId: Long,
    val resultSerial: Long,
    val status: String,
    val clientMs: Double = 0.0,
    val serverQueueMs: Double? = null,
    val serverPreprocessMs: Double? = null,
    val serverInferenceMs: Double? = null,
    val serverVoteMs: Double? = null,
    val serverTotalMs: Double? = null,
    val error: String? = null,
)

interface RecognitionController {
    fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        ctx: RecognitionContext,
    ): RecognitionFrame

    /** Current utterance epoch; results from an older epoch must be dropped. */
    fun currentEpoch(): Long = 0L

    /** Last window id submitted (manual TRANS suppresses results up to it). */
    fun lastWindowId(): Long = -1L

    /** Thread-safe invalidation without touching the window deque. */
    fun invalidate() {}

    fun clear()

    /** Called on 3 s auto-flush; invalidates in-flight results. */
    fun endUtterance() {}

    fun suppressBefore(windowId: Long) {}

    fun close()
}

class LocalRecognitionController(
    context: Context,
    private val onEvent: (RecognitionEvent, RecognitionContext) -> Unit,
) : RecognitionController {

    private val recognizer = SignRecognizer(context)
    private var lastSerial = -1L

    override fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        ctx: RecognitionContext,
    ): RecognitionFrame {
        val rec = recognizer.processFrame(handResult, faceResult, timestampMs)
        if (rec.inferenceCount != lastSerial) {
            lastSerial = rec.inferenceCount
            onEvent(
                RecognitionEvent(
                    topLabel = rec.topLabel,
                    topConf = rec.topConf,
                    top3 = rec.top3,
                    hasHand = rec.hasHand,
                    inferred = true,
                    source = "local",
                    windowId = rec.inferenceCount,
                    resultSerial = rec.inferenceCount,
                    status = "ok",
                ),
                ctx,
            )
        }
        return RecognitionFrame(rec.hasHand, rec.detectedHands, rec.faceDots)
    }

    override fun clear() {
        recognizer.clearBuffer()
        lastSerial = -1L
    }

    override fun close() = recognizer.close()
}
