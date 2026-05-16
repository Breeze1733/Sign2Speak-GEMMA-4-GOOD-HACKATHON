package com.sign2speak.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

// ─── App state ────────────────────────────────────────────────────────────────

data class AppState(
    val currentSign: String = "",
    val currentConf: Float = 0f,
    val top3: List<Pair<String, Float>> = emptyList(),
    val pendingGlosses: List<String> = emptyList(),
    val translatedSentence: String = "",
    val isMuted: Boolean = false,
    val style: String = "friendly",
    val isTranslating: Boolean = false,
    val handDetected: Boolean = false,
    val fps: Float = 0f,
    val detectedHands: List<DetectedHand> = emptyList(),
    val faceDots: List<LandmarkPoint> = emptyList(),
    val debugLogs: List<String> = emptyList(),
    val cameraWidth: Int = 720,
    val cameraHeight: Int = 1280,
)

// ─── ViewModel ────────────────────────────────────────────────────────────────

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(AppState())
    val state: StateFlow<AppState> = _state.asStateFlow()

    private val signRecognizer: SignRecognizer by lazy { SignRecognizer(application) }
    private val glossBuffer    = GlossBuffer()
    private val gloss2Text: Gloss2TextEngine by lazy { Gloss2TextEngine(application) }
    private val tts: TtsEngine by lazy { TtsEngine(application) }

    // Single-threaded pipeline: MediaPipe VIDEO mode requires sequential calls
    private val pipelineExecutor   = Executors.newSingleThreadExecutor()
    private val pipelineDispatcher = pipelineExecutor.asCoroutineDispatcher()
    private val pipelineScope      = CoroutineScope(SupervisorJob() + pipelineDispatcher)

    init {
        // Charge Gemma en arrière-plan dès le démarrage (évite freeze au 1er signe)
        viewModelScope.launch(Dispatchers.IO) { gloss2Text.initialize() }
    }

    // Pipeline-only mutable state (never touched from other threads)
    private var lastStableLabel       = ""
    private var lastTranslatedSentence = ""
    private var lastFrameTimeMs       = 0L
    private val fpsSamples            = ArrayDeque<Float>(30)
    private var frameCount            = 0

    // ── Public API ────────────────────────────────────────────────────────────

    fun onFrameProcessed(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        imgW: Int = 720,
        imgH: Int = 1280,
    ) {
        if (imgW > 0 && imgH > 0 && (_state.value.cameraWidth != imgW || _state.value.cameraHeight != imgH)) {
            _state.update { it.copy(cameraWidth = imgW, cameraHeight = imgH) }
            log("CAM: ${imgW}×${imgH}")
        }
        pipelineScope.launch { processFrame(handResult, faceResult, timestampMs) }
    }

    /** Force-flush current pending glosses to translation (skips the 3 s timeout). */
    fun manualTranslate() {
        pipelineScope.launch {
            val glosses = glossBuffer.currentGlosses().also { glossBuffer.clear() }
            if (glosses.isEmpty()) { log("TRANS: no glosses"); return@launch }
            startTranslation(glosses)
        }
    }

    fun toggleMute() {
        _state.update { it.copy(isMuted = !it.isMuted) }
        if (_state.value.isMuted) tts.stop()
    }

    fun setStyle(style: String) { _state.update { it.copy(style = style) } }

    fun clearBuffer() {
        pipelineScope.launch {
            signRecognizer.clearBuffer()
            glossBuffer.clear()
            lastStableLabel        = ""
            lastTranslatedSentence = ""
            _state.update {
                it.copy(
                    currentSign = "", currentConf = 0f, top3 = emptyList(),
                    pendingGlosses = emptyList(), translatedSentence = "",
                    isTranslating = false, handDetected = false,
                    detectedHands = emptyList(), faceDots = emptyList(),
                )
            }
            log("Buffer cleared")
        }
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────

    private fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
    ) {
        // FPS
        val now = System.currentTimeMillis()
        if (lastFrameTimeMs > 0L) {
            fpsSamples.addLast(1000f / (now - lastFrameTimeMs).coerceAtLeast(1L))
            if (fpsSamples.size > 30) fpsSamples.removeFirst()
        }
        lastFrameTimeMs = now
        val avgFps = if (fpsSamples.isNotEmpty()) fpsSamples.average().toFloat() else 0f
        frameCount++

        // Sign recognition (overlay extraction happens inside SignRecognizer)
        val rec         = signRecognizer.processFrame(handResult, faceResult, timestampMs)
        val hasHand     = rec.hasHand
        val currentSign = rec.topLabel
        val currentConf = rec.topConf

        // Log every 60 frames or on sign change
        if (frameCount % 60 == 0) {
            log("Frame $frameCount | ${avgFps.toInt()} fps | hands: ${rec.detectedHands.size}")
        }
        if (currentSign.isNotEmpty() && currentSign != lastStableLabel) {
            log("Sign: $currentSign (${(currentConf * 100).toInt()}%)")
        }

        // Gloss accumulation
        if (currentSign.isNotEmpty() && currentSign != lastStableLabel) {
            if (lastStableLabel.isNotEmpty()) { glossBuffer.push(lastStableLabel); log("Push: $lastStableLabel") }
            lastStableLabel = currentSign
        } else if (currentSign.isEmpty() && lastStableLabel.isNotEmpty()) {
            glossBuffer.push(lastStableLabel); log("Push: $lastStableLabel")
            lastStableLabel = ""
        }

        // Auto-flush after 3 s silence
        glossBuffer.tick()?.takeIf { it.isNotEmpty() }?.let { startTranslation(it) }

        _state.update {
            it.copy(
                currentSign    = currentSign,
                currentConf    = currentConf,
                top3           = rec.top3,
                pendingGlosses = glossBuffer.currentGlosses(),
                handDetected   = hasHand,
                fps            = avgFps,
                detectedHands  = rec.detectedHands,
                faceDots       = rec.faceDots,
            )
        }
    }

    private fun startTranslation(glosses: List<String>) {
        val style = _state.value.style
        log("→ Translate [${glosses.joinToString(" ")}] [$style]")
        viewModelScope.launch(Dispatchers.Default) {
            _state.update { it.copy(isTranslating = true) }
            val sentence = gloss2Text.translate(glosses, style)
            log("← Gemma: $sentence")
            _state.update { it.copy(translatedSentence = sentence, isTranslating = false) }
            if (sentence.isNotBlank() && sentence != lastTranslatedSentence) {
                lastTranslatedSentence = sentence
                if (!_state.value.isMuted) tts.speak(sentence)
            }
        }
    }

    // ── Debug log (thread-safe via StateFlow.update) ──────────────────────────

    private fun log(msg: String) {
        _state.update { s -> s.copy(debugLogs = (s.debugLogs + msg).takeLast(8)) }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        pipelineExecutor.shutdown()
        signRecognizer.close()
        tts.shutdown()
        gloss2Text.close()
    }
}
