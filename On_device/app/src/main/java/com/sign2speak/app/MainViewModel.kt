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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

// ─── History entry ────────────────────────────────────────────────────────────

data class HistoryEntry(
    val id: Long,
    val glosses: List<String>,
    val interpretation: String,
)

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
    val history: List<HistoryEntry> = emptyList(),
    val heardText: String = "",
    val isListening: Boolean = false,
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
    private val speechEngine: SpeechEngine by lazy { SpeechEngine(application) }

    private val historyFile = File(application.cacheDir, "interpretation_history.json")

    private val pipelineExecutor   = Executors.newSingleThreadExecutor()
    private val pipelineDispatcher = pipelineExecutor.asCoroutineDispatcher()
    private val pipelineScope      = CoroutineScope(SupervisorJob() + pipelineDispatcher)

    private var perfLastInference = -1L
    private var perfSentenceSeq = 1L
    private var perfWordOrdinal = 0
    private var perfLastEpisode = -1L
    private var perfWordRecorded = false
    private var perfWordEnded = false
    private var lastStableLabel        = ""
    private var lastTranslatedSentence = ""
    private var lastFrameTimeMs        = 0L
    private val fpsSamples             = ArrayDeque<Float>(30)
    private var frameCount             = 0
    private var lastHandSeenMs         = 0L

    init {
        PerfTrace.begin("", "app_start")
        viewModelScope.launch(Dispatchers.IO) { gloss2Text.initialize() }
        viewModelScope.launch(Dispatchers.IO) {
            _state.update { it.copy(history = loadHistory()) }
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    fun onFrameProcessed(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        imgW: Int = 720,
        imgH: Int = 1280,
        perf: PerfFrame = PerfFrame(PerfTrace.current, PerfTrace.now(), PerfTrace.now(), -1),
    ) {
        if (imgW > 0 && imgH > 0 && (_state.value.cameraWidth != imgW || _state.value.cameraHeight != imgH)) {
            _state.update { it.copy(cameraWidth = imgW, cameraHeight = imgH) }
        }
        val enqueuedNs = PerfTrace.now()
        pipelineScope.launch { processFrame(handResult, faceResult, timestampMs, perf, enqueuedNs) }
    }

    fun beginPerfTrial(expected: String) {
        // Clear first on the existing pipeline, then mark the new trial.
        clearBuffer()
        pipelineScope.launch {
            perfSentenceSeq = 1L
            perfWordOrdinal = 0
            perfLastEpisode = -1L
            PerfTrace.begin(expected)
        }
    }

    fun markPerfGestureEnd() { PerfTrace.event("manual_gesture_end") }

    fun manualTranslate() {
        val submittedNs = PerfTrace.now()
        val submittedTrial = PerfTrace.current
        pipelineScope.launch {
            val glosses = glossBuffer.currentGlosses().also { glossBuffer.clear() }
            if (glosses.isEmpty()) return@launch
            startTranslation(glosses, "manual", submittedNs, submittedTrial)
        }
    }

    fun toggleMute() {
        _state.update { it.copy(isMuted = !it.isMuted) }
        if (_state.value.isMuted) tts.stop()
    }

    fun setStyle(style: String) { _state.update { it.copy(style = style) } }

    /** Démarre / arrête l'écoute de la personne entendante. Appelé depuis le main thread. */
    fun toggleListening() {
        if (_state.value.isListening) {
            speechEngine.stopListening()
            _state.update { it.copy(isListening = false) }
        } else {
            _state.update { it.copy(heardText = "") }
            speechEngine.startListening(
                onPartial     = { text -> _state.update { it.copy(heardText = text) } },
                onResult      = { text -> _state.update { it.copy(heardText = text, isListening = false) } },
                onStateChange = { listening -> _state.update { it.copy(isListening = listening) } },
            )
        }
    }

    /** Rejoue vocalement une interprétation de l'historique. */
    fun replaySentence(sentence: String) {
        if (!_state.value.isMuted) tts.speak(sentence)
    }

    fun clearBuffer() {
        pipelineScope.launch {
            signRecognizer.clearBuffer()
            glossBuffer.clear()
            lastStableLabel        = ""
            lastTranslatedSentence = ""
            lastHandSeenMs         = 0L
            _state.update {
                it.copy(
                    currentSign = "", currentConf = 0f, top3 = emptyList(),
                    pendingGlosses = emptyList(), translatedSentence = "",
                    isTranslating = false, handDetected = false,
                    detectedHands = emptyList(), faceDots = emptyList(),
                )
            }
        }
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────

    private fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        perf: PerfFrame,
        enqueuedNs: Long,
    ) {
        val processingNs = PerfTrace.now()
        val now = System.currentTimeMillis()
        if (lastFrameTimeMs > 0L) {
            fpsSamples.addLast(1000f / (now - lastFrameTimeMs).coerceAtLeast(1L))
            if (fpsSamples.size > 30) fpsSamples.removeFirst()
        }
        lastFrameTimeMs = now
        val avgFps = if (fpsSamples.isNotEmpty()) fpsSamples.average().toFloat() else 0f
        frameCount++

        val rec         = signRecognizer.processFrame(handResult, faceResult, timestampMs)
        val hasHand     = rec.hasHand
        val currentSign = rec.topLabel
        val currentConf = rec.topConf
        if (BuildConfig.TIMING_ENABLED && hasHand && perf.episodeId != perfLastEpisode) {
            perfLastEpisode = perf.episodeId
            perfWordOrdinal++
            perfWordRecorded = false
            perfWordEnded = false
            PerfTrace.event("word_started", perf.trial,
                "sentence_id" to perfSentenceSeq, "word_index" to perfWordOrdinal,
                "episode_id" to perf.episodeId, "start_ns" to perf.firstHandNs.toString(),
                "hand_seen_ns" to perf.firstHandDetectedNs.toString())
        }
        if (!hasHand && perfWordOrdinal > 0 && !perfWordEnded) {
            perfWordEnded = true
            PerfTrace.event("word_hand_removed", perf.trial,
                "sentence_id" to perfSentenceSeq, "word_index" to perfWordOrdinal,
                "episode_id" to perf.episodeId, "removed_ns" to perf.observedNs.toString())
        }
        if (rec.inferenceSerial != perfLastInference) {
            perfLastInference = rec.inferenceSerial
            if (currentSign.isNotEmpty()) {
                PerfTrace.event("recognition_result", perf.trial,
                    "frame_id" to perf.frameId, "label" to currentSign,
                    "sentence_id" to perfSentenceSeq, "word_index" to perfWordOrdinal,
                    "episode_id" to perf.episodeId,
                    "confidence" to currentConf, "accepted_threshold" to (currentConf >= 0.5f),
                    "frame_observed_ns" to perf.observedNs.toString(),
                    "pipeline_queue_ms" to (processingNs - enqueuedNs) / 1e6,
                    "classification_call_ms" to (PerfTrace.now() - processingNs) / 1e6)
                if (hasHand && currentConf >= 0.5f && !perfWordRecorded && perfWordOrdinal > 0) {
                    perfWordRecorded = true
                    PerfTrace.event("word_recognized", perf.trial,
                        "sentence_id" to perfSentenceSeq, "word_index" to perfWordOrdinal,
                        "episode_id" to perf.episodeId, "label" to currentSign,
                        "confidence" to currentConf)
                }
            }
        }

        // Seuil 50% : on n'accepte un nouveau signe que s'il est suffisamment confiant.
        if (currentSign.isNotEmpty() && currentConf >= 0.5f && currentSign != lastStableLabel) {
            if (lastStableLabel.isNotEmpty()) glossBuffer.push(lastStableLabel)
            lastStableLabel = currentSign
        } else if (currentSign.isEmpty() && lastStableLabel.isNotEmpty()) {
            glossBuffer.push(lastStableLabel)
            lastStableLabel = ""
        }

        // Mise à jour du timer "main visible".
        if (hasHand) {
            lastHandSeenMs = now
        } else if (lastHandSeenMs > 0L && now - lastHandSeenMs >= 3_000L) {
            // Aucune main depuis 3 s → envoyer à Gemma si des mots sont en attente.
            val glosses = glossBuffer.currentGlosses()
            if (glosses.isNotEmpty()) {
                glossBuffer.clear()
                lastStableLabel = ""
                lastHandSeenMs  = 0L
                startTranslation(glosses, "auto")
            } else {
                lastHandSeenMs = 0L
            }
        }

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

    private fun startTranslation(
        glosses: List<String>, trigger: String,
        submittedNs: Long = PerfTrace.now(), trial: PerfTrial = PerfTrace.current,
    ) {
        val requestId = PerfTrace.nextRequest()
        val sentenceId = perfSentenceSeq
        val wordCount = perfWordOrdinal
        perfSentenceSeq++
        perfWordOrdinal = 0
        PerfTrace.event("sentence_submitted", trial, "request_id" to requestId,
            "sentence_id" to sentenceId, "word_count" to wordCount,
            "trigger" to trigger, "submitted_ns" to submittedNs.toString(),
            "glosses" to glosses.joinToString(" "))
        val style = _state.value.style
        viewModelScope.launch(Dispatchers.Default) {
            _state.update { it.copy(isTranslating = true) }
            PerfTrace.event("generation_started", trial, "request_id" to requestId)
            val sentence = gloss2Text.translate(glosses, style, trial, requestId)
            PerfTrace.event("sentence_returned", trial, "request_id" to requestId,
                "text" to sentence, "submit_to_return_ms" to (PerfTrace.now() - submittedNs) / 1e6)
            val entry      = HistoryEntry(System.currentTimeMillis(), glosses, sentence)
            val newHistory = (_state.value.history + entry).takeLast(20)
            _state.update { it.copy(translatedSentence = sentence, isTranslating = false, history = newHistory) }
            viewModelScope.launch(Dispatchers.IO) { saveHistory(newHistory) }
            // TTS uniquement si le LLM est chargé — pas le fallback texte brut
            if (gloss2Text.isAvailable && sentence.isNotBlank() && sentence != lastTranslatedSentence) {
                lastTranslatedSentence = sentence
                if (!_state.value.isMuted) tts.speak(sentence, trial, requestId)
            }
        }
    }

    // ── History persistence ───────────────────────────────────────────────────

    private fun loadHistory(): List<HistoryEntry> = try {
        if (!historyFile.exists()) emptyList()
        else {
            val arr = JSONArray(historyFile.readText())
            (0 until arr.length()).map { i ->
                val o  = arr.getJSONObject(i)
                val ga = o.getJSONArray("glosses")
                HistoryEntry(
                    id             = o.getLong("id"),
                    glosses        = (0 until ga.length()).map { j -> ga.getString(j) },
                    interpretation = o.getString("interpretation"),
                )
            }
        }
    } catch (_: Exception) { emptyList() }

    private fun saveHistory(history: List<HistoryEntry>) {
        try {
            val arr = JSONArray()
            history.forEach { e ->
                val o = JSONObject()
                o.put("id", e.id)
                val ga = JSONArray().also { a -> e.glosses.forEach { a.put(it) } }
                o.put("glosses", ga)
                o.put("interpretation", e.interpretation)
                arr.put(o)
            }
            historyFile.writeText(arr.toString())
        } catch (_: Exception) {}
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        pipelineExecutor.shutdown()
        signRecognizer.close()
        tts.shutdown()
        gloss2Text.close()
        speechEngine.destroy()
    }
}
