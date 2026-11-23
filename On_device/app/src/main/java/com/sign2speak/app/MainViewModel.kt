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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

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
    val pipelineMode: String = "p0",          // p0 local | p1 remote G2T | p2 remote M3+G2T
    val remoteUrl: String = "",
    val remoteHasToken: Boolean = false,
    val lastSource: String = "",             // local | remote | local_fallback
    val lastStatus: String = "",
    val lastM3Status: String = "off",        // off | starting | ready | ok 18ms | busy | timeout ...
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

    @Volatile private var controller: RecognitionController? = null
    private var controllerMode: String? = null
    private val glossBuffer    = GlossBuffer()
    private val gloss2Text: Gloss2TextEngine by lazy { Gloss2TextEngine(application) }
    private val tts: TtsEngine by lazy { TtsEngine(application) }
    private val speechEngine: SpeechEngine by lazy { SpeechEngine(application) }

    private val historyFile = File(application.cacheDir, "interpretation_history.json")

    private val pipelineExecutor   = Executors.newSingleThreadExecutor()
    private val pipelineDispatcher = pipelineExecutor.asCoroutineDispatcher()
    private val pipelineScope      = CoroutineScope(SupervisorJob() + pipelineDispatcher)

    // ── P0/P1 translation selection ───────────────────────────────────────────
    private val localGlossToText: LocalGlossToText by lazy { LocalGlossToText(gloss2Text) }
    private val configStore: TranslationConfigStore by lazy { TranslationConfigStore(application) }
    private lateinit var storedConfig: StoredTranslationConfig
    /** Serialises translations so sentences are produced/displayed in submission order. */
    private val translationMutex = Mutex()
    private val appSessionId  = UUID.randomUUID().toString().take(8)
    /** Bumped on clear/cancel; responses from an older epoch are logged and dropped. */
    private val sessionEpoch  = AtomicLong(0L)

    private var perfSentenceSeq = 1L
    private var perfWordOrdinal = 0
    private var perfLastEpisode = -1L
    private var perfWordEnded = false
    private var lastStableLabel        = ""
    private var currentSign            = ""
    private var currentConf            = 0f
    private var currentTop3: List<Pair<String, Float>> = emptyList()
    private var lastRecognitionSerial  = -1L
    private var lastWordRecordedIndex  = -1
    private val classifierVersion: String by lazy {
        try {
            val raw = application.assets.open("model_meta.json")
                .bufferedReader().use { it.readText() }
            LandmarkLayout.modelVersion(JSONObject(raw), BuildConfig.CLASSIFIER_MODEL_VERSION)
        } catch (_: Exception) {
            BuildConfig.CLASSIFIER_MODEL_VERSION
        }
    }
    private var lastTranslatedSentence = ""
    private var lastFrameTimeMs        = 0L
    private val fpsSamples             = ArrayDeque<Float>(30)
    private var frameCount             = 0
    private var lastHandSeenMs         = 0L

    init {
        storedConfig = configStore.load()
        _state.update {
            it.copy(
                pipelineMode    = storedConfig.pipelineMode,
                remoteUrl       = storedConfig.remoteUrl,
                remoteHasToken  = storedConfig.remoteToken.isNotBlank(),
                lastM3Status    = if (storedConfig.pipelineMode == "p2") "starting" else "off",
            )
        }
        PerfTrace.begin("", "app_start")
        // P2 deliberately does not load the 2.58 GB Gemma model: M5 is remote.
        if (storedConfig.pipelineMode != "p2") {
            viewModelScope.launch(Dispatchers.IO) { gloss2Text.initialize() }
        }
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
        pipelineScope.launch { processFrame(handResult, faceResult, timestampMs, perf) }
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
            controller?.suppressBefore(controller?.lastWindowId() ?: -1L)
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

    fun setPipelineMode(mode: String) {
        val normalized = when (mode) {
            "p0", "local"  -> "p0"
            "p1", "remote" -> "p1"
            "p2"           -> "p2"
            else           -> "p0"
        }
        if (normalized == _state.value.pipelineMode) return
        storedConfig = storedConfig.copy(pipelineMode = normalized)
        configStore.savePipelineMode(normalized)
        sessionEpoch.incrementAndGet()
        controller?.invalidate()
        if (normalized == "p2") gloss2Text.close()
        _state.update {
            it.copy(
                pipelineMode = normalized,
                lastM3Status = if (normalized == "p2") "starting" else "off",
            )
        }
        PerfTrace.event("pipeline_mode", PerfTrace.current, "mode" to normalized)
        pipelineScope.launch {
            controller?.close()
            controller = null
            controllerMode = null
            glossBuffer.clear()
            lastStableLabel = ""
            currentSign = ""
            currentConf = 0f
            currentTop3 = emptyList()
            lastRecognitionSerial = -1L
            lastWordRecordedIndex = -1
            ensureController(normalized)
            _state.update {
                it.copy(
                    currentSign = "", currentConf = 0f, top3 = emptyList(),
                    pendingGlosses = emptyList(),
                    lastM3Status = if (normalized == "p2") "ready" else "off",
                )
            }
        }
    }

    /** Saves the remote endpoint. Pass a blank token to keep the existing one. */
    fun updateRemoteSettings(url: String, token: String?) {
        val newToken = token?.takeIf { it.isNotBlank() } ?: storedConfig.remoteToken
        storedConfig = storedConfig.copy(remoteUrl = url.trim(), remoteToken = newToken)
        configStore.saveRemote(storedConfig.remoteUrl, storedConfig.remoteToken)
        _state.update {
            it.copy(
                remoteUrl      = storedConfig.remoteUrl,
                remoteHasToken = storedConfig.remoteToken.isNotBlank(),
            )
        }
    }

    fun remoteSettingsForDialog(): Pair<String, Boolean> =
        storedConfig.remoteUrl to storedConfig.remoteToken.isNotBlank()

    /** Pipeline-thread only. Creates the controller matching the current mode. */
    private fun ensureController(mode: String): RecognitionController {
        val existing = controller
        if (existing != null && controllerMode == mode) return existing
        existing?.close()
        val created: RecognitionController = when (mode) {
            "p2" -> RemoteRecognitionController(
                configProvider = {
                    RemoteRecognitionConfig(
                        baseUrl = storedConfig.remoteUrl,
                        token = storedConfig.remoteToken,
                        modelVersion = classifierVersion,
                    )
                },
                sessionId = appSessionId,
                deliver = { event, ctx ->
                    pipelineScope.launch { handleRecognition(event, ctx) }
                },
                onStatus = { status ->
                    _state.update { it.copy(lastM3Status = status) }
                },
            )
            else -> LocalRecognitionController(getApplication()) { event, ctx ->
                handleRecognition(event, ctx)
            }
        }
        controller = created
        controllerMode = mode
        return created
    }

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
        // Invalidate any in-flight sentence: its response belongs to an older session.
        sessionEpoch.incrementAndGet()
        controller?.invalidate()
        pipelineScope.launch {
            controller?.clear()
            glossBuffer.clear()
            lastStableLabel        = ""
            lastTranslatedSentence = ""
            lastHandSeenMs         = 0L
            lastFrameTimeMs        = 0L
            fpsSamples.clear()
            frameCount             = 0
            currentSign            = ""
            currentConf            = 0f
            currentTop3            = emptyList()
            lastRecognitionSerial  = -1L
            lastWordRecordedIndex  = -1
            _state.update {
                it.copy(
                    currentSign = "", currentConf = 0f, top3 = emptyList(),
                    pendingGlosses = emptyList(), translatedSentence = "",
                    isTranslating = false, lastSource = "", lastStatus = "",
                    handDetected = false, fps = 0f,
                    detectedHands = emptyList(), faceDots = emptyList(),
                    lastM3Status = if (it.pipelineMode == "p2") "ready" else "off",
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

        val hasHand = HolisticAssembler.hasHand(handResult)
        if (BuildConfig.TIMING_ENABLED && hasHand && perf.episodeId != perfLastEpisode) {
            perfLastEpisode = perf.episodeId
            perfWordOrdinal++
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

        val ctl = ensureController(_state.value.pipelineMode)
        val ctx = RecognitionContext(
            perf = perf,
            trial = perf.trial,
            sentenceId = perfSentenceSeq,
            wordIndex = perfWordOrdinal,
            submittedNs = processingNs,
            windowEndObservedNs = perf.observedNs,
            recognitionEpoch = ctl.currentEpoch(),
        )
        val frame = ctl.processFrame(handResult, faceResult, timestampMs, ctx)

        // Mise à jour du timer "main visible".
        if (hasHand) {
            lastHandSeenMs = now
        } else if (lastHandSeenMs > 0L && now - lastHandSeenMs >= 3_000L) {
            // Aucune main depuis 3 s → envoyer à Gemma si des mots sont en attente.
            ctl.endUtterance()
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
                top3           = currentTop3,
                pendingGlosses = glossBuffer.currentGlosses(),
                handDetected   = hasHand,
                fps            = avgFps,
                detectedHands  = frame.detectedHands,
                faceDots       = frame.faceDots,
            )
        }
    }

    /**
     * M3 -> M4: shared by local inference (called synchronously on the pipeline
     * thread) and remote responses (posted to the pipeline dispatcher).
     */
    private fun handleRecognition(event: RecognitionEvent, ctx: RecognitionContext) {
        if (!event.inferred) {
            _state.update { it.copy(lastM3Status = event.status) }
            return
        }
        if (event.topLabel.isNotEmpty()) {
            if (event.resultSerial == lastRecognitionSerial) return
            lastRecognitionSerial = event.resultSerial
        }

        currentSign = event.topLabel
        currentConf = event.topConf
        currentTop3 = event.top3

        if (BuildConfig.TIMING_ENABLED && currentSign.isNotEmpty()) {
            val callMs = if (event.source == "remote") {
                event.clientMs
            } else {
                (PerfTrace.now() - ctx.submittedNs) / 1e6
            }
            PerfTrace.event("recognition_result", ctx.trial,
                "frame_id" to ctx.perf.frameId, "label" to currentSign,
                "sentence_id" to ctx.sentenceId, "word_index" to ctx.wordIndex,
                "episode_id" to ctx.perf.episodeId,
                "confidence" to currentConf, "accepted_threshold" to (currentConf >= 0.5f),
                "source" to event.source, "window_id" to event.windowId,
                "frame_observed_ns" to ctx.windowEndObservedNs.toString(),
                "pipeline_queue_ms" to ((ctx.submittedNs - ctx.perf.observedNs) / 1e6),
                "classification_call_ms" to callMs)
            if (event.hasHand && currentConf >= 0.5f &&
                ctx.wordIndex > 0 && ctx.wordIndex != lastWordRecordedIndex
            ) {
                lastWordRecordedIndex = ctx.wordIndex
                PerfTrace.event("word_recognized", ctx.trial,
                    "sentence_id" to ctx.sentenceId, "word_index" to ctx.wordIndex,
                    "episode_id" to ctx.perf.episodeId, "label" to currentSign,
                    "confidence" to currentConf, "source" to event.source)
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

        if (event.source == "remote") {
            _state.update {
                it.copy(
                    currentSign = currentSign,
                    currentConf = currentConf,
                    top3 = currentTop3,
                    pendingGlosses = glossBuffer.currentGlosses(),
                    lastM3Status = event.status,
                )
            }
        } else {
            _state.update {
                it.copy(
                    currentSign = currentSign,
                    currentConf = currentConf,
                    top3 = currentTop3,
                    pendingGlosses = glossBuffer.currentGlosses(),
                )
            }
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
        val mode = _state.value.pipelineMode
        val epochAtSubmit = sessionEpoch.get()
        val sessionId = "$appSessionId-$epochAtSubmit"

        viewModelScope.launch(Dispatchers.Default) {
            // One translation at a time keeps sentence order stable (proposal 7.2).
            translationMutex.withLock {
                if (sessionEpoch.get() != epochAtSubmit) {
                    PerfTrace.event("stale_response", trial, "request_id" to requestId,
                        "reason" to "session_changed_before_start")
                    return@withLock
                }
                _state.update { it.copy(isTranslating = true) }
                PerfTrace.event("generation_started", trial, "request_id" to requestId, "mode" to mode)

                val outcome = runTranslation(mode, glosses, style, trial, requestId, sessionId, sentenceId)

                if (sessionEpoch.get() != epochAtSubmit) {
                    PerfTrace.event("stale_response", trial, "request_id" to requestId,
                        "reason" to "session_changed_after_response", "source" to outcome.source)
                    _state.update { it.copy(isTranslating = false) }
                    return@withLock
                }

                PerfTrace.event("sentence_returned", trial, "request_id" to requestId,
                    "text" to outcome.text, "source" to outcome.source,
                    "status" to outcome.status,
                    "submit_to_return_ms" to (PerfTrace.now() - submittedNs) / 1e6,
                    "client_ms" to (outcome.clientMs ?: -1.0),
                    "server_queue_ms" to (outcome.serverQueueMs ?: -1.0),
                    "server_inference_ms" to (outcome.serverInferenceMs ?: -1.0))

                val sentence = outcome.text
                if (sentence.isNotBlank()) {
                    val entry      = HistoryEntry(System.currentTimeMillis(), glosses, sentence)
                    val newHistory = (_state.value.history + entry).takeLast(20)
                    _state.update {
                        it.copy(
                            translatedSentence = sentence,
                            isTranslating      = false,
                            history            = newHistory,
                            lastSource         = outcome.source,
                            lastStatus         = outcome.status,
                        )
                    }
                    viewModelScope.launch(Dispatchers.IO) { saveHistory(newHistory) }
                    // Speak real sentences only; never speak the raw-gloss fallback.
                    val speakable = outcome.isRealSentence &&
                        (outcome.source != "local" || gloss2Text.isAvailable)
                    if (speakable && sentence != lastTranslatedSentence) {
                        lastTranslatedSentence = sentence
                        if (!_state.value.isMuted) tts.speak(sentence, trial, requestId)
                    }
                } else {
                    // No sentence (remote failure with no local model): show the glosses.
                    _state.update {
                        it.copy(
                            isTranslating      = false,
                            translatedSentence = glosses.joinToString(" ").uppercase(),
                            lastSource         = outcome.source,
                            lastStatus         = outcome.status,
                        )
                    }
                }
            }
        }
    }

    private suspend fun runTranslation(
        mode: String, glosses: List<String>, style: String,
        trial: PerfTrial, requestId: String, sessionId: String, sentenceIndex: Long,
    ): TranslationResult {
        if (mode == "p0") {
            return localGlossToText.translate(
                glosses, style, trial, requestId, sessionId, sentenceIndex
            )
        }

        val remote = RemoteGlossToText(
            RemoteConfig(
                url          = storedConfig.remoteUrl,
                token        = storedConfig.remoteToken,
                modelVersion = BuildConfig.REMOTE_MODEL_VERSION,
            )
        )
        val remoteResult = remote.translate(glosses, style, trial, requestId, sessionId, sentenceIndex)
        if (!remoteResult.isRemoteFailure) return remoteResult

        // P2 deliberately does not load the 2.58 GB local Gemma model, so there
        // is no local fallback: the existing raw-gloss branch is shown instead.
        if (mode == "p2") return remoteResult

        PerfTrace.event("remote_fallback", trial, "request_id" to requestId,
            "reason" to remoteResult.status, "error" to (remoteResult.error ?: ""))
        val local = localGlossToText.translate(
            glosses, style, trial, requestId, sessionId, sentenceIndex
        )
        return local.copy(source = "local_fallback")
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
        controller?.close()
        controller = null
        pipelineExecutor.shutdown()
        tts.shutdown()
        gloss2Text.close()
        speechEngine.destroy()
    }
}
