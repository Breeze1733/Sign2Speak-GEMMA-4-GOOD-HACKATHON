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
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

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
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(AppState())
    val state: StateFlow<AppState> = _state.asStateFlow()

    // Composants du pipeline (lazy pour ne pas bloquer la création du ViewModel)
    private val signRecognizer: SignRecognizer by lazy {
        SignRecognizer(application)
    }
    private val glossBuffer: GlossBuffer = GlossBuffer()
    private val gloss2TextEngine: Gloss2TextEngine by lazy {
        Gloss2TextEngine(application)
    }
    private val ttsEngine: TtsEngine by lazy {
        TtsEngine(application)
    }

    // Dispatcher single-threaded pour la logique pipeline (glosses, état interne)
    // Cela évite tout problème de concurrence sur lastStableLabel / glossBuffer
    // sans avoir besoin de Mutex explicite.
    private val pipelineExecutor = Executors.newSingleThreadExecutor()
    private val pipelineDispatcher = pipelineExecutor.asCoroutineDispatcher()
    private val pipelineScope = CoroutineScope(SupervisorJob() + pipelineDispatcher)

    // État interne (accédé uniquement depuis pipelineDispatcher)
    private var lastStableLabel: String = ""
    private var lastTranslatedSentence: String = ""

    // FPS tracking (accédé uniquement depuis pipelineDispatcher)
    private var lastFrameTimeMs: Long = 0L
    private val fpsSamples = ArrayDeque<Float>(30)

    /**
     * Appelée par CameraManager pour chaque frame (depuis l'executor CameraX).
     * On dispatch immédiatement vers pipelineDispatcher pour la thread-safety.
     */
    fun onFrameProcessed(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
    ) {
        pipelineScope.launch {
            processPipelineFrame(handResult, faceResult, timestampMs)
        }
    }

    /** Pipeline principal — tourne sur pipelineDispatcher (single-threaded). */
    private suspend fun processPipelineFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
    ) {
        // Calcul FPS
        val now = System.currentTimeMillis()
        if (lastFrameTimeMs > 0L) {
            val dt = (now - lastFrameTimeMs).coerceAtLeast(1L)
            val fps = 1000f / dt
            fpsSamples.addLast(fps)
            if (fpsSamples.size > 30) fpsSamples.removeFirst()
        }
        lastFrameTimeMs = now
        val avgFps = if (fpsSamples.isNotEmpty()) fpsSamples.average().toFloat() else 0f

        // Traitement via SignRecognizer (synchrone, déjà sur pipelineDispatcher)
        val recognition = signRecognizer.processFrame(handResult, faceResult, timestampMs)

        val hasHand    = recognition.hasHand
        val currentSign = recognition.topLabel
        val currentConf = recognition.topConf
        val top3       = recognition.top3

        // Logique d'accumulation des glosses (identique à live_winner.py)
        if (currentSign.isNotEmpty() && currentSign != lastStableLabel) {
            if (lastStableLabel.isNotEmpty()) {
                glossBuffer.push(lastStableLabel)
            }
            lastStableLabel = currentSign
        } else if (currentSign.isEmpty() && lastStableLabel.isNotEmpty()) {
            glossBuffer.push(lastStableLabel)
            lastStableLabel = ""
        }

        // Check GlossBuffer timeout
        val flushedGlosses = glossBuffer.tick()
        val pendingGlosses = glossBuffer.currentGlosses()

        if (flushedGlosses != null && flushedGlosses.isNotEmpty()) {
            // Lance la traduction dans un coroutine séparé (Dispatchers.Default)
            // pour ne pas bloquer le pipeline
            val style = _state.value.style
            viewModelScope.launch(Dispatchers.Default) {
                _state.value = _state.value.copy(isTranslating = true)
                val sentence = gloss2TextEngine.translate(flushedGlosses, style)
                onTranslationReady(sentence)
            }
        }

        // Mise à jour de l'état (MutableStateFlow est thread-safe)
        _state.value = _state.value.copy(
            currentSign    = currentSign,
            currentConf    = currentConf,
            top3           = top3,
            pendingGlosses = pendingGlosses,
            handDetected   = hasHand,
            fps            = avgFps,
        )
    }

    /** Appelée lorsque la traduction Gemma est prête. */
    private fun onTranslationReady(sentence: String) {
        _state.value = _state.value.copy(
            translatedSentence = sentence,
            isTranslating      = false,
        )
        // TTS : parle seulement si nouvelle phrase et non muet
        if (sentence.isNotBlank() && sentence != lastTranslatedSentence) {
            lastTranslatedSentence = sentence
            if (!_state.value.isMuted) {
                ttsEngine.speak(sentence)
            }
        }
    }

    fun toggleMute() {
        val newMuted = !_state.value.isMuted
        _state.value = _state.value.copy(isMuted = newMuted)
        if (newMuted) {
            ttsEngine.stop()
        }
    }

    fun setStyle(style: String) {
        _state.value = _state.value.copy(style = style)
    }

    fun clearBuffer() {
        pipelineScope.launch {
            signRecognizer.clearBuffer()
            glossBuffer.clear()
            lastStableLabel        = ""
            lastTranslatedSentence = ""
            _state.value = _state.value.copy(
                currentSign        = "",
                currentConf        = 0f,
                top3               = emptyList(),
                pendingGlosses     = emptyList(),
                translatedSentence = "",
                isTranslating      = false,
                handDetected       = false,
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        pipelineExecutor.shutdown()
        signRecognizer.close()
        ttsEngine.shutdown()
        gloss2TextEngine.close()
    }
}
