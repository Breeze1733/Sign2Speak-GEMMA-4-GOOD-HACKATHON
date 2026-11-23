package com.sign2speak.app

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.concurrent.ConcurrentHashMap
import android.util.Log
import java.util.Locale

private const val TAG = "TtsEngine"

/**
 * TtsEngine — synthèse vocale Android native (TextToSpeech).
 *
 * Utilise la voix en_US avec un débit normal.
 * Appeler [shutdown] dans onCleared() du ViewModel.
 */
class TtsEngine(context: Context) {

    private var tts: TextToSpeech? = null
    private var isReady = false
    private val perfRequests = ConcurrentHashMap<String, Pair<PerfTrial, String>>()

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(TAG, "Langue en_US non supportée sur cet appareil")
                } else {
                    if (BuildConfig.TIMING_ENABLED) tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(id: String) {
                            perfRequests[id]?.let { (trial, request) ->
                                PerfTrace.event("tts_start_callback", trial, "request_id" to request)
                            }
                        }
                        override fun onDone(id: String) { finish(id, "tts_done") }
                        @Deprecated("Deprecated in Java")
                        override fun onError(id: String) { finish(id, "tts_error") }
                        override fun onStop(id: String, interrupted: Boolean) { finish(id, "tts_stopped") }
                        private fun finish(id: String, event: String) {
                            perfRequests.remove(id)?.let { (trial, request) ->
                                PerfTrace.event(event, trial, "request_id" to request)
                            }
                        }
                    })
                    tts?.setSpeechRate(1.0f)
                    tts?.setPitch(1.0f)
                    isReady = true
                    Log.i(TAG, "TTS initialisé (en_US)")
                }
            } else {
                Log.e(TAG, "Initialisation TTS échouée : status=$status")
            }
        }
    }

    /**
     * Prononce le texte donné.
     * QUEUE_FLUSH interrompt tout audio en cours et commence immédiatement.
     */
    fun speak(text: String, trial: PerfTrial = PerfTrace.current, request: String = "") {
        if (!BuildConfig.TIMING_ENABLED) {
            if (!isReady || text.isBlank()) return
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "s2s_utterance")
            return
        }
        if (!isReady || text.isBlank()) {
            PerfTrace.event("tts_skipped", trial, "request_id" to request)
            return
        }
        val utterance = "s2s_" + PerfTrace.nextRequest()
        perfRequests[utterance] = trial to request
        PerfTrace.event("tts_requested", trial, "request_id" to request)
        val status = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utterance)
        if (status != TextToSpeech.SUCCESS) {
            perfRequests.remove(utterance)
            PerfTrace.event("tts_error", trial, "request_id" to request)
        }
    }

    /**
     * Arrête la lecture en cours sans vider la queue complètement.
     */
    fun stop() {
        if (!isReady) return
        tts?.stop()
    }

    /**
     * Libère les ressources TTS. À appeler dans onCleared() ou onDestroy().
     */
    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (e: Exception) {
            Log.e(TAG, "Erreur fermeture TTS : ${e.message}")
        } finally {
            tts     = null
            isReady = false
        }
    }
}
