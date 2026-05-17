package com.sign2speak.app

import android.content.Context
import android.speech.tts.TextToSpeech
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

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(TAG, "Langue en_US non supportée sur cet appareil")
                } else {
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
    fun speak(text: String) {
        if (!isReady || text.isBlank()) return
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "s2s_utterance")
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
