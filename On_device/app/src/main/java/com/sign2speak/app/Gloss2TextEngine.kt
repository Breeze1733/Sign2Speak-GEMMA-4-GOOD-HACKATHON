package com.sign2speak.app

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

private const val TAG = "Gloss2TextEngine"

/**
 * Gloss2TextEngine — traduit une séquence de glosses ASL en phrase anglaise
 * via MediaPipe LLM Inference (Gemma 4 E2B / 2B).
 *
 * Le modèle GGUF doit être placé dans le stockage externe de l'appareil Android :
 *   /sdcard/gemma4-2b-it-q4.gguf
 *   (ou autre chemin passé en paramètre)
 *
 * Si le modèle est absent, [translate] retourne les glosses concaténés directement
 * (fallback gracieux).
 *
 * NOTE API : LlmInference de mediapipe-tasks-genai 0.10.14 — l'API peut
 * légèrement varier selon la version. Adaptez les builder methods si nécessaire.
 */
class Gloss2TextEngine(
    private val context: Context,
    private val modelPath: String = "/sdcard/gemma4-2b-it-q4.gguf",
) {

    private var llm: LlmInference? = null

    /** Vrai si le modèle LLM a été chargé avec succès. */
    val isAvailable: Boolean get() = llm != null

    // Prompt système (identique à live_winner.py)
    private val systemPromptTemplate = """
You are an ASL (American Sign Language) interpreter. You receive a sequence of ASL glosses and must convert them into natural English sentences.

ASL grammar is different from English:
- ASL uses topic-comment structure (topic first)
- Time indicators come first ("YESTERDAY I GO STORE" = "I went to the store yesterday")
- No articles (a, the) or "to be" verbs in ASL
- Questions use facial grammar, the word order may differ

Rules:
- Output ONLY the English sentence, nothing else
- Keep it concise and natural
- If glosses are unclear, make the most reasonable interpretation
- Style: {style}

Style descriptions:
- professional: formal, clear, suitable for medical/business contexts
- friendly: casual, warm, conversational
- simple: short sentences, easy vocabulary, suitable for children
""".trimIndent()

    init {
        tryLoadLlm()
    }

    private fun tryLoadLlm() {
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            Log.w(TAG, "Modèle LLM introuvable : $modelPath — mode fallback activé")
            return
        }
        try {
            // NOTE : LlmInference.LlmInferenceOptions API (mediapipe-tasks-genai 0.10.14)
            // Vérifier la disponibilité de la méthode maxNewTokens() vs setMaxTokens()
            // selon la version exacte du SDK.
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(50)
                .setTopK(40)
                .setTemperature(0.3f)
                .setRandomSeed(42)
                .build()
            llm = LlmInference.createFromOptions(context, options)
            Log.i(TAG, "LLM Gemma chargé depuis $modelPath")
        } catch (e: Exception) {
            Log.e(TAG, "Impossible de charger le LLM : ${e.message}")
            llm = null
        }
    }

    /**
     * Traduit une liste de glosses ASL en phrase anglaise.
     * Suspend function — sûre à appeler depuis un coroutine context.
     *
     * @param glosses  Liste de glosses (ex. ["HELLO", "MY", "NAME"])
     * @param style    Style de traduction : "professional" | "friendly" | "simple"
     * @return         Phrase anglaise traduite, ou les glosses concaténées en fallback.
     */
    suspend fun translate(glosses: List<String>, style: String = "friendly"): String =
        withContext(Dispatchers.Default) {
            if (glosses.isEmpty()) return@withContext ""

            val glossStr = glosses.joinToString(" ")
            Log.i(TAG, "Traduction [$style] : $glossStr")

            val engine = llm
            if (engine == null) {
                // Fallback : retourne les glosses concaténés en minuscules
                Log.w(TAG, "LLM indisponible — retour direct des glosses")
                return@withContext glossStr.lowercase()
                    .split(" ")
                    .joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
            }

            val systemPrompt = systemPromptTemplate.replace("{style}", style)
            val fullPrompt   = "<start_of_turn>user\n$systemPrompt\n\nASL glosses: $glossStr<end_of_turn>\n<start_of_turn>model\n"

            try {
                val result = engine.generateResponse(fullPrompt)
                // Nettoyage : retire les balises éventuelles de fin de tour
                result
                    .replace("<end_of_turn>", "")
                    .replace("<eos>", "")
                    .trim()
                    .takeIf { it.isNotBlank() } ?: glossStr
            } catch (e: Exception) {
                Log.e(TAG, "Erreur génération LLM : ${e.message}")
                glossStr
            }
        }

    fun close() {
        try { llm?.close() } catch (_: Exception) {}
        llm = null
    }
}
