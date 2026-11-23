package com.sign2speak.app

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

private const val TAG = "Gloss2TextEngine"

/**
 * Gloss2TextEngine — traduit une séquence de glosses ASL en phrase anglaise
 * via LiteRT-LM + Gemma 4 E2B (inférence on-device).
 *
 * Modèle requis : gemma-4-E2B-it.litertlm (2.58 GB)
 * Source : https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
 *
 * Installation :
 *   adb shell mkdir -p /sdcard/Android/data/com.sign2speak.app/files/
 *   adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.sign2speak.app/files/
 *
 * Si le modèle est absent, [translate] retourne les glosses concaténés (fallback).
 */
class Gloss2TextEngine(private val context: Context) {

    private var engine: Engine? = null

    @Volatile private var isInitialized = false

    val isAvailable: Boolean get() = engine != null

    private val modelPath: String
        get() = "${context.getExternalFilesDir(null)}/gemma-4-E2B-it.litertlm"

    private val systemPromptBase = """
You are an ASL (American Sign Language) interpreter. You receive a sequence of ASL glosses and must convert them into natural English sentences.

ASL grammar is different from English:
- ASL uses topic-comment structure (topic first)
- Time indicators come first ("YESTERDAY I GO STORE" = "I went to the store yesterday")
- No articles (a, the) or "to be" verbs in ASL

Rules:
- Output ONLY the English sentence, nothing else
- Keep it concise and natural
- If glosses are unclear, make the most reasonable interpretation

Examples of semantic inference with our current 50-sign vocabulary:

"WATER PLEASE" → "Can I have some water, please?"
"HUNGRY FOOD PLEASE" → "I'm hungry. Could I have some food, please?"
"THIRSTY WATER PLEASE" → "I'm thirsty. Could I have some water?"
"SICK" → "I feel sick."
"WHERE MOM" → "Where is my mom?"
"WHERE DAD" → "Where is my dad?"
"WHO PERSON" → "Who is this person?"
"WHY SAD" → "Why are you sad?"
"HAPPY" → "I'm happy."
"MAD" → "I'm angry."
"GO NOW" → "I’m leaving now."
"SEE TOMORROW" → "See you tomorrow."
"WAIT PLEASE" → "Please wait."
"THANKYOU" → "Thank you."
"HELLO" → "Hello."
"BYE" → "Goodbye."
"GO SLEEP NOW" → "I'm going to sleep now."
"WAKE MORNING" → "Wake me up in the morning."
"FOOD FINISH" → "The food is finished."
"OPEN PLEASE" → "Can you open it, please?"
"CLOSE PLEASE" → "Can you close it, please?"
"LISTEN PLEASE" → "Please listen."
"LOOK PLEASE" → "Please look."
"TALK LATER" → "Can we talk later?"
"JOB INTERVIEW ME GO" → "I'm going to a job interview."
"JOB INTERVIEW FINISH REJECT AGAIN" → "The job interview is over. I got rejected again."
""".trimIndent()

    /** Initialise Gemma — à appeler depuis un coroutine IO (peut prendre ~10 s). */
    suspend fun initialize() = withContext(Dispatchers.IO) {
        if (isInitialized) return@withContext
        isInitialized = true

        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            Log.w(TAG, "Modèle introuvable : $modelPath — fallback activé")
            return@withContext
        }

        engine = tryCreateEngine(gpu = true) ?: tryCreateEngine(gpu = false)
    }

    private suspend fun tryCreateEngine(gpu: Boolean): Engine? = try {
        val backend = if (gpu) Backend.GPU() else Backend.CPU()
        val config = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            cacheDir = context.cacheDir.absolutePath,
        )
        Engine(config).also { eng ->
            eng.initialize()
            Log.i(TAG, "Gemma 4 E2B chargé (${if (gpu) "GPU" else "CPU"})")
        }
    } catch (e: Exception) {
        Log.w(TAG, "${if (gpu) "GPU" else "CPU"} failed : ${e.message}")
        null
    }

    /**
     * Traduit une liste de glosses ASL en phrase anglaise.
     * Initialise Gemma automatiquement au premier appel.
     */
    suspend fun translate(glosses: List<String>, style: String = "friendly",
        perfTrial: PerfTrial = PerfTrace.current, perfRequest: String = ""): String =
        withContext(Dispatchers.IO) {
            if (glosses.isEmpty()) return@withContext ""
            if (!isInitialized) initialize()

            val glossStr = glosses.joinToString(" ")
            Log.i(TAG, "Traduction [$style] : $glossStr")

            val eng = engine ?: run {
                PerfTrace.event("generation_status", perfTrial, "request_id" to perfRequest, "status" to "model_unavailable")
                return@withContext fallback(glossStr)
            }

            val styleNote = when (style) {
                "professional" -> "formal, clear, suitable for medical/business contexts"
                "simple"       -> "short sentences, easy vocabulary, suitable for children"
                else           -> "casual, warm, conversational"
            }

            try {
                val conversation = eng.createConversation(
                    ConversationConfig(
                        systemInstruction = Contents.of("$systemPromptBase\nStyle: $styleNote."),
                        samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.3),
                    )
                )
                val response = conversation.sendMessage("ASL glosses: $glossStr")
                conversation.close()
                val generated = response.toString()
                    .replace("<end_of_turn>", "")
                    .replace("<eos>", "")
                    .trim()
                    .takeIf { it.isNotBlank() }
                PerfTrace.event("generation_status", perfTrial, "request_id" to perfRequest,
                    "status" to if (generated == null) "empty_output" else "generated")
                generated ?: glossStr
            } catch (e: Exception) {
                PerfTrace.event("generation_status", perfTrial, "request_id" to perfRequest, "status" to "generation_error", "error" to (e.message ?: ""))
                Log.e(TAG, "Erreur génération Gemma : ${e.message}")
                fallback(glossStr)
            }
        }

    private fun fallback(glossStr: String) = glossStr
        .split(" ")
        .joinToString(" ") { it.lowercase().replaceFirstChar { c -> c.uppercase() } }

    fun close() {
        try { engine?.close() } catch (_: Exception) {}
        engine = null
    }
}
