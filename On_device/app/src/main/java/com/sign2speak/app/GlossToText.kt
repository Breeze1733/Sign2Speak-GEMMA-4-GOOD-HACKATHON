package com.sign2speak.app

import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Unified gloss -> sentence interface (proposal section 7.1).
 *
 * Two implementations:
 *  - [LocalGlossToText]  wraps the on-device LiteRT-LM engine (P0).
 *  - [RemoteGlossToText] posts the gloss sequence to the P1 service (P1).
 *
 * MainViewModel picks one at runtime; a failed remote call falls back to local.
 */
data class TranslationResult(
    val text: String,
    val status: String,
    val source: String,               // local | remote | local_fallback
    val requestId: String,
    val clientMs: Double? = null,
    val serverQueueMs: Double? = null,
    val serverInferenceMs: Double? = null,
    val error: String? = null,
) {
    val isRemoteFailure: Boolean get() = status.startsWith("remote_")
    val isRealSentence: Boolean get() = status == "generated" && text.isNotBlank()
}

interface GlossToText {
    val mode: String
    val isAvailable: Boolean

    suspend fun translate(
        glosses: List<String>,
        style: String,
        perfTrial: PerfTrial,
        requestId: String,
        sessionId: String,
        sentenceIndex: Long,
    ): TranslationResult

    fun close()
}

// ─── Local (on-device LiteRT-LM) ─────────────────────────────────────────────

class LocalGlossToText(private val engine: Gloss2TextEngine) : GlossToText {

    override val mode: String = "local"
    override val isAvailable: Boolean get() = engine.isAvailable

    override suspend fun translate(
        glosses: List<String>,
        style: String,
        perfTrial: PerfTrial,
        requestId: String,
        sessionId: String,
        sentenceIndex: Long,
    ): TranslationResult {
        val started = SystemClock.elapsedRealtime()
        val text = engine.translate(glosses, style, perfTrial, requestId)
        val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()
        val status = if (engine.isAvailable) "generated" else "fallback"
        PerfTrace.event("local_response", perfTrial, "request_id" to requestId,
            "status" to status, "client_ms" to elapsed, "text" to text)
        return TranslationResult(
            text = text,
            status = status,
            source = "local",
            requestId = requestId,
            clientMs = elapsed,
        )
    }

    override fun close() = engine.close()
}

// ─── Remote (P1 HTTP service) ────────────────────────────────────────────────

data class RemoteConfig(
    val url: String,                 // base, e.g. http://121.41.19.165:8080
    val token: String,
    val modelVersion: String = "gemma-4-E2B-it.litertlm",
    val connectTimeoutMs: Int = 5_000,
    val readTimeoutMs: Int = 60_000,
) {
    val isUsable: Boolean get() = url.isNotBlank()

    /** Accepts a base URL or a full /translate or /recognize URL. */
    fun baseUrl(): String {
        val trimmed = url.trim().trimEnd('/')
        return when {
            trimmed.endsWith("/translate") -> trimmed.removeSuffix("/translate").trimEnd('/')
            trimmed.endsWith("/recognize") -> trimmed.removeSuffix("/recognize").trimEnd('/')
            else -> trimmed
        }
    }

    fun endpoint(): String = "${baseUrl()}/translate"
}

class RemoteGlossToText(private val config: RemoteConfig) : GlossToText {

    override val mode: String = "remote"
    override val isAvailable: Boolean get() = config.isUsable

    override suspend fun translate(
        glosses: List<String>,
        style: String,
        perfTrial: PerfTrial,
        requestId: String,
        sessionId: String,
        sentenceIndex: Long,
    ): TranslationResult = withContext(Dispatchers.IO) {
        if (!config.isUsable) {
            return@withContext TranslationResult(
                "", "remote_unconfigured", "remote", requestId, error = "url is blank",
            )
        }

        val body = JSONObject()
            .put("request_id", requestId)
            .put("session_id", sessionId)
            .put("sentence_index", sentenceIndex)
            .put("glosses", JSONArray(glosses.map { it.lowercase() }))
            .put("style", style)
            .put("model_version", config.modelVersion)
            .toString()

        PerfTrace.event("remote_request", perfTrial, "request_id" to requestId,
            "url" to config.endpoint(), "session_id" to sessionId,
            "sentence_index" to sentenceIndex, "glosses" to glosses.joinToString(" "))

        val started = SystemClock.elapsedRealtime()
        var connection: HttpURLConnection? = null
        try {
            connection = (URL(config.endpoint()).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = config.connectTimeoutMs
                readTimeout = config.readTimeoutMs
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                if (config.token.isNotBlank()) {
                    setRequestProperty("Authorization", "Bearer ${config.token}")
                }
            }
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()

            if (code !in 200..299) {
                PerfTrace.event("remote_http_error", perfTrial, "request_id" to requestId,
                    "http_code" to code, "client_ms" to elapsed, "body" to raw.take(300))
                return@withContext TranslationResult(
                    "", "remote_http_$code", "remote", requestId,
                    clientMs = elapsed, error = raw.take(300),
                )
            }

            val json = JSONObject(raw)
            val text = json.optString("sentence", "")
            val status = json.optString("status", "generated")
            val queueMs = json.optDouble("queue_ms", Double.NaN).takeUnless { it.isNaN() }
            val inferenceMs = json.optDouble("inference_ms", Double.NaN).takeUnless { it.isNaN() }
            PerfTrace.event("remote_response", perfTrial, "request_id" to requestId,
                "http_code" to code, "client_ms" to elapsed, "status" to status,
                "queue_ms" to (queueMs ?: -1.0), "inference_ms" to (inferenceMs ?: -1.0),
                "sentence" to text)
            TranslationResult(
                text = text,
                status = status,
                source = "remote",
                requestId = requestId,
                clientMs = elapsed,
                serverQueueMs = queueMs,
                serverInferenceMs = inferenceMs,
            )
        } catch (e: java.net.SocketTimeoutException) {
            val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()
            PerfTrace.event("remote_timeout", perfTrial, "request_id" to requestId,
                "client_ms" to elapsed, "error" to (e.message ?: ""))
            TranslationResult("", "remote_timeout", "remote", requestId,
                clientMs = elapsed, error = e.message)
        } catch (e: Exception) {
            val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()
            PerfTrace.event("remote_error", perfTrial, "request_id" to requestId,
                "client_ms" to elapsed, "error" to (e.message ?: e.javaClass.simpleName))
            TranslationResult("", "remote_error", "remote", requestId,
                clientMs = elapsed, error = e.message ?: e.javaClass.simpleName)
        } finally {
            connection?.disconnect()
        }
    }

    override fun close() = Unit
}
