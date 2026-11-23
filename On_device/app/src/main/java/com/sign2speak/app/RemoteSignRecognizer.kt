package com.sign2speak.app

import android.os.SystemClock
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

data class RemoteRecognitionConfig(
    val baseUrl: String,
    val token: String,
    val modelVersion: String,
    val connectTimeoutMs: Int = 3_000,
    val readTimeoutMs: Int = 5_000,
    val coordCount: Int = P2Protocol.COORD_COUNT,
) {
    val isUsable: Boolean get() = baseUrl.isNotBlank()

    fun endpoint(): String {
        val trimmed = baseUrl.trim().trimEnd('/')
        return when {
            trimmed.endsWith("/recognize") -> trimmed
            trimmed.endsWith("/translate") -> trimmed.removeSuffix("/translate") + "/recognize"
            else -> "$trimmed/recognize"
        }
    }
}

/**
 * P2 remote M3 client: keeps the 64-frame window locally, uploads at the same
 * cadence as the local classifier, allows at most one in-flight request, and
 * drops stale responses by epoch/window id.
 */
class RemoteSignRecognizer(
    private val configProvider: () -> RemoteRecognitionConfig,
    private val sessionId: String,
    private val onEvent: (RecognitionEvent, RecognitionContext) -> Unit,
    private val onStatus: (String) -> Unit = {},
) {
    private val window = FrameWindow()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val windowId = AtomicLong(0L)
    private val episodeId = AtomicLong(0L)
    private val lastSubmitted = AtomicLong(-1L)
    private val lastApplied = AtomicLong(-1L)
    private val suppressBefore = AtomicLong(-1L)
    private val inFlight = AtomicBoolean(false)
    private val resetRequested = AtomicBoolean(true)

    private var frameSeq = 0L
    private var lastHandSeenMs = 0L
    private var handLostNotified = false
    @Volatile private var closed = false

    fun currentEpoch(): Long = episodeId.get()

    fun lastWindowId(): Long = lastSubmitted.get()

    fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        ctx: RecognitionContext,
    ): RecognitionFrame {
        val holistic = HolisticAssembler.assemble(handResult, faceResult)
        val hasHand = HolisticAssembler.hasHand(handResult)
        window.add(holistic, ctx.perf.frameId, timestampMs, hasHand)

        val now = SystemClock.elapsedRealtime()
        if (hasHand) {
            lastHandSeenMs = now
            handLostNotified = false
        }

        val attempt = frameSeq % LandmarkLayout.INFER_EVERY_N == 0L &&
            window.size >= LandmarkLayout.MIN_FRAMES_FOR_INFER
        frameSeq++

        // Do not upload no-hand windows: the server would clear its vote state
        // for a window with no useful landmarks. A sustained loss is signalled
        // once, locally, and the next hand window starts a new episode.
        if (hasHand && attempt) {
            submit(ctx)
        } else if (!hasHand && lastHandSeenMs > 0L &&
            now - lastHandSeenMs >= 200L && !handLostNotified
        ) {
            handLostNotified = true
            endUtterance()
            onEvent(
                RecognitionEvent(
                    topLabel = "",
                    topConf = 0f,
                    top3 = emptyList(),
                    hasHand = false,
                    inferred = true,
                    source = "remote",
                    windowId = lastSubmitted.get(),
                    resultSerial = -1L,
                    status = "no_hand",
                ),
                ctx.copy(recognitionEpoch = episodeId.get()),
            )
        }

        return RecognitionFrame(
            hasHand = hasHand,
            detectedHands = HolisticAssembler.extractHandOverlay(handResult),
            faceDots = HolisticAssembler.extractFaceOverlay(faceResult),
        )
    }

    private fun submit(ctx: RecognitionContext) {
        if (closed) return
        val id = windowId.incrementAndGet()
        lastSubmitted.set(id)
        if (!inFlight.compareAndSet(false, true)) {
            PerfTrace.event(
                "m3_window_dropped",
                ctx.trial,
                "window_id" to id,
                "frame_count" to window.size,
                "reason" to "busy",
            )
            onStatus("busy")
            return
        }

        val config = configProvider()
        if (!config.isUsable) {
            inFlight.set(false)
            PerfTrace.event("m3_window_error", ctx.trial, "window_id" to id, "error" to "url is blank")
            onStatus("unconfigured")
            return
        }

        val snapshot = window.snapshot(coordCount = config.coordCount)
        val epoch = episodeId.get()
        val reset = resetRequested.getAndSet(false)
        val context = ctx.copy(recognitionEpoch = epoch)

        scope.launch {
            val started = SystemClock.elapsedRealtime()
            var connection: HttpURLConnection? = null
            try {
                val body = P2Protocol.encode(
                    snapshot = snapshot,
                    windowId = id,
                    episodeId = epoch,
                    resetVote = reset,
                    pointCount = P2Protocol.POINT_COUNT,
                    coordCount = config.coordCount,
                )
                val bodySha = sha256(body)
                PerfTrace.event(
                    "m3_window_submitted",
                    context.trial,
                    "window_id" to id,
                    "episode_id" to epoch,
                    "frame_count" to snapshot.frameCount,
                    "bytes" to body.size,
                    "body_sha256" to bodySha,
                    "has_hand" to snapshot.hasHandCurrentFrame,
                    "recognition_epoch" to epoch,
                )

                connection = (URL(config.endpoint()).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = config.connectTimeoutMs
                    readTimeout = config.readTimeoutMs
                    doOutput = true
                    setRequestProperty("Content-Type", "application/octet-stream")
                    setRequestProperty("Accept", "application/json")
                    setRequestProperty("X-S2S-Session-Id", sessionId)
                    setRequestProperty("X-S2S-Request-Id", "m3-$id")
                    setRequestProperty("X-S2S-Model-Version", config.modelVersion)
                    if (config.token.isNotBlank()) {
                        setRequestProperty("Authorization", "Bearer ${config.token}")
                    }
                    setFixedLengthStreamingMode(body.size)
                }
                connection.outputStream.use { it.write(body) }

                val code = connection.responseCode
                val stream = if (code in 200..299) connection.inputStream else connection.errorStream
                val raw = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
                val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()

                if (code !in 200..299) {
                    PerfTrace.event(
                        "m3_window_error",
                        context.trial,
                        "window_id" to id,
                        "http_code" to code,
                        "client_ms" to elapsed,
                        "body" to raw.take(300),
                    )
                    onStatus("http_$code")
                    return@launch
                }

                if (context.recognitionEpoch != episodeId.get()) {
                    PerfTrace.event(
                        "m3_window_stale",
                        context.trial,
                        "window_id" to id,
                        "reason" to "episode_changed",
                    )
                    onStatus("stale")
                    return@launch
                }
                if (id <= lastApplied.get() || id <= suppressBefore.get()) {
                    PerfTrace.event(
                        "m3_window_stale",
                        context.trial,
                        "window_id" to id,
                        "reason" to "older_or_suppressed",
                    )
                    onStatus("stale")
                    return@launch
                }

                val json = JSONObject(raw)
                val top3 = parseTop3(json.optJSONArray("top3"))
                val event = RecognitionEvent(
                    topLabel = json.optString("label", ""),
                    topConf = json.optDouble("confidence", 0.0).toFloat(),
                    top3 = top3,
                    hasHand = json.optBoolean("has_hand", true),
                    inferred = json.optBoolean("inferred", true),
                    source = "remote",
                    windowId = id,
                    resultSerial = id,
                    status = json.optString("status", "ok"),
                    clientMs = elapsed,
                    serverQueueMs = optDoubleOrNull(json, "queue_ms"),
                    serverPreprocessMs = optDoubleOrNull(json, "preprocess_ms"),
                    serverInferenceMs = optDoubleOrNull(json, "inference_ms"),
                    serverVoteMs = optDoubleOrNull(json, "vote_ms"),
                    serverTotalMs = optDoubleOrNull(json, "total_ms"),
                    error = null,
                )
                lastApplied.set(id)
                PerfTrace.event(
                    "m3_window_returned",
                    context.trial,
                    "window_id" to id,
                    "episode_id" to context.recognitionEpoch,
                    "status" to event.status,
                    "inferred" to event.inferred,
                    "label" to event.topLabel,
                    "confidence" to event.topConf,
                    "client_ms" to elapsed,
                    "server_queue_ms" to (event.serverQueueMs ?: -1.0),
                    "server_preprocess_ms" to (event.serverPreprocessMs ?: -1.0),
                    "server_inference_ms" to (event.serverInferenceMs ?: -1.0),
                    "server_vote_ms" to (event.serverVoteMs ?: -1.0),
                    "server_total_ms" to (event.serverTotalMs ?: -1.0),
                    "bytes_down" to raw.toByteArray(Charsets.UTF_8).size,
                )
                onStatus(if (event.inferred) "ok ${event.serverTotalMs?.toInt() ?: 0}ms" else event.status)
                onEvent(event, context)
            } catch (e: SocketTimeoutException) {
                val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()
                PerfTrace.event(
                    "m3_window_timeout",
                    context.trial,
                    "window_id" to id,
                    "client_ms" to elapsed,
                    "error" to (e.message ?: ""),
                )
                onStatus("timeout")
            } catch (e: Exception) {
                val elapsed = (SystemClock.elapsedRealtime() - started).toDouble()
                PerfTrace.event(
                    "m3_window_error",
                    context.trial,
                    "window_id" to id,
                    "client_ms" to elapsed,
                    "error" to (e.message ?: e.javaClass.simpleName),
                )
                onStatus("error")
            } finally {
                connection?.disconnect()
                inFlight.set(false)
            }
        }
    }

    fun invalidate() {
        episodeId.incrementAndGet()
        resetRequested.set(true)
    }

    fun endUtterance() {
        episodeId.incrementAndGet()
        resetRequested.set(true)
    }

    fun suppressBefore(id: Long) {
        suppressBefore.set(max(suppressBefore.get(), id))
    }

    fun clear() {
        invalidate()
        window.clear()
        frameSeq = 0L
        lastHandSeenMs = 0L
        handLostNotified = false
    }

    fun close() {
        closed = true
        scope.cancel()
    }

    private fun parseTop3(array: JSONArray?): List<Pair<String, Float>> {
        if (array == null) return emptyList()
        return buildList {
            for (i in 0 until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                add(item.optString("label", "?") to item.optDouble("confidence", 0.0).toFloat())
            }
        }
    }

    private fun optDoubleOrNull(json: JSONObject, key: String): Double? {
        if (!json.has(key)) return null
        val value = json.optDouble(key, Double.NaN)
        return if (value.isNaN()) null else value
    }

    private fun sha256(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        val out = StringBuilder(digest.size * 2)
        for (b in digest) out.append("%02x".format(b.toInt() and 0xff))
        return out.toString()
    }
}

class RemoteRecognitionController(
    configProvider: () -> RemoteRecognitionConfig,
    sessionId: String,
    private val deliver: (RecognitionEvent, RecognitionContext) -> Unit,
    onStatus: (String) -> Unit,
) : RecognitionController {

    private val remote = RemoteSignRecognizer(
        configProvider = configProvider,
        sessionId = sessionId,
        onEvent = deliver,
        onStatus = onStatus,
    )

    override fun processFrame(
        handResult: HandLandmarkerResult?,
        faceResult: FaceLandmarkerResult?,
        timestampMs: Long,
        ctx: RecognitionContext,
    ): RecognitionFrame = remote.processFrame(handResult, faceResult, timestampMs, ctx)

    override fun currentEpoch(): Long = remote.currentEpoch()

    override fun lastWindowId(): Long = remote.lastWindowId()

    override fun invalidate() = remote.invalidate()

    override fun clear() = remote.clear()

    override fun endUtterance() = remote.endUtterance()

    override fun suppressBefore(windowId: Long) = remote.suppressBefore(windowId)

    override fun close() = remote.close()
}
