package com.sign2speak.app

import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong

data class PerfTrial(val id: Long, val expected: String)
data class PerfFrame(val trial: PerfTrial, val observedNs: Long, val detectedNs: Long, val frameId: Long,
    val episodeId: Long = 0, val firstHandNs: Long = 0, val firstHandDetectedNs: Long = 0) {
    companion object {
        val Disabled = PerfFrame(PerfTrial(0, ""), 0, 0, -1)
    }
}

/** The timing build logs events without exporting images or landmarks. */
object PerfTrace {
    private val session = UUID.randomUUID().toString()
    private val trials = AtomicLong()
    private val requests = AtomicLong()
    @Volatile var current = PerfTrial(0, "")
        private set
    private var handTrial = 0L
    private var wasHand = false
    private var episodeId = 0L
    private var firstHandNs = 0L
    private var firstHandDetectedNs = 0L
    private var absentSinceNs = 0L

    fun now(): Long = if (BuildConfig.TIMING_ENABLED) SystemClock.elapsedRealtimeNanos() else 0L
    fun nextRequest(): String = requests.incrementAndGet().toString()

    @Synchronized fun begin(expected: String, reason: String = "manual_test_marker") {
        if (!BuildConfig.TIMING_ENABLED) return
        if (current.id != 0L) event("trial_end", current)
        current = PerfTrial(trials.incrementAndGet(), expected.trim().lowercase())
        event("trial_start", current, "start_definition" to reason)
    }

    @Synchronized fun observeHand(frame: PerfFrame, hasHand: Boolean): PerfFrame {
        if (frame.trial.id == 0L) return frame
        if (handTrial != frame.trial.id) {
            handTrial = frame.trial.id; wasHand = false
            firstHandNs = 0L; absentSinceNs = 0L
        }
        if (hasHand != wasHand) {
            if (hasHand) {
                // Ignore brief tracking dropouts; a new word needs >=200 ms without hands.
                if (firstHandNs == 0L || (absentSinceNs > 0L && frame.observedNs - absentSinceNs >= 200_000_000L)) {
                    episodeId++
                    firstHandNs = frame.observedNs
                    firstHandDetectedNs = frame.detectedNs
                }
                absentSinceNs = 0L
            } else {
                absentSinceNs = frame.observedNs
            }
            event(if (hasHand) "hand_seen" else "hand_lost", frame.trial,
                "frame_id" to frame.frameId,
                "episode_id" to episodeId,
                "frame_observed_ns" to frame.observedNs.toString(),
                "detection_finished_ns" to frame.detectedNs.toString())
            wasHand = hasHand
        }
        return frame.copy(episodeId = episodeId, firstHandNs = firstHandNs,
            firstHandDetectedNs = firstHandDetectedNs)
    }

    fun event(name: String, trial: PerfTrial = current, vararg fields: Pair<String, Any>) {
        if (trial.id == 0L) return
        val json = JSONObject().put("schema", 1).put("session", session)
            .put("trial_id", trial.id).put("expected", trial.expected)
            .put("event", name).put("t_ns", now().toString())
            .put("wall_ms", System.currentTimeMillis())
        for ((key, value) in fields) json.put(key, value)
        Log.i("S2S_PERF", json.toString())
    }
}
