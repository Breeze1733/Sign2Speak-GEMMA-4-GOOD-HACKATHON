package com.sign2speak.app

import android.content.Context

/** Persisted runtime pipeline: P0 local, P1 remote G2T, P2 remote M3+G2T. */
data class StoredTranslationConfig(
    val pipelineMode: String,   // "p0" | "p1" | "p2"
    val remoteUrl: String,
    val remoteToken: String,
)

class TranslationConfigStore(context: Context) {

    private val prefs = context.getSharedPreferences("s2s_translation", Context.MODE_PRIVATE)

    fun load(): StoredTranslationConfig {
        val stored = prefs.getString(KEY_PIPELINE_MODE, null)
        val mode = if (stored != null) normalize(stored) else migrateLegacy(
            prefs.getString(KEY_LEGACY_MODE, null)
        )
        return StoredTranslationConfig(
            pipelineMode = mode,
            remoteUrl = prefs.getString(KEY_URL, BuildConfig.REMOTE_URL) ?: BuildConfig.REMOTE_URL,
            remoteToken = prefs.getString(KEY_TOKEN, BuildConfig.REMOTE_TOKEN) ?: BuildConfig.REMOTE_TOKEN,
        )
    }

    fun savePipelineMode(mode: String) {
        val normalized = normalize(mode)
        prefs.edit()
            .putString(KEY_PIPELINE_MODE, normalized)
            // Keep the legacy key so an older APK can still read the choice.
            .putString(KEY_LEGACY_MODE, if (normalized == "p0") "local" else "remote")
            .apply()
    }

    fun saveRemote(url: String, token: String) {
        prefs.edit().putString(KEY_URL, url).putString(KEY_TOKEN, token).apply()
    }

    private fun normalize(mode: String): String = when (mode) {
        "p0", "local"  -> "p0"
        "p1", "remote" -> "p1"
        "p2"           -> "p2"
        else           -> "p0"
    }

    private fun migrateLegacy(mode: String?): String =
        if (mode == "remote") "p1" else "p0"

    companion object {
        private const val KEY_PIPELINE_MODE = "pipeline_mode"
        private const val KEY_LEGACY_MODE = "mode"
        private const val KEY_URL = "remote_url"
        private const val KEY_TOKEN = "remote_token"
    }
}
