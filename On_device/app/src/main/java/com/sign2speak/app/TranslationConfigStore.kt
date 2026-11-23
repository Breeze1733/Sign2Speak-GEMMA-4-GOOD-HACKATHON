package com.sign2speak.app

import android.content.Context

/** Persisted runtime choice between P0 local and P1 remote gloss-to-text. */
data class StoredTranslationConfig(
    val mode: String,          // "local" | "remote"
    val remoteUrl: String,
    val remoteToken: String,
)

class TranslationConfigStore(context: Context) {

    private val prefs = context.getSharedPreferences("s2s_translation", Context.MODE_PRIVATE)

    fun load(): StoredTranslationConfig = StoredTranslationConfig(
        mode = prefs.getString(KEY_MODE, DEFAULT_MODE) ?: DEFAULT_MODE,
        remoteUrl = prefs.getString(KEY_URL, BuildConfig.REMOTE_URL) ?: BuildConfig.REMOTE_URL,
        remoteToken = prefs.getString(KEY_TOKEN, BuildConfig.REMOTE_TOKEN) ?: BuildConfig.REMOTE_TOKEN,
    )

    fun saveMode(mode: String) {
        prefs.edit().putString(KEY_MODE, mode).apply()
    }

    fun saveRemote(url: String, token: String) {
        prefs.edit().putString(KEY_URL, url).putString(KEY_TOKEN, token).apply()
    }

    companion object {
        private const val KEY_MODE = "mode"
        private const val KEY_URL = "remote_url"
        private const val KEY_TOKEN = "remote_token"
        private const val DEFAULT_MODE = "local"
    }
}
