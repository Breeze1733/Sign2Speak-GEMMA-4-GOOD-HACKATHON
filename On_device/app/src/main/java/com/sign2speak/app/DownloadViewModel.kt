package com.sign2speak.app

import android.app.Application
import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

private const val MODEL_FILENAME = "gemma-4-E2B-it.litertlm"

private val MODEL_URLS = listOf(
    "https://hf-mirror.com/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
    "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
)

sealed class DownloadState {
    object Idle        : DownloadState()
    data class Downloading(val progress: Float, val downloadedMb: Float, val totalMb: Float) : DownloadState()
    object Done        : DownloadState()
    data class Error(val message: String) : DownloadState()
}

class DownloadViewModel(application: Application) : AndroidViewModel(application) {

    private val modelFile: File
        get() = File(getApplication<Application>().getExternalFilesDir(null), MODEL_FILENAME)

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState.asStateFlow()

    private val _modelReady = MutableStateFlow(modelFile.exists())
    val modelReady: StateFlow<Boolean> = _modelReady.asStateFlow()

    fun startDownload() {
        if (_downloadState.value is DownloadState.Downloading) return
        downloadFrom(0)
    }

    /** Tries [MODEL_URLS] in order; a failed attempt is removed before the next. */
    private fun downloadFrom(index: Int) {
        if (index >= MODEL_URLS.size) {
            _downloadState.value = DownloadState.Error("All download sources failed")
            return
        }

        val app = getApplication<Application>()
        val dm  = app.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

        // Clear any partial file so a retry starts clean.
        runCatching { modelFile.delete() }

        val request = DownloadManager.Request(Uri.parse(MODEL_URLS[index]))
            .setTitle("Sign2Speak — Gemma 4 model")
            .setDescription(
                if (index == 0) "Downloading Gemma 4 E2B (2.58 GB)…"
                else "Retrying from source ${index + 1} of ${MODEL_URLS.size}…"
            )
            .setDestinationInExternalFilesDir(app, null, MODEL_FILENAME)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)

        val downloadId = dm.enqueue(request)
        viewModelScope.launch {
            val ok = pollProgress(dm, downloadId)
            if (!ok) {
                runCatching { dm.remove(downloadId) }
                runCatching { modelFile.delete() }
                downloadFrom(index + 1)
            }
        }
    }

    /** Returns true when the model is complete, false when this source failed. */
    private suspend fun pollProgress(dm: DownloadManager, id: Long): Boolean {
        _downloadState.value = DownloadState.Downloading(0f, 0f, 2580f)

        while (true) {
            delay(800)
            val cursor = dm.query(DownloadManager.Query().setFilterById(id))

            if (!cursor.moveToFirst()) {
                cursor.close()
                return false
            }

            val status     = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            val downloaded = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
            val total      = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
            cursor.close()

            when (status) {
                DownloadManager.STATUS_SUCCESSFUL -> {
                    _modelReady.value    = true
                    _downloadState.value = DownloadState.Done
                    return true
                }
                DownloadManager.STATUS_FAILED -> return false
                else -> {
                    val dlMb    = downloaded / 1_048_576f
                    val totalMb = if (total > 0) total / 1_048_576f else 2580f
                    val prog    = if (total > 0) downloaded.toFloat() / total.toFloat() else 0f
                    _downloadState.value = DownloadState.Downloading(prog, dlMb, totalMb)
                }
            }
        }
    }
}
