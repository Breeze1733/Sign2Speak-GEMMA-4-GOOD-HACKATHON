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

private const val MODEL_URL =
    "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"
private const val MODEL_FILENAME = "gemma-4-E2B-it.litertlm"

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

        val app = getApplication<Application>()
        val dm  = app.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

        val request = DownloadManager.Request(Uri.parse(MODEL_URL))
            .setTitle("Sign2Speak — Gemma 4 model")
            .setDescription("Downloading Gemma 4 E2B (2.58 GB)…")
            .setDestinationInExternalFilesDir(app, null, MODEL_FILENAME)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)

        val downloadId = dm.enqueue(request)
        viewModelScope.launch { pollProgress(dm, downloadId) }
    }

    private suspend fun pollProgress(dm: DownloadManager, id: Long) {
        _downloadState.value = DownloadState.Downloading(0f, 0f, 2580f)

        while (true) {
            delay(800)
            val cursor = dm.query(DownloadManager.Query().setFilterById(id))

            if (!cursor.moveToFirst()) {
                cursor.close()
                _downloadState.value = DownloadState.Error("Téléchargement annulé")
                return
            }

            val status     = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            val downloaded = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
            val total      = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
            val reason     = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
            cursor.close()

            when (status) {
                DownloadManager.STATUS_SUCCESSFUL -> {
                    _modelReady.value    = true
                    _downloadState.value = DownloadState.Done
                    return
                }
                DownloadManager.STATUS_FAILED -> {
                    _downloadState.value = DownloadState.Error("Échec (code $reason)")
                    return
                }
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
