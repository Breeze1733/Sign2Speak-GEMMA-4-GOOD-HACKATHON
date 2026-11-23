package com.sign2speak.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sign2speak.app.DownloadState
import com.sign2speak.app.DownloadViewModel

private val Dark   = Color(0xFF121212)
private val Green  = Color(0xFF00E676)
private val Gray   = Color(0xFFBDBDBD)
private val Red    = Color(0xFFEF5350)
private val Scrim  = Color(0xFF1E1E1E)

@Composable
fun ModelDownloadScreen(viewModel: DownloadViewModel, onSkip: () -> Unit = {}) {
    val state by viewModel.downloadState.collectAsState()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Dark),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {

            // ── Logo / titre ──────────────────────────────────────────────────
            Text(
                text       = "Sign2Speak",
                color      = Green,
                fontSize   = 36.sp,
                fontWeight = FontWeight.ExtraBold,
            )
            Text(
                text      = "Bridging silence and sound",
                color     = Gray,
                fontSize  = 14.sp,
                textAlign = TextAlign.Center,
            )

            Spacer(Modifier.height(16.dp))

            // ── Model info ────────────────────────────────────────────────────
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Scrim, RoundedCornerShape(12.dp))
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("Gemma 4 E2B — on-device LLM", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                Text("2.58 GB · needed for on-device translation (P0)", color = Gray, fontSize = 13.sp, textAlign = TextAlign.Center)
                Text("Remote P1 can run without it", color = Gray, fontSize = 12.sp)
            }

            Spacer(Modifier.height(8.dp))

            // ── State-driven UI ───────────────────────────────────────────────
            when (val s = state) {

                is DownloadState.Idle -> {
                    Button(
                        onClick = { viewModel.startDownload() },
                        colors  = ButtonDefaults.buttonColors(containerColor = Green),
                        shape   = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth().height(52.dp),
                    ) {
                        Text("Download Gemma model", color = Dark, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }

                is DownloadState.Downloading -> {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        LinearProgressIndicator(
                            progress = { s.progress },
                            modifier = Modifier.fillMaxWidth().height(8.dp),
                            color    = Green,
                            trackColor = Scrim,
                        )
                        Text(
                            text  = "${"%.0f".format(s.downloadedMb)} MB / ${"%.0f".format(s.totalMb)} MB  (${"%.0f".format(s.progress * 100)}%)",
                            color = Gray,
                            fontSize = 13.sp,
                        )
                        Text("Do not close the app…", color = Gray, fontSize = 12.sp)
                    }
                }

                is DownloadState.Done -> {
                    CircularProgressIndicator(color = Green, modifier = Modifier.size(32.dp))
                    Text("Model ready — loading…", color = Green, fontSize = 14.sp)
                }

                is DownloadState.Error -> {
                    Text(s.message, color = Red, fontSize = 13.sp, textAlign = TextAlign.Center)
                    Spacer(Modifier.height(4.dp))
                    Button(
                        onClick = { viewModel.startDownload() },
                        colors  = ButtonDefaults.buttonColors(containerColor = Red),
                        shape   = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth().height(52.dp),
                    ) {
                        Text("Retry", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }

            TextButton(
                onClick  = onSkip,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    "Skip — recognition / remote P1 only",
                    color    = Gray,
                    fontSize = 13.sp,
                )
            }

            Spacer(Modifier.height(24.dp))

            // ── Gemma terms notice ────────────────────────────────────────────
            Text(
                text      = "This app uses Gemma 4 by Google.\nSubject to the Gemma Terms of Use:\nai.google.dev/gemma/terms",
                color     = Color(0xFF616161),
                fontSize  = 10.sp,
                textAlign = TextAlign.Center,
                lineHeight = 15.sp,
            )
        }
    }
}
