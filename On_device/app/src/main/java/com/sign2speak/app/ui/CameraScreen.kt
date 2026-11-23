package com.sign2speak.app.ui

import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import com.sign2speak.app.FramePackSource
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.sign2speak.app.AppState
import com.sign2speak.app.BuildConfig
import com.sign2speak.app.CameraManager
import com.sign2speak.app.DetectedHand
import com.sign2speak.app.HistoryEntry
import com.sign2speak.app.LandmarkPoint
import com.sign2speak.app.MainViewModel
import java.io.File

// ─── MediaPipe hand skeleton connections (21 keypoints) ──────────────────────
private val HAND_CONNECTIONS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,            // thumb
    0 to 5, 5 to 6, 6 to 7, 7 to 8,             // index
    0 to 9, 9 to 10, 10 to 11, 11 to 12,        // middle
    0 to 13, 13 to 14, 14 to 15, 15 to 16,      // ring
    0 to 17, 17 to 18, 18 to 19, 19 to 20,      // pinky
    5 to 9, 9 to 13, 13 to 17,                   // palm
)

// ─── Palette ──────────────────────────────────────────────────────────────────
private val Green  = Color(0xFF00E676)
private val Orange = Color(0xFFFF9800)
private val Cyan   = Color(0xFF00B0FF)
private val Yellow = Color(0xFFFFD600)
private val White  = Color.White
private val Gray   = Color(0xFFBDBDBD)
private val Dark   = Color(0xFF121212)
private val Scrim  = Color(0xB3000000)   // 70 % black

// ─── Root screen ─────────────────────────────────────────────────────────────

@Composable
fun CameraScreen(
    viewModel: MainViewModel,
    initialBinUri: Uri? = null,
    hasCameraPermission: Boolean = true,
    onRequestCameraPermission: () -> Unit = {},
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    var activeBinUri by remember { mutableStateOf(initialBinUri) }
    var binRunCounter by remember { mutableStateOf(0) }
    var videoPreviewBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var binFileName by remember { mutableStateOf("") }
    var binFrameIdx by remember { mutableStateOf(0) }
    var binTotalFrames by remember { mutableStateOf(0) }
    var isBinPlaying by remember { mutableStateOf(false) }
    var binError by remember { mutableStateOf<String?>(null) }
    var showRemoteDialog by remember { mutableStateOf(false) }
    var showClipsDialog by remember { mutableStateOf(false) }
    val clipFiles = remember(showClipsDialog) {
        context.getExternalFilesDir(null)
            ?.listFiles { f -> f.isFile && f.name.endsWith(".bin", ignoreCase = true) }
            ?.sortedBy { it.name }
            ?: emptyList()
    }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            activeBinUri = uri
            binRunCounter++
        }
    }

    val cameraManager = remember {
        CameraManager(context) { hand, face, ts, imgW, imgH, perf ->
            viewModel.onFrameProcessed(hand, face, ts, imgW, imgH, perf)
        }
    }
    val framePackSource = remember {
        FramePackSource(
            context = context,
            onFrame = { hand, face, ts, imgW, imgH, perf, bmp, idx, total ->
                viewModel.onFrameProcessed(hand, face, ts, imgW, imgH, perf)
                mainHandler.post {
                    videoPreviewBitmap = bmp
                    binFrameIdx = idx
                    binTotalFrames = total
                }
            },
            onStateChange = { name, playing, err ->
                mainHandler.post {
                    binFileName = name
                    isBinPlaying = playing
                    binError = err
                }
            },
        )
    }
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    LaunchedEffect(Unit) {
        cameraManager.initialize()
    }
    LaunchedEffect(activeBinUri, binRunCounter, hasCameraPermission) {
        val uri = activeBinUri
        if (uri != null) {
            cameraManager.stopCamera()
            videoPreviewBitmap = null
            binFrameIdx = 0
            binTotalFrames = 0
            binError = null
            viewModel.clearBuffer()
            framePackSource.start(uri)
        } else if (hasCameraPermission) {
            framePackSource.stop()
            videoPreviewBitmap = null
            binError = null
            viewModel.clearBuffer()
            cameraManager.startCamera(lifecycleOwner, previewView)
        } else {
            framePackSource.stop()
            cameraManager.stopCamera()
        }
    }
    DisposableEffect(Unit) {
        onDispose {
            cameraManager.close()
            framePackSource.close()
        }
    }

    val isBinMode = activeBinUri != null

    Box(modifier = Modifier.fillMaxSize().background(Dark)) {

        // ── 1. Camera preview or imported .bin frame preview ────────────────
        if (!isBinMode) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        } else {
            val previewBmp = videoPreviewBitmap
            if (previewBmp != null) {
                Image(
                    bitmap             = previewBmp.asImageBitmap(),
                    contentDescription = "Imported frame pack preview",
                    contentScale       = ContentScale.Fit,
                    modifier           = Modifier.fillMaxSize(),
                )
            }
        }

        // ── 2. Landmark skeleton overlay ─────────────────────────────────────
        LandmarkOverlay(
            detectedHands = state.detectedHands,
            faceDots      = state.faceDots,
            cameraWidth   = state.cameraWidth,
            cameraHeight  = state.cameraHeight,
            isCameraMode  = !isBinMode,
        )

        // ── 3. HUD ───────────────────────────────────────────────────────────
        Box(modifier = Modifier.fillMaxSize()) {

            // Top-left: status + history
            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .statusBarsPadding()
                    .padding(8.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                StatusPanel(state)
                if (BuildConfig.TIMING_ENABLED) PerfTestPanel(viewModel)
                HistoryPanel(
                    history  = state.history,
                    onReplay = { entry -> viewModel.replaySentence(entry.interpretation) },
                )
            }

            // Top-right: .bin source controls + top-3 predictions
            Column(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .statusBarsPadding()
                    .padding(8.dp),
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                SourceControlPanel(
                    isBinMode      = isBinMode,
                    isBinPlaying   = isBinPlaying,
                    binFileName    = binFileName,
                    binFrameIdx    = binFrameIdx,
                    binTotalFrames = binTotalFrames,
                    binError       = binError,
                    onPickBin      = { filePickerLauncher.launch(arrayOf("*/*")) },
                    onReplayBin    = { binRunCounter++ },
                    onSwitchCamera = {
                        if (hasCameraPermission) activeBinUri = null
                        else onRequestCameraPermission()
                    },
                    onOpenClips = { showClipsDialog = true },
                )
                TranslationModePanel(
                    mode         = state.pipelineMode,
                    lastSource   = state.lastSource,
                    lastStatus   = state.lastStatus,
                    lastM3Status = state.lastM3Status,
                    onSelect     = { viewModel.setPipelineMode(it) },
                    onConfigure  = { showRemoteDialog = true },
                )
                Top3Panel(state)
            }

            // Top-center: heard text from hearing person
            if (state.isListening || state.heardText.isNotEmpty()) {
                HeardTextPanel(
                    text        = state.heardText,
                    isListening = state.isListening,
                    modifier    = Modifier
                        .align(Alignment.TopCenter)
                        .statusBarsPadding()
                        .padding(horizontal = 100.dp, vertical = 8.dp),
                )
            }

            // Current sign (large, center-bottom)
            if (state.currentSign.isNotEmpty()) {
                CurrentSignDisplay(
                    state = state,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 168.dp),
                )
            }

            // AI spinner
            if (state.isTranslating) {
                TranslatingSpinner(
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .padding(end = 12.dp),
                )
            }

            // Glosses + translation sentence
            BottomInfoPanel(
                state = state,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 64.dp),
            )

            // Control bar
            ControlBar(
                state       = state,
                onMute      = { viewModel.toggleMute() },
                onStyle     = { viewModel.setStyle(it) },
                onTranslate = { viewModel.manualTranslate() },
                onClear     = { viewModel.clearBuffer() },
                onListen    = { viewModel.toggleListening() },
                modifier    = Modifier.align(Alignment.BottomCenter),
            )

            if (showRemoteDialog) {
                RemoteSettingsDialog(
                    initialUrl = state.remoteUrl,
                    hasToken   = state.remoteHasToken,
                    onDismiss  = { showRemoteDialog = false },
                    onSave     = { url, token ->
                        viewModel.updateRemoteSettings(url, token)
                        showRemoteDialog = false
                    },
                )
            }

            if (showClipsDialog) {
                ClipLibraryDialog(
                    files       = clipFiles,
                    packageName = context.packageName,
                    onDismiss   = { showClipsDialog = false },
                    onPick      = { file ->
                        activeBinUri = Uri.fromFile(file)
                        binRunCounter++
                        showClipsDialog = false
                    },
                )
            }
        }
    }
}

// ─── Landmark skeleton Canvas ─────────────────────────────────────────────────

@Composable
private fun LandmarkOverlay(
    detectedHands: List<DetectedHand>,
    faceDots     : List<LandmarkPoint>,
    cameraWidth  : Int,
    cameraHeight : Int,
    isCameraMode : Boolean,
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height

        fun lmToScreen(lm: LandmarkPoint): Offset {
            return if (isCameraMode) {
                // FILL_CENTER — calcul toujours en espace portrait (narrow×tall)
                // pour être robuste quelle que soit l'orientation du bitmap caméra.
                val camNarrow = minOf(cameraWidth, cameraHeight).toFloat()  // 720
                val camTall   = maxOf(cameraWidth, cameraHeight).toFloat()  // 1280
                val scale = maxOf(w / camNarrow, h / camTall)
                val dispW = camNarrow * scale   // ex. 1350 sur écran 1080×2400
                val dispH = camTall   * scale   // ex. 2400
                val cropX = (dispW - w) / 2f   // ex. 135 px — utilisé pour compenser le crop FILL_CENTER
                // (1-lm.x) : miroir caméra frontale sur axe X
                Offset((1f - lm.x) * dispW - cropX, lm.y * dispH)
            } else {
                // Mode .bin : Image utilise ContentScale.Fit sans miroir
                val srcW = cameraWidth.toFloat().coerceAtLeast(1f)
                val srcH = cameraHeight.toFloat().coerceAtLeast(1f)
                val scale = minOf(w / srcW, h / srcH)
                val dispW = srcW * scale
                val dispH = srcH * scale
                val offsetX = (w - dispW) / 2f
                val offsetY = (h - dispH) / 2f
                Offset(offsetX + lm.x * dispW, offsetY + lm.y * dispH)
            }
        }

        // Face: lip, nose, eye dots (cyan)
        faceDots.forEach { pt ->
            drawCircle(
                color  = Cyan.copy(alpha = 0.75f),
                radius = 3.dp.toPx(),
                center = lmToScreen(pt),
            )
        }

        // Hands: skeleton + joints
        detectedHands.forEach { hand ->
            val baseColor = if (hand.isLeft) Green else Orange
            val pts = hand.points

            // Bones
            HAND_CONNECTIONS.forEach { (i, j) ->
                if (i < pts.size && j < pts.size) {
                    drawLine(
                        color       = baseColor.copy(alpha = 0.65f),
                        start       = lmToScreen(pts[i]),
                        end         = lmToScreen(pts[j]),
                        strokeWidth = 3.dp.toPx(),
                    )
                }
            }

            // Joints: filled circle + white centre
            pts.forEach { pt ->
                val center = lmToScreen(pt)
                drawCircle(baseColor, radius = 6.dp.toPx(), center = center)
                drawCircle(White.copy(alpha = 0.7f), radius = 2.5.dp.toPx(), center = center)
            }
        }
    }
}

// ─── Status panel (top-left) ──────────────────────────────────────────────────

@Composable
private fun StatusPanel(state: AppState) {
    Column(
        modifier = Modifier
            .background(Scrim, RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp),
    ) {
        Text("FPS: ${state.fps.toInt()}", color = Green, fontSize = 13.sp, fontWeight = FontWeight.Medium)

        // Hand present indicator
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(8.dp).clip(CircleShape)
                    .background(if (state.handDetected) Green else Color.Red)
            )
            Spacer(Modifier.width(4.dp))
            Text(
                text  = if (state.handDetected) "HAND" else "NO HAND",
                color = if (state.handDetected) Green else Gray,
                fontSize = 12.sp,
            )
        }

        // Per-hand label
        state.detectedHands.forEach { hand ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier.size(6.dp).clip(CircleShape)
                        .background(if (hand.isLeft) Green else Orange)
                )
                Spacer(Modifier.width(4.dp))
                Text(
                    text = if (hand.isLeft) "LEFT  ✓" else "RIGHT",
                    color = if (hand.isLeft) Green else Orange,
                    fontSize = 11.sp,
                    fontWeight = if (hand.isLeft) FontWeight.Bold else FontWeight.Normal,
                )
            }
        }

        // Reminder if no left hand
        if (state.detectedHands.none { it.isLeft }) {
            Text(
                text     = "→ Show left hand",
                color    = Yellow.copy(alpha = 0.85f),
                fontSize = 10.sp,
            )
        }
    }
}

// ─── Source control panel (.bin import / camera switch) ──────────────────────

@Composable
private fun SourceControlPanel(
    isBinMode: Boolean,
    isBinPlaying: Boolean,
    binFileName: String,
    binFrameIdx: Int,
    binTotalFrames: Int,
    binError: String?,
    onPickBin: () -> Unit,
    onReplayBin: () -> Unit,
    onSwitchCamera: () -> Unit,
    onOpenClips: () -> Unit,
) {
    Column(
        modifier = Modifier
            .background(Scrim, RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.End,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .background(Color(0xFF00695C), RoundedCornerShape(6.dp))
                    .clickable { onPickBin() }
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            ) {
                Text(".BIN", color = White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
            Box(
                modifier = Modifier
                    .background(Color(0xFF283593), RoundedCornerShape(6.dp))
                    .clickable { onOpenClips() }
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            ) {
                Text("CLIPS", color = White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }

            if (isBinMode) {
                if (!isBinPlaying) {
                    Box(
                        modifier = Modifier
                            .background(Color(0xFF1565C0), RoundedCornerShape(6.dp))
                            .clickable { onReplayBin() }
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                    ) {
                        Text("REPLAY", color = White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
                Box(
                    modifier = Modifier
                        .background(Color(0xFF37474F), RoundedCornerShape(6.dp))
                        .clickable { onSwitchCamera() }
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                ) {
                    Text("CAM", color = White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        if (isBinMode) {
            val statusLine = when {
                binError != null -> binError
                binTotalFrames > 0 -> "$binFileName ($binFrameIdx/$binTotalFrames)"
                else -> binFileName.ifEmpty { "Loading .bin…" }
            }
            Text(
                text = statusLine,
                color = if (binError != null) Color(0xFFEF5350) else Cyan,
                fontSize = 10.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.widthIn(max = 180.dp),
            )
        }
    }
}

// ─── Top-3 predictions (top-right) ───────────────────────────────────────────

@Composable
private fun Top3Panel(state: AppState) {
    if (state.top3.isEmpty()) return
    Column(
        modifier = Modifier
            .background(Scrim, RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.End,
    ) {
        state.top3.forEachIndexed { i, (label, conf) ->
            Text(
                text       = "${label.uppercase()}  ${(conf * 100).toInt()}%",
                color      = if (i == 0) White else Gray,
                fontSize   = if (i == 0) 15.sp else 12.sp,
                fontWeight = if (i == 0) FontWeight.Bold else FontWeight.Normal,
            )
        }
    }
}

// ─── History panel (top-left) ─────────────────────────────────────────────────

@Composable
private fun HistoryPanel(history: List<HistoryEntry>, onReplay: (HistoryEntry) -> Unit) {
    if (history.isEmpty()) return
    LazyColumn(
        modifier = Modifier
            .widthIn(max = 260.dp)
            .heightIn(max = 220.dp)
            .background(Scrim, RoundedCornerShape(8.dp))
            .padding(6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
        reverseLayout = true,
    ) {
        items(history, key = { it.id }) { entry ->
            HistoryRow(entry = entry, onReplay = { onReplay(entry) })
        }
    }
}

@Composable
private fun HistoryRow(entry: HistoryEntry, onReplay: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 6.dp)) {
            Text(
                text       = entry.glosses.joinToString(" | ").uppercase(),
                color      = Cyan.copy(alpha = 0.75f),
                fontSize   = 8.sp,
                fontFamily = FontFamily.Monospace,
                maxLines   = 1,
                overflow   = TextOverflow.Ellipsis,
            )
            Text(
                text       = entry.interpretation,
                color      = White,
                fontSize   = 11.sp,
                lineHeight = 14.sp,
                maxLines   = 2,
                overflow   = TextOverflow.Ellipsis,
            )
        }
        Box(
            modifier = Modifier
                .size(26.dp)
                .background(Color(0xFF0D47A1), CircleShape)
                .clickable { onReplay() },
            contentAlignment = Alignment.Center,
        ) {
            Text("▶", color = White, fontSize = 9.sp)
        }
    }
}

// ─── Current sign ─────────────────────────────────────────────────────────────

@Composable
private fun CurrentSignDisplay(state: AppState, modifier: Modifier) {
    val color = when {
        state.currentConf > 0.5f -> Green
        state.currentConf > 0.3f -> Yellow
        else                     -> Orange
    }
    Box(
        modifier = modifier
            .background(Scrim, RoundedCornerShape(12.dp))
            .padding(horizontal = 24.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text       = state.currentSign.uppercase(),
                color      = color,
                fontSize   = 54.sp,
                fontWeight = FontWeight.ExtraBold,
            )
            Text(
                text     = "${(state.currentConf * 100).toInt()}%",
                color    = color.copy(alpha = 0.8f),
                fontSize = 18.sp,
            )
        }
    }
}

// ─── AI spinner ───────────────────────────────────────────────────────────────

@Composable
private fun TranslatingSpinner(modifier: Modifier) {
    val rot by rememberInfiniteTransition(label = "spin").animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(900, easing = LinearEasing), RepeatMode.Restart),
        label = "rot",
    )
    Box(
        modifier = modifier.size(36.dp).background(Scrim, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        CircularProgressIndicator(Modifier.size(28.dp).rotate(rot), color = Cyan, strokeWidth = 3.dp)
        Text("AI", color = Cyan, fontSize = 8.sp, fontWeight = FontWeight.Bold)
    }
}

// ─── Bottom info panel (glosses + translation) ────────────────────────────────

@Composable
private fun BottomInfoPanel(state: AppState, modifier: Modifier) {
    Column(
        modifier = modifier.fillMaxWidth().padding(horizontal = 12.dp),
        horizontalAlignment = Alignment.Start,
    ) {
        if (state.pendingGlosses.isNotEmpty()) {
            Text(
                text       = state.pendingGlosses.joinToString("  ").uppercase(),
                color      = White,
                fontSize   = 16.sp,
                fontWeight = FontWeight.Medium,
                maxLines   = 2,
                overflow   = TextOverflow.Ellipsis,
                modifier   = Modifier
                    .fillMaxWidth()
                    .background(Scrim, RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            )
            Spacer(Modifier.height(4.dp))
        }
        if (state.translatedSentence.isNotEmpty()) {
            Text(
                text       = state.translatedSentence,
                color      = Green,
                fontSize   = 15.sp,
                fontWeight = FontWeight.SemiBold,
                maxLines   = 2,
                overflow   = TextOverflow.Ellipsis,
                modifier   = Modifier
                    .fillMaxWidth()
                    .background(Scrim, RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            )
        }
    }
}

// ─── Heard text panel (top-center) — texte de la personne entendante ─────────

@Composable
private fun HeardTextPanel(text: String, isListening: Boolean, modifier: Modifier) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(Scrim, RoundedCornerShape(10.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            if (isListening && text.isEmpty()) {
                Text(
                    text       = "Listening…",
                    color      = Cyan,
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.Medium,
                    textAlign  = TextAlign.Center,
                )
            }
            if (text.isNotEmpty()) {
                Text(
                    text       = text,
                    color      = White,
                    fontSize   = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    textAlign  = TextAlign.Center,
                    maxLines   = 3,
                    overflow   = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

// ─── Pipeline mode (P0 local / P1 remote G2T / P2 remote M3+G2T) ─────────────

@Composable
private fun TranslationModePanel(
    mode         : String,
    lastSource   : String,
    lastStatus   : String,
    lastM3Status : String,
    onSelect     : (String) -> Unit,
    onConfigure  : () -> Unit,
) {
    Column(
        modifier = Modifier
            .background(Scrim, RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.End,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            ModeChip("P0 LOCAL", mode == "p0", Color(0xFF4E342E)) { onSelect("p0") }
            ModeChip("P1 G2T", mode == "p1", Color(0xFF00695C)) { onSelect("p1") }
            ModeChip("P2 M3", mode == "p2", Color(0xFF283593)) { onSelect("p2") }
            Box(
                modifier = Modifier
                    .background(Color(0xFF263238), RoundedCornerShape(6.dp))
                    .clickable { onConfigure() }
                    .padding(horizontal = 7.dp, vertical = 4.dp),
            ) {
                Text("CFG", color = White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        if (lastSource.isNotEmpty()) {
            Text(
                text = "last: $lastSource ${lastStatus.take(20)}",
                color = Cyan,
                fontSize = 9.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.widthIn(max = 230.dp),
            )
        }
        if (mode == "p2") {
            Text(
                text = "M3: ${lastM3Status.take(20)}",
                color = Color(0xFF80CBC4),
                fontSize = 9.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.widthIn(max = 230.dp),
            )
        }
    }
}

@Composable
private fun ModeChip(
    label       : String,
    active      : Boolean,
    activeColor : Color,
    onClick     : () -> Unit,
) {
    Box(
        modifier = Modifier
            .background(
                if (active) activeColor else Color(0xFF263238),
                RoundedCornerShape(6.dp),
            )
            .clickable { onClick() }
            .padding(horizontal = 7.dp, vertical = 4.dp),
    ) {
        Text(
            text       = label,
            color      = White,
            fontSize   = 10.sp,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
private fun RemoteSettingsDialog(
    initialUrl: String,
    hasToken  : Boolean,
    onDismiss : () -> Unit,
    onSave    : (url: String, token: String) -> Unit,
) {
    var url   by remember { mutableStateOf(initialUrl) }
    var token by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("P1/P2 remote service") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value         = url,
                    onValueChange = { url = it },
                    label         = { Text("Base URL") },
                    singleLine    = true,
                )
                OutlinedTextField(
                    value         = token,
                    onValueChange = { token = it },
                    label         = { Text(if (hasToken) "Token (blank = keep)" else "Token") },
                    singleLine    = true,
                )
                Text(
                    "P1: POST /translate · P2: POST /recognize\nBearer token · cleartext HTTP",
                    color = Gray,
                    fontSize = 11.sp,
                )
            }
        },
        confirmButton = {
            TextButton(onClick = { onSave(url, token) }) { Text("SAVE") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("CANCEL") }
        },
    )
}

// ─── Local clip library (bypasses the system file picker) ────────────────────

@Composable
private fun ClipLibraryDialog(
    files      : List<File>,
    packageName: String,
    onDismiss  : () -> Unit,
    onPick     : (File) -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Local test clips") },
        text = {
            if (files.isEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("No .bin files in the app folder.", color = Gray, fontSize = 13.sp)
                    Text(
                        "adb push <clip>.bin /sdcard/Android/data/$packageName/files/",
                        color = Cyan,
                        fontSize = 11.sp,
                    )
                }
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    files.forEach { file ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { onPick(file) }
                                .padding(vertical = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = file.name,
                                color = White,
                                fontSize = 13.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                text = "${file.length() / (1024 * 1024)} MB",
                                color = Gray,
                                fontSize = 11.sp,
                                modifier = Modifier.padding(start = 8.dp),
                            )
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("CLOSE") } },
    )
}

// ─── Control bar ─────────────────────────────────────────────────────────────

@Composable
private fun ControlBar(
    state      : AppState,
    onMute     : () -> Unit,
    onStyle    : (String) -> Unit,
    onTranslate: () -> Unit,
    onClear    : () -> Unit,
    onListen   : () -> Unit,
    modifier   : Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(Scrim)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        BarBtn(
            label   = if (state.isMuted) "MUTE" else "SPK",
            onClick = onMute,
            color   = if (state.isMuted) Color(0xFF424242) else Color(0xFF1B5E20),
        )

        // MIC button — red pulsing when active
        BarBtn(
            label   = if (state.isListening) "● MIC" else "MIC",
            onClick = onListen,
            color   = if (state.isListening) Color(0xFFB71C1C) else Color(0xFF4A148C),
            bold    = state.isListening,
        )

        listOf("professional" to "P", "friendly" to "F", "simple" to "S").forEach { (key, lbl) ->
            BarBtn(
                label   = lbl,
                onClick = { onStyle(key) },
                color   = if (state.style == key) Color(0xFF1565C0) else Color(0xFF263238),
                bold    = state.style == key,
            )
        }

        BarBtn(
            label   = "TRANS",
            onClick = onTranslate,
            color   = if (state.pendingGlosses.isNotEmpty()) Color(0xFF6A1B9A) else Color(0xFF263238),
            enabled = state.pendingGlosses.isNotEmpty() && !state.isTranslating,
        )

        BarBtn(label = "CLR", onClick = onClear, color = Color(0xFF4E342E))
    }
}

@Composable
private fun RowScope.BarBtn(
    label  : String,
    onClick: () -> Unit,
    color  : Color,
    bold   : Boolean = false,
    enabled: Boolean = true,
) {
    Button(
        onClick  = onClick,
        enabled  = enabled,
        colors   = ButtonDefaults.buttonColors(
            containerColor         = color,
            disabledContainerColor = color.copy(alpha = 0.35f),
        ),
        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
        modifier = Modifier.weight(1f).height(36.dp),
    ) {
        Text(
            text       = label,
            fontSize   = 10.sp,
            fontWeight = if (bold) FontWeight.ExtraBold else FontWeight.Bold,
            color      = White,
            maxLines   = 1,
        )
    }
}
