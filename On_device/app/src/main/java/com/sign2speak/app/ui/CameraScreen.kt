package com.sign2speak.app.ui

import androidx.camera.view.PreviewView
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.sign2speak.app.AppState
import com.sign2speak.app.CameraManager
import com.sign2speak.app.MainViewModel

// ─── Couleurs Sign2Speak ──────────────────────────────────────────────────────
private val ColorGreen     = Color(0xFF00E676)
private val ColorYellow    = Color(0xFFFFD600)
private val ColorOrange    = Color(0xFFFF6D00)
private val ColorWhite     = Color.White
private val ColorGray      = Color(0xFFBDBDBD)
private val ColorDark      = Color(0xFF121212)
private val ColorSemiBlack = Color(0x99000000)

/**
 * CameraScreen — écran principal plein écran.
 *
 * Layout :
 *   - CameraX PreviewView en plein fond
 *   - Overlay transparent par-dessus avec toutes les infos
 *   - Contrôles en bas (mute + styles + clear)
 */
@Composable
fun CameraScreen(viewModel: MainViewModel) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // CameraManager — créé une seule fois, fermé au DisposableEffect
    val cameraManager = remember {
        CameraManager(context) { handResult, faceResult, timestampMs ->
            viewModel.onFrameProcessed(handResult, faceResult, timestampMs)
        }
    }

    val previewView = remember { PreviewView(context) }

    // Initialisation et démarrage de la caméra
    LaunchedEffect(Unit) {
        cameraManager.initialize()
        cameraManager.startCamera(lifecycleOwner, previewView)
    }

    // Nettoyage à la sortie du composable
    DisposableEffect(Unit) {
        onDispose {
            cameraManager.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(ColorDark)) {
        // ── Fond caméra ─────────────────────────────────────────────────────
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // ── Overlay transparent ──────────────────────────────────────────────
        Box(modifier = Modifier.fillMaxSize()) {

            // ── Top-left : FPS + Hand indicator ─────────────────────────────
            TopLeftInfoPanel(state = state, modifier = Modifier.align(Alignment.TopStart))

            // ── Top-right : Top 3 prédictions ────────────────────────────────
            Top3Panel(state = state, modifier = Modifier.align(Alignment.TopEnd))

            // ── Centre-bas : signe courant ────────────────────────────────────
            CurrentSignDisplay(
                state = state,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 160.dp)
            )

            // ── Spinner IA en cours ───────────────────────────────────────────
            if (state.isTranslating) {
                TranslatingSpinner(
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .padding(end = 12.dp)
                )
            }

            // ── Bande basse : glosses + traduction ────────────────────────────
            BottomInfoPanel(
                state = state,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 60.dp)
            )

            // ── Barre de contrôles (tout en bas) ──────────────────────────────
            ControlBar(
                state    = state,
                onMute   = { viewModel.toggleMute() },
                onStyle  = { viewModel.setStyle(it) },
                onClear  = { viewModel.clearBuffer() },
                modifier = Modifier.align(Alignment.BottomCenter)
            )
        }
    }
}

// ─── Composants ──────────────────────────────────────────────────────────────

@Composable
private fun TopLeftInfoPanel(state: AppState, modifier: Modifier) {
    Column(
        modifier = modifier
            .padding(8.dp)
            .background(ColorSemiBlack, RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    ) {
        Text(
            text = "FPS: ${state.fps.toInt()}",
            color = ColorGreen,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(if (state.handDetected) ColorGreen else Color.Red)
            )
            Spacer(Modifier.width(4.dp))
            Text(
                text = if (state.handDetected) "HAND" else "---",
                color = if (state.handDetected) ColorGreen else ColorGray,
                fontSize = 12.sp,
            )
        }
    }
}

@Composable
private fun Top3Panel(state: AppState, modifier: Modifier) {
    if (state.top3.isEmpty()) return
    Column(
        modifier = modifier
            .padding(8.dp)
            .background(ColorSemiBlack, RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.End
    ) {
        state.top3.forEachIndexed { idx, (label, conf) ->
            val fontSize = if (idx == 0) 15.sp else 13.sp
            val color    = if (idx == 0) ColorWhite else ColorGray
            val fw       = if (idx == 0) FontWeight.Bold else FontWeight.Normal
            Text(
                text = "${label.uppercase()}  ${(conf * 100).toInt()}%",
                color = color,
                fontSize = fontSize,
                fontWeight = fw,
            )
        }
    }
}

@Composable
private fun CurrentSignDisplay(state: AppState, modifier: Modifier) {
    if (state.currentSign.isEmpty()) return

    val signColor = when {
        state.currentConf > 0.5f -> ColorGreen
        state.currentConf > 0.3f -> ColorYellow
        else                     -> ColorOrange
    }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Fond léger pour lisibilité
        Box(
            modifier = Modifier
                .background(ColorSemiBlack, RoundedCornerShape(12.dp))
                .padding(horizontal = 20.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = state.currentSign.uppercase(),
                    color = signColor,
                    fontSize = 52.sp,
                    fontWeight = FontWeight.ExtraBold,
                    lineHeight = 56.sp,
                )
                Text(
                    text = "${(state.currentConf * 100).toInt()}%",
                    color = signColor.copy(alpha = 0.8f),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                )
            }
        }
    }
}

@Composable
private fun BottomInfoPanel(state: AppState, modifier: Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp),
        horizontalAlignment = Alignment.Start
    ) {
        // Glosses en attente
        if (state.pendingGlosses.isNotEmpty()) {
            Text(
                text = state.pendingGlosses.joinToString("  ").uppercase(),
                color = ColorWhite,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ColorSemiBlack, RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            )
            Spacer(Modifier.height(4.dp))
        }

        // Phrase traduite
        if (state.translatedSentence.isNotEmpty()) {
            Text(
                text = state.translatedSentence,
                color = ColorGreen,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ColorSemiBlack, RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            )
        }
    }
}

@Composable
private fun TranslatingSpinner(modifier: Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "spinner")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue  = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "spin"
    )
    Box(
        modifier = modifier
            .size(36.dp)
            .background(ColorSemiBlack, CircleShape),
        contentAlignment = Alignment.Center
    ) {
        CircularProgressIndicator(
            modifier = Modifier
                .size(28.dp)
                .rotate(rotation),
            color = Color(0xFF00B0FF),
            strokeWidth = 3.dp,
        )
        Text(
            text = "AI",
            color = Color(0xFF00B0FF),
            fontSize = 8.sp,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
private fun ControlBar(
    state   : AppState,
    onMute  : () -> Unit,
    onStyle : (String) -> Unit,
    onClear : () -> Unit,
    modifier: Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(ColorSemiBlack)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment     = Alignment.CenterVertically,
    ) {
        // Bouton Mute
        Button(
            onClick = onMute,
            colors  = ButtonDefaults.buttonColors(
                containerColor = if (state.isMuted) Color(0xFF424242) else Color(0xFF1B5E20)
            ),
            modifier = Modifier.width(56.dp)
        ) {
            Text(
                text     = if (state.isMuted) "MUTE" else "SPK",
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color    = ColorWhite,
                maxLines = 1,
            )
        }

        // Boutons de style
        val styles = listOf("professional" to "P", "friendly" to "F", "simple" to "S")
        styles.forEach { (styleKey, label) ->
            val isSelected = state.style == styleKey
            FilledTonalButton(
                onClick = { onStyle(styleKey) },
                colors  = ButtonDefaults.filledTonalButtonColors(
                    containerColor = if (isSelected) Color(0xFF1565C0) else Color(0xFF263238),
                    contentColor   = ColorWhite,
                ),
                modifier = Modifier.width(48.dp)
            ) {
                Text(
                    text       = label,
                    fontSize   = 14.sp,
                    fontWeight = if (isSelected) FontWeight.ExtraBold else FontWeight.Normal,
                )
            }
        }

        // Bouton Clear
        Button(
            onClick = onClear,
            colors  = ButtonDefaults.buttonColors(containerColor = Color(0xFF4E342E)),
            modifier = Modifier.width(56.dp)
        ) {
            Text(
                text     = "CLR",
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color    = ColorWhite,
            )
        }
    }
}
