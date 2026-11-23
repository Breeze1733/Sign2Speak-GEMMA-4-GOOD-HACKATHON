package com.sign2speak.app

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sign2speak.app.ui.CameraScreen
import com.sign2speak.app.ui.ModelDownloadScreen

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()
    private val downloadViewModel: DownloadViewModel by viewModels()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        cameraPermissionGranted = results[Manifest.permission.CAMERA] == true
    }

    private var cameraPermissionGranted by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        cameraPermissionGranted = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        if (!cameraPermissionGranted || checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
        }

        setContent {
            MaterialTheme(colorScheme = androidx.compose.material3.darkColorScheme()) {

                val modelReady by downloadViewModel.modelReady.collectAsState()
                var initialBinUri by remember { mutableStateOf<Uri?>(null) }
                // Remote P1 and bare recognition do not need the 2.58 GB on-device model.
                var skipModelGate by remember { mutableStateOf(false) }

                val filePickerLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.OpenDocument()
                ) { uri ->
                    if (uri != null) {
                        initialBinUri = uri
                    }
                }

                when {
                    // ── 1. Modèle absent → écran de téléchargement (skip possible) ──
                    !modelReady && !skipModelGate ->
                        ModelDownloadScreen(downloadViewModel) { skipModelGate = true }

                    // ── 2. Modèle OK mais caméra refusée (hors mode .bin) ────
                    !cameraPermissionGranted && initialBinUri == null -> {
                        var permissionRequested by remember { mutableStateOf(false) }
                        Box(
                            modifier = Modifier.fillMaxSize().background(Color.Black),
                            contentAlignment = Alignment.Center,
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Text(
                                    "Sign2Speak",
                                    color = Color.White,
                                    fontSize = 28.sp,
                                )
                                Text(
                                    "La permission caméra est requise\npour détecter la langue des signes.",
                                    color = Color.Gray,
                                    fontSize = 15.sp,
                                    modifier = Modifier.padding(bottom = 12.dp),
                                    textAlign = TextAlign.Center,
                                )
                                Button(onClick = {
                                    permissionRequested = true
                                    permissionLauncher.launch(
                                        arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
                                    )
                                }) {
                                    Text(if (permissionRequested) "Relancer la demande" else "Autoriser la caméra")
                                }
                                Spacer(Modifier.height(4.dp))
                                OutlinedButton(onClick = {
                                    filePickerLauncher.launch(arrayOf("*/*"))
                                }) {
                                    Text("Importer un fichier .bin")
                                }
                            }
                        }
                    }

                    // ── 3. Caméra OK ou mode .bin actif ──────────────────────
                    else -> CameraScreen(
                        viewModel                 = viewModel,
                        initialBinUri             = initialBinUri,
                        hasCameraPermission       = cameraPermissionGranted,
                        onRequestCameraPermission = {
                            permissionLauncher.launch(
                                arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
                            )
                        },
                    )
                }
            }
        }
    }
}
