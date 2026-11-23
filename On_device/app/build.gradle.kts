plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

val timingEnabled = providers.gradleProperty("sign2speakTiming").orNull == "true"

// P1 remote gloss-to-text defaults (overridable at runtime in the app's CFG dialog).
val remoteUrl = providers.gradleProperty("s2sRemoteUrl")
    .orElse("http://121.41.19.165:8080").get()
val remoteToken = providers.gradleProperty("s2sRemoteToken").orElse("").get()
val remoteModelVersion = providers.gradleProperty("s2sRemoteModel")
    .orElse("gemma-4-E2B-it.litertlm").get()
val classifierModelVersion = providers.gradleProperty("s2sClassifierVersion")
    .orElse("islr-conv1d-transformer-v1").get()

android {
    namespace = "com.sign2speak.app"
    compileSdk = 36

    defaultConfig {
        applicationId = if (timingEnabled) "com.sign2speak.app.perf" else "com.sign2speak.app"
        manifestPlaceholders["appLabel"] = if (timingEnabled) "Sign2Speak Timing" else "@string/app_name"
        buildConfigField("boolean", "TIMING_ENABLED", timingEnabled.toString())
        buildConfigField("String", "REMOTE_URL", "\"$remoteUrl\"")
        buildConfigField("String", "REMOTE_TOKEN", "\"$remoteToken\"")
        buildConfigField("String", "REMOTE_MODEL_VERSION", "\"$remoteModelVersion\"")
        buildConfigField("String", "CLASSIFIER_MODEL_VERSION", "\"$classifierModelVersion\"")
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
        }
    }

    buildFeatures {
        buildConfig = true
        compose = true
    }

    // Nécessaire pour les gros fichiers assets (classifier.tflite, modèles gemma .litertlm)
    androidResources {
        noCompress += listOf("tflite", "gguf", "litertlm", "task", "bin")
    }

    // Éviter les conflits de fichiers dupliqués provenant de mediapipe et litert
    packaging {
        resources {
            excludes += listOf(
                "META-INF/LICENSE",
                "META-INF/LICENSE.txt",
                "META-INF/NOTICE",
                "META-INF/NOTICE.txt",
                "META-INF/*.kotlin_module",
                "META-INF/DEPENDENCIES",
                "/META-INF/{AL2.0,LGPL2.1}",
            )
            pickFirsts += listOf(
                "lib/x86/libc++_shared.so",
                "lib/x86_64/libc++_shared.so",
                "lib/armeabi-v7a/libc++_shared.so",
                "lib/arm64-v8a/libc++_shared.so",
            )
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    // CameraX
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)

    // MediaPipe Tasks (vision : HandLandmarker + FaceLandmarker)
    implementation(libs.mediapipe.tasks.vision)
    // LiteRT-LM — Gemma 4 E2B on-device (remplace MediaPipe LLM Inference)
    val liteRtLmVersion = if (timingEnabled) "0.19.0" else "latest.release"
    implementation("com.google.ai.edge.litertlm:litertlm-android:$liteRtLmVersion")

    // TFLite runtime (arm64-v8a + x86_64)
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)

    // ViewModel + Compose intégration
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}
