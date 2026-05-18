# Gemma Sign2Speak v1.0

**Real-time American Sign Language (ASL) interpreter running fully on-device.**

---

## What it does

- Detects hand & face landmarks via **MediaPipe** — no internet needed
- Recognizes ASL signs with a custom **TFLite classifier** (Conv1D + Transformer, trained on Kaggle ISLR)
- Translates sign sequences into natural English sentences using **Gemma 4 E2B on-device**
- Sign2Speak” is currently limited to 50 words, but it can be retrained with the full database to reach a vocabulary of 250 words.
- Speaks the result aloud via **Android TTS**
- Sign2Speak also includes a microphone feature that records and transcribes the speaker’s voice into written text displayed on the screen. This allows signers to understand the other person and enables a complete two-way conversation.

---

## Install

1. Download `Sign2Speak.apk` from the [Releases](https://github.com/Pirodax/Sign2Speak-GEMMA-4-GOOD-HACKATHON/releases) page
2. On your Android device: **Settings → Apps → Special app access → Install unknown apps** → allow your browser
3. Open the APK and tap **Install**
4. Launch **Sign2Speak**
5. On first launch, tap **"Download Gemma model"** (2.58 GB — use Wi-Fi)
6. Done — works **fully offline** after that

> Tested on **Pixel 7a (Android 13)** — 8 GB RAM recommended

See [README_INSTALL.md](README_INSTALL.md) for ADB fallback instructions and usage details.

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Kotlin + Jetpack Compose |
| Landmark detection | MediaPipe Tasks (hand + face landmarker) |
| Sign classifier | TensorFlow Lite (custom Conv1D + Transformer) |
| On-device LLM | LiteRT-LM + Gemma 4 E2B (2.58 GB, downloaded at first launch) |
| TTS | Android TextToSpeech |

> This app uses **Gemma 4** by Google, subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). See [NOTICE_GEMMA.txt](NOTICE_GEMMA.txt).

---

## Full pipeline

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TRAINING  (sign2speak-training/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Kaggle ISLR dataset (40 GB, ~94k Parquet files)
            │
            ▼  preconvert.py
  asl_converted/   (.npy sequences, 543×3 landmarks)
            │
            ▼  train_winner.py
  model_winner/    (Keras weights, Conv1D + Transformer)
            │
            ▼  convert_to_tflite.py
  classifier.tflite  →  On_device/app/src/main/assets/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE A — PC prototype  (sign2speak-PC-version/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Webcam (30 fps)
    │
    ▼  MediaPipe Tasks  (hand + face, Python)
  543 landmarks / frame
    │
    ▼  Sliding window → preprocess → TF model
  Predicted sign  (50 classes, confidence ≥ 20%)
    │
    ▼  Gloss buffer  (3 s silence → flush)
  HELLO  THANK-YOU  NAME
    │
    ▼  Ollama  (local HTTP :11434)
       gemma4:e2b
    │
    ▼  Piper TTS  (offline neural voice)
  "Hi, my name is …"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE B — Android app  (On_device/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Camera (CameraX)
    │
    ▼  MediaPipe Tasks  (hand + face, on-device)
  543 landmarks / frame
    │
    ▼  Sliding window → preprocess → TFLite interpreter
  Predicted sign  (confidence ≥ 50%)
    │
    ▼  Gloss buffer  (5 s no hand → flush)
  HELLO  THANK-YOU  NAME
    │
    ▼  LiteRT-LM
       Gemma 4 E2B  (2.58 GB, on-device, fully offline)
    │
    ▼  Android TextToSpeech
  "Hi, my name is …"
```

