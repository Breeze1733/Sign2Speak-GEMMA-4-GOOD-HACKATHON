# Sign2Speak — Installation Guide

Sign2Speak is a real-time ASL (American Sign Language) interpreter that runs **fully on-device**.  
It uses MediaPipe for hand/face detection and Gemma 4 E2B for gloss-to-sentence translation.

---

## Requirements

| What | Minimum |
|------|---------|
| Android version | Android 10 (API 29) |
| RAM | 8 GB recommended |
| Free storage | **3 GB** (for the Gemma model) |
| Connection | Wi-Fi recommended for the first download |

> Tested on **Pixel 7a (Android 13)**. Any recent mid-range or flagship Android should work.

---

## Quick install (recommended)

1. **Download** `Sign2Speak.apk` from the [GitHub Releases](../../releases/latest) page.
2. On your Android device, go to **Settings → Apps → Special app access → Install unknown apps** and allow your browser or file manager.
3. Open the APK and tap **Install**.
4. Launch **Sign2Speak**.
5. On first launch, tap **"Download Gemma model"** (2.58 GB — use Wi-Fi).
6. Wait for the download to complete (progress bar shown).
7. The app opens automatically once the model is ready.

From step 6 onward the app works **offline**.

---

## Fallback: manual install via ADB

If the in-app download fails (network issues, corporate proxy, etc.), you can push the model manually.

### Step 1 — Install the APK
```bash
adb install -r Sign2Speak.apk
```

### Step 2 — Download the model on your computer
```
https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

### Step 3 — Push the model to the device
```bash
adb shell mkdir -p /sdcard/Android/data/com.sign2speak.app/files/
adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.sign2speak.app/files/gemma-4-E2B-it.litertlm
```

### Step 4 — Launch
```bash
adb shell am start -n com.sign2speak.app/.MainActivity
```

---

## How to use Sign2Speak

| Action | Result |
|--------|--------|
| Show your hand to the camera | The app detects signs in real-time |
| Hold a sign clearly | Recognized sign appears in the gloss list |
| Lower your hand for 5 seconds | Glosses are sent to Gemma → natural sentence |
| Tap **TRANS** | Manual translation of current glosses |
| Tap **CLR** | Clear all pending glosses |
| Tap **SPK / MUTE** | Toggle text-to-speech |
| Tap **MIC** | Hearing person speaks → text displayed |
| Tap **P / F / S** | Switch style: Professional / Friendly / Simple |

---

## Gemma Terms of Use

This app uses **Gemma 4** by Google.  
Gemma is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).  
See `NOTICE_GEMMA.txt` for the full notice.
