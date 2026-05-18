# Sign2Speak — PC Version

Real-time ASL sign recognition from your webcam.  
The model (Conv1D + Transformer, trained on Kaggle ISLR) reads hand and face landmarks frame by frame and outputs the recognized sign. When you stop signing, the accumulated glosses are sent to **Gemma 4** (via Ollama) to produce a natural English sentence.

---

## What it does

1. **Webcam capture** at 30 fps via OpenCV.
2. **Landmark extraction** — MediaPipe Tasks detects up to 2 hands (every frame) and 1 face (every 2 frames). This produces a 543-point skeleton per frame.
3. **Sign recognition** — a sliding window of frames is preprocessed (118 key landmarks, nose normalization, velocity + acceleration → 708 features) and fed to the TensorFlow model. Only predictions above the confidence threshold (default 20 %) are kept.
4. **Gloss accumulation** — confirmed signs are queued. After 3 seconds of silence they are flushed as a gloss sequence (e.g. `HELLO | THANK-YOU | NAME`).
5. **Translation** — the gloss sequence is sent to Gemma 4 running locally via Ollama. Gemma returns a natural English sentence.
6. **Text-to-speech** — the sentence is spoken aloud by Piper TTS (offline, neural voice).

---

## Requirements

| Dependency | Install |
|---|---|
| Python 3.9 + | [python.org](https://www.python.org) |
| TensorFlow 2.x | `pip install tensorflow` |
| MediaPipe | `pip install mediapipe` |
| OpenCV | `pip install opencv-python` |
| NumPy | `pip install numpy` |
| Piper TTS *(optional)* | `pip install piper-tts sounddevice` |
| Ollama *(optional)* | [ollama.com](https://ollama.com) |

> Piper and Ollama are both optional. Without them the sign recognition still works in full — you just lose text-to-speech and sentence translation.

Install everything at once:

```bash
pip install tensorflow mediapipe opencv-python numpy piper-tts sounddevice
```

---

## First run — what gets downloaded automatically

On first launch the script downloads two small files into `./models/` (created automatically):

| File | Size | Source |
|---|---|---|
| `hand_landmarker.task` | ~8 MB | Google MediaPipe CDN |
| `face_landmarker.task` | ~8 MB | Google MediaPipe CDN |
| `en_US-lessac-medium.onnx` | ~65 MB | Hugging Face (rhasspy/piper-voices) |
| `en_US-lessac-medium.onnx.json` | <1 MB | Hugging Face |

These are cached locally; subsequent launches are instant.

---

## Ollama setup (for translation)

Ollama runs Gemma 4 locally on your machine. It acts as a local HTTP server that `live_winner.py` calls on port 11434.

**Install Ollama**: download from [ollama.com](https://ollama.com) and run the installer.

**Pull the model** (one-time, ~1.5 GB):

```bash
ollama pull gemma4:e2b
```

**Start the server** (runs in background automatically after install, or manually):

```bash
ollama serve
```

**Without Ollama**: the script detects that port 11434 is unreachable and silently skips translation. Sign recognition, the gloss list, and the history panel all continue to work normally. No error message is shown on screen.

---

## Run

```bash
cd sign2speak-PC-version
python live_winner.py
```

By default the model is loaded from `../model_winner`. You can point to a different directory:

```bash
python live_winner.py --model /path/to/model_winner
python live_winner.py --threshold 0.35   # raise confidence threshold
```

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1` | Translation style: **Professional** (formal, medical/business) |
| `2` | Translation style: **Friendly** (casual, conversational) — default |
| `3` | Translation style: **Simple** (short sentences, easy vocabulary) |
| `m` | Toggle TTS mute / unmute |
| `c` | Clear frame buffer, gloss list, and translation |
| `q` | Quit |

---

## On-screen display

```
FPS: 28          [top-left]        HELLO: 82%   ← top-3 predictions [top-right]
HAND             [detected]        YES:    6%
                                   BYE:    3%

                     ┌─────────────┐
                     │  History    │  ← confirmed word history panel
                     │  THANK-YOU  │
                     │  HELLO      │
                     └─────────────┘

HELLO            ← current sign (large, bottom area)
82%

HELLO  THANK-YOU ← accumulated glosses (waiting for translation)

Hello, thank you ! ← translated sentence (green bar)
[F] SPK          ← style indicator + TTS status [bottom-right]
```

---

## Pipeline position

This script is **Phase A** of the Sign2Speak project — the PC prototype used to validate the model before the Android port.

```
sign2speak-PC-version/live_winner.py   ← you are here
        ↓ validated model
On_device/  (Android app with on-device Gemma 4 via LiteRT)
```

The trained model lives in `model_winner/` at the repo root (produced by `sign2speak-training/train_winner.py`).
