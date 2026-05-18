"""
Sign2Speak — Live webcam inference with the Kaggle winning model
================================================================
Adapts the winner's preprocessing pipeline (118 landmarks, nose normalization,
velocity + acceleration = 708 features) to MediaPipe Tasks for real-time use.

Usage:
    python live_winner.py
    or
    python live_winner.py --model ../model_winner
"""

import argparse
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*was passed an input with a mask.*",
    category=UserWarning,
)
import json
import time
import urllib.request
import urllib.error
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
)


# Model configuration
# =============================================================================

ROWS_PER_FRAME = 543

NOSE  = [1, 2, 98, 327]
LIP   = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
         291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
         78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
         95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
REYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE  = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)   # 118
CHANNELS  = 6 * NUM_NODES          # 708

MODELS_DIR = Path(__file__).parent / "models"

# FPS optimizations
FACE_EVERY_N = 2    # run face detector every N frames (face = 77/118 landmarks)
DETECT_SCALE = 0.8  # 0.8 = good speed/robustness tradeoff for fast movements

# Drawing helpers — pure OpenCV (no mediapipe.framework dependency)
# =============================================================================

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
# color per finger (thumb → pinky)
_FINGER_COLORS = [
    (0, 176, 255), (0, 255, 118), (255, 176, 0), (255, 0, 176), (176, 0, 255),
]
_FINGER_RANGES = [(1,5),(5,9),(9,13),(13,17),(17,21)]

def draw_hand_landmarks(image, hand_result):
    if not hand_result or not hand_result.hand_landmarks:
        return
    h, w = image.shape[:2]
    for hand_lms in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], (220, 220, 220), 1, cv2.LINE_AA)
        for color, (start, end) in zip(_FINGER_COLORS, _FINGER_RANGES):
            for i in range(start, end):
                cv2.circle(image, pts[i], 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, pts[0], 5, (255, 255, 255), -1, cv2.LINE_AA)


def draw_face_landmarks(image, face_result):
    """Draw only the 77 face landmarks used by the model."""
    if not face_result or not face_result.face_landmarks:
        return
    h, w = image.shape[:2]
    lms = face_result.face_landmarks[0]
    for idx in LIP:
        lm = lms[idx]
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 2, (0, 220, 255), -1, cv2.LINE_AA)
    for idx in LEYE + REYE:
        lm = lms[idx]
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 2, (180, 255, 180), -1, cv2.LINE_AA)
    for idx in NOSE:
        lm = lms[idx]
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 2, (255, 220, 0), -1, cv2.LINE_AA)


def _draw_word_history(image, word_history):
    if not word_history:
        return
    img_h, img_w = image.shape[:2]
    words   = list(word_history)
    n       = len(words)
    line_h  = 30
    pad     = 10
    panel_w = 210
    panel_h = n * line_h + pad * 2 + 14

    x0 = img_w - panel_w - 8
    y0 = img_h // 2 - panel_h // 2

    # semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.rectangle(image, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

    cv2.putText(image, "History", (x0 + 6, y0 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    for i, word in enumerate(reversed(words)):
        alpha = max(0.25, 1.0 - i * 0.13)
        v     = int(255 * alpha)
        color = (v, v, v)
        thick = 2 if i == 0 else 1
        y = y0 + pad + 14 + (i + 1) * line_h - 4
        cv2.putText(image, word.upper(), (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, thick)



# Gloss2Text — ASL gloss → English sentence via Gemma 4 (Ollama)
# =============================================================================

_OLLAMA_URL  = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "gemma4:e2b"
_FLUSH_TIMEOUT = 3.0   # seconds of silence before flushing glosses

_SYSTEM_PROMPT = """\
You are an ASL (American Sign Language) interpreter. You receive a sequence of ASL glosses \
and must convert them into natural English sentences.

ASL grammar is different from English:
- ASL uses topic-comment structure (topic first)
- Time indicators come first ("YESTERDAY I GO STORE" = "I went to the store yesterday")
- No articles (a, the) or "to be" verbs in ASL
- Questions use facial grammar, the word order may differ

Rules:
- Output ONLY the English sentence, nothing else
- Keep it concise and natural
- If glosses are unclear, make the most reasonable interpretation
- Style: {style}

Style descriptions:
- professional: formal, clear, suitable for medical/business contexts
- friendly: casual, warm, conversational
- simple: short sentences, easy vocabulary, suitable for children\
"""

_STYLES = {"1": "professional", "2": "friendly", "3": "simple"}


def _call_ollama(glosses: list, style: str) -> str:
    gloss_str = " ".join(glosses)
    system    = _SYSTEM_PROMPT.format(style=style)
    payload   = json.dumps({
        "model":  _OLLAMA_MODEL,
        "prompt": f"ASL glosses: {gloss_str}",
        "system": system,
        "stream": False,
    }).encode()
    try:
        req = urllib.request.Request(
            _OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError:
        # Ollama not running — sign recognition still works, translation unavailable
        return ""
    except Exception as e:
        print(f"[Ollama] Error: {e}")
        return ""


class Gloss2Text:
    def __init__(self):
        self.style        = "friendly"
        self._glosses     = []          # glosses being accumulated
        self._last_time   = None        # timestamp of the last gloss
        self._sentence    = ""          # last translated sentence
        self._pending     = None        # in-flight Ollama future
        self._executor    = ThreadPoolExecutor(max_workers=1)

    def push(self, gloss: str):
        """Add a gloss and reset the silence timer."""
        if not self._glosses or self._glosses[-1] != gloss:
            self._glosses.append(gloss)
        self._last_time = time.time()

    def tick(self) -> tuple:
        """Called every frame. Returns (current_glosses, translated_sentence)."""
        # Collect result if the background thread is done
        if self._pending is not None and self._pending.done():
            result = self._pending.result()
            if result:
                self._sentence = result
                print(f"  Gemma → {result}")
            self._pending = None

        # Flush if silence timeout expired and no translation in progress
        if (self._glosses
                and self._last_time is not None
                and time.time() - self._last_time >= _FLUSH_TIMEOUT
                and self._pending is None):
            glosses_snapshot = list(self._glosses)
            print(f"\nGlosses: {' | '.join(glosses_snapshot)}  [{self.style}]")
            self._pending   = self._executor.submit(
                _call_ollama, glosses_snapshot, self.style
            )
            self._glosses   = []
            self._last_time = None

        return list(self._glosses), self._sentence

    @property
    def is_loading(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def set_style(self, key: str):
        self.style = _STYLES.get(key, "friendly")

    def clear(self):
        self._glosses   = []
        self._last_time = None
        self._sentence  = ""
        # does not cancel in-flight future (daemon thread)

    def shutdown(self):
        self._executor.shutdown(wait=False)


_PIPER_MODEL = str(MODELS_DIR / "en_US-lessac-medium.onnx")
_PIPER_RATE  = 22050   # output sample rate for lessac-medium

_PIPER_URLS = {
    "en_US-lessac-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
}


def _ensure_piper_model():
    import urllib.request
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in _PIPER_URLS.items():
        dest = MODELS_DIR / fname
        if not dest.exists():
            print(f"  Downloading Piper voice {fname} (~80 MB)…")
            urllib.request.urlretrieve(url, dest)
            print(f"  ✓ {fname}")


class TTSEngine:
    """Piper TTS (offline neural voice) running in a dedicated thread."""
    def __init__(self):
        self._queue  = queue.Queue()
        self._muted  = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        _ensure_piper_model()
        try:
            import sounddevice as sd
        except ImportError:
            print("[TTS] sounddevice not found — pip install sounddevice")
            return

        while True:
            phrase = self._queue.get()
            if phrase is None:
                break
            if not self._muted and phrase.strip():
                try:
                    proc = subprocess.Popen(
                        ["piper", "--model", _PIPER_MODEL, "--output-raw"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    raw, _ = proc.communicate(input=phrase.encode())
                    if raw:
                        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        sd.play(audio, samplerate=_PIPER_RATE)
                        sd.wait()
                except FileNotFoundError:
                    print("[TTS] Piper not found — pip install piper-tts")
                except Exception as e:
                    print(f"[TTS] Error: {e}")

    def speak(self, phrase: str):
        # Drain the queue to avoid queueing stale phrases
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(phrase)

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        state = "muted" if self._muted else "active"
        print(f"[TTS] {state}")
        return self._muted

    @property
    def muted(self) -> bool:
        return self._muted

    def shutdown(self):
        self._queue.put(None)


# MediaPipe Tasks assembler → Holistic 543-landmark format
# =============================================================================

class HolisticAssembler:
    def __init__(self):
        self._hand  = None
        self._pose  = None
        self._face  = None
        self._face_cache   = None
        self._face_counter = FACE_EVERY_N - 1  # trigger on the very first frame

    def __enter__(self):
        self._ensure_models()
        self._hand = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.2,
                min_tracking_confidence=0.2,
            ))
        self._face = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            ))
        return self

    def __exit__(self, *args):
        if self._hand: self._hand.close()
        if self._face: self._face.close()

    def _ensure_models(self):
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urls = {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        }
        for fname, url in urls.items():
            dest = MODELS_DIR / fname
            if not dest.exists():
                print(f"  Downloading {fname}...")
                urllib.request.urlretrieve(url, dest)

    def extract(self, frame_bgr, timestamp_ms):
        h, w = frame_bgr.shape[:2]
        small    = cv2.resize(frame_bgr, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        holistic = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

        # Face: only every FACE_EVERY_N frames (face = 77/118 landmarks, expensive)
        self._face_counter = (self._face_counter + 1) % FACE_EVERY_N
        if self._face_counter == 0:
            self._face_cache = self._face.detect_for_video(mp_image, timestamp_ms)
        face_result = self._face_cache
        if face_result and face_result.face_landmarks:
            for i, lm in enumerate(face_result.face_landmarks[0]):
                if i < 468:
                    holistic[i] = [lm.x, lm.y, lm.z]

        # Hands: every frame
        hand_result = self._hand.detect_for_video(mp_image, timestamp_ms)
        has_hand = False
        if hand_result.hand_landmarks:
            has_hand = True
            for h_idx, handedness_list in enumerate(hand_result.handedness):
                label     = handedness_list[0].category_name
                landmarks = hand_result.hand_landmarks[h_idx]
                if label == "Left":
                    offset = 468  # left_hand
                else:
                    offset = 522  # right_hand
                for i, lm in enumerate(landmarks):
                    holistic[offset + i] = [lm.x, lm.y, lm.z]

        return holistic, has_hand, hand_result, face_result


# Preprocessing (identical to the winning solution)
# =============================================================================

def preprocess_sequence(frames_543):
    x    = frames_543[np.newaxis, ...]
    nose = x[:, :, [17], :]
    mean = np.nanmean(nose, axis=(1, 2), keepdims=True)
    mean = np.where(np.isnan(mean), 0.5, mean)

    x    = x[:, :, POINT_LANDMARKS, :]
    diff = x - mean
    std  = np.sqrt(np.nanmean(diff * diff, axis=(1, 2), keepdims=True))
    std  = np.where(std < 1e-6, 1.0, std)
    x    = (x - mean) / std

    T = x.shape[1]
    x = x[..., :2]

    dx = np.zeros_like(x)
    if T > 1:
        dx[:, :-1] = x[:, 1:] - x[:, :-1]

    dx2 = np.zeros_like(x)
    if T > 2:
        dx2[:, :-2] = x[:, 2:] - x[:, :-2]

    x = np.concatenate([
        x.reshape(1, T, 2 * NUM_NODES),
        dx.reshape(1, T, 2 * NUM_NODES),
        dx2.reshape(1, T, 2 * NUM_NODES),
    ], axis=-1)

    x = np.nan_to_num(x, nan=0.0)
    return x[0]   # (T, 708)


# Inference (runs in a background thread)
# =============================================================================

def _run_inference(model, inp_arr, idx_to_label):
    logits   = model(inp_arr, training=False).numpy()[0]
    probs    = _softmax(logits)
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3     = [(idx_to_label.get(int(i), "?"), float(probs[i])) for i in top3_idx]
    return {"top_idx": top_idx, "top_conf": top_conf, "top3": top3}


# Live loop
# =============================================================================

def run_live(model_dir: Path, threshold: float = 0.2):
    print("📦 Loading model...")

    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    num_classes  = meta["num_classes"]
    max_len      = meta["max_len"]

    import tensorflow as tf

    class ECA(tf.keras.layers.Layer):
        def __init__(self, kernel_size=5, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.kernel_size = kernel_size
            self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False)
        def call(self, inputs, mask=None):
            nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
            nn = tf.expand_dims(nn, -1)
            nn = self.conv(nn)
            nn = tf.squeeze(nn, -1)
            nn = tf.nn.sigmoid(nn)
            return inputs * nn[:, None, :]

    class CausalDWConv1D(tf.keras.layers.Layer):
        def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False, **kwargs):
            super().__init__(**kwargs)
            self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))
            self.dw_conv    = tf.keras.layers.DepthwiseConv1D(kernel_size, strides=1,
                              dilation_rate=dilation_rate, padding="valid", use_bias=use_bias)
            self.supports_masking = True
        def call(self, inputs):
            return self.dw_conv(self.causal_pad(inputs))

    class MultiHeadSelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
            super().__init__(**kwargs)
            self.dim   = dim
            self.scale = dim ** -0.5
            self.num_heads = num_heads
            self.qkv   = tf.keras.layers.Dense(3 * dim, use_bias=False)
            self.drop  = tf.keras.layers.Dropout(dropout)
            self.proj  = tf.keras.layers.Dense(dim, use_bias=False)
            self.supports_masking = True
        def call(self, inputs, mask=None):
            qkv = self.qkv(inputs)
            qkv = tf.keras.layers.Permute((2, 1, 3))(
                tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
            q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
            attn = tf.matmul(q, k, transpose_b=True) * self.scale
            if mask is not None:
                attn = tf.where(mask[:, None, None, :], attn, -1e9)
            attn = tf.nn.softmax(attn, axis=-1)
            attn = self.drop(attn)
            x = attn @ v
            x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
            return self.proj(x)

    def Conv1DBlock(ch, ks, drop_rate=0.0, expand_ratio=2, activation="swish", name=None):
        if name is None:
            name = str(tf.keras.backend.get_uid("cb"))
        def apply(inputs):
            skip = inputs
            x = tf.keras.layers.Dense(tf.keras.backend.int_shape(inputs)[-1] * expand_ratio,
                                      activation=activation, name=name + "_exp")(inputs)
            x = CausalDWConv1D(ks, name=name + "_dw")(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + "_bn")(x)
            x = ECA(name=name + "_eca")(x)
            x = tf.keras.layers.Dense(ch, name=name + "_proj")(x)
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
            if tf.keras.backend.int_shape(inputs)[-1] == ch:
                x = tf.keras.layers.add([x, skip])
            return x
        return apply

    def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation="swish"):
        def apply(inputs):
            x = tf.keras.layers.BatchNormalization(momentum=0.95)(inputs)
            x = MultiHeadSelfAttention(dim, num_heads, attn_dropout)(x)
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
            x = tf.keras.layers.Add()([inputs, x])
            attn_out = x
            x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
            x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
            x = tf.keras.layers.Dense(dim, use_bias=False)(x)
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
            return tf.keras.layers.Add()([attn_out, x])
        return apply

    dim = meta.get("dim", 192)
    inp = tf.keras.Input((max_len, CHANNELS))
    x   = tf.keras.layers.Masking(mask_value=-100.0)(inp)
    x   = tf.keras.layers.Dense(dim, use_bias=False, name="stem")(x)
    x   = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
    ks  = 17
    x   = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x   = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x   = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x   = TransformerBlock(dim, expand=2)(x)
    x   = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x   = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x   = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x   = TransformerBlock(dim, expand=2)(x)
    x   = tf.keras.layers.Dense(dim * 2)(x)
    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Dense(num_classes, name="classifier")(x)
    model = tf.keras.Model(inp, x)
    model.load_weights(str(model_dir / "weights.weights.h5"))
    print(f"   {num_classes} signs, max_len={max_len}, dim={dim}")

    # ─── Buffers ─────────────────────────────────────────────────────────────
    PAD                = -100.0
    frame_buffer       = deque(maxlen=max_len)
    prediction_history = deque(maxlen=8)
    word_history       = deque(maxlen=7)   # history of confirmed words
    last_stable_label  = ""

    # Gloss2Text + TTS
    g2t            = Gloss2Text()
    tts            = TTSEngine()
    last_translated = ""
    top3               = []
    current_label      = ""
    current_conf       = 0.0

    # background thread for TF inference (non-blocking)
    executor    = ThreadPoolExecutor(max_workers=1)
    infer_future = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\n{'─'*45}")
    print(f"  Confidence threshold : {threshold:.0%}")
    print(f"  Keyboard shortcuts:")
    print(f"    1  → style Professional")
    print(f"    2  → style Friendly (default)")
    print(f"    3  → style Simple")
    print(f"    m  → mute / unmute TTS")
    print(f"    c  → clear buffer, glosses and translation")
    print(f"    q  → quit")
    print(f"{'─'*45}\n")

    frame_idx = 0
    fps_list  = deque(maxlen=30)

    with HolisticAssembler() as assembler:
        while True:
            t0  = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx * 33  # steady timestamps = more stable tracking

            holistic, has_hand, hand_res, face_res = assembler.extract(frame, timestamp_ms)
            frame_buffer.append(holistic)

            # ── Collect previous inference result ─────────────────────────
            if infer_future is not None and infer_future.done():
                result       = infer_future.result()
                infer_future = None
                top3         = result["top3"]
                top_idx      = result["top_idx"]
                top_conf     = result["top_conf"]

                if top_conf >= threshold and has_hand:
                    prediction_history.append(top_idx)
                    votes    = Counter(prediction_history)
                    best_idx, _ = votes.most_common(1)[0]
                    current_label = idx_to_label.get(best_idx, "?")
                    current_conf  = top_conf
                else:
                    if not has_hand:
                        prediction_history.clear()
                    current_label = ""
                    current_conf  = 0.0

            # ── Submit new inference (non-blocking) ───────────────────────
            if infer_future is None and frame_idx % 3 == 0 and len(frame_buffer) >= 10:
                seq  = np.array(list(frame_buffer), dtype=np.float32)
                pts  = seq[:, POINT_LANDMARKS, :]
                valid = ~np.all(np.isnan(pts), axis=(1, 2))
                seq  = seq[valid] if valid.any() else seq

                if seq.shape[0] >= 3:
                    features = preprocess_sequence(seq)
                    T = features.shape[0]
                    if T > max_len:
                        features = features[-max_len:]
                    elif T < max_len:
                        pad      = np.full((max_len - T, CHANNELS), PAD, dtype=np.float32)
                        features = np.concatenate([features, pad])
                    infer_future = executor.submit(
                        _run_inference, model, features[np.newaxis, ...], idx_to_label
                    )

            # ── Word history + gloss accumulation ────────────────────────
            if current_label and current_label != last_stable_label:
                if last_stable_label:
                    word_history.append(last_stable_label)
                    g2t.push(last_stable_label)
                last_stable_label = current_label
            elif not current_label and last_stable_label:
                word_history.append(last_stable_label)
                g2t.push(last_stable_label)
                last_stable_label = ""

            pending_glosses, translated = g2t.tick()

            # TTS: speak as soon as a new sentence arrives
            if translated and translated != last_translated:
                last_translated = translated
                tts.speak(translated)

            # ── Drawing ──────────────────────────────────────────────────
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            draw_hand_landmarks(annotated, hand_res)
            draw_face_landmarks(annotated, face_res)

            # FPS
            dt = time.time() - t0
            fps_list.append(1.0 / dt if dt > 0 else 0)
            avg_fps = float(np.mean(fps_list))

            cv2.putText(annotated, f"FPS: {avg_fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "HAND" if has_hand else "---"
            cv2.putText(annotated, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if has_hand else (0, 0, 255), 2)

            if infer_future is not None:
                cv2.putText(annotated, "...", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            # Current prediction (positioned to leave room for glosses/translation)
            if current_label and current_conf >= threshold:
                c = ((0, 255, 0)   if current_conf > 0.5
                     else (0, 255, 255) if current_conf > 0.3
                     else (0, 165, 255))
                cv2.putText(annotated, current_label.upper(), (10, h - 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, 4)
                cv2.putText(annotated, f"{current_conf:.0%}", (10, h - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

            # Accumulated glosses (waiting to be sent to Ollama)
            if pending_glosses:
                gloss_str = "  ".join(pending_glosses)
                cv2.putText(annotated, gloss_str, (10, h - 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Sentence translated by Gemma 4
            if translated:
                # semi-transparent bar across full width
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
                # truncate if too long for the frame width
                max_chars = w // 11
                display_txt = translated if len(translated) <= max_chars else translated[:max_chars - 1] + "…"
                cv2.putText(annotated, display_txt, (8, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

            # Style indicator + mute status (bottom-right corner)
            cv2.putText(annotated, f"[{g2t.style[0].upper()}]", (w - 45, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            mute_label = "MUT" if tts.muted else "SPK"
            mute_color = (60, 60, 60) if tts.muted else (0, 220, 120)
            cv2.putText(annotated, mute_label, (w - 90, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, mute_color, 1)

            # Ollama spinner while waiting for translation
            if g2t.is_loading:
                cx, cy  = w - 20, h - 60
                r       = 10
                angle   = (frame_idx * 9) % 360
                cv2.ellipse(annotated, (cx, cy), (r, r), angle, 0, 270,
                            (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, "AI", (cx - 7, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

            # Top 3 predictions
            for i, (label, conf) in enumerate(top3):
                cv2.putText(annotated, f"{label}: {conf:.0%}", (w - 250, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Word history panel
            _draw_word_history(annotated, word_history)

            cv2.imshow("Sign2Speak_Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                frame_buffer.clear()
                prediction_history.clear()
                word_history.clear()
                g2t.clear()
                current_label   = ""
                last_stable_label = ""
                last_translated = ""
            elif key in (ord("1"), ord("2"), ord("3")):
                g2t.set_style(chr(key))
            elif key == ord("m"):
                tts.toggle_mute()

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    g2t.shutdown()
    tts.shutdown()


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Live webcam inference")
    parser.add_argument("--model",     type=str,   default="../model_winner")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    run_live(Path(args.model), args.threshold)


if __name__ == "__main__":
    main()
