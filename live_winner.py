"""
Sign2Speak — Live Test pour le modèle gagnant Kaggle
=====================================================
Adapte le preprocessing du gagnant (118 landmarks, normalisation nez,
vélocité + accélération = 708 features) pour MediaPipe Tasks en live.

Usage:
    python live_winner.py --model ./model_winner
    python live_winner.py --model ./model_winner --threshold 0.3
"""

import argparse
import json
import time
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

# =============================================================================
# Configuration du modèle gagnant
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

# FPS optimisations
FACE_EVERY_N = 2    # face detector toutes les N frames (réduit à 2 : face = 77/118 landmarks)
DETECT_SCALE = 0.8  # 0.8 = bon compromis vitesse/robustesse aux mouvements rapides

# =============================================================================
# Drawing helpers — pur OpenCV (pas de dépendance mediapipe.framework)
# =============================================================================

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
# couleur par doigt (pouce→auriculaire)
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
    """Affiche uniquement les 77 points visage utilisés par le modèle."""
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

    # fond semi-transparent
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.rectangle(image, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

    cv2.putText(image, "Historique", (x0 + 6, y0 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    for i, word in enumerate(reversed(words)):
        alpha = max(0.25, 1.0 - i * 0.13)
        v     = int(255 * alpha)
        color = (v, v, v)
        thick = 2 if i == 0 else 1
        y = y0 + pad + 14 + (i + 1) * line_h - 4
        cv2.putText(image, word.upper(), (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, thick)


# =============================================================================
# Assemblage MediaPipe Tasks → format Holistic 543
# =============================================================================

class HolisticAssembler:
    def __init__(self):
        self._hand  = None
        self._pose  = None
        self._face  = None
        self._face_cache   = None
        self._face_counter = FACE_EVERY_N - 1  # appel dès la 1ère frame

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

        # Face: seulement toutes les FACE_EVERY_N frames
        self._face_counter = (self._face_counter + 1) % FACE_EVERY_N
        if self._face_counter == 0:
            self._face_cache = self._face.detect_for_video(mp_image, timestamp_ms)
        face_result = self._face_cache
        if face_result and face_result.face_landmarks:
            for i, lm in enumerate(face_result.face_landmarks[0]):
                if i < 468:
                    holistic[i] = [lm.x, lm.y, lm.z]

        # Mains (chaque frame)
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


# =============================================================================
# Preprocessing (identique au gagnant)
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


# =============================================================================
# Inférence (appelée dans un thread séparé)
# =============================================================================

def _run_inference(model, inp_arr, idx_to_label):
    logits   = model(inp_arr, training=False).numpy()[0]
    probs    = _softmax(logits)
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3     = [(idx_to_label.get(int(i), "?"), float(probs[i])) for i in top3_idx]
    return {"top_idx": top_idx, "top_conf": top_conf, "top3": top3}


# =============================================================================
# Live
# =============================================================================

def run_live(model_dir: Path, threshold: float = 0.2):
    print("📦 Chargement du modèle...")

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
    print(f"   {num_classes} signes, max_len={max_len}, dim={dim}")

    # ─── Buffers ────────────────────────────────────────────────────────────
    PAD                = -100.0
    frame_buffer       = deque(maxlen=max_len)
    prediction_history = deque(maxlen=8)
    word_history       = deque(maxlen=7)   # historique des mots confirmés
    last_stable_label  = ""
    top3               = []
    current_label      = ""
    current_conf       = 0.0

    # thread pour l'inférence TF (non-bloquant)
    executor    = ThreadPoolExecutor(max_workers=1)
    infer_future = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam introuvable")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\n📸 'q' quitter | 'c' clear | Seuil: {threshold:.0%}")

    frame_idx = 0
    fps_list  = deque(maxlen=30)

    with HolisticAssembler() as assembler:
        while True:
            t0  = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx * 33  # timestamps réguliers = tracking plus stable

            holistic, has_hand, hand_res, face_res = assembler.extract(frame, timestamp_ms)
            frame_buffer.append(holistic)

            # ── Résultat inférence précédente ─────────────────────────────
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

            # ── Soumettre nouvelle inférence (non-bloquant) ───────────────
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

            # ── Historique des mots ───────────────────────────────────────
            if current_label and current_label != last_stable_label:
                if last_stable_label:
                    word_history.append(last_stable_label)
                last_stable_label = current_label
            elif not current_label and last_stable_label:
                word_history.append(last_stable_label)
                last_stable_label = ""

            # ── Dessin ───────────────────────────────────────────────────
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

            status = "MAIN" if has_hand else "---"
            cv2.putText(annotated, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if has_hand else (0, 0, 255), 2)

            if infer_future is not None:
                cv2.putText(annotated, "...", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            # Prédiction courante
            if current_label and current_conf >= threshold:
                c = ((0, 255, 0)   if current_conf > 0.5
                     else (0, 255, 255) if current_conf > 0.3
                     else (0, 165, 255))
                cv2.putText(annotated, current_label.upper(), (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, 4)
                cv2.putText(annotated, f"{current_conf:.0%}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

            # Top 3
            for i, (label, conf) in enumerate(top3):
                cv2.putText(annotated, f"{label}: {conf:.0%}", (w - 250, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Historique
            _draw_word_history(annotated, word_history)

            cv2.imshow("Sign2Speak_Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                frame_buffer.clear()
                prediction_history.clear()
                word_history.clear()
                current_label     = ""
                last_stable_label = ""

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Live test modèle gagnant")
    parser.add_argument("--model",     type=str,   default="./model_winner")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    run_live(Path(args.model), args.threshold)


if __name__ == "__main__":
    main()
