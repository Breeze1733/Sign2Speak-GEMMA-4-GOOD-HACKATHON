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
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np

# On utilise notre extracteur MediaPipe Tasks existant pour les mains/pose/face
# MAIS on doit aussi extraire les landmarks dans le format Holistic (543 landmarks)
# pour matcher le preprocessing du gagnant.
# Solution: utiliser MediaPipe Tasks pour les 3 modèles séparément,
# puis réassembler dans le format 543 landmarks.

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
)
from train_winner import ECA, CausalDWConv1D, MultiHeadSelfAttention
from train_winner import ECA

# =============================================================================
# Configuration du modèle gagnant
# =============================================================================

ROWS_PER_FRAME = 543

# Landmarks utilisés par le gagnant (indices dans le format Holistic 543)
NOSE = [1, 2, 98, 327]
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)  # 118
CHANNELS = 6 * NUM_NODES           # 708

MODELS_DIR = Path(__file__).parent / "models"

# =============================================================================
# Assemblage MediaPipe Tasks → format Holistic 543
# =============================================================================

class HolisticAssembler:
    """
    Utilise les 3 modèles MediaPipe Tasks séparément
    et réassemble le résultat en format Holistic (543, 3).
    """
    def __init__(self):
        self._hand = None
        self._pose = None
        self._face = None

    def __enter__(self):
        self._ensure_models()

        self._hand = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            ))
        self._pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.3,
                min_tracking_confidence=0.3,
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
        if self._pose: self._pose.close()
        if self._face: self._face.close()

    def _ensure_models(self):
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urls = {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        }
        for fname, url in urls.items():
            dest = MODELS_DIR / fname
            if not dest.exists():
                print(f"  ⬇️  {fname}...")
                urllib.request.urlretrieve(url, dest)

    def extract(self, frame_bgr, timestamp_ms):
        """
        Extrait et réassemble en format Holistic (543, 3).
        Retourne aussi les résultats bruts pour le dessin.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Array (543, 3) rempli de NaN
        holistic = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

        # Face (0-467)
        face_result = self._face.detect_for_video(mp_image, timestamp_ms)
        if face_result.face_landmarks:
            for i, lm in enumerate(face_result.face_landmarks[0]):
                if i < 468:
                    holistic[i] = [lm.x, lm.y, lm.z]

        # Left hand (468-488) et Right hand (522-542)
        hand_result = self._hand.detect_for_video(mp_image, timestamp_ms)
        has_hand = False
        if hand_result.hand_landmarks:
            has_hand = True
            for h_idx, handedness_list in enumerate(hand_result.handedness):
                label = handedness_list[0].category_name
                landmarks = hand_result.hand_landmarks[h_idx]
                # Caméra POV: "Right" = main gauche du signeur face caméra
                if label == "Right":
                    offset = 468  # left_hand dans Holistic
                else:
                    offset = 522  # right_hand
                for i, lm in enumerate(landmarks):
                    holistic[offset + i] = [lm.x, lm.y, lm.z]

        # Pose (489-521) — seulement 33 landmarks dans Holistic, mais Tasks en a 33 aussi
        pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_landmarks:
            for i, lm in enumerate(pose_result.pose_landmarks[0]):
                if i < 33:
                    holistic[489 + i] = [lm.x, lm.y, lm.z]

        return holistic, has_hand, hand_result, pose_result, face_result


# =============================================================================
# Preprocessing (identique au gagnant)
# =============================================================================

def preprocess_sequence(frames_543):
    """
    Applique le preprocessing du gagnant sur une séquence (n_frames, 543, 3).
    Retourne (n_frames, 708).
    """
    x = frames_543[np.newaxis, ...]  # (1, T, 543, 3)

    # Centrer sur le nez (landmark 17)
    nose = x[:, :, [17], :]
    mean = np.nanmean(nose, axis=(1, 2), keepdims=True)
    mean = np.where(np.isnan(mean), 0.5, mean)

    # Sélectionner 118 landmarks
    x = x[:, :, POINT_LANDMARKS, :]

    # Normaliser
    diff = x - mean
    std = np.sqrt(np.nanmean(diff * diff, axis=(1, 2), keepdims=True))
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std

    # Garder x, y seulement
    T = x.shape[1]
    x = x[..., :2]

    # Vélocité
    if T > 1:
        dx = np.zeros_like(x)
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
    else:
        dx = np.zeros_like(x)

    # Accélération
    if T > 2:
        dx2 = np.zeros_like(x)
        dx2[:, :-2] = x[:, 2:] - x[:, :-2]
    else:
        dx2 = np.zeros_like(x)

    # Concat et reshape → (T, 708)
    x = np.concatenate([
        x.reshape(1, T, 2 * NUM_NODES),
        dx.reshape(1, T, 2 * NUM_NODES),
        dx2.reshape(1, T, 2 * NUM_NODES),
    ], axis=-1)

    x = np.nan_to_num(x, nan=0.0)
    return x[0]  # (T, 708)


# =============================================================================
# Live
# =============================================================================

def run_live(model_dir: Path, threshold: float = 0.2):
    print("📦 Chargement du modèle...")

    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    idx_to_label = {int(k): v for k, v in meta['idx_to_label'].items()}
    num_classes = meta['num_classes']
    max_len = meta['max_len']

    # Charger le modèle TF
    model = tf.keras.models.load_model(
        str(model_dir / "best_model.keras"),
        compile=False,
        custom_objects={
            'CausalDWConv1D': CausalDWConv1D,
            'ECA': ECA,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
        }
    )   
    print(f"   {num_classes} signes, max_len={max_len}")

    # Buffer
    PAD = -100.0
    frame_buffer = deque(maxlen=max_len)
    prediction_history = deque(maxlen=8)
    top3 = []
    current_label = ""
    current_conf = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam introuvable")
        return

    print(f"\n📸 'q' quitter | 'c' clear | Seuil: {threshold:.0%}")

    frame_idx = 0
    fps_list = []

    with HolisticAssembler() as assembler:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx * 33

            # Extraire (543, 3)
            holistic, has_hand, hand_res, pose_res, face_res = assembler.extract(frame, timestamp_ms)
            frame_buffer.append(holistic)

            # Prédire toutes les 3 frames
            if frame_idx % 3 == 0 and len(frame_buffer) >= 10:
                # Construire la séquence
                seq = np.array(list(frame_buffer), dtype=np.float32)  # (T, 543, 3)

                # Filtrer les frames sans landmarks utiles
                pts = seq[:, POINT_LANDMARKS, :]
                valid = ~np.all(np.isnan(pts), axis=(1, 2))
                seq = seq[valid] if valid.any() else seq

                if seq.shape[0] >= 3:
                    # Preprocessing du gagnant
                    features = preprocess_sequence(seq)  # (T, 708)

                    # Pad/truncate
                    T = features.shape[0]
                    if T > max_len:
                        features = features[-max_len:]
                    elif T < max_len:
                        pad = np.full((max_len - T, CHANNELS), PAD, dtype=np.float32)
                        features = np.concatenate([features, pad])

                    # Inférence
                    inp = features[np.newaxis, ...]  # (1, max_len, 708)
                    logits = model(inp, training=False).numpy()[0]
                    probs = _softmax(logits)

                    top_idx = np.argmax(probs)
                    top_conf = probs[top_idx]

                    top3_idx = np.argsort(probs)[-3:][::-1]
                    top3 = [(idx_to_label.get(i, "?"), probs[i]) for i in top3_idx]

                    if top_conf >= threshold and has_hand:
                        prediction_history.append(top_idx)
                        votes = Counter(prediction_history)
                        best_idx, _ = votes.most_common(1)[0]
                        current_label = idx_to_label.get(best_idx, "?")
                        current_conf = top_conf
                    else:
                        if not has_hand:
                            prediction_history.clear()
                        current_label = ""
                        current_conf = 0.0

            # --- Affichage ---
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # Dessiner les mains
            if hand_res and hand_res.hand_landmarks:
                from mediapipe.framework.formats import landmark_pb2
                from mediapipe import solutions as mp_sol
                for hand_lms in hand_res.hand_landmarks:
                    proto = landmark_pb2.NormalizedLandmarkList()
                    for lm in hand_lms:
                        proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
                    mp_sol.drawing_utils.draw_landmarks(
                        annotated, proto, mp_sol.hands.HAND_CONNECTIONS,
                        mp_sol.drawing_styles.get_default_hand_landmarks_style(),
                        mp_sol.drawing_styles.get_default_hand_connections_style())

            # FPS
            dt = time.time() - t0
            fps_list.append(1.0 / dt if dt > 0 else 0)
            if len(fps_list) > 30: fps_list.pop(0)

            cv2.putText(annotated, f"FPS: {np.mean(fps_list):.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "MAIN" if has_hand else "---"
            cv2.putText(annotated, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if has_hand else (0, 0, 255), 2)

            # Prédiction
            if current_label and current_conf >= threshold:
                c = (0, 255, 0) if current_conf > 0.5 else (0, 255, 255) if current_conf > 0.3 else (0, 165, 255)
                cv2.putText(annotated, current_label.upper(), (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, 4)
                cv2.putText(annotated, f"{current_conf:.0%}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

            # Top 3
            for i, (label, conf) in enumerate(top3):
                cv2.putText(annotated, f"{label}: {conf:.0%}", (w - 250, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Sign2Speak — Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'):
                frame_buffer.clear()
                prediction_history.clear()
                current_label = ""

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Live test modèle gagnant")
    parser.add_argument('--model', type=str, default='./model_winner')
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    run_live(Path(args.model), args.threshold)


if __name__ == "__main__":
    main()