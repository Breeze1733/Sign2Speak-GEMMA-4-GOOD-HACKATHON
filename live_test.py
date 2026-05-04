"""
Sign2Speak — Test Live : Reconnaissance ASL en temps réel
==========================================================
Utilise la webcam + le classifieur entraîné pour reconnaître
les signes ASL en direct.

Usage:
    python live_test.py --model ./model
    python live_test.py --model ./model --threshold 0.3
"""

import argparse
import json
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keypoint_extractor import KeypointExtractor, ExtractionConfig


# =============================================================================
# Configuration
# =============================================================================

BUFFER_SIZE = 60          # frames dans le buffer glissant (= MAX_SEQUENCE_LENGTH)
MIN_HAND_FRAMES = 10      # nb minimum de frames avec main détectée pour prédire
PREDICTION_INTERVAL = 5   # prédire toutes les N frames (pas à chaque frame)
CONFIDENCE_THRESHOLD = 0.15  # seuil de confiance minimum pour afficher


# =============================================================================
# Live Recognition
# =============================================================================

def run_live(
    model_dir: Path,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    """Lance la reconnaissance ASL en temps réel sur webcam."""

    # Charger le modèle et les métadonnées
    print("📦 Chargement du modèle...")
    
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        print(f"❌ {meta_path} introuvable")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    idx_to_label = {int(k): v for k, v in meta['idx_to_label'].items()}
    max_seq_len = meta['max_seq_len']
    n_features = meta['n_features']
    n_classes = meta['n_classes']

    print(f"   {n_classes} signes, séquence={max_seq_len} frames, {n_features} features")

    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        model_path = model_dir / "final_model.keras"
    model = keras.models.load_model(str(model_path))
    print(f"   Modèle chargé: {model_path.name}")

    # Extracteur de keypoints
    config = ExtractionConfig(
        use_hands=True,
        use_pose=True,
        use_face=True,
        face_subset_only=True,
        normalize=True,
    )

    # Buffer glissant
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    hand_detected_count = 0

    # État d'affichage
    current_prediction = ""
    current_confidence = 0.0
    top3_predictions = []
    prediction_history = deque(maxlen=10)  # lissage temporel

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return

    print("\n📸 Webcam ouverte!")
    print("   Fais un signe ASL devant la caméra")
    print("   'q' pour quitter | 'c' pour clear le buffer")
    print(f"   Seuil de confiance: {threshold:.0%}")

    frame_idx = 0
    fps_list = []

    with KeypointExtractor(config) as extractor:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx * 33

            # Extraire les keypoints
            data = extractor.extract_frame(frame, timestamp_ms)
            keypoints = extractor.extract_and_flatten(frame, timestamp_ms, data=data)

            # Tracker si une main est détectée
            has_hand = data['left_hand'] is not None or data['right_hand'] is not None
            if has_hand:
                hand_detected_count += 1
            
            # Ajouter au buffer
            frame_buffer.append(keypoints)

            # Prédire périodiquement
            if frame_idx % PREDICTION_INTERVAL == 0 and len(frame_buffer) >= MIN_HAND_FRAMES:
                # Construire la séquence
                sequence = np.array(list(frame_buffer), dtype=np.float32)
                n_frames = sequence.shape[0]

                # Padding si nécessaire
                if n_frames < max_seq_len:
                    pad = np.zeros((max_seq_len - n_frames, n_features), dtype=np.float32)
                    sequence = np.concatenate([sequence, pad], axis=0)
                elif n_frames > max_seq_len:
                    sequence = sequence[-max_seq_len:]  # garder les plus récentes

                # Prédiction
                input_tensor = np.expand_dims(sequence, axis=0)  # (1, seq_len, features)
                predictions = model.predict(input_tensor, verbose=0)[0]

                top_idx = np.argmax(predictions)
                top_conf = predictions[top_idx]

                # Top 3
                top3_idx = np.argsort(predictions)[-3:][::-1]
                top3_predictions = [
                    (idx_to_label.get(idx, "?"), predictions[idx])
                    for idx in top3_idx
                ]

                # Lissage temporel: ajouter à l'historique
                if top_conf >= threshold:
                    prediction_history.append(top_idx)
                    
                    # Vote majoritaire sur les dernières prédictions
                    if prediction_history:
                        from collections import Counter
                        votes = Counter(prediction_history)
                        best_label_idx, best_count = votes.most_common(1)[0]
                        
                        current_prediction = idx_to_label.get(best_label_idx, "?")
                        current_confidence = top_conf
                        stability = best_count / len(prediction_history)
                    else:
                        current_prediction = ""
                        current_confidence = 0.0
                else:
                    current_prediction = ""
                    current_confidence = 0.0

            # --- Affichage ---
            annotated = extractor.draw_landmarks(frame, data)

            # FPS
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)

            # Infos en haut
            cv2.putText(annotated, f"FPS: {avg_fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            hand_status = "MAIN DETECTEE" if has_hand else "Pas de main"
            hand_color = (0, 255, 0) if has_hand else (0, 0, 255)
            cv2.putText(annotated, hand_status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

            cv2.putText(annotated, f"Buffer: {len(frame_buffer)}/{BUFFER_SIZE}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Prédiction principale — gros texte en bas
            if current_prediction and current_confidence >= threshold:
                # Nettoyer le label (enlever les suffixes numériques type HELLO1 → HELLO)
                display_label = current_prediction.rstrip('0123456789')
                
                # Couleur selon la confiance
                if current_confidence > 0.5:
                    color = (0, 255, 0)     # vert = confiant
                elif current_confidence > 0.3:
                    color = (0, 255, 255)   # jaune = moyen
                else:
                    color = (0, 165, 255)   # orange = faible

                h = annotated.shape[0]
                cv2.putText(annotated, display_label, (10, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
                cv2.putText(annotated, f"{current_confidence:.0%}", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Top 3 predictions à droite
            if top3_predictions:
                w = annotated.shape[1]
                for i, (label, conf) in enumerate(top3_predictions):
                    label_clean = label.rstrip('0123456789')
                    text = f"{label_clean}: {conf:.0%}"
                    cv2.putText(annotated, text, (w - 250, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Sign2Speak — Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                frame_buffer.clear()
                prediction_history.clear()
                current_prediction = ""
                hand_detected_count = 0
                print("🔄 Buffer vidé")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Terminé")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Test live ASL")
    parser.add_argument('--model', type=str, default='./model',
                        help='Dossier du modèle entraîné')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                        help=f'Seuil de confiance (défaut: {CONFIDENCE_THRESHOLD})')

    args = parser.parse_args()
    run_live(Path(args.model), threshold=args.threshold)


if __name__ == "__main__":
    main()