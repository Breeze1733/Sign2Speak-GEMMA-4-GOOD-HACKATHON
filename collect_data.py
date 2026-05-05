"""
Sign2Speak — Collecte de données ASL via webcam
=================================================
Enregistre des séquences de keypoints directement depuis ta webcam.
C'est la méthode utilisée par tous les projets à 95%+ de précision.

Principe:
    1. Le script affiche le signe à faire
    2. Tu fais le signe devant la caméra
    3. Il enregistre 30 frames de keypoints
    4. Répète N fois par signe
    5. Passe au signe suivant

Usage:
    # Collecte interactive (mode guidé)
    python collect_data.py --output ./my_data --sequences 30 --frames 30

    # Collecte pour des signes spécifiques
    python collect_data.py --output ./my_data --signs HELLO THANK-YOU HELP DOCTOR

    # Ajouter des données à une collecte existante
    python collect_data.py --output ./my_data --signs WATER FOOD --append

    # Voir les stats de la collecte actuelle
    python collect_data.py --output ./my_data --stats
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from keypoint_extractor import KeypointExtractor, ExtractionConfig


# =============================================================================
# Signes recommandés pour le MVP (20-30 signes du quotidien)
# =============================================================================

DEFAULT_SIGNS = [
    # Salutations
    "HELLO", "THANK-YOU", "PLEASE", "SORRY", "YES", "NO",
    # Urgence / santé
    "HELP", "DOCTOR", "PAIN", "WATER",
    # Questions
    "WHAT", "WHERE", "WHO", "HOW",
    # Actions
    "WANT", "NEED", "EAT", "DRINK", "GO", "STOP",
    # Social
    "GOOD", "BAD", "NAME", "UNDERSTAND",
    # Identité
    "MY", "YOUR",
    #Lettres
    "L","U","D","O",
]

FRAMES_PER_SEQUENCE = 30
SEQUENCES_PER_SIGN = 30


# =============================================================================
# Collecte
# =============================================================================

def collect_data(
    output_dir: Path,
    signs: list[str],
    sequences_per_sign: int = SEQUENCES_PER_SIGN,
    frames_per_sequence: int = FRAMES_PER_SEQUENCE,
    append: bool = False,
):
    """Collecte interactive de données ASL via webcam."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Config MediaPipe — même config qu'on utilisera en live
    config = ExtractionConfig(
        use_hands=True,
        use_pose=True,
        use_face=True,
        face_subset_only=True,
        normalize=True,
    )

    # Vérifier les données existantes
    existing = {}
    if append:
        for sign_dir in output_dir.iterdir():
            if sign_dir.is_dir() and sign_dir.name != "__pycache__":
                count = len(list(sign_dir.glob("*.npy")))
                if count > 0:
                    existing[sign_dir.name] = count

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return

    # Prendre la résolution
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📸 Webcam ouverte ({w}x{h})")
    print(f"🔧 {config.features_per_frame()} features/frame")
    print(f"📋 {len(signs)} signes à collecter, {sequences_per_sign} séquences × {frames_per_sequence} frames chacune")
    print(f"\n   Contrôles:")
    print(f"   ESPACE = enregistrer une séquence")
    print(f"   S      = passer au signe suivant")
    print(f"   R      = refaire la dernière séquence")
    print(f"   Q      = quitter et sauvegarder")
    print(f"\n   Appuie sur une touche pour commencer...")

    cv2.namedWindow("Sign2Speak — Collecte", cv2.WINDOW_NORMAL)

    # Attendre que l'utilisateur soit prêt
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Appuie sur une touche pour commencer", (30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Sign2Speak — Collecte", frame)
        if cv2.waitKey(1) & 0xFF != 255:
            break

    frame_idx = 0  # compteur global pour les timestamps MediaPipe

    with KeypointExtractor(config) as extractor:
        for sign_idx, sign in enumerate(signs):
            sign_dir = output_dir / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            # Compter les séquences existantes
            existing_count = len(list(sign_dir.glob("*.npy")))
            if append and existing_count >= sequences_per_sign:
                print(f"\n⏭️  {sign} — déjà {existing_count} séquences, on passe")
                continue

            start_seq = existing_count if append else 0
            seq_num = start_seq

            print(f"\n{'='*50}")
            print(f"  Signe {sign_idx + 1}/{len(signs)} : {sign}")
            print(f"  Séquences : {seq_num}/{sequences_per_sign}")
            print(f"{'='*50}")

            skip_sign = False

            while seq_num < sequences_per_sign:
                if skip_sign:
                    break

                # --- Phase d'attente: montrer le signe à faire ---
                recording = False
                show_ready = True

                while show_ready:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp_ms = frame_idx * 33
                    data = extractor.extract_frame(frame, timestamp_ms)
                    annotated = extractor.draw_landmarks(frame, data)
                    frame_idx += 1

                    has_hand = data['left_hand'] is not None or data['right_hand'] is not None

                    # UI
                    # Bande du haut — info signe
                    cv2.rectangle(annotated, (0, 0), (w, 80), (40, 40, 40), -1)
                    cv2.putText(annotated, f"Signe: {sign}", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(annotated, f"Sequence {seq_num + 1}/{sequences_per_sign}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                    # Progrès global
                    progress_text = f"[{sign_idx + 1}/{len(signs)}] Total: {sign_idx * sequences_per_sign + seq_num}/{len(signs) * sequences_per_sign}"
                    cv2.putText(annotated, progress_text, (w - 400, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                    # Bande du bas — instructions
                    cv2.rectangle(annotated, (0, h - 50), (w, h), (40, 40, 40), -1)
                    if has_hand:
                        cv2.putText(annotated, "ESPACE = enregistrer | S = passer | Q = quitter",
                                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated, "Montre tes mains... | S = passer | Q = quitter",
                                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    cv2.imshow("Sign2Speak — Collecte", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        recording = True
                        show_ready = False
                    elif key == ord('s'):
                        skip_sign = True
                        show_ready = False
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        _save_metadata(output_dir, signs, config, sequences_per_sign, frames_per_sequence)
                        print(f"\n💾 Collecte sauvegardée dans {output_dir}")
                        return

                if skip_sign:
                    break

                # --- Phase d'enregistrement: capturer N frames ---
                if recording:
                    sequence = []
                    hands_detected_count = 0

                    for frame_count in range(frames_per_sequence):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        timestamp_ms = frame_idx * 33
                        data = extractor.extract_frame(frame, timestamp_ms)
                        keypoints = extractor.extract_and_flatten(frame, timestamp_ms, data=data)
                        annotated = extractor.draw_landmarks(frame, data)
                        frame_idx += 1

                        has_hand = data['left_hand'] is not None or data['right_hand'] is not None
                        if has_hand:
                            hands_detected_count += 1

                        sequence.append(keypoints)

                        # UI pendant l'enregistrement
                        cv2.rectangle(annotated, (0, 0), (w, 80), (0, 0, 150), -1)
                        cv2.putText(annotated, f"REC: {sign}", (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                        # Barre de progression
                        progress = (frame_count + 1) / frames_per_sequence
                        bar_w = int(progress * (w - 20))
                        cv2.rectangle(annotated, (10, 55), (10 + bar_w, 70), (0, 0, 255), -1)
                        cv2.rectangle(annotated, (10, 55), (w - 10, 70), (200, 200, 200), 2)

                        cv2.putText(annotated, f"{frame_count + 1}/{frames_per_sequence}",
                                    (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cv2.imshow("Sign2Speak — Collecte", annotated)
                        cv2.waitKey(1)

                    # Sauvegarder la séquence
                    if sequence:
                        seq_array = np.array(sequence, dtype=np.float32)
                        npy_path = sign_dir / f"seq_{seq_num:04d}.npy"
                        np.save(npy_path, seq_array)

                        hand_pct = (hands_detected_count / frames_per_sequence) * 100
                        quality = "✅" if hand_pct > 70 else "⚠️" if hand_pct > 40 else "❌"
                        print(f"   {quality} Seq {seq_num + 1}: {seq_array.shape} | Mains: {hand_pct:.0f}%")

                        seq_num += 1

                    # Petite pause entre les séquences
                    time.sleep(0.3)

            print(f"   ✅ {sign} terminé — {seq_num} séquences")

    cap.release()
    cv2.destroyAllWindows()

    # Sauvegarder les métadonnées
    _save_metadata(output_dir, signs, config, sequences_per_sign, frames_per_sequence)
    print(f"\n✅ Collecte terminée!")
    show_stats(output_dir)


def _save_metadata(output_dir: Path, signs: list[str], config: ExtractionConfig,
                   sequences_per_sign: int, frames_per_sequence: int):
    """Sauvegarde les métadonnées de la collecte."""
    # Compter ce qu'on a vraiment
    actual_signs = []
    label_to_idx = {}
    idx = 0

    for sign_dir in sorted(output_dir.iterdir()):
        if sign_dir.is_dir() and sign_dir.name != "__pycache__":
            count = len(list(sign_dir.glob("*.npy")))
            if count > 0:
                actual_signs.append(sign_dir.name)
                label_to_idx[sign_dir.name] = idx
                idx += 1

    meta = {
        'dataset': 'self-collected',
        'signs': actual_signs,
        'signs_count': len(actual_signs),
        'sequences_per_sign': sequences_per_sign,
        'frames_per_sequence': frames_per_sequence,
        'features_per_frame': config.features_per_frame(),
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
        'config': {
            'use_hands': config.use_hands,
            'use_pose': config.use_pose,
            'use_face': config.use_face,
            'face_subset_only': config.face_subset_only,
            'normalize': config.normalize,
        },
    }

    # Sauvegarder labels.json (pour compatibilité avec le classifieur)
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)


def show_stats(output_dir: Path):
    """Affiche les stats de la collecte."""
    print(f"\n📊 Stats collecte — {output_dir}")

    total_sequences = 0
    signs = []

    for sign_dir in sorted(output_dir.iterdir()):
        if sign_dir.is_dir() and sign_dir.name != "__pycache__":
            npys = list(sign_dir.glob("*.npy"))
            count = len(npys)
            if count > 0:
                signs.append((sign_dir.name, count))
                total_sequences += count

                # Vérifier la shape d'un sample
                if npys:
                    sample = np.load(npys[0])

    print(f"   Signes: {len(signs)}")
    print(f"   Total séquences: {total_sequences}")
    if signs:
        sample = np.load(list(sorted(output_dir.iterdir())[0].glob("*.npy"))[0] if sorted(output_dir.iterdir())[0].is_dir() else None)

    for sign, count in signs:
        bar = "█" * count + "░" * max(0, 30 - count)
        print(f"   {sign:15s} | {bar} {count}")


# =============================================================================
# Préparer les données pour l'entraînement (split train/val/test)
# =============================================================================

def prepare_for_training(data_dir: Path, output_dir: Path, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Organise les données collectées en splits train/val/test
    compatibles avec sign_classifier.py.
    """
    import random
    random.seed(42)

    print(f"\n📦 Préparation des splits train/val/test...")

    # Charger les labels
    labels_path = data_dir / "labels.json"
    if not labels_path.exists():
        print("❌ labels.json introuvable. Lance d'abord la collecte.")
        return

    with open(labels_path) as f:
        label_to_idx = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {'train': 0, 'val': 0, 'test': 0}

    for sign_name in label_to_idx:
        sign_dir = data_dir / sign_name
        if not sign_dir.exists():
            continue

        npys = sorted(sign_dir.glob("*.npy"))
        random.shuffle(npys)

        n = len(npys)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        splits = {
            'train': npys[:n_train],
            'val': npys[n_train:n_train + n_val],
            'test': npys[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dest_dir = output_dir / split_name / sign_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                import shutil
                shutil.copy2(f, dest_dir / f.name)
                stats[split_name] += 1

    # Copier les métadonnées
    import shutil
    shutil.copy2(data_dir / "labels.json", output_dir / "labels.json")

    # Mettre à jour metadata
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    meta['stats'] = {'extracted': sum(stats.values()), 'skipped': 0, 'errors': 0}
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"   Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    print(f"   ✅ Sauvegardé dans {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Collecte de données ASL")
    parser.add_argument('--output', type=str, default='./my_data',
                        help='Dossier de sortie')
    parser.add_argument('--signs', nargs='+', default=None,
                        help='Liste des signes à collecter (défaut: 26 signes MVP)')
    parser.add_argument('--sequences', type=int, default=SEQUENCES_PER_SIGN,
                        help=f'Séquences par signe (défaut: {SEQUENCES_PER_SIGN})')
    parser.add_argument('--frames', type=int, default=FRAMES_PER_SEQUENCE,
                        help=f'Frames par séquence (défaut: {FRAMES_PER_SEQUENCE})')
    parser.add_argument('--append', action='store_true',
                        help='Ajouter aux données existantes')
    parser.add_argument('--stats', action='store_true',
                        help='Afficher les stats de la collecte')
    parser.add_argument('--prepare', action='store_true',
                        help='Préparer les splits train/val/test pour l\'entraînement')
    parser.add_argument('--prepared-output', type=str, default='./my_keypoints',
                        help='Dossier de sortie des splits (défaut: ./my_keypoints)')

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.stats:
        show_stats(output_dir)
        return

    if args.prepare:
        prepare_for_training(output_dir, Path(args.prepared_output))
        return

    signs = args.signs if args.signs else DEFAULT_SIGNS

    collect_data(
        output_dir=output_dir,
        signs=signs,
        sequences_per_sign=args.sequences,
        frames_per_sequence=args.frames,
        append=args.append,
    )


if __name__ == "__main__":
    main()
