"""
Sign2Speak — Dataset Pipeline : ASL Citizen
============================================
Adapté à la structure réelle du dataset:
    data/ASL_Citizen/
        videos/          ← toutes les vidéos en vrac (.mp4)
        splits/
            train.csv    ← CSV avec colonnes: Participant ID, Video file, Gloss, ASL-LEX Code
            val.csv
            test.csv

Usage:
    # Explorer les signes disponibles
    python dataset_pipeline.py --explore --data-dir ./data

    # Test rapide sur 5 signes (VALIDER AVANT LE BATCH)
    python dataset_pipeline.py --extract --data-dir ./data --output ./keypoints --max-signs 5 --max-videos-per-sign 5

    # Extraction complète MVP (100 signes)
    python dataset_pipeline.py --extract --data-dir ./data --output ./keypoints --max-signs 100

    # Extraction custom
    python dataset_pipeline.py --extract --data-dir ./data --output ./keypoints --max-signs 50 --max-videos-per-sign 30
"""

import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np

from keypoint_extractor import KeypointExtractor, ExtractionConfig


# =============================================================================
# Configuration
# =============================================================================

# Les signes prioritaires pour la communication quotidienne
# On les matche avec les glosses du dataset (qui sont en MAJUSCULES)
MVP_SIGNS_PRIORITY = [
    # Salutations / bases
    "HELLO", "GOODBYE", "THANK-YOU", "PLEASE", "SORRY", "YES", "NO",
    "HELP", "WELCOME", "HI",
    # Identité
    "NAME", "MY", "YOUR", "I", "YOU", "HE", "SHE", "WE", "THEY",
    # Questions
    "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW",
    # Urgence / santé
    "DOCTOR", "HOSPITAL", "MEDICINE", "PAIN", "SICK", "EMERGENCY",
    "HURT", "CALL", "HELP",
    # Nourriture / commerce
    "WATER", "FOOD", "EAT", "DRINK", "HUNGRY", "MONEY", "PAY",
    "BUY", "COFFEE", "RESTAURANT",
    # Déplacement
    "GO", "STOP", "LEFT", "RIGHT", "HERE", "THERE", "HOME", "WORK",
    "CAR", "BUS", "TRAIN", "WALK",
    # Temps
    "TODAY", "TOMORROW", "YESTERDAY", "NOW", "LATER", "MORNING",
    "AFTERNOON", "NIGHT", "TIME", "WAIT",
    # Actions courantes
    "WANT", "NEED", "LIKE", "KNOW", "UNDERSTAND",
    "AGAIN", "SLOW", "FAST", "MORE", "FINISH", "START",
    # Social
    "FRIEND", "FAMILY", "HAPPY", "SAD", "LOVE", "NICE",
    "GOOD", "BAD", "BIG", "SMALL", "NEW", "OLD",
    # Chiffres
    "ONE", "TWO", "THREE", "FOUR", "FIVE",
]


# =============================================================================
# Chargement des CSV
# =============================================================================

def load_splits(data_dir: Path) -> dict:
    """
    Charge les 3 CSV (train, val, test) et construit un index.
    
    Returns:
        dict avec:
            'all_entries': liste de dicts {participant, video_file, gloss, asl_lex_code, split}
            'gloss_to_videos': {gloss: [video_file, ...]}
            'gloss_counts': Counter des glosses
            'videos_dir': Path vers le dossier vidéos
    """
    asl_dir = data_dir / "ASL_Citizen"
    splits_dir = asl_dir / "splits"
    videos_dir = asl_dir / "videos"

    all_entries = []
    gloss_to_videos = defaultdict(list)

    for split_name in ['train', 'val', 'test']:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"⚠️  {csv_path} introuvable")
            continue

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    'participant': row['Participant ID'],
                    'video_file': row['Video file'],
                    'gloss': row['Gloss'].strip(),
                    'asl_lex_code': row.get('ASL-LEX Code', ''),
                    'split': split_name,
                }
                all_entries.append(entry)
                gloss_to_videos[entry['gloss']].append(entry)

    gloss_counts = Counter(entry['gloss'] for entry in all_entries)

    return {
        'all_entries': all_entries,
        'gloss_to_videos': dict(gloss_to_videos),
        'gloss_counts': gloss_counts,
        'videos_dir': videos_dir,
    }


# =============================================================================
# Exploration
# =============================================================================

def explore_dataset(data_dir: Path):
    """Affiche les stats du dataset et les signes disponibles."""
    print(f"\n🔍 Chargement du dataset...")
    data = load_splits(data_dir)

    total = len(data['all_entries'])
    n_glosses = len(data['gloss_counts'])
    print(f"\n📊 Dataset ASL Citizen:")
    print(f"   Total vidéos: {total}")
    print(f"   Signes distincts: {n_glosses}")

    # Split breakdown
    splits = Counter(e['split'] for e in data['all_entries'])
    for split, count in sorted(splits.items()):
        print(f"   {split}: {count} vidéos")

    # Signeurs
    participants = set(e['participant'] for e in data['all_entries'])
    print(f"   Signeurs: {len(participants)}")

    # Top 30 signes les plus fréquents
    print(f"\n📋 Top 30 signes (par nombre de vidéos):")
    for gloss, count in data['gloss_counts'].most_common(30):
        print(f"   {gloss:25s} → {count} vidéos")

    # Matcher avec nos signes MVP
    print(f"\n🎯 Correspondance avec les signes MVP prioritaires:")
    available = set(data['gloss_counts'].keys())
    matched = []
    not_found = []

    for sign in MVP_SIGNS_PRIORITY:
        # Chercher exact match
        if sign in available:
            matched.append((sign, data['gloss_counts'][sign]))
        else:
            # Chercher variantes (ex: THANK-YOU vs THANKYOU, HELLO vs HELLO1)
            variants = [g for g in available if sign in g or g in sign]
            if variants:
                best = max(variants, key=lambda g: data['gloss_counts'][g])
                matched.append((best, data['gloss_counts'][best]))
            else:
                not_found.append(sign)

    # Dédupliquer
    seen = set()
    matched_unique = []
    for sign, count in matched:
        if sign not in seen:
            seen.add(sign)
            matched_unique.append((sign, count))

    print(f"   ✅ Trouvés: {len(matched_unique)} signes")
    for sign, count in matched_unique[:20]:
        print(f"      {sign:25s} → {count} vidéos")
    if len(matched_unique) > 20:
        print(f"      ... et {len(matched_unique) - 20} autres")

    if not_found:
        print(f"   ❌ Non trouvés ({len(not_found)}): {not_found[:15]}")

    # Vérifier que les fichiers vidéo existent
    print(f"\n🔎 Vérification des fichiers vidéo...")
    videos_dir = data['videos_dir']
    sample = data['all_entries'][:10]
    found = 0
    for entry in sample:
        path = videos_dir / entry['video_file']
        if path.exists():
            found += 1
    print(f"   {found}/{len(sample)} fichiers test trouvés dans {videos_dir}")

    return data


# =============================================================================
# Sélection des signes
# =============================================================================

def select_signs(data: dict, max_signs: int = 100) -> list[str]:
    """
    Sélectionne les signes à extraire:
    1. D'abord les signes MVP prioritaires qui existent dans le dataset
    2. Puis complète avec les plus fréquents
    """
    available = set(data['gloss_counts'].keys())
    selected = []
    seen = set()

    # 1. Signes MVP prioritaires
    for sign in MVP_SIGNS_PRIORITY:
        if len(selected) >= max_signs:
            break

        if sign in available and sign not in seen:
            selected.append(sign)
            seen.add(sign)
        else:
            # Variantes
            variants = [g for g in available if sign in g or g in sign]
            if variants:
                best = max(variants, key=lambda g: data['gloss_counts'][g])
                if best not in seen:
                    selected.append(best)
                    seen.add(best)

    # 2. Compléter avec les plus fréquents
    if len(selected) < max_signs:
        for gloss, count in data['gloss_counts'].most_common():
            if len(selected) >= max_signs:
                break
            if gloss not in seen:
                selected.append(gloss)
                seen.add(gloss)

    return selected


# =============================================================================
# Extraction des keypoints
# =============================================================================

def extract_keypoints(
    data_dir: Path,
    output_dir: Path,
    max_signs: int = 100,
    max_videos_per_sign: int = 50,
):
    """
    Extrait les keypoints pour les signes sélectionnés.
    
    Structure de sortie:
        output_dir/
            labels.json          ← {gloss: index}
            metadata.json        ← config, stats
            train/
                HELLO/
                    vid_001.npy  ← (n_frames, 298)
                    vid_002.npy
                DOCTOR/
                    ...
            val/
                ...
            test/
                ...
    """
    print(f"\n🔧 Préparation de l'extraction...")

    # Charger les données
    data = load_splits(data_dir)
    videos_dir = data['videos_dir']

    # Sélectionner les signes
    selected_signs = select_signs(data, max_signs)
    print(f"   Signes sélectionnés: {len(selected_signs)}")
    print(f"   Premiers: {selected_signs[:10]}")

    # Créer le mapping label → index
    label_to_idx = {sign: idx for idx, sign in enumerate(selected_signs)}
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    # Regrouper les vidéos par signe et par split
    sign_videos = defaultdict(lambda: defaultdict(list))
    for entry in data['all_entries']:
        if entry['gloss'] in label_to_idx:
            sign_videos[entry['gloss']][entry['split']].append(entry)

    # Stats avant extraction
    total_to_extract = 0
    for sign in selected_signs:
        for split in ['train', 'val', 'test']:
            count = len(sign_videos[sign][split])
            total_to_extract += min(count, max_videos_per_sign)
    print(f"   Vidéos à extraire: ~{total_to_extract}")

    # Extraction
    config = ExtractionConfig(
        use_hands=True,
        use_pose=True,
        use_face=True,
        face_subset_only=True,
        normalize=True,
    )
    print(f"   Features/frame: {config.features_per_frame()}")

    stats = {'extracted': 0, 'skipped': 0, 'errors': 0}

    with KeypointExtractor(config) as extractor:
        for sign_idx, sign in enumerate(selected_signs):
            for split in ['train', 'val', 'test']:
                entries = sign_videos[sign][split][:max_videos_per_sign]
                if not entries:
                    continue

                # Dossier de sortie
                sign_dir = output_dir / split / sign
                sign_dir.mkdir(parents=True, exist_ok=True)

                for i, entry in enumerate(entries):
                    video_path = videos_dir / entry['video_file']
                    npy_name = f"{entry['participant']}_{video_path.stem}.npy"
                    npy_path = sign_dir / npy_name

                    # Skip si déjà fait
                    if npy_path.exists():
                        stats['skipped'] += 1
                        continue

                    # Vérifier que la vidéo existe
                    if not video_path.exists():
                        stats['errors'] += 1
                        continue

                    try:
                        cap = cv2.VideoCapture(str(video_path))
                        if not cap.isOpened():
                            stats['errors'] += 1
                            continue

                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        all_keypoints = []
                        frame_idx = 0

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            timestamp_ms = int(frame_idx * (1000.0 / fps))
                            keypoints = extractor.extract_and_flatten(frame, timestamp_ms)
                            all_keypoints.append(keypoints)
                            frame_idx += 1

                        cap.release()

                        if all_keypoints:
                            sequence = np.array(all_keypoints, dtype=np.float32)
                            np.save(npy_path, sequence)
                            stats['extracted'] += 1
                        else:
                            stats['errors'] += 1

                    except Exception as e:
                        stats['errors'] += 1
                        if stats['errors'] <= 5:
                            print(f"      ⚠️  {entry['video_file']}: {e}")
                        continue

                # Log de progression
                done = stats['extracted'] + stats['skipped']
                print(f"   [{sign_idx+1}/{len(selected_signs)}] {sign:20s} | "
                      f"{split:5s}: {len(entries)} vids | "
                      f"Total: {done}/{total_to_extract} "
                      f"({'%.0f' % (done/total_to_extract*100) if total_to_extract else 0}%)")

    # Sauvegarder les métadonnées
    metadata = {
        'dataset': 'ASL Citizen',
        'signs': selected_signs,
        'signs_count': len(selected_signs),
        'features_per_frame': config.features_per_frame(),
        'stats': stats,
        'config': {
            'use_hands': config.use_hands,
            'use_pose': config.use_pose,
            'use_face': config.use_face,
            'face_subset_only': config.face_subset_only,
            'normalize': config.normalize,
        },
        'label_to_index': label_to_idx,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Extraction terminée!")
    print(f"   Extraits: {stats['extracted']}")
    print(f"   Skippés (déjà faits): {stats['skipped']}")
    print(f"   Erreurs: {stats['errors']}")
    print(f"   Sortie: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Pipeline ASL Citizen")
    parser.add_argument('--explore', action='store_true',
                        help='Explorer le dataset')
    parser.add_argument('--extract', action='store_true',
                        help='Extraire les keypoints')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Dossier du dataset (défaut: ./data)')
    parser.add_argument('--output', type=str, default='./keypoints',
                        help='Sortie des keypoints (défaut: ./keypoints)')
    parser.add_argument('--max-signs', type=int, default=100,
                        help='Nombre de signes (défaut: 100)')
    parser.add_argument('--max-videos-per-sign', type=int, default=50,
                        help='Vidéos max par signe (défaut: 50)')

    args = parser.parse_args()

    if not (args.explore or args.extract):
        parser.print_help()
        return

    if args.explore:
        explore_dataset(Path(args.data_dir))

    if args.extract:
        extract_keypoints(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output),
            max_signs=args.max_signs,
            max_videos_per_sign=args.max_videos_per_sign,
        )


if __name__ == "__main__":
    main()