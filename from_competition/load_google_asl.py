"""
Sign2Speak — Loader Google ASL Signs Dataset (Kaggle)
=====================================================
Charge les fichiers Parquet pré-vectorisés du dataset Google
Isolated Sign Language Recognition et les convertit en .npy
compatibles avec notre classifieur.

Le dataset contient:
- 94 477 vidéos de 21 signeurs sourds
- 250 signes (mots courants)
- Landmarks MediaPipe Holistic déjà extraits (543 landmarks × 3 coords)

On sélectionne un subset de ~130 landmarks (comme le gagnant Kaggle)
pour un meilleur rapport signal/bruit.

Usage:
    # Explorer le dataset
    python load_google_asl.py --explore --data-dir ./asl-signs

    # Convertir en .npy pour l'entraînement (50 signes MVP)
    python load_google_asl.py --convert --data-dir ./asl-signs --output ./google_keypoints --max-signs 50

    # Convertir + fusionner avec données perso
    python load_google_asl.py --convert --data-dir ./asl-signs --output ./combined_keypoints --max-signs 50 --merge-with ./my_keypoints
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("⚠️  Installe pandas et pyarrow: pip install pandas pyarrow")
    exit(1)


# =============================================================================
# Sélection des 130 landmarks optimaux (basé sur le gagnant Kaggle)
# =============================================================================

# Colonnes MediaPipe Holistic dans le Parquet:
# x/y/z_face_0..467 (468 face), x/y/z_left_hand_0..20 (21), 
# x/y/z_pose_0..32 (33), x/y/z_right_hand_0..20 (21)
# Total: 543 landmarks × 3 coords = 1629 colonnes

# Subset optimal ~130 landmarks:
# - Mains complètes: 21 × 2 = 42 landmarks
# - Pose upper body: 13 landmarks (0-12: tête, épaules, coudes, poignets, hanches)
# - Lèvres: ~40 landmarks (contour externe + interne)
# - Sourcils: ~10 landmarks
# Total: ~105 landmarks × 3 = 315 features

# Indices de pose à garder (upper body uniquement)
POSE_INDICES = list(range(0, 25))  # 0-24: tout le upper body + hanches

# Indices de face à garder (lèvres + sourcils + contour yeux)
FACE_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
FACE_LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                    95, 88, 178, 87, 14, 317, 402, 318, 324]
FACE_BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
FACE_EYES = [33, 133, 362, 263, 159, 145, 386, 374]  # coins des yeux

# Dédupliquer
FACE_INDICES = sorted(set(FACE_LIPS_OUTER + FACE_LIPS_INNER + FACE_BROWS + FACE_EYES))

# Mains: toutes (0-20)
HAND_INDICES = list(range(21))


def get_landmark_columns():
    """Retourne la liste ordonnée des colonnes à extraire du Parquet."""
    cols = []
    
    # Mains (toutes)
    for hand in ['left_hand', 'right_hand']:
        for idx in HAND_INDICES:
            for axis in ['x', 'y', 'z']:
                cols.append(f"{axis}_{hand}_{idx}")
    
    # Pose (upper body)
    for idx in POSE_INDICES:
        for axis in ['x', 'y', 'z']:
            cols.append(f"{axis}_pose_{idx}")
    
    # Face (subset)
    for idx in FACE_INDICES:
        for axis in ['x', 'y', 'z']:
            cols.append(f"{axis}_face_{idx}")
    
    return cols


def get_features_count():
    """Nombre total de features par frame."""
    return len(HAND_INDICES) * 3 * 2 + len(POSE_INDICES) * 3 + len(FACE_INDICES) * 3


# =============================================================================
# Signes MVP
# =============================================================================

MVP_SIGNS = [
    "hello", "thank you", "please", "sorry", "yes", "no",
    "help", "doctor", "water", "food",
    "what", "where", "who", "how", "why",
    "want", "need", "go", "stop", "eat",
    "drink", "good", "bad", "name", "understand",
    "my", "your", "friend", "family", "happy",
    "sad", "love", "know", "wait", "finish",
    "more", "again", "here", "there", "home",
    "work", "money", "time", "morning", "night",
    "call", "come", "sit", "walk", "like",
]


# =============================================================================
# Explorer
# =============================================================================

def explore(data_dir: Path):
    """Explore le dataset Google ASL Signs."""
    print(f"\n🔍 Exploration de {data_dir}...")

    # Trouver les fichiers
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    csv_files = sorted(data_dir.rglob("*.csv"))
    json_files = sorted(data_dir.rglob("*.json"))

    print(f"\n📂 Fichiers trouvés:")
    print(f"   Parquet: {len(parquet_files)}")
    print(f"   CSV: {len(csv_files)}")
    print(f"   JSON: {len(json_files)}")

    # Lire le mapping des signes
    for jf in json_files:
        if 'sign_to_prediction' in jf.name or 'map' in jf.name:
            with open(jf) as f:
                sign_map = json.load(f)
            print(f"\n📋 Mapping des signes ({jf.name}): {len(sign_map)} signes")
            # Montrer quelques exemples
            examples = list(sign_map.items())[:20]
            for sign, idx in examples:
                print(f"   {sign:20s} → {idx}")
            if len(sign_map) > 20:
                print(f"   ... et {len(sign_map) - 20} autres")

    # Lire le CSV de metadata
    for cf in csv_files:
        if 'train' in cf.name:
            df = pd.read_csv(cf)
            print(f"\n📊 Metadata ({cf.name}):")
            print(f"   Colonnes: {list(df.columns)}")
            print(f"   Lignes: {len(df)}")
            print(f"   Premières lignes:")
            print(df.head())

            if 'sign' in df.columns:
                sign_counts = df['sign'].value_counts()
                print(f"\n   Signes distincts: {len(sign_counts)}")
                print(f"   Vidéos/signe: min={sign_counts.min()}, max={sign_counts.max()}, median={int(sign_counts.median())}")

            if 'participant_id' in df.columns:
                print(f"   Signeurs: {df['participant_id'].nunique()}")

    # Lire un Parquet pour voir la structure
    if parquet_files:
        sample = parquet_files[0]
        df_sample = pd.read_parquet(sample)
        print(f"\n📐 Structure Parquet ({sample.name}):")
        print(f"   Shape: {df_sample.shape}")
        print(f"   Colonnes: {len(df_sample.columns)}")
        print(f"   Premières colonnes: {list(df_sample.columns[:10])}")
        if 'frame' in df_sample.columns:
            print(f"   Frames: {df_sample['frame'].nunique()}")

        # Vérifier quelles colonnes landmark existent
        expected_cols = get_landmark_columns()
        found = [c for c in expected_cols if c in df_sample.columns]
        missing = [c for c in expected_cols if c not in df_sample.columns]
        print(f"\n   Landmarks subset (130): {len(found)} trouvés, {len(missing)} manquants")
        if missing[:5]:
            print(f"   Exemples manquants: {missing[:5]}")

    # Matcher avec nos signes MVP
    if json_files:
        for jf in json_files:
            if 'sign_to_prediction' in jf.name or 'map' in jf.name:
                with open(jf) as f:
                    sign_map = json.load(f)
                available = set(sign_map.keys())
                matched = [s for s in MVP_SIGNS if s in available]
                not_found = [s for s in MVP_SIGNS if s not in available]
                print(f"\n🎯 Match MVP:")
                print(f"   ✅ Trouvés: {len(matched)}")
                print(f"   ❌ Non trouvés: {not_found[:15]}")


# =============================================================================
# Conversion Parquet → npy
# =============================================================================

def convert(
    data_dir: Path,
    output_dir: Path,
    max_signs: int = 50,
    max_videos_per_sign: int = 999,
    merge_with: Path = None,
):
    """Convertit les Parquet en .npy structurés pour l'entraînement."""
    print(f"\n🔧 Conversion du dataset Google ASL Signs...")

    # Trouver le CSV de metadata
    csv_path = None
    for cf in sorted(data_dir.rglob("*.csv")):
        if 'train' in cf.name:
            csv_path = cf
            break

    if not csv_path:
        print("❌ train.csv introuvable")
        return

    # Trouver le mapping des signes
    sign_map = None
    for jf in sorted(data_dir.rglob("*.json")):
        if 'sign_to_prediction' in jf.name or 'map' in jf.name:
            with open(jf) as f:
                sign_map = json.load(f)
            break

    if not sign_map:
        print("❌ sign_to_prediction_index_map.json introuvable")
        return

    # Charger metadata
    meta_df = pd.read_csv(csv_path)
    print(f"   {len(meta_df)} vidéos, {meta_df['sign'].nunique()} signes")

    # Sélectionner les signes
    available_signs = set(sign_map.keys())
    selected = []

    # D'abord les MVP
    for s in MVP_SIGNS:
        if s in available_signs and len(selected) < max_signs:
            selected.append(s)

    # Compléter avec les plus fréquents
    if len(selected) < max_signs:
        sign_counts = meta_df['sign'].value_counts()
        for s in sign_counts.index:
            if s not in selected and len(selected) < max_signs:
                selected.append(s)

    print(f"   {len(selected)} signes sélectionnés")

    # Créer le mapping label → index
    label_to_idx = {s: i for i, s in enumerate(sorted(selected))}
    idx_to_label = {i: s for s, i in label_to_idx.items()}

    # Colonnes à extraire
    landmark_cols = get_landmark_columns()
    n_features = get_features_count()
    print(f"   {n_features} features/frame ({len(landmark_cols)} colonnes)")

    # Trouver le dossier des parquet
    parquet_dir = None
    for d in data_dir.rglob("*"):
        if d.is_dir() and list(d.glob("*.parquet")):
            parquet_dir = d
            break

    if not parquet_dir:
        print("❌ Dossier de fichiers Parquet introuvable")
        return

    # Filtrer les vidéos pour les signes sélectionnés
    filtered = meta_df[meta_df['sign'].isin(selected)].copy()
    
    # Limiter par signe
    limited = []
    for sign in selected:
        sign_rows = filtered[filtered['sign'] == sign]
        limited.append(sign_rows.head(max_videos_per_sign))
    filtered = pd.concat(limited)

    print(f"   {len(filtered)} vidéos à convertir")

    # Split train/val/test (70/15/15)
    random.seed(42)
    
    # Grouper par participant pour éviter le data leakage
    participants = filtered['participant_id'].unique().tolist()
    random.shuffle(participants)
    n_p = len(participants)
    test_participants = set(participants[:max(1, int(n_p * 0.15))])
    val_participants = set(participants[max(1, int(n_p * 0.15)):max(2, int(n_p * 0.30))])
    train_participants = set(participants) - test_participants - val_participants

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}

    # Traiter chaque vidéo
    for i, (_, row) in enumerate(filtered.iterrows()):
        sign = row['sign']
        path_col = 'path' if 'path' in row.index else 'file_id'
        
        if path_col not in row.index:
            # Chercher la colonne qui donne le chemin du parquet
            for col in row.index:
                if col not in ['sign', 'participant_id', 'sequence_id']:
                    if isinstance(row[col], str) and '.parquet' in str(row[col]):
                        path_col = col
                        break

        # Construire le chemin du parquet
        parquet_path = None
        if path_col in row.index:
            candidate = data_dir / str(row[path_col])
            if candidate.exists():
                parquet_path = candidate
            else:
                # Essayer dans le sous-dossier
                for ext in ['', '.parquet']:
                    for base in [parquet_dir, data_dir]:
                        p = base / f"{row[path_col]}{ext}"
                        if p.exists():
                            parquet_path = p
                            break

        # Fallback: chercher par sequence_id ou index
        if parquet_path is None and 'sequence_id' in row.index:
            seq_id = row['sequence_id']
            candidates = list(parquet_dir.glob(f"*{seq_id}*.parquet"))
            if candidates:
                parquet_path = candidates[0]

        if parquet_path is None:
            # Le fichier parquet est peut-être le même pour tous (un gros fichier)
            # avec une colonne sequence_id pour filtrer
            if i == 0:
                big_parquets = list(parquet_dir.glob("*.parquet"))
                if big_parquets:
                    print(f"   📦 Détection de structure: {len(big_parquets)} fichier(s) Parquet")
                    # Lire le premier pour comprendre la structure
                    sample_df = pd.read_parquet(big_parquets[0], columns=['frame'] if 'frame' in pd.read_parquet(big_parquets[0], columns=[]).columns else None)
                    print(f"   Colonnes disponibles: {list(sample_df.columns[:15])}")

        if parquet_path is None:
            stats['errors'] += 1
            continue

        try:
            # Lire le Parquet
            df = pd.read_parquet(parquet_path)

            # Filtrer les colonnes disponibles
            available_cols = [c for c in landmark_cols if c in df.columns]
            if not available_cols:
                stats['errors'] += 1
                continue

            # Extraire les landmarks
            landmarks = df[available_cols].values.astype(np.float32)

            # Remplacer NaN par 0
            landmarks = np.nan_to_num(landmarks, nan=0.0)

            if landmarks.shape[0] == 0:
                stats['errors'] += 1
                continue

            # Déterminer le split
            pid = row['participant_id']
            if pid in test_participants:
                split = 'test'
            elif pid in val_participants:
                split = 'val'
            else:
                split = 'train'

            # Sauvegarder
            sign_dir = output_dir / split / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            seq_id = row.get('sequence_id', i)
            npy_path = sign_dir / f"{seq_id}.npy"
            np.save(npy_path, landmarks)

            stats[split] += 1

        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"   ⚠️  Erreur: {e}")
            continue

        if (i + 1) % 500 == 0:
            total = stats['train'] + stats['val'] + stats['test']
            print(f"   [{i+1}/{len(filtered)}] Convertis: {total} | Erreurs: {stats['errors']}")

    # Sauvegarder les métadonnées
    meta = {
        'dataset': 'google-asl-signs',
        'signs': sorted(selected),
        'signs_count': len(selected),
        'features_per_frame': n_features,
        'landmark_columns': landmark_cols[:10],  # échantillon
        'n_landmarks': len(HAND_INDICES) * 2 + len(POSE_INDICES) + len(FACE_INDICES),
        'stats': stats,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
    }

    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Conversion terminée!")
    print(f"   Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    print(f"   Erreurs: {stats['errors']}")
    print(f"   Features/frame: {n_features}")
    print(f"   Sortie: {output_dir}")

    # Merge avec données perso si demandé
    if merge_with and merge_with.exists():
        print(f"\n🔗 Fusion avec {merge_with}...")
        merge_personal_data(merge_with, output_dir)


def merge_personal_data(personal_dir: Path, target_dir: Path):
    """Fusionne les données perso dans le dataset Google."""
    import shutil

    personal_labels = None
    labels_file = personal_dir / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            personal_labels = json.load(f)

    # Charger le mapping cible
    with open(target_dir / "labels.json") as f:
        target_labels = json.load(f)

    copied = 0
    for split in ['train', 'val', 'test']:
        split_dir = personal_dir / split
        if not split_dir.exists():
            continue
        for sign_dir in split_dir.iterdir():
            if not sign_dir.is_dir():
                continue
            sign_name = sign_dir.name
            # Mapper le nom si nécessaire
            if sign_name in target_labels:
                dest = target_dir / split / sign_name
                dest.mkdir(parents=True, exist_ok=True)
                for npy in sign_dir.glob("*.npy"):
                    dest_file = dest / f"personal_{npy.name}"
                    if not dest_file.exists():
                        shutil.copy2(npy, dest_file)
                        copied += 1

    print(f"   ✅ {copied} séquences personnelles ajoutées")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Loader Google ASL Signs")
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--data-dir', type=str, default='./asl-signs')
    parser.add_argument('--output', type=str, default='./google_keypoints')
    parser.add_argument('--max-signs', type=int, default=50)
    parser.add_argument('--max-videos-per-sign', type=int, default=999)
    parser.add_argument('--merge-with', type=str, default=None)

    args = parser.parse_args()

    if args.explore:
        explore(Path(args.data_dir))

    if args.convert:
        merge = Path(args.merge_with) if args.merge_with else None
        convert(
            Path(args.data_dir), Path(args.output),
            args.max_signs, args.max_videos_per_sign, merge
        )

    if not (args.explore or args.convert):
        parser.print_help()


if __name__ == "__main__":
    main()
