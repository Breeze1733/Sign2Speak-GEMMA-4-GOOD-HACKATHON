"""
Sign2Speak — Loader Google ASL Signs v3 (FIXED)
================================================
Correctifs critiques:
1. Aligné EXACTEMENT sur keypoint_extractor.py (298 features)
   - Mêmes UPPER_BODY_INDICES (19 landmarks × 4 avec visibility=1.0)
   - Mêmes FACE_KEY_INDICES (32 landmarks × 3)
   - Même ordre: [left_hand, right_hand, pose, face]
2. Normalisation par épaules (comme keypoint_extractor._normalize)
3. Landmarks manquants → NaN pendant le traitement, 0.0 après normalisation

Usage:
    python load_google_asl.py --explore --data-dir ../asl-signs
    python load_google_asl.py --convert --data-dir ../asl-signs --output ./google_keypoints --max-signs 50
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# =============================================================================
# Landmarks — IDENTIQUES à keypoint_extractor.py
# =============================================================================

N_HAND = 21
HAND_INDICES = list(range(21))

UPPER_BODY_INDICES = [
    0,                        # nez
    2, 5,                     # yeux
    7, 8,                     # oreilles
    9, 10,                    # coins bouche
    11, 12, 13, 14, 15, 16,   # épaules, coudes, poignets
    17, 18, 19, 20,           # index et pinky tips
    23, 24,                   # hanches
]

LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
BROW_INDICES = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
]
FACE_KEY_INDICES = LIPS_INDICES + BROW_INDICES

# Format du vecteur de features (IDENTIQUE à keypoint_extractor):
# left_hand:  21 × 3 = 63   (offset 0)
# right_hand: 21 × 3 = 63   (offset 63)
# pose:       19 × 4 = 76   (offset 126)  ← x,y,z + visibility
# face:       32 × 3 = 96   (offset 202)
# TOTAL = 298
N_FEATURES = N_HAND * 3 * 2 + len(UPPER_BODY_INDICES) * 4 + len(FACE_KEY_INDICES) * 3  # = 298

# Offsets dans le vecteur
OFF_LEFT_HAND = 0
OFF_RIGHT_HAND = N_HAND * 3
OFF_POSE = N_HAND * 3 * 2
OFF_FACE = OFF_POSE + len(UPPER_BODY_INDICES) * 4

# Position des épaules dans UPPER_BODY_INDICES (pour normalisation)
IDX_L_SHOULDER = UPPER_BODY_INDICES.index(11)  # = 7
IDX_R_SHOULDER = UPPER_BODY_INDICES.index(12)  # = 8


# =============================================================================
# Signes MVP (vérifiés dans le JSON du dataset)
# =============================================================================

MVP_SIGNS = [
    "hello", "bye", "thankyou", "please", "yes", "no",
    "happy", "sad", "mad", "like", "hate", "fine", "sick", "hungry", "thirsty",
    "go", "drink", "give", "wait", "finish", "look", "see", "talk",
    "think", "listen", "sleep", "wake", "stay", "open", "close", "find",
    "where", "who", "why",
    "mom", "dad", "boy", "girl", "child", "person",
    "water", "food", "milk",
    "now", "time", "morning", "night", "tomorrow", "yesterday", "later",
    "home", "there", "up", "down", "outside",
    "have", "haveto", "can", "not",
]


# =============================================================================
# Conversion d'un Parquet → numpy array (298 features, normalisé)
# =============================================================================

def parquet_to_numpy(parquet_path: Path) -> np.ndarray:
    """
    Convertit un Parquet long en array (n_frames, 298) normalisé.
    Format aligné sur keypoint_extractor.py.
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return np.array([])

    frames = sorted(df['frame'].unique())
    n_frames = len(frames)
    frame_to_row = {f: i for i, f in enumerate(frames)}

    # Pré-alloquer avec NaN (pour distinguer "absent" de "zéro")
    result = np.full((n_frames, N_FEATURES), np.nan, dtype=np.float32)

    # --- Extraire les mains ---
    for hand_name, offset in [('left_hand', OFF_LEFT_HAND), ('right_hand', OFF_RIGHT_HAND)]:
        hand_df = df[(df['type'] == hand_name) & (df['landmark_index'].isin(HAND_INDICES))]
        if hand_df.empty:
            continue

        for _, row in hand_df.iterrows():
            fr = frame_to_row.get(row['frame'])
            lm = row['landmark_index']
            if fr is not None and 0 <= lm < N_HAND:
                col = offset + lm * 3
                result[fr, col] = row['x']
                result[fr, col + 1] = row['y']
                result[fr, col + 2] = row['z']

    # --- Extraire la pose (avec visibility=1.0 ajouté) ---
    pose_set = set(UPPER_BODY_INDICES)
    idx_to_pos = {idx: pos for pos, idx in enumerate(UPPER_BODY_INDICES)}
    pose_df = df[(df['type'] == 'pose') & (df['landmark_index'].isin(pose_set))]

    if not pose_df.empty:
        for _, row in pose_df.iterrows():
            fr = frame_to_row.get(row['frame'])
            pos = idx_to_pos.get(row['landmark_index'])
            if fr is not None and pos is not None:
                col = OFF_POSE + pos * 4
                result[fr, col] = row['x']
                result[fr, col + 1] = row['y']
                result[fr, col + 2] = row['z']
                result[fr, col + 3] = 1.0  # visibility (Google n'a pas cette info)

    # --- Extraire le visage ---
    face_set = set(FACE_KEY_INDICES)
    face_idx_to_pos = {idx: pos for pos, idx in enumerate(FACE_KEY_INDICES)}
    face_df = df[(df['type'] == 'face') & (df['landmark_index'].isin(face_set))]

    if not face_df.empty:
        for _, row in face_df.iterrows():
            fr = frame_to_row.get(row['frame'])
            pos = face_idx_to_pos.get(row['landmark_index'])
            if fr is not None and pos is not None:
                col = OFF_FACE + pos * 3
                result[fr, col] = row['x']
                result[fr, col + 1] = row['y']
                result[fr, col + 2] = row['z']

    # --- Normalisation par épaules (comme keypoint_extractor._normalize) ---
    for fr in range(n_frames):
        # Récupérer les épaules
        l_sh_col = OFF_POSE + IDX_L_SHOULDER * 4
        r_sh_col = OFF_POSE + IDX_R_SHOULDER * 4

        l_shoulder = result[fr, l_sh_col:l_sh_col + 3]
        r_shoulder = result[fr, r_sh_col:r_sh_col + 3]

        # Si épaules non détectées, skip la normalisation pour cette frame
        if np.any(np.isnan(l_shoulder)) or np.any(np.isnan(r_shoulder)):
            continue

        center = (l_shoulder + r_shoulder) / 2.0
        scale = np.linalg.norm(l_shoulder - r_shoulder)

        if scale < 1e-6:
            continue

        # Normaliser les mains (x, y, z par groupes de 3)
        for offset in [OFF_LEFT_HAND, OFF_RIGHT_HAND]:
            for lm in range(N_HAND):
                col = offset + lm * 3
                vals = result[fr, col:col + 3]
                if not np.any(np.isnan(vals)):
                    result[fr, col:col + 3] = (vals - center) / scale

        # Normaliser la pose (x, y, z par groupes de 4, skip visibility)
        for lm in range(len(UPPER_BODY_INDICES)):
            col = OFF_POSE + lm * 4
            vals = result[fr, col:col + 3]
            if not np.any(np.isnan(vals)):
                result[fr, col:col + 3] = (vals - center) / scale
            # visibility (col+3) reste inchangée

        # Normaliser le visage (x, y, z par groupes de 3)
        for lm in range(len(FACE_KEY_INDICES)):
            col = OFF_FACE + lm * 3
            vals = result[fr, col:col + 3]
            if not np.any(np.isnan(vals)):
                result[fr, col:col + 3] = (vals - center) / scale

    # Remplacer NaN par 0 (landmarks non détectés)
    result = np.nan_to_num(result, nan=0.0)

    return result


# =============================================================================
# Explorer
# =============================================================================

def explore(data_dir: Path):
    print(f"\n🔍 Exploration de {data_dir}...")

    sign_map = None
    for jf in data_dir.rglob("*.json"):
        with open(jf) as f:
            sign_map = json.load(f)
        print(f"📋 {len(sign_map)} signes")
        break

    csv_path = next(data_dir.rglob("*.csv"), None)
    if csv_path:
        meta_df = pd.read_csv(csv_path)
        print(f"📊 {len(meta_df)} vidéos, {meta_df['sign'].nunique()} signes")
        print(f"   Vidéos/signe: min={meta_df['sign'].value_counts().min()}, max={meta_df['sign'].value_counts().max()}")

    # Test de conversion
    parquet_files = list(data_dir.rglob("*.parquet"))
    if parquet_files:
        print(f"\n🧪 Test conversion sur {parquet_files[0].name}...")
        arr = parquet_to_numpy(parquet_files[0])
        print(f"   Shape: {arr.shape}")
        print(f"   Features attendus: {N_FEATURES}")
        print(f"   Zeros: {(arr == 0).sum()}/{arr.size} ({(arr == 0).sum() / arr.size * 100:.0f}%)")
        print(f"   Non-zero range: [{arr[arr != 0].min():.3f}, {arr[arr != 0].max():.3f}]" if (arr != 0).any() else "   Tout est zéro!")
        # Vérifier que les mains ont des données
        lh = arr[:, OFF_LEFT_HAND:OFF_RIGHT_HAND]
        rh = arr[:, OFF_RIGHT_HAND:OFF_POSE]
        pose = arr[:, OFF_POSE:OFF_FACE]
        face = arr[:, OFF_FACE:]
        print(f"   Left hand non-zero: {(lh != 0).any(axis=1).sum()}/{len(arr)} frames")
        print(f"   Right hand non-zero: {(rh != 0).any(axis=1).sum()}/{len(arr)} frames")
        print(f"   Pose non-zero: {(pose != 0).any(axis=1).sum()}/{len(arr)} frames")
        print(f"   Face non-zero: {(face != 0).any(axis=1).sum()}/{len(arr)} frames")

    # Match MVP
    if sign_map:
        available = set(sign_map.keys())
        matched = [s for s in MVP_SIGNS if s in available]
        not_found = [s for s in MVP_SIGNS if s not in available]
        print(f"\n🎯 MVP: {len(matched)}/{len(MVP_SIGNS)} trouvés")
        if not_found:
            print(f"   ❌ {not_found}")


# =============================================================================
# Conversion complète
# =============================================================================

def convert(data_dir: Path, output_dir: Path, max_signs: int = 50,
            max_videos_per_sign: int = 999):
    print(f"\n🔧 Conversion Google ASL Signs → format Sign2Speak...")
    print(f"   Format cible: {N_FEATURES} features/frame (aligné sur keypoint_extractor.py)")

    csv_path = next(data_dir.rglob("*.csv"))
    meta_df = pd.read_csv(csv_path)

    sign_map_path = next(data_dir.rglob("*.json"))
    with open(sign_map_path) as f:
        sign_map = json.load(f)

    available = set(sign_map.keys())

    # Sélectionner les signes
    selected = [s for s in MVP_SIGNS if s in available]
    if len(selected) < max_signs:
        for s in meta_df['sign'].value_counts().index:
            if s not in selected and len(selected) < max_signs:
                selected.append(s)
    selected = selected[:max_signs]

    label_to_idx = {s: i for i, s in enumerate(sorted(selected))}
    print(f"   {len(selected)} signes sélectionnés")

    # Filtrer et limiter
    filtered = meta_df[meta_df['sign'].isin(selected)]
    parts = [filtered[filtered['sign'] == s].head(max_videos_per_sign) for s in selected]
    filtered = pd.concat(parts)
    print(f"   {len(filtered)} vidéos à convertir")

    # Split par participant
    random.seed(42)
    participants = filtered['participant_id'].unique().tolist()
    random.shuffle(participants)
    n_p = len(participants)
    test_p = set(participants[:max(1, int(n_p * 0.15))])
    val_p = set(participants[max(1, int(n_p * 0.15)):max(2, int(n_p * 0.30))])
    train_p = set(participants) - test_p - val_p
    print(f"   Signeurs: {len(train_p)} train, {len(val_p)} val, {len(test_p)} test")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}

    for i, (_, row) in enumerate(filtered.iterrows()):
        sign = row['sign']
        parquet_path = data_dir / row['path']

        if not parquet_path.exists():
            stats['errors'] += 1
            continue

        try:
            arr = parquet_to_numpy(parquet_path)

            if arr.size == 0 or arr.shape[0] < 3:
                stats['errors'] += 1
                continue

            # Vérification: le array doit avoir 298 features
            assert arr.shape[1] == N_FEATURES, f"Expected {N_FEATURES}, got {arr.shape[1]}"

            # Split
            pid = row['participant_id']
            split = 'test' if pid in test_p else ('val' if pid in val_p else 'train')

            sign_dir = output_dir / split / sign
            sign_dir.mkdir(parents=True, exist_ok=True)
            np.save(sign_dir / f"{row['sequence_id']}.npy", arr)
            stats[split] += 1

        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"   ⚠️  {e}")
            continue

        if (i + 1) % 1000 == 0:
            done = stats['train'] + stats['val'] + stats['test']
            print(f"   [{i+1}/{len(filtered)}] {done} convertis | Erreurs: {stats['errors']}")

    # Sauvegarder
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    meta = {
        'dataset': 'google-asl-signs',
        'signs': sorted(selected),
        'signs_count': len(selected),
        'features_per_frame': N_FEATURES,
        'stats': stats,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Terminé!")
    print(f"   Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    print(f"   Erreurs: {stats['errors']}")
    print(f"   Features/frame: {N_FEATURES}")


def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Loader Google ASL Signs")
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--data-dir', type=str, default='../asl-signs')
    parser.add_argument('--output', type=str, default='./google_keypoints')
    parser.add_argument('--max-signs', type=int, default=50)
    parser.add_argument('--max-videos-per-sign', type=int, default=999)
    args = parser.parse_args()

    if args.explore:
        explore(Path(args.data_dir))
    if args.convert:
        convert(Path(args.data_dir), Path(args.output), args.max_signs, args.max_videos_per_sign)
    if not (args.explore or args.convert):
        parser.print_help()

if __name__ == "__main__":
    main()