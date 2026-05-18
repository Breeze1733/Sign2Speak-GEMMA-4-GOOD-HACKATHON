"""
Sign2Speak — Parquet → NPZ pre-conversion
==========================================
Reads all Parquet files once, converts them to numpy arrays
in (n_frames, 543, 3) format and saves them to individual .npy files.

Training is then instant — no Parquet reads per epoch.

Usage:
    python3 preconvert.py --data-dir ./asl-signs --output ./asl_data.npz --max-signs 50
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROWS_PER_FRAME = 543

# MVP signs
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


def load_parquet(path):
    """Load a long-format Parquet file → array (n_frames, 543, 3)."""
    df = pd.read_parquet(path)
    if df.empty:
        return None

    frames = sorted(df['frame'].unique())
    n_frames = len(frames)
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    result = np.full((n_frames, ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

    type_offsets = {'face': 0, 'left_hand': 468, 'pose': 489, 'right_hand': 522}

    # Vectorized for speed
    fr_indices = df['frame'].map(frame_to_idx).values
    types = df['type'].values
    lm_indices = df['landmark_index'].values.astype(int)
    x_vals = df['x'].values.astype(np.float32)
    y_vals = df['y'].values.astype(np.float32)
    z_vals = df['z'].values.astype(np.float32)

    for j in range(len(df)):
        offset = type_offsets.get(types[j])
        if offset is None:
            continue
        global_idx = offset + lm_indices[j]
        if global_idx < ROWS_PER_FRAME:
            fr = fr_indices[j]
            result[fr, global_idx, 0] = x_vals[j]
            result[fr, global_idx, 1] = y_vals[j]
            result[fr, global_idx, 2] = z_vals[j]

    return result


def main():
    parser = argparse.ArgumentParser(description="Parquet → NPZ pre-conversion")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='./asl_data.npz')
    parser.add_argument('--max-signs', type=int, default=50)
    parser.add_argument('--max-per-sign', type=int, default=999)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load metadata
    train_df = pd.read_csv(data_dir / 'train.csv')
    with open(data_dir / 'sign_to_prediction_index_map.json') as f:
        sign_map = json.load(f)

    print(f"📊 {len(train_df)} videos, {len(sign_map)} signs")

    # Select signs
    available = set(sign_map.keys())
    selected = [s for s in MVP_SIGNS if s in available]
    if len(selected) < args.max_signs:
        for s in train_df['sign'].value_counts().index:
            if s not in selected and len(selected) < args.max_signs:
                selected.append(s)
    selected = selected[:args.max_signs]
    label_to_idx = {s: i for i, s in enumerate(sorted(selected))}
    print(f"   {len(selected)} signs selected")

    # Filter
    filtered = train_df[train_df['sign'].isin(selected)].copy()
    parts = [filtered[filtered['sign'] == s].head(args.max_per_sign) for s in selected]
    filtered = pd.concat(parts)
    print(f"   {len(filtered)} videos to convert")

    # Split by participant (prevents data leakage)
    random.seed(42)
    participants = filtered['participant_id'].unique().tolist()
    random.shuffle(participants)
    n_p = len(participants)
    test_p = set(participants[:max(1, int(n_p * 0.15))])
    val_p = set(participants[max(1, int(n_p * 0.15)):max(2, int(n_p * 0.30))])

    # Convert
    all_data = []  # list of (array, label, split)
    errors = 0

    for i, (_, row) in enumerate(filtered.iterrows()):
        path = data_dir / row['path']
        if not path.exists():
            errors += 1
            continue

        try:
            arr = load_parquet(path)
            if arr is None or arr.shape[0] < 3:
                errors += 1
                continue

            label = label_to_idx[row['sign']]
            pid = row['participant_id']

            if pid in test_p:
                split = 2  # test
            elif pid in val_p:
                split = 1  # val
            else:
                split = 0  # train

            all_data.append((arr, label, split))

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   ⚠️  {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"   [{i+1}/{len(filtered)}] {len(all_data)} converted | Errors: {errors}")

    print(f"\n   Total: {len(all_data)} sequences | Errors: {errors}")

    # Save — arrays have variable length so we save them individually
    labels = np.array([d[1] for d in all_data], dtype=np.int32)
    splits = np.array([d[2] for d in all_data], dtype=np.int32)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences
    for idx, (arr, label, split) in enumerate(all_data):
        np.save(output_dir / f"seq_{idx:05d}.npy", arr)

    # Save index
    index = {
        'n_sequences': len(all_data),
        'labels': labels.tolist(),
        'splits': splits.tolist(),  # 0=train, 1=val, 2=test
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
        'num_classes': len(label_to_idx),
        'format': '(n_frames, 543, 3)',
    }
    with open(output_dir / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)

    # Stats
    train_count = sum(1 for s in splits if s == 0)
    val_count = sum(1 for s in splits if s == 1)
    test_count = sum(1 for s in splits if s == 2)
    print(f"\n✅ Saved to {output_dir}/")
    print(f"   Train: {train_count} | Val: {val_count} | Test: {test_count}")
    print(f"   {len(all_data)} .npy files + index.json")


if __name__ == "__main__":
    main()
