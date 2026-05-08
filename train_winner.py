"""
Sign2Speak — Entraînement basé sur la solution 1ère place Kaggle
================================================================
Adapté de: hoyso48/Google-Isolated-Sign-Language-Recognition-1st-place-solution
Licence: MIT

Architecture: Conv1D blocks + Transformer blocks
Preprocessing: normalisation nez + vélocité + accélération (708 features)
Landmarks: 118 (lèvres + mains + nez + yeux)

Usage:
    python train_winner.py --data-dir /mnt/c/Users/Ludo/dev/Hackathon/Sign2Speak-GEMMA-4-GOOD-HACKATHON/asl-signs --output ./model_winner --epochs 100 --max-signs 50
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

# =============================================================================
# Seed
# =============================================================================

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =============================================================================
# Landmarks (identiques au gagnant)
# =============================================================================

ROWS_PER_FRAME = 543
PAD = -100.0

NOSE = [1, 2, 98, 327]
LNOSE = [98]
RNOSE = [327]

LIP = [0,
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

LHAND = list(range(468, 489))
RHAND = list(range(522, 543))

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(POINT_LANDMARKS)  # 118
CHANNELS = 6 * NUM_NODES           # 708

# Signes MVP
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
# Chargement des Parquet → numpy (format 543 landmarks × 3)
# =============================================================================

def load_parquet_to_array(parquet_path):
    """
    Charge un Parquet long et retourne un array (n_frames, 543, 3).
    Format identique à ce que le gagnant utilise.
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return None

    frames = sorted(df['frame'].unique())
    n_frames = len(frames)
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    # Pré-allouer avec NaN (comme le dataset original)
    result = np.full((n_frames, ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

    # Mapping type → offset dans les 543 landmarks
    type_offsets = {
        'face': 0,        # 0-467
        'left_hand': 468,  # 468-488
        'pose': 489,       # 489-521 (33 landmarks, mais le dataset n'a que 0-32)
        'right_hand': 522, # 522-542
    }

    for _, row in df.iterrows():
        fr = frame_to_idx.get(row['frame'])
        if fr is None:
            continue
        lm_type = row['type']
        offset = type_offsets.get(lm_type)
        if offset is None:
            continue
        lm_idx = int(row['landmark_index'])
        global_idx = offset + lm_idx
        if global_idx < ROWS_PER_FRAME:
            result[fr, global_idx, 0] = row['x']
            result[fr, global_idx, 1] = row['y']
            result[fr, global_idx, 2] = row['z']

    return result

# =============================================================================
# Preprocessing (identique au gagnant)
# =============================================================================

def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims
    ) / tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims
    )

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

def preprocess_coordinates(coords):
    """
    Preprocessing du gagnant:
    1. Centre sur le nez (landmark 17)
    2. Normalise par std
    3. Sélectionne 118 landmarks
    4. Garde x, y seulement
    5. Ajoute vélocité et accélération
    → 708 features par frame
    """
    if tf.rank(coords) == 3:
        x = coords[None, ...]
    else:
        x = coords

    # Centre sur le nez
    mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
    mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)

    # Sélectionne les 118 landmarks
    x = tf.gather(x, POINT_LANDMARKS, axis=2)

    # Normalise
    std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
    x = (x - mean) / std

    # Garde x, y seulement
    length = tf.shape(x)[1]
    x = x[..., :2]

    # Vélocité (frame[t+1] - frame[t])
    dx = tf.cond(
        tf.shape(x)[1] > 1,
        lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0,0],[0,1],[0,0],[0,0]]),
        lambda: tf.zeros_like(x)
    )

    # Accélération (frame[t+2] - frame[t])
    dx2 = tf.cond(
        tf.shape(x)[1] > 2,
        lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0,0],[0,2],[0,0],[0,0]]),
        lambda: tf.zeros_like(x)
    )

    x = tf.concat([
        tf.reshape(x, (-1, length, 2 * NUM_NODES)),
        tf.reshape(dx, (-1, length, 2 * NUM_NODES)),
        tf.reshape(dx2, (-1, length, 2 * NUM_NODES)),
    ], axis=-1)

    x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)

    return x[0]  # remove batch dim

# =============================================================================
# Augmentations (du gagnant)
# =============================================================================

def interp1d_(x, target_len):
    target_len = tf.maximum(1, target_len)
    x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
    return x

def resample(x, rate=(0.8, 1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x

def flip_lr(x):
    x_coord, y, z = tf.unstack(x, axis=-1)
    x_coord = 1 - x_coord
    new_x = tf.stack([x_coord, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    # Swap hands
    lhand = tf.gather(new_x, LHAND, axis=0)
    rhand = tf.gather(new_x, RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND)[..., None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND)[..., None], lhand)
    # Swap lips
    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP)[..., None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP)[..., None], llip)
    # Swap eyes
    leye = tf.gather(new_x, LEYE, axis=0)
    reye = tf.gather(new_x, REYE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LEYE)[..., None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(REYE)[..., None], leye)
    # Swap nose
    lnose = tf.gather(new_x, LNOSE, axis=0)
    rnose = tf.gather(new_x, RNOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LNOSE)[..., None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RNOSE)[..., None], lnose)
    new_x = tf.transpose(new_x, [1, 0, 2])
    return new_x

def spatial_random_affine(xyz, scale=(0.8,1.2), shear=(-0.15,0.15),
                           shift=(-0.1,0.1), degree=(-30,30)):
    center = tf.constant([0.5, 0.5])
    if scale is not None:
        s = tf.random.uniform((), *scale)
        xyz = s * xyz
    if shear is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = tf.identity([[1., shear_x], [shear_y, 1.]])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
        xyz = tf.concat([xy, z], axis=-1)
    if degree is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        xy -= center
        deg = tf.random.uniform((), *degree)
        radian = deg / 180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([[c, s], [-s, c]])
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = tf.concat([xy, z], axis=-1)
    if shift is not None:
        sh = tf.random.uniform((), *shift)
        xyz = xyz + sh
    return xyz

def temporal_crop(x, length):
    l = tf.shape(x)[0]
    offset = tf.random.uniform((), 0, tf.clip_by_value(l - length, 1, length), dtype=tf.int32)
    x = x[offset:offset + length]
    return x

def temporal_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    l = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l - mask_size, 1, l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(
        x, tf.range(mask_offset, mask_offset + mask_size)[..., None],
        tf.fill([mask_size, 543, 3], mask_value)
    )
    return x

def spatial_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x < x[..., 0]) & (x[..., 0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[..., 1]) & (x[..., 1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[..., None], mask_value, x)
    return x

def augment_fn(x, max_len=None):
    if tf.random.uniform(()) < 0.8:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.5:
        x = flip_lr(x)
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if tf.random.uniform(()) < 0.75:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.5:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.5:
        x = spatial_mask(x)
    return x

# =============================================================================
# Modèle (identique au gagnant)
# =============================================================================

class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False,
                 depthwise_initializer='glorot_uniform', name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size, strides=1, dilation_rate=dilation_rate,
            padding='valid', use_bias=use_bias,
            depthwise_initializer=depthwise_initializer, name=name + '_dwconv'
        )
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0,
                expand_ratio=2, se_ratio=0.25, activation='swish', name=None):
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(channels_expand, use_bias=True, activation=activation, name=name + '_expand')(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, use_bias=False, name=name + '_conv')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)
        x = ECA(name=name + '_eca')(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True, name=name + '_project')(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + '_drop')(x)

        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x
    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :]
        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x

def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(inputs)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def get_model(num_classes, max_len=64, dim=192):
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD, input_shape=(max_len, CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim * 2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes, name='classifier')(x)
    return tf.keras.Model(inp, x)

# =============================================================================
# Dataset tf.data
# =============================================================================

def create_dataset(parquet_paths, labels, num_classes, max_len=64,
                   augment=False, batch_size=64, shuffle=False):
    """Crée un tf.data.Dataset depuis les paths Parquet."""

    def generator():
        indices = list(range(len(parquet_paths)))
        if shuffle:
            random.shuffle(indices)
        for i in indices:
            path = parquet_paths[i]
            label = labels[i]
            try:
                arr = load_parquet_to_array(Path(path))
                if arr is None or arr.shape[0] < 3:
                    continue
                yield arr, label
            except Exception:
                continue

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, ROWS_PER_FRAME, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    def process(coords, label):
        # Filter nan frames
        mask = tf.math.logical_not(
            tf.reduce_all(tf.math.is_nan(
                tf.gather(coords, POINT_LANDMARKS, axis=1)
            ), axis=[-2, -1])
        )
        coords = tf.boolean_mask(coords, mask, axis=0)

        if augment:
            coords = augment_fn(coords, max_len=max_len)

        coords = tf.ensure_shape(coords, (None, ROWS_PER_FRAME, 3))
        features = preprocess_coordinates(coords)

        # Truncate / pad
        length = tf.shape(features)[0]
        if length > max_len:
            features = features[:max_len]
        
        return tf.cast(features, tf.float32), tf.one_hot(label, num_classes)

    ds = ds.map(process, tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.padded_batch(
        batch_size,
        padding_values=(PAD, 0.0),
        padded_shapes=([max_len, CHANNELS], [num_classes]),
        drop_remainder=True,
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Entraînement architecture gagnante Kaggle")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Chemin vers le dossier asl-signs (avec train.csv et train_landmark_files/)')
    parser.add_argument('--output', type=str, default='./model_winner')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--max-signs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=192)

    args = parser.parse_args()
    seed_everything(42)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- GPU setup ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU détecté: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  Pas de GPU, entraînement sur CPU (lent)")

    # --- Charger metadata ---
    train_df = pd.read_csv(data_dir / 'train.csv')
    with open(data_dir / 'sign_to_prediction_index_map.json') as f:
        sign_map = json.load(f)

    print(f"📊 Dataset: {len(train_df)} vidéos, {len(sign_map)} signes")

    # --- Sélectionner les signes ---
    available = set(sign_map.keys())
    selected = [s for s in MVP_SIGNS if s in available]
    if len(selected) < args.max_signs:
        for s in train_df['sign'].value_counts().index:
            if s not in selected and len(selected) < args.max_signs:
                selected.append(s)
    selected = selected[:args.max_signs]

    # Créer un nouveau mapping d'indices
    label_to_idx = {s: i for i, s in enumerate(sorted(selected))}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(label_to_idx)
    print(f"   {num_classes} signes sélectionnés")

    # --- Filtrer et splitter ---
    filtered = train_df[train_df['sign'].isin(selected)].copy()
    filtered['label'] = filtered['sign'].map(label_to_idx)

    # Split par participant
    random.seed(42)
    participants = filtered['participant_id'].unique().tolist()
    random.shuffle(participants)
    n_p = len(participants)
    test_p = set(participants[:max(1, int(n_p * 0.15))])
    val_p = set(participants[max(1, int(n_p * 0.15)):max(2, int(n_p * 0.30))])
    train_p = set(participants) - test_p - val_p

    train_df_split = filtered[filtered['participant_id'].isin(train_p)]
    val_df_split = filtered[filtered['participant_id'].isin(val_p)]
    test_df_split = filtered[filtered['participant_id'].isin(test_p)]

    print(f"   Train: {len(train_df_split)} | Val: {len(val_df_split)} | Test: {len(test_df_split)}")

    # Paths complets
    train_paths = [str(data_dir / p) for p in train_df_split['path'].values]
    train_labels = train_df_split['label'].values.tolist()
    val_paths = [str(data_dir / p) for p in val_df_split['path'].values]
    val_labels = val_df_split['label'].values.tolist()
    test_paths = [str(data_dir / p) for p in test_df_split['path'].values]
    test_labels = test_df_split['label'].values.tolist()

    # --- Datasets ---
    print(f"\n🔧 Création des datasets (max_len={args.max_len}, features={CHANNELS})...")
    train_ds = create_dataset(train_paths, train_labels, num_classes,
                               max_len=args.max_len, augment=True,
                               batch_size=args.batch_size, shuffle=True)
    val_ds = create_dataset(val_paths, val_labels, num_classes,
                             max_len=args.max_len, augment=False,
                             batch_size=args.batch_size, shuffle=False)

    # Test rapide
    print("   Test du pipeline...")
    for x, y in train_ds.take(1):
        print(f"   Batch shape: x={x.shape}, y={y.shape}")
        break

    # --- Modèle ---
    print(f"\n🏗️  Construction du modèle (dim={args.dim})...")
    model = get_model(num_classes, max_len=args.max_len, dim=args.dim)
    model.summary()

    # --- Compilation ---
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.epochs * len(train_paths) // args.batch_size,
        alpha=1e-6,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['categorical_accuracy'],
    )

    # --- Callbacks ---
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / 'best_model.keras'),
            monitor='val_categorical_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / 'training_log.csv')),
    ]

    # --- Entraînement ---
    print(f"\n🚀 Entraînement ({args.epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Évaluation ---
    print(f"\n📊 Évaluation sur le test set...")
    test_ds = create_dataset(test_paths, test_labels, num_classes,
                              max_len=args.max_len, augment=False,
                              batch_size=args.batch_size, shuffle=False)
    results = model.evaluate(test_ds, verbose=0)
    print(f"   Test loss: {results[0]:.4f}")
    print(f"   Test accuracy: {results[1]:.1%}")

    # --- Sauvegarder ---
    model.save(str(output_dir / 'final_model.keras'))

    meta = {
        'num_classes': num_classes,
        'max_len': args.max_len,
        'channels': CHANNELS,
        'num_nodes': NUM_NODES,
        'dim': args.dim,
        'point_landmarks': POINT_LANDMARKS,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
        'test_accuracy': float(results[1]),
    }
    with open(output_dir / 'model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Sauvegardé dans {output_dir}/")
    print(f"   Pour tester en live (sur Windows): adapter le preprocessing dans live_test.py")


if __name__ == "__main__":
    main()
