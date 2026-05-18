"""
Sign2Speak — Entraînement architecture gagnante (v2 — données pré-converties)
==============================================================================
Lit les .npy pré-convertis (format 543×3 par frame) depuis asl_converted/.
Plus de lecture Parquet = entraînement instantané.

Usage:
    python3 train_winner.py --data-dir ./asl_converted --output ./model_winner --epochs 100
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
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
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)  # 118
CHANNELS = 6 * NUM_NODES           # 708

# =============================================================================
# Preprocessing (identique au gagnant)
# =============================================================================
def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / \
           tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

def preprocess(coords):
    x = coords[None, ...] if tf.rank(coords) == 3 else coords
    mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
    mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
    x = tf.gather(x, POINT_LANDMARKS, axis=2)
    std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
    x = (x - mean) / std
    length = tf.shape(x)[1]
    x = x[..., :2]
    dx = tf.cond(tf.shape(x)[1] > 1,
                 lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0,0],[0,1],[0,0],[0,0]]),
                 lambda: tf.zeros_like(x))
    dx2 = tf.cond(tf.shape(x)[1] > 2,
                  lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0,0],[0,2],[0,0],[0,0]]),
                  lambda: tf.zeros_like(x))
    x = tf.concat([tf.reshape(x, (-1, length, 2*NUM_NODES)),
                    tf.reshape(dx, (-1, length, 2*NUM_NODES)),
                    tf.reshape(dx2, (-1, length, 2*NUM_NODES))], axis=-1)
    x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
    return x[0]

# =============================================================================
# Augmentations (du gagnant)
# =============================================================================
def interp1d_(x, target_len):
    target_len = tf.maximum(1, target_len)
    return tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')

def resample(x, rate=(0.8, 1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    new_size = tf.cast(rate * tf.cast(tf.shape(x)[0], tf.float32), tf.int32)
    return interp1d_(x, new_size)

def flip_lr(x):
    x_c, y, z = tf.unstack(x, axis=-1)
    x_c = 1 - x_c
    new_x = tf.stack([x_c, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    for src, dst in [(LHAND, RHAND), (LLIP, RLIP), (LEYE, REYE), (LNOSE, RNOSE)]:
        a = tf.gather(new_x, src, axis=0)
        b = tf.gather(new_x, dst, axis=0)
        new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(src)[..., None], b)
        new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(dst)[..., None], a)
    return tf.transpose(new_x, [1, 0, 2])

def spatial_random_affine(xyz):
    s = tf.random.uniform((), 0.8, 1.2)
    xyz = s * xyz
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    deg = tf.random.uniform((), -30., 30.)
    r = deg / 180 * 3.14159
    c, sn = tf.math.cos(r), tf.math.sin(r)
    rot = tf.identity([[c, sn], [-sn, c]])
    xy = (xy - 0.5) @ rot + 0.5
    sh = tf.random.uniform((), -0.1, 0.1)
    return tf.concat([xy + sh, z], axis=-1)

def temporal_mask(x):
    l = tf.shape(x)[0]
    ms = tf.cast(tf.cast(l, tf.float32) * tf.random.uniform((), 0.2, 0.4), tf.int32)
    mo = tf.random.uniform((), 0, tf.clip_by_value(l - ms, 1, l), dtype=tf.int32)
    return tf.tensor_scatter_nd_update(x, tf.range(mo, mo + ms)[..., None],
                                        tf.fill([ms, 543, 3], float('nan')))

def augment_fn(x, max_len):
    if tf.random.uniform(()) < 0.8:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.5:
        x = flip_lr(x)
    l = tf.shape(x)[0]
    if l > max_len:
        offset = tf.random.uniform((), 0, l - max_len, dtype=tf.int32)
        x = x[offset:offset + max_len]
    if tf.random.uniform(()) < 0.75:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.5:
        x = temporal_mask(x)
    return x

# =============================================================================
# Modèle (identique au gagnant)
# =============================================================================
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False)
    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        return inputs * nn[:, None, :]

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False, name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(kernel_size, strides=1,
                       dilation_rate=dilation_rate, padding='valid', use_bias=use_bias)
        self.supports_masking = True
    def call(self, inputs):
        return self.dw_conv(self.causal_pad(inputs))

def Conv1DBlock(ch, ks, drop_rate=0.0, expand_ratio=2, activation='swish', name=None):
    if name is None: name = str(tf.keras.backend.get_uid("cb"))
    def apply(inputs):
        skip = inputs
        x = tf.keras.layers.Dense(tf.keras.backend.int_shape(inputs)[-1] * expand_ratio,
                                   activation=activation, name=name+'_exp')(inputs)
        x = CausalDWConv1D(ks, name=name+'_dw')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name+'_bn')(x)
        x = ECA(name=name+'_eca')(x)
        x = tf.keras.layers.Dense(ch, name=name+'_proj')(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        if tf.keras.backend.int_shape(inputs)[-1] == ch:
            x = tf.keras.layers.add([x, skip])
        return x
    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True
    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2,1,3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim*3//self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim//self.num_heads]*3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None:
            attn = tf.where(mask[:, None, None, :], attn, -1e9)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop(attn)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2,1,3))(x))
        return self.proj(x)

def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(inputs)
        x = MultiHeadSelfAttention(dim, num_heads, attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        return tf.keras.layers.Add()([attn_out, x])
    return apply

def get_model(num_classes, max_len=64, dim=192):
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD)(inp)
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
    ks = 17
    x = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)
    x = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ks, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)
    x = tf.keras.layers.Dense(dim * 2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes, name='classifier')(x)
    return tf.keras.Model(inp, x)

# =============================================================================
# Dataset depuis .npy pré-convertis
# =============================================================================
def create_dataset(npy_dir, indices, labels, num_classes, max_len=64,
                   augment=False, batch_size=64, shuffle=False):
    paths = [str(npy_dir / f"seq_{i:05d}.npy") for i in indices]

    def generator():
        order = list(range(len(paths)))
        if shuffle:
            random.shuffle(order)
        for i in order:
            try:
                arr = np.load(paths[i])
                if arr.shape[0] < 3:
                    continue
                yield arr.astype(np.float32), labels[i]
            except:
                continue

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((None, ROWS_PER_FRAME, 3), tf.float32),
            tf.TensorSpec((), tf.int32),
        ),
    )

    def process(coords, label):
        # Filtrer les frames entièrement NaN
        pts = tf.gather(coords, POINT_LANDMARKS, axis=1)
        mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(pts), axis=[-2, -1]))
        coords = tf.boolean_mask(coords, mask, axis=0)
        coords = tf.ensure_shape(coords, (None, ROWS_PER_FRAME, 3))

        if augment:
            coords = augment_fn(coords, max_len)

        features = preprocess(coords)
        # Truncate
        features = features[:max_len]
        return tf.cast(features, tf.float32), tf.one_hot(label, num_classes)

    ds = ds.map(process, tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(512)
    ds = ds.padded_batch(batch_size, padding_values=(PAD, 0.0),
                          padded_shapes=([max_len, CHANNELS], [num_classes]),
                          drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Dossier asl_converted/')
    parser.add_argument('--output', type=str, default='./model_winner')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=192)
    args = parser.parse_args()

    seed_everything(42)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU: {gpus}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print("⚠️  CPU only")

    # Index
    with open(data_dir / 'index.json') as f:
        index = json.load(f)

    num_classes = index['num_classes']
    labels = index['labels']
    splits = index['splits']
    idx_to_label = index['idx_to_label']
    label_to_idx = index['label_to_idx']

    train_idx = [i for i, s in enumerate(splits) if s == 0]
    val_idx = [i for i, s in enumerate(splits) if s == 1]
    test_idx = [i for i, s in enumerate(splits) if s == 2]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"📊 {num_classes} classes | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Datasets
    print(f"🔧 Features: {CHANNELS} | Max len: {args.max_len}")
    train_ds = create_dataset(data_dir, train_idx, train_labels, num_classes,
                               args.max_len, augment=True, batch_size=args.batch_size, shuffle=True)
    val_ds = create_dataset(data_dir, val_idx, val_labels, num_classes,
                             args.max_len, augment=False, batch_size=args.batch_size)

    # Test pipeline
    for x, y in train_ds.take(1):
        print(f"   Batch: x={x.shape}, y={y.shape}")
        break

    # Modèle
    model = get_model(num_classes, args.max_len, args.dim)
    n_params = sum(p.numpy().size for p in model.trainable_variables)
    print(f"🏗️  Modèle: {n_params:,} paramètres")

    steps = args.epochs * len(train_idx) // args.batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, steps, alpha=1e-6)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['categorical_accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output_dir / 'best_model.keras'),
                                            monitor='val_categorical_accuracy',
                                            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                          patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(output_dir / 'training_log.csv')),
    ]

    print(f"\n🚀 Entraînement ({args.epochs} epochs)...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    # Test
    print(f"\n📊 Test...")
    test_ds = create_dataset(data_dir, test_idx, test_labels, num_classes,
                              args.max_len, augment=False, batch_size=args.batch_size)
    results = model.evaluate(test_ds, verbose=0)
    print(f"   Test accuracy: {results[1]:.1%}")

    # Save
    model.save(str(output_dir / 'final_model.keras'))
    meta = {
        'num_classes': num_classes, 'max_len': args.max_len,
        'channels': CHANNELS, 'num_nodes': NUM_NODES, 'dim': args.dim,
        'point_landmarks': POINT_LANDMARKS,
        'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label,
        'test_accuracy': float(results[1]),
    }
    with open(output_dir / 'model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Sauvegardé dans {output_dir}/")

if __name__ == "__main__":
    main()