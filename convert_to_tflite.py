"""
convert_to_tflite.py — Convertit le modèle gagnant Keras vers TFLite (.tflite)
pour l'application Android Sign2Speak.

Usage:
    python convert_to_tflite.py

Sortie:
    On_device/app/src/main/assets/classifier.tflite       (float32 + SELECT_TF_OPS)
    On_device/app/src/main/assets/classifier_int8.tflite  (quantifié INT8, plus léger)
    On_device/app/src/main/assets/model_meta.json         (copie des métadonnées)

Pré-requis:
    pip install tensorflow>=2.16.0
"""

import json
import shutil
import os
import numpy as np
from pathlib import Path

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODEL_DIR  = ROOT_DIR / "model_winner"
ASSETS_DIR = ROOT_DIR / "On_device" / "app" / "src" / "main" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Métadonnées ─────────────────────────────────────────────────────────────
with open(MODEL_DIR / "model_meta.json") as f:
    meta = json.load(f)

num_classes = meta["num_classes"]   # 50
max_len     = meta["max_len"]       # 64
CHANNELS    = meta.get("channels", 708)
dim         = meta.get("dim", 192)

print(f"num_classes={num_classes}, max_len={max_len}, channels={CHANNELS}, dim={dim}")

# ─── Architecture (identique à live_winner.py) ───────────────────────────────
import tensorflow as tf

class ECA(tf.keras.layers.Layer):
    """Efficient Channel Attention."""
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(
            1, kernel_size=kernel_size, padding="same", use_bias=False
        )

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        return inputs * nn[:, None, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg


class CausalDWConv1D(tf.keras.layers.Layer):
    """Causal depthwise convolution 1D."""
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias      = use_bias
        self.causal_pad    = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0)
        )
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size, strides=1,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=use_bias,
        )

    def call(self, inputs):
        return self.dw_conv(self.causal_pad(inputs))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "kernel_size":   self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "use_bias":      self.use_bias,
        })
        return cfg


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention."""
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dim       = dim
        self.scale     = dim ** -0.5
        self.num_heads = num_heads
        self.dropout   = dropout
        self.qkv  = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape(
                (-1, self.num_heads, self.dim * 3 // self.num_heads)
            )(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None:
            attn = tf.where(mask[:, None, None, :], attn, -1e9)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop(attn)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2, 1, 3))(x)
        )
        return self.proj(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "dim":       self.dim,
            "num_heads": self.num_heads,
            "dropout":   self.dropout,
        })
        return cfg


def Conv1DBlock(ch, ks, drop_rate=0.0, expand_ratio=2, activation="swish", name=None):
    if name is None:
        name = str(tf.keras.backend.get_uid("cb"))

    def apply(inputs):
        skip = inputs
        x = tf.keras.layers.Dense(
            tf.keras.backend.int_shape(inputs)[-1] * expand_ratio,
            activation=activation, name=name + "_exp"
        )(inputs)
        x = CausalDWConv1D(ks, name=name + "_dw")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + "_bn")(x)
        x = ECA(name=name + "_eca")(x)
        x = tf.keras.layers.Dense(ch, name=name + "_proj")(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        if tf.keras.backend.int_shape(inputs)[-1] == ch:
            x = tf.keras.layers.add([x, skip])
        return x

    return apply


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2,
                     drop_rate=0.2, activation="swish"):
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


# ─── Construction du modèle ──────────────────────────────────────────────────
print("Reconstruction de l'architecture...")
ks  = 17
inp = tf.keras.Input((max_len, CHANNELS))
x   = tf.keras.layers.Masking(mask_value=-100.0)(inp)
x   = tf.keras.layers.Dense(dim, use_bias=False, name="stem")(x)
x   = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
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

# Chargement des poids
weights_path = str(MODEL_DIR / "weights.weights.h5")
print(f"Chargement des poids depuis {weights_path}...")
model.load_weights(weights_path)
print(f"Modèle chargé : {model.count_params():,} paramètres")

# ─── Test rapide de sanity check ─────────────────────────────────────────────
dummy = np.random.randn(1, max_len, CHANNELS).astype(np.float32)
out   = model(dummy, training=False).numpy()
print(f"Sanity check : shape_out={out.shape}, max_prob={float(np.max(tf.nn.softmax(out[0]))):.3f}")

# ─── Conversion TFLite float32 + SELECT_TF_OPS ───────────────────────────────
print("\nConversion TFLite (float32 + SELECT_TF_OPS)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

out_path = ASSETS_DIR / "classifier.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)
size_mb = os.path.getsize(out_path) / 1e6
print(f"  → {out_path}  ({size_mb:.1f} Mo)")

# ─── Conversion TFLite INT8 quantifié ────────────────────────────────────────
print("\nConversion TFLite INT8 (représentative dataset)...")

# Dataset représentatif : bruit gaussien + séquences paddées réalistes
def representative_dataset_gen():
    # Quelques patterns réalistes
    rng = np.random.default_rng(42)
    for _ in range(100):
        seq = rng.standard_normal((1, max_len, CHANNELS)).astype(np.float32)
        # Simuler du padding
        pad_len = rng.integers(0, 30)
        seq[0, max_len - pad_len:] = -100.0
        yield [seq]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter_int8.inference_input_type  = tf.float32   # garder float32 en entrée
converter_int8.inference_output_type = tf.float32   # garder float32 en sortie
converter_int8._experimental_lower_tensor_list_ops = False

try:
    tflite_int8 = converter_int8.convert()
    out_int8 = ASSETS_DIR / "classifier_int8.tflite"
    with open(out_int8, "wb") as f:
        f.write(tflite_int8)
    size_int8 = os.path.getsize(out_int8) / 1e6
    print(f"  → {out_int8}  ({size_int8:.1f} Mo)")
except Exception as e:
    print(f"  [WARN] Quantification INT8 échouée : {e}")
    print("  Le modèle float32 sera utilisé à la place.")

# ─── Copie model_meta.json ───────────────────────────────────────────────────
meta_dst = ASSETS_DIR / "model_meta.json"
shutil.copy(MODEL_DIR / "model_meta.json", meta_dst)
print(f"\nCopie model_meta.json → {meta_dst}")

# ─── Vérification rapide du TFLite ───────────────────────────────────────────
print("\nVérification du modèle TFLite...")
interp = tf.lite.Interpreter(model_path=str(out_path))
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"  Input  : {inp_det['shape']}  dtype={inp_det['dtype']}")
print(f"  Output : {out_det['shape']}  dtype={out_det['dtype']}")

test_in = dummy.copy()
interp.set_tensor(inp_det['index'], test_in)
interp.invoke()
tflite_out = interp.get_tensor(out_det['index'])
print(f"  Prédiction TFLite : classe={int(np.argmax(tflite_out[0]))}")

print("\nConversion terminée avec succès.")
print(f"Assets disponibles dans : {ASSETS_DIR}")
print()
print("RAPPEL — Placez aussi dans On_device/app/src/main/assets/ :")
print("  hand_landmarker.task  (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)")
print("  face_landmarker.task  (https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)")
print()
print("Pour Gemma 4 (LLM), placez le fichier .gguf sur l'appareil Android dans /sdcard/gemma4-2b-it-q4.gguf")
