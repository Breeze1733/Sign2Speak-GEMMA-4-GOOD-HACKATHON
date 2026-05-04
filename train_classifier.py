"""
Sign2Speak — Étape 2 : Classifieur LSTM de signes ASL
======================================================
Entraîne un LSTM bidirectionnel sur les keypoints extraits
pour reconnaître les signes ASL en temps réel.

Pipeline:
    keypoints/*.npy → padding/truncation → LSTM BiDir → softmax → label du signe

Usage:
    # Entraînement standard
    python train_classifier.py --data ./keypoints --output ./model

    # Entraînement rapide (test)
    python train_classifier.py --data ./keypoints --output ./model --epochs 10 --quick

    # Reprendre un entraînement
    python train_classifier.py --data ./keypoints --output ./model --resume ./model/checkpoint.keras
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MAX_SEQUENCE_LENGTH = 60    # frames max par séquence (padding/truncation)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50


# =============================================================================
# Chargement des données
# =============================================================================

def load_dataset(data_dir: Path, split: str, label_to_idx: dict, max_seq_len: int = MAX_SEQUENCE_LENGTH):
    """
    Charge tous les .npy d'un split et retourne X, y.
    
    Args:
        data_dir: dossier keypoints/
        split: 'train', 'val', ou 'test'
        label_to_idx: mapping {gloss: index}
        max_seq_len: longueur de séquence cible
    
    Returns:
        X: np.array (n_samples, max_seq_len, n_features)
        y: np.array (n_samples,) — indices des classes
        sample_paths: liste des chemins pour debug
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        print(f"⚠️  Dossier {split_dir} introuvable")
        return np.array([]), np.array([]), []
    
    X_list = []
    y_list = []
    paths = []
    skipped = 0
    
    for sign_dir in sorted(split_dir.iterdir()):
        if not sign_dir.is_dir():
            continue
        
        sign_name = sign_dir.name
        if sign_name not in label_to_idx:
            continue
        
        label_idx = label_to_idx[sign_name]
        
        for npy_file in sorted(sign_dir.glob("*.npy")):
            try:
                sequence = np.load(npy_file)  # (n_frames, n_features)
                
                if sequence.ndim != 2 or sequence.shape[0] == 0:
                    skipped += 1
                    continue
                
                # Padding / Truncation
                n_frames, n_features = sequence.shape
                
                if n_frames > max_seq_len:
                    # Truncation: garder le milieu (le signe est souvent centré)
                    start = (n_frames - max_seq_len) // 2
                    sequence = sequence[start:start + max_seq_len]
                elif n_frames < max_seq_len:
                    # Padding: zéros à la fin
                    pad = np.zeros((max_seq_len - n_frames, n_features), dtype=np.float32)
                    sequence = np.concatenate([sequence, pad], axis=0)
                
                X_list.append(sequence)
                y_list.append(label_idx)
                paths.append(str(npy_file))
                
            except Exception as e:
                skipped += 1
                continue
    
    if not X_list:
        return np.array([]), np.array([]), []
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"   {split}: {X.shape[0]} séquences, shape {X.shape}, "
          f"{len(set(y_list))} classes, {skipped} skippés")
    
    return X, y, paths


# =============================================================================
# Data Augmentation
# =============================================================================

def augment_sequence(sequence: np.ndarray, noise_std: float = 0.02, 
                      time_shift_max: int = 5, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """
    Augmentation de données pour les séquences de keypoints.
    Appliquée uniquement pendant l'entraînement.
    """
    augmented = sequence.copy()
    
    # 1. Bruit gaussien léger
    noise = np.random.normal(0, noise_std, augmented.shape).astype(np.float32)
    augmented += noise
    
    # 2. Décalage temporel (shift aléatoire)
    if time_shift_max > 0:
        shift = np.random.randint(-time_shift_max, time_shift_max + 1)
        if shift > 0:
            augmented = np.concatenate([np.zeros((shift, augmented.shape[1]), dtype=np.float32),
                                         augmented[:-shift]], axis=0)
        elif shift < 0:
            augmented = np.concatenate([augmented[-shift:],
                                         np.zeros((-shift, augmented.shape[1]), dtype=np.float32)], axis=0)
    
    # 3. Scale aléatoire
    scale = np.random.uniform(*scale_range)
    augmented *= scale
    
    return augmented


class AugmentedDataGenerator(keras.utils.Sequence):
    """Générateur avec augmentation à la volée."""
    
    def __init__(self, X, y, batch_size=BATCH_SIZE, augment=True, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices].copy()
        batch_y = self.y[batch_indices]
        
        if self.augment:
            for i in range(len(batch_X)):
                batch_X[i] = augment_sequence(batch_X[i])
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# Modèle LSTM
# =============================================================================

def build_model(n_features: int, n_classes: int, max_seq_len: int = MAX_SEQUENCE_LENGTH) -> keras.Model:
    """
    LSTM bidirectionnel avec attention simple.
    
    Architecture:
        Input (max_seq_len, n_features)
        → Masking (ignore les frames paddées)
        → LSTM BiDir 128 (return_sequences)
        → LSTM BiDir 64
        → Dropout 0.3
        → Dense 128 ReLU
        → Dropout 0.3
        → Dense n_classes Softmax
    """
    inputs = layers.Input(shape=(max_seq_len, n_features), name="keypoints_input")
    
    # Masking: ignore les frames de padding (tous des zéros)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # LSTM bidirectionnel
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2"
    )(x)
    
    # Classifier head
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', name="fc_1")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax', name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Sign2Speak_LSTM")
    return model


# =============================================================================
# Entraînement
# =============================================================================

def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    quick: bool = False,
    resume: str = None,
):
    """Entraîne le classifieur."""
    
    # Charger le mapping des labels
    labels_file = data_dir / "labels.json"
    if not labels_file.exists():
        print(f"❌ {labels_file} introuvable. Lance d'abord dataset_pipeline.py --extract")
        return
    
    with open(labels_file) as f:
        label_to_idx = json.load(f)
    
    n_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print(f"📊 {n_classes} classes (signes)")
    
    # Charger les données
    print(f"\n📂 Chargement des données...")
    max_seq_len = 30 if quick else MAX_SEQUENCE_LENGTH
    
    X_train, y_train, _ = load_dataset(data_dir, 'train', label_to_idx, max_seq_len)
    X_val, y_val, _ = load_dataset(data_dir, 'val', label_to_idx, max_seq_len)
    X_test, y_test, _ = load_dataset(data_dir, 'test', label_to_idx, max_seq_len)
    
    if X_train.size == 0:
        print("❌ Aucune donnée d'entraînement trouvée")
        return
    
    n_features = X_train.shape[2]
    print(f"\n📐 Dimensions:")
    print(f"   Features/frame: {n_features}")
    print(f"   Séquence max: {max_seq_len} frames")
    print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # Vérifier la distribution des classes
    train_counts = Counter(y_train)
    min_count = min(train_counts.values())
    max_count = max(train_counts.values())
    print(f"   Samples/classe: min={min_count}, max={max_count}")
    
    # Class weights pour gérer le déséquilibre
    total = len(y_train)
    class_weights = {
        idx: total / (n_classes * count)
        for idx, count in train_counts.items()
    }
    
    # Construire le modèle
    print(f"\n🏗️  Construction du modèle...")
    if resume:
        model = keras.models.load_model(resume)
        print(f"   Modèle chargé depuis {resume}")
    else:
        model = build_model(n_features, n_classes, max_seq_len)
    
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    # Callbacks
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]
    
    # Data generators
    train_gen = AugmentedDataGenerator(X_train, y_train, batch_size, augment=True)
    val_gen = AugmentedDataGenerator(X_val, y_val, batch_size, augment=False, shuffle=False)
    
    # Entraînement
    print(f"\n🚀 Entraînement ({epochs} epochs)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Évaluation sur le test set
    print(f"\n📊 Évaluation sur le test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test loss: {test_loss:.4f}")
    
    # Top-5 accuracy
    predictions = model.predict(X_test, verbose=0)
    top5_correct = 0
    for i in range(len(y_test)):
        top5_indices = np.argsort(predictions[i])[-5:]
        if y_test[i] in top5_indices:
            top5_correct += 1
    top5_acc = top5_correct / len(y_test)
    print(f"   Top-5 accuracy: {top5_acc:.4f}")
    
    # Matrice de confusion — top erreurs
    from collections import defaultdict
    errors = defaultdict(int)
    for i in range(len(y_test)):
        pred = np.argmax(predictions[i])
        if pred != y_test[i]:
            true_label = idx_to_label[y_test[i]]
            pred_label = idx_to_label[pred]
            errors[f"{true_label} → {pred_label}"] += 1
    
    if errors:
        print(f"\n   Top confusions:")
        for pair, count in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f"      {pair}: {count}")
    
    # Sauvegarder le modèle final
    model.save(str(output_dir / "final_model.keras"))
    
    # Sauvegarder les métadonnées du modèle
    model_meta = {
        'n_classes': n_classes,
        'n_features': n_features,
        'max_seq_len': max_seq_len,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'test_accuracy': float(test_acc),
        'top5_accuracy': float(top5_acc),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
    }
    with open(output_dir / "model_meta.json", 'w') as f:
        json.dump(model_meta, f, indent=2)
    
    print(f"\n💾 Modèle sauvegardé dans {output_dir}/")
    
    # Export TFLite
    export_tflite(model, output_dir, max_seq_len, n_features)
    
    return model, history


# =============================================================================
# Export TFLite (pour mobile)
# =============================================================================

def export_tflite(model: keras.Model, output_dir: Path, max_seq_len: int, n_features: int):
    """Exporte le modèle en TFLite pour déploiement mobile."""
    print(f"\n📱 Export TFLite...")
    
    try:
        # Conversion standard
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        tflite_path = output_dir / "sign_classifier.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   ✅ {tflite_path} ({len(tflite_model) / 1e6:.1f} Mo)")
        
        # Version quantifiée (plus petite, plus rapide sur mobile)
        converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_q.target_spec.supported_types = [tf.float16]
        tflite_q = converter_q.convert()
        
        tflite_q_path = output_dir / "sign_classifier_fp16.tflite"
        with open(tflite_q_path, 'wb') as f:
            f.write(tflite_q)
        print(f"   ✅ {tflite_q_path} ({len(tflite_q) / 1e6:.1f} Mo) [quantifié fp16]")
        
    except Exception as e:
        print(f"   ⚠️  Erreur export TFLite: {e}")
        print(f"   Le modèle .keras reste utilisable, l'export TFLite pourra être refait.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Entraînement classifieur ASL")
    parser.add_argument('--data', type=str, default='./keypoints',
                        help='Dossier des keypoints extraits')
    parser.add_argument('--output', type=str, default='./model',
                        help='Dossier de sortie du modèle')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Nombre d\'epochs (défaut: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (défaut: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (défaut: {LEARNING_RATE})')
    parser.add_argument('--quick', action='store_true',
                        help='Mode rapide (séquences courtes, pour tester)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Chemin vers un modèle .keras à reprendre')
    
    args = parser.parse_args()
    
    # Seed pour reproductibilité
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    train(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        quick=args.quick,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()