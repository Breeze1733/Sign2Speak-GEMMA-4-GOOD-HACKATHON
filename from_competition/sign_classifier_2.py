"""
Sign2Speak — Classifieur ASL v2 (PyTorch GPU)
===============================================
Améliorations vs v1:
- FingerDropout (augmentation du gagnant Kaggle)
- Architecture LSTM + Attention améliorée
- Export ONNX + LiteRT (bonus hackathon $10K)
- Compatible Google ASL dataset (Parquet) et données perso
- Mixed precision training (2x plus rapide sur RTX)

Usage:
    python sign_classifier_2.py train --data ./google_keypoints --output ./model_v2 --epochs 100
    python sign_classifier_2.py live --model ./model_v2
    python sign_classifier_2.py export --model ./model_v2
"""

import argparse
import json
import time
import random
import shutil
from pathlib import Path
from collections import Counter, deque, defaultdict

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from keypoint_extractor import KeypointExtractor, ExtractionConfig


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MAX_SEQ_LEN = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Structure des features (pour FingerDropout)
# Quand on utilise 298 features (notre extracteur):
#   left_hand: 0:63, right_hand: 63:126, pose: 126:198, face: 198:298
# Quand on utilise le Google dataset (~315 features):
#   left_hand: 0:63, right_hand: 63:126, pose: 126:201, face: 201:end


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset avec FingerDropout
# =============================================================================

class SignDataset(Dataset):
    """Dataset avec augmentations avancées incluant FingerDropout."""

    def __init__(self, data_dir: Path, split: str, label_to_idx: dict,
                 max_seq_len: int = MAX_SEQ_LEN, augment: bool = False):
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.samples = []

        split_dir = data_dir / split
        if not split_dir.exists():
            return

        for sign_dir in sorted(split_dir.iterdir()):
            if not sign_dir.is_dir() or sign_dir.name not in label_to_idx:
                continue
            label_idx = label_to_idx[sign_dir.name]
            for npy_file in sorted(sign_dir.glob("*.npy")):
                self.samples.append((npy_file, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)

        if seq.ndim != 2 or seq.shape[0] == 0:
            seq = np.zeros((self.max_seq_len, 298), dtype=np.float32)

        n_features = seq.shape[1]

        # Augmentations
        if self.augment:
            seq = self._augment(seq, n_features)

        # Interpoler à longueur fixe au lieu de pad/trunc
        # (meilleur que le padding pour capturer le timing du signe)
        n_frames = seq.shape[0]
        if n_frames != self.max_seq_len and n_frames > 0:
            indices = np.linspace(0, n_frames - 1, self.max_seq_len)
            seq = np.array([
                seq[int(np.floor(i))] * (1 - (i % 1)) + seq[min(int(np.ceil(i)), n_frames - 1)] * (i % 1)
                for i in indices
            ], dtype=np.float32)

        # Masque (tout à 1 car on interpole au lieu de padder)
        mask = np.ones(self.max_seq_len, dtype=np.float32)

        # Remplacer NaN
        seq = np.nan_to_num(seq, nan=0.0)

        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(mask),
        )

    def _augment(self, seq: np.ndarray, n_features: int) -> np.ndarray:
        aug = seq.copy()

        # --- FingerDropout (inspiration gagnant Kaggle) ---
        # Drop une main entière aléatoirement (20% de chance chacune)
        if random.random() < 0.2:
            aug[:, 0:63] = 0.0   # drop left hand
        if random.random() < 0.2:
            aug[:, 63:126] = 0.0  # drop right hand

        # --- Bruit gaussien ---
        if random.random() < 0.5:
            noise_std = random.uniform(0.005, 0.02)
            aug += np.random.normal(0, noise_std, aug.shape).astype(np.float32)

        # --- TimeStretch (accélérer/ralentir) ---
        if random.random() < 0.4:
            factor = random.uniform(0.8, 1.2)
            n = len(aug)
            new_n = max(5, int(n * factor))
            indices = np.linspace(0, n - 1, new_n)
            aug = np.array([
                aug[int(np.floor(i))] * (1 - (i % 1)) + aug[min(int(np.ceil(i)), n - 1)] * (i % 1)
                for i in indices
            ], dtype=np.float32)

        # --- Spatial scale ---
        if random.random() < 0.3:
            scale = random.uniform(0.85, 1.15)
            aug *= scale

        # --- Temporal shift ---
        if random.random() < 0.3:
            shift = random.randint(-3, 3)
            if shift > 0:
                aug = np.concatenate([np.zeros((shift, aug.shape[1]), dtype=np.float32), aug[:-shift]])
            elif shift < 0:
                aug = np.concatenate([aug[-shift:], np.zeros((-shift, aug.shape[1]), dtype=np.float32)])

        # --- Spatial jitter (déplacer légèrement toutes les coords) ---
        if random.random() < 0.3:
            offset = np.random.uniform(-0.02, 0.02, (1, aug.shape[1])).astype(np.float32)
            aug += offset

        return aug


# =============================================================================
# Modèle amélioré
# =============================================================================

class SignClassifier(nn.Module):
    """
    LSTM bidirectionnel + Attention multi-head simplifiée.
    Optimisé pour export ONNX/LiteRT.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, n_heads: int = 4):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        # Input projection (réduit la dimensionalité si besoin)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention multi-head simplifiée
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size // n_heads),
                nn.Tanh(),
                nn.Linear(hidden_size // n_heads, 1),
            )
            for _ in range(n_heads)
        ])

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2 * n_heads),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 * n_heads, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x, mask=None):
        # Input projection
        x = self.input_proj(x)  # (B, T, hidden)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Multi-head attention
        contexts = []
        for head in self.attention_heads:
            weights = head(lstm_out).squeeze(-1)  # (B, T)
            if mask is not None:
                weights = weights.masked_fill(mask == 0, float('-inf'))
            weights = F.softmax(weights, dim=1)
            ctx = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, hidden*2)
            contexts.append(ctx)

        # Concatenate heads
        context = torch.cat(contexts, dim=1)  # (B, hidden*2*n_heads)

        # Classify
        return self.classifier(context)


# =============================================================================
# Entraînement
# =============================================================================

def train(data_dir: Path, output_dir: Path, epochs: int = EPOCHS,
          batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
    set_seed()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go")

    # Labels
    with open(data_dir / "labels.json") as f:
        label_to_idx = json.load(f)
    n_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Datasets
    print(f"\n📂 Chargement...")
    train_ds = SignDataset(data_dir, 'train', label_to_idx, augment=True)
    val_ds = SignDataset(data_dir, 'val', label_to_idx, augment=False)
    test_ds = SignDataset(data_dir, 'test', label_to_idx, augment=False)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if len(train_ds) == 0:
        print("❌ Aucune donnée d'entraînement")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Déterminer n_features
    sample_seq, _, _ = train_ds[0]
    n_features = sample_seq.shape[1]
    print(f"   Features: {n_features} | Classes: {n_classes}")

    # Class weights
    label_counts = Counter(label for _, label in train_ds.samples)
    total = len(train_ds)
    weights = torch.tensor([
        total / (n_classes * label_counts.get(i, 1))
        for i in range(n_classes)
    ], dtype=torch.float32).to(DEVICE)

    # Modèle
    model = SignClassifier(n_features, n_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  Modèle: {total_params:,} paramètres")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Mixed precision
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    patience = 25
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n🚀 Entraînement ({epochs} epochs, AMP={'ON' if use_amp else 'OFF'})...")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)

            optimizer.zero_grad()

            if use_amp:
                with autocast('cuda'):
                    logits = model(batch_x, batch_mask)
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_mask in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                batch_mask = batch_mask.to(DEVICE)
                logits = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            improved = " ⭐"
            patience_counter = 0
        else:
            patience_counter += 1

        lr_current = optimizer.param_groups[0]['lr']
        print(f"   Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_acc:.1%} ({train_loss:.3f}) | "
              f"Val: {val_acc:.1%} ({val_loss:.3f}) | "
              f"LR: {lr_current:.1e}{improved}")

        if patience_counter >= patience:
            print(f"\n   ⏹️  Early stopping")
            break

    # Best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    # Test
    print(f"\n📊 Évaluation sur le test set...")
    test_acc, test_loss, top5_acc, confusions = evaluate(model, test_loader, criterion, idx_to_label)
    print(f"   Test accuracy: {test_acc:.1%}")
    print(f"   Top-5 accuracy: {top5_acc:.1%}")
    print(f"   Best val accuracy: {best_val_acc:.1%}")

    if confusions:
        print(f"\n   Top confusions:")
        for pair, count in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
            print(f"      {pair}: {count}")

    # Save
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    meta = {
        'n_classes': n_classes, 'n_features': n_features,
        'max_seq_len': MAX_SEQ_LEN, 'hidden_size': 128,
        'num_layers': 2, 'n_heads': 4,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
        'test_accuracy': test_acc, 'top5_accuracy': top5_acc,
        'best_val_accuracy': best_val_acc,
        'epochs_trained': len(history['train_loss']),
    }
    with open(output_dir / "model_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f)

    print(f"\n💾 Sauvegardé dans {output_dir}/")

    # Export
    export_onnx(model, output_dir, n_features)
    export_litert(output_dir)

    return model


def evaluate(model, loader, criterion, idx_to_label):
    model.eval()
    total_loss = correct = top5_correct = total = 0
    confusions = defaultdict(int)

    with torch.no_grad():
        for batch_x, batch_y, batch_mask in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(1)
            correct += (preds == batch_y).sum().item()
            top5 = logits.topk(5, dim=1).indices
            for i in range(len(batch_y)):
                if batch_y[i] in top5[i]:
                    top5_correct += 1
                if preds[i] != batch_y[i]:
                    confusions[f"{idx_to_label.get(batch_y[i].item(), '?')} → {idx_to_label.get(preds[i].item(), '?')}"] += 1
            total += len(batch_y)

    return correct / total, total_loss / total, top5_correct / total, dict(confusions)


# =============================================================================
# Export ONNX
# =============================================================================

def export_onnx(model, output_dir: Path, n_features: int):
    print(f"\n📱 Export ONNX...")
    model_cpu = model.cpu().eval()

    dummy_x = torch.randn(1, MAX_SEQ_LEN, n_features)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN)

    onnx_path = output_dir / "sign_classifier_2.onnx"

    try:
        torch.onnx.export(
            model_cpu, (dummy_x, dummy_mask), str(onnx_path),
            input_names=['keypoints', 'mask'],
            output_names=['logits'],
            dynamic_axes={'keypoints': {0: 'batch'}, 'mask': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=17,
        )
        print(f"   ✅ {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} Mo)")
    except Exception as e:
        print(f"   ⚠️  ONNX export error: {e}")

    model.to(DEVICE)


# =============================================================================
# Export LiteRT (TFLite) — bonus hackathon $10K
# =============================================================================

def export_litert(output_dir: Path):
    """Convertit ONNX → TFLite via onnx2tf ou ai_edge_torch."""
    onnx_path = output_dir / "sign_classifier_2.onnx"
    if not onnx_path.exists():
        print("   ⚠️  ONNX non trouvé, skip LiteRT")
        return

    print(f"\n📱 Export LiteRT (TFLite)...")

    # Méthode 1: via ai_edge_torch (recommandé par Google)
    try:
        import ai_edge_torch
        print("   Utilisation de ai_edge_torch...")
        # ai_edge_torch convertit directement depuis PyTorch
        meta_path = output_dir / "model_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        model = SignClassifier(
            meta['n_features'], meta['n_classes'],
            hidden_size=meta.get('hidden_size', 128),
            num_layers=meta.get('num_layers', 2),
            n_heads=meta.get('n_heads', 4),
        )
        model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location='cpu', weights_only=True))
        model.eval()

        sample = (torch.randn(1, MAX_SEQ_LEN, meta['n_features']),
                  torch.ones(1, MAX_SEQ_LEN))

        edge_model = ai_edge_torch.convert(model, sample)
        tflite_path = output_dir / "sign_classifier_2.tflite"
        edge_model.export(str(tflite_path))
        print(f"   ✅ {tflite_path} ({tflite_path.stat().st_size / 1e6:.1f} Mo)")
        return
    except ImportError:
        pass

    # Méthode 2: via onnx-tf + tflite
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        print("   Utilisation de onnx-tf...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(output_dir / "tf_model"))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(output_dir / "tf_model"))
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

        tflite_path = output_dir / "sign_classifier_2.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   ✅ {tflite_path} ({len(tflite_model) / 1e6:.1f} Mo)")
        return
    except ImportError:
        pass

    print("   ℹ️  Pour l'export LiteRT, installe: pip install ai-edge-torch")
    print("   Ou: pip install onnx-tf tensorflow")
    print("   L'ONNX reste utilisable via ONNX Runtime Mobile.")


# =============================================================================
# Live Test
# =============================================================================

def live(model_dir: Path, threshold: float = 0.15, use_face: bool = False):
    print("📦 Chargement du modèle...")

    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    idx_to_label = {int(k): v for k, v in meta['idx_to_label'].items()}
    n_features = meta['n_features']
    n_classes = meta['n_classes']
    max_seq_len = meta['max_seq_len']

    model = SignClassifier(
        n_features, n_classes,
        hidden_size=meta.get('hidden_size', 128),
        num_layers=meta.get('num_layers', 2),
        n_heads=meta.get('n_heads', 4),
    ).to(DEVICE)

    weights_path = model_dir / "best_model.pt"
    if not weights_path.exists():
        weights_path = model_dir / "final_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"   {n_classes} signes, {n_features} features, device={DEVICE}")

    config = ExtractionConfig(
        use_hands=True, use_pose=True,
        use_face=use_face, face_subset_only=True, normalize=True,
    )

    expected_features = n_features
    actual_features = config.features_per_frame()
    pad_features = max(0, expected_features - actual_features)
    trim_features = max(0, actual_features - expected_features)

    frame_buffer = deque(maxlen=max_seq_len)
    prediction_history = deque(maxlen=8)
    top3 = []
    current_label = ""
    current_conf = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam introuvable")
        return

    print(f"\n📸 'q' quitter | 'c' clear | Face: {'ON' if use_face else 'OFF'}")

    frame_idx = 0
    fps_list = []

    with KeypointExtractor(config) as extractor:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx * 33
            data = extractor.extract_frame(frame, timestamp_ms)
            keypoints = extractor.extract_and_flatten(frame, timestamp_ms, data=data)

            # Ajuster les features si nécessaire
            if pad_features > 0:
                keypoints = np.concatenate([keypoints, np.zeros(pad_features, dtype=np.float32)])
            elif trim_features > 0:
                keypoints = keypoints[:expected_features]

            has_hand = data['left_hand'] is not None or data['right_hand'] is not None
            frame_buffer.append(keypoints)

            # Prédiction
            if frame_idx % 3 == 0 and len(frame_buffer) >= 10:
                seq = np.array(list(frame_buffer), dtype=np.float32)
                n = seq.shape[0]

                # Interpolation à longueur fixe
                if n != max_seq_len and n > 1:
                    indices = np.linspace(0, n - 1, max_seq_len)
                    seq = np.array([
                        seq[int(np.floor(i))] * (1 - (i % 1)) + seq[min(int(np.ceil(i)), n - 1)] * (i % 1)
                        for i in indices
                    ], dtype=np.float32)

                mask = np.ones(max_seq_len, dtype=np.float32)
                seq = np.nan_to_num(seq, nan=0.0)

                with torch.no_grad():
                    x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
                    m = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)
                    logits = model(x, m)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                top_idx = np.argmax(probs)
                top_conf = probs[top_idx]

                top3_idx = np.argsort(probs)[-3:][::-1]
                top3 = [(idx_to_label.get(i, "?"), probs[i]) for i in top3_idx]

                if top_conf >= threshold and has_hand:
                    prediction_history.append(top_idx)
                    votes = Counter(prediction_history)
                    best_idx, _ = votes.most_common(1)[0]
                    current_label = idx_to_label.get(best_idx, "?")
                    current_conf = top_conf
                else:
                    if not has_hand:
                        prediction_history.clear()
                    current_label = ""
                    current_conf = 0.0

            # Affichage
            annotated = extractor.draw_landmarks(frame, data)
            h, w = annotated.shape[:2]

            dt = time.time() - t0
            fps_list.append(1.0 / dt if dt > 0 else 0)
            if len(fps_list) > 30: fps_list.pop(0)

            cv2.putText(annotated, f"FPS: {np.mean(fps_list):.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "MAIN" if has_hand else "---"
            cv2.putText(annotated, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if has_hand else (0, 0, 255), 2)

            if current_label and current_conf >= threshold:
                display = current_label.rstrip('0123456789').replace('-', ' ')
                c = (0, 255, 0) if current_conf > 0.5 else (0, 255, 255) if current_conf > 0.3 else (0, 165, 255)
                cv2.putText(annotated, display, (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, 4)
                cv2.putText(annotated, f"{current_conf:.0%}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

            for i, (label, conf) in enumerate(top3):
                cv2.putText(annotated, f"{label}: {conf:.0%}", (w - 250, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Sign2Speak — Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'):
                frame_buffer.clear()
                prediction_history.clear()
                current_label = ""

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Classifieur ASL v2")
    sub = parser.add_subparsers(dest='command')

    p_train = sub.add_parser('train')
    p_train.add_argument('--data', type=str, default='./google_keypoints')
    p_train.add_argument('--output', type=str, default='./model_v2')
    p_train.add_argument('--epochs', type=int, default=EPOCHS)
    p_train.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p_train.add_argument('--lr', type=float, default=LEARNING_RATE)

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--data', type=str, default='./google_keypoints')
    p_eval.add_argument('--model', type=str, default='./model_v2')

    p_live = sub.add_parser('live')
    p_live.add_argument('--model', type=str, default='./model_v2')
    p_live.add_argument('--threshold', type=float, default=0.15)
    p_live.add_argument('--face', action='store_true')

    p_export = sub.add_parser('export')
    p_export.add_argument('--model', type=str, default='./model_v2')

    args = parser.parse_args()

    if args.command == 'train':
        set_seed()
        train(Path(args.data), Path(args.output), args.epochs, args.batch_size, args.lr)

    elif args.command == 'eval':
        model_dir = Path(args.model)
        with open(model_dir / "model_meta.json") as f:
            meta = json.load(f)
        label_to_idx = meta['label_to_idx']
        idx_to_label = {int(k): v for k, v in meta['idx_to_label'].items()}
        test_ds = SignDataset(Path(args.data), 'test', label_to_idx)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        model = SignClassifier(meta['n_features'], meta['n_classes'],
                                hidden_size=meta.get('hidden_size', 128),
                                num_layers=meta.get('num_layers', 2),
                                n_heads=meta.get('n_heads', 4)).to(DEVICE)
        model.load_state_dict(torch.load(model_dir / "best_model.pt",
                                          map_location=DEVICE, weights_only=True))
        criterion = nn.CrossEntropyLoss()
        acc, loss, top5, conf = evaluate(model, test_loader, criterion, idx_to_label)
        print(f"Test: {acc:.1%} | Top-5: {top5:.1%}")

    elif args.command == 'live':
        live(Path(args.model), args.threshold, use_face=args.face)

    elif args.command == 'export':
        model_dir = Path(args.model)
        with open(model_dir / "model_meta.json") as f:
            meta = json.load(f)
        model = SignClassifier(meta['n_features'], meta['n_classes'],
                                hidden_size=meta.get('hidden_size', 128),
                                num_layers=meta.get('num_layers', 2),
                                n_heads=meta.get('n_heads', 4))
        model.load_state_dict(torch.load(model_dir / "best_model.pt",
                                          map_location='cpu', weights_only=True))
        model.eval()
        export_onnx(model, model_dir, meta['n_features'])
        export_litert(model_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
