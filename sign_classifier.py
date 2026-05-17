"""
Sign2Speak — Classifieur ASL PyTorch (GPU)
==========================================
Entraîne un LSTM bidirectionnel sur GPU, évalue, exporte en ONNX,
et teste en live sur webcam. Tout dans un seul script.

Usage:
    # Entraîner
    python sign_classifier.py train --data ./keypoints --output ./model --epochs 100

    # Évaluer un modèle existant
    python sign_classifier.py eval --data ./keypoints --model ./model

    # Test live webcam
    python sign_classifier.py live --model ./model

    # Export ONNX pour mobile
    python sign_classifier.py export --model ./model
"""

import argparse
import json
import time
import random
from pathlib import Path
from collections import Counter, deque, defaultdict

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from keypoint_extractor import KeypointExtractor, ExtractionConfig


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MAX_SEQ_LEN = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset
# =============================================================================

class SignDataset(Dataset):
    """Dataset de séquences de keypoints ASL."""

    def __init__(self, data_dir: Path, split: str, label_to_idx: dict,
                 max_seq_len: int = MAX_SEQ_LEN, augment: bool = False):
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.samples = []  # (sequence_path, label_idx)

        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠️  {split_dir} introuvable")
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
        sequence = np.load(path).astype(np.float32)  # (n_frames, n_features)

        if sequence.ndim != 2 or sequence.shape[0] == 0:
            # Fallback: séquence vide
            sequence = np.zeros((self.max_seq_len, 298), dtype=np.float32)

        # Augmentation
        if self.augment:
            sequence = self._augment(sequence)

        # Padding / Truncation
        n_frames = sequence.shape[0]
        if n_frames > self.max_seq_len:
            start = (n_frames - self.max_seq_len) // 2
            sequence = sequence[start:start + self.max_seq_len]
        elif n_frames < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - n_frames, sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)

        # Masque: 1 pour les vraies frames, 0 pour le padding
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:min(n_frames, self.max_seq_len)] = 1.0

        return (
            torch.from_numpy(sequence),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(mask),
        )

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Augmentation: bruit + shift temporel + scale + mirror horizontal."""
        aug = seq.copy()

        # Bruit gaussien
        if random.random() < 0.5:
            aug += np.random.normal(0, 0.015, aug.shape).astype(np.float32)

        # Décalage temporel
        if random.random() < 0.3:
            shift = random.randint(-3, 3)
            if shift > 0:
                aug = np.concatenate([np.zeros((shift, aug.shape[1]), dtype=np.float32), aug[:-shift]])
            elif shift < 0:
                aug = np.concatenate([aug[-shift:], np.zeros((-shift, aug.shape[1]), dtype=np.float32)])

        # Scale
        if random.random() < 0.5:
            aug *= random.uniform(0.9, 1.1)

        # Speed perturbation (drop ou dupliquer des frames)
        if random.random() < 0.3:
            n = len(aug)
            if random.random() < 0.5 and n > 10:
                # Accélérer: supprimer des frames aléatoires
                keep = sorted(random.sample(range(n), max(n - 3, 5)))
                aug = aug[keep]
            else:
                # Ralentir: dupliquer des frames
                dup_idx = sorted(random.sample(range(n), min(3, n)))
                aug = np.insert(aug, dup_idx, aug[dup_idx], axis=0)

        return aug


# =============================================================================
# Modèle
# =============================================================================

class SignClassifier(nn.Module):
    """
    LSTM bidirectionnel avec attention et classification de signes ASL.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        # Normalisation de l'input
        self.input_norm = nn.LayerNorm(n_features)

        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention simple
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: (batch, seq_len) — 1 pour vraies frames, 0 pour padding
        Returns:
            logits: (batch, n_classes)
        """
        # Normalisation
        x = self.input_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)

        # Classification
        logits = self.classifier(context)
        return logits


# =============================================================================
# Entraînement
# =============================================================================

def train(data_dir: Path, output_dir: Path, epochs: int = EPOCHS,
          batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
    """Entraîne le classifieur sur GPU."""
    set_seed()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go")

    # Labels
    labels_file = data_dir / "labels.json"
    with open(labels_file) as f:
        label_to_idx = json.load(f)
    n_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Datasets
    print(f"\n📂 Chargement des données...")
    train_ds = SignDataset(data_dir, 'train', label_to_idx, augment=True)
    val_ds = SignDataset(data_dir, 'val', label_to_idx, augment=False)
    test_ds = SignDataset(data_dir, 'test', label_to_idx, augment=False)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Déterminer n_features depuis un sample
    sample_seq, _, _ = train_ds[0]
    n_features = sample_seq.shape[1]
    print(f"   Features/frame: {n_features}")
    print(f"   Classes: {n_classes}")

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
    print(model)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = 20
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n🚀 Entraînement ({epochs} epochs, batch={batch_size}, lr={lr})...")

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

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Checkpoint
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

        # Early stopping
        if patience_counter >= patience:
            print(f"\n   ⏹️  Early stopping (patience={patience})")
            break

    # Charger le meilleur modèle
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    # --- Test ---
    print(f"\n📊 Évaluation sur le test set...")
    test_acc, test_loss, top5_acc, confusions = evaluate(model, test_loader, criterion, idx_to_label)

    print(f"   Test accuracy: {test_acc:.1%}")
    print(f"   Top-5 accuracy: {top5_acc:.1%}")
    print(f"   Best val accuracy: {best_val_acc:.1%}")

    if confusions:
        print(f"\n   Top confusions:")
        for pair, count in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
            print(f"      {pair}: {count}")

    # Sauvegarder
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    meta = {
        'n_classes': n_classes,
        'n_features': n_features,
        'max_seq_len': MAX_SEQ_LEN,
        'hidden_size': 128,
        'num_layers': 2,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(k): v for k, v in idx_to_label.items()},
        'test_accuracy': test_acc,
        'top5_accuracy': top5_acc,
        'best_val_accuracy': best_val_acc,
        'epochs_trained': len(history['train_loss']),
    }
    with open(output_dir / "model_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # Historique
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f)

    print(f"\n💾 Sauvegardé dans {output_dir}/")
    return model


def evaluate(model, loader, criterion, idx_to_label):
    """Évalue le modèle et retourne les métriques."""
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
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

            # Top-5
            top5 = logits.topk(5, dim=1).indices
            for i in range(len(batch_y)):
                if batch_y[i] in top5[i]:
                    top5_correct += 1
                if preds[i] != batch_y[i]:
                    true_l = idx_to_label.get(batch_y[i].item(), "?")
                    pred_l = idx_to_label.get(preds[i].item(), "?")
                    confusions[f"{true_l} → {pred_l}"] += 1

            total += len(batch_y)

    return correct / total, total_loss / total, top5_correct / total, dict(confusions)


# =============================================================================
# Live Test
# =============================================================================

def live(model_dir: Path, threshold: float = 0.15, use_face: bool = False):
    """Test live sur webcam."""
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
    ).to(DEVICE)

    # Charger les poids
    weights_path = model_dir / "best_model.pt"
    if not weights_path.exists():
        weights_path = model_dir / "final_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"   {n_classes} signes, {n_features} features, device={DEVICE}")

    # Extracteur — même config que l'entraînement
    config = ExtractionConfig(
        use_hands=True,
        use_pose=True,
        use_face=use_face,
        face_subset_only=True,
        normalize=True,
    )

    # Si le modèle a été entraîné avec face mais qu'on tourne sans,
    # il faut padder les features manquantes
    expected_features = n_features
    actual_features = config.features_per_frame()
    pad_features = expected_features - actual_features if expected_features > actual_features else 0

    # Buffer
    frame_buffer = deque(maxlen=max_seq_len)
    prediction_history = deque(maxlen=8)
    top3 = []
    current_label = ""
    current_conf = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam introuvable")
        return

    print(f"\n📸 Webcam ouverte — 'q' quitter, 'c' clear")
    print(f"   Face: {'ON' if use_face else 'OFF'} | Seuil: {threshold:.0%}")

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

            # Padding si nécessaire (config live ≠ config entraînement)
            if pad_features > 0:
                keypoints = np.concatenate([keypoints, np.zeros(pad_features, dtype=np.float32)])

            has_hand = data['left_hand'] is not None or data['right_hand'] is not None
            frame_buffer.append(keypoints)

            # Prédiction toutes les 3 frames
            if frame_idx % 3 == 0 and len(frame_buffer) >= 10:
                sequence = np.array(list(frame_buffer), dtype=np.float32)
                n = sequence.shape[0]

                # Padding/truncation
                if n < max_seq_len:
                    pad = np.zeros((max_seq_len - n, expected_features), dtype=np.float32)
                    sequence = np.concatenate([sequence, pad])
                elif n > max_seq_len:
                    sequence = sequence[-max_seq_len:]

                mask = np.zeros(max_seq_len, dtype=np.float32)
                mask[:min(n, max_seq_len)] = 1.0

                # Inférence
                with torch.no_grad():
                    x = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)
                    m = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)
                    logits = model(x, m)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                top_idx = np.argmax(probs)
                top_conf = probs[top_idx]

                # Top 3
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3 = [(idx_to_label.get(i, "?"), probs[i]) for i in top3_idx]

                # Lissage
                if top_conf >= threshold and has_hand:
                    prediction_history.append(top_idx)
                    if prediction_history:
                        votes = Counter(prediction_history)
                        best_idx, _ = votes.most_common(1)[0]
                        current_label = idx_to_label.get(best_idx, "?")
                        current_conf = top_conf
                else:
                    if not has_hand:
                        prediction_history.clear()
                    current_label = ""
                    current_conf = 0.0

            # --- Affichage ---
            annotated = extractor.draw_landmarks(frame, data)
            h, w = annotated.shape[:2]

            # FPS
            dt = time.time() - t0
            fps_list.append(1.0 / dt if dt > 0 else 0)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)

            cv2.putText(annotated, f"FPS: {avg_fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "MAIN" if has_hand else "---"
            color = (0, 255, 0) if has_hand else (0, 0, 255)
            cv2.putText(annotated, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Prédiction principale
            if current_label and current_conf >= threshold:
                display = current_label.rstrip('0123456789')
                c = (0, 255, 0) if current_conf > 0.5 else (0, 255, 255) if current_conf > 0.3 else (0, 165, 255)
                cv2.putText(annotated, display, (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, 4)
                cv2.putText(annotated, f"{current_conf:.0%}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

            # Top 3
            for i, (label, conf) in enumerate(top3):
                txt = f"{label.rstrip('0123456789')}: {conf:.0%}"
                cv2.putText(annotated, txt, (w - 250, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Sign2Speak — Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                frame_buffer.clear()
                prediction_history.clear()
                current_label = ""
                print("🔄 Clear")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# Export ONNX
# =============================================================================

def export_onnx(model_dir: Path):
    """Exporte le modèle en ONNX pour déploiement mobile."""
    print("📱 Export ONNX...")

    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    model = SignClassifier(
        meta['n_features'], meta['n_classes'],
        hidden_size=meta.get('hidden_size', 128),
        num_layers=meta.get('num_layers', 2),
    )

    weights_path = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()

    # Dummy inputs
    dummy_x = torch.randn(1, meta['max_seq_len'], meta['n_features'])
    dummy_mask = torch.ones(1, meta['max_seq_len'])

    onnx_path = model_dir / "sign_classifier.onnx"
    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        str(onnx_path),
        input_names=['keypoints', 'mask'],
        output_names=['logits'],
        dynamic_axes={
            'keypoints': {0: 'batch'},
            'mask': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=17,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"   ✅ {onnx_path} ({size_mb:.1f} Mo)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Classifieur ASL PyTorch")
    sub = parser.add_subparsers(dest='command')

    # Train
    p_train = sub.add_parser('train')
    p_train.add_argument('--data', type=str, default='./keypoints')
    p_train.add_argument('--output', type=str, default='./model')
    p_train.add_argument('--epochs', type=int, default=EPOCHS)
    p_train.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p_train.add_argument('--lr', type=float, default=LEARNING_RATE)

    # Eval
    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--data', type=str, default='./keypoints')
    p_eval.add_argument('--model', type=str, default='./model')

    # Live
    p_live = sub.add_parser('live')
    p_live.add_argument('--model', type=str, default='./model')
    p_live.add_argument('--threshold', type=float, default=0.15)
    p_live.add_argument('--face', action='store_true', help='Activer le face landmarker (plus lent)')

    # Export
    p_export = sub.add_parser('export')
    p_export.add_argument('--model', type=str, default='./model')

    args = parser.parse_args()

    if args.command == 'train':
        set_seed()
        train(Path(args.data), Path(args.output), args.epochs, args.batch_size, args.lr)

    elif args.command == 'eval':
        # Quick eval
        model_dir = Path(args.model)
        with open(model_dir / "model_meta.json") as f:
            meta = json.load(f)
        label_to_idx = meta['label_to_idx']
        idx_to_label = {int(k): v for k, v in meta['idx_to_label'].items()}
        test_ds = SignDataset(Path(args.data), 'test', label_to_idx)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        model = SignClassifier(meta['n_features'], meta['n_classes'],
                                hidden_size=meta.get('hidden_size', 128),
                                num_layers=meta.get('num_layers', 2)).to(DEVICE)
        model.load_state_dict(torch.load(model_dir / "best_model.pt",
                                          map_location=DEVICE, weights_only=True))
        criterion = nn.CrossEntropyLoss()
        acc, loss, top5, conf = evaluate(model, test_loader, criterion, idx_to_label)
        print(f"Test: {acc:.1%} | Top-5: {top5:.1%}")

    elif args.command == 'live':
        live(Path(args.model), args.threshold, use_face=args.face)

    elif args.command == 'export':
        export_onnx(Path(args.model))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
