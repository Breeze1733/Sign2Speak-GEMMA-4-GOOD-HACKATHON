# Sign2Speak — Training Pipeline

ASL sign language recognition model trained on the [Kaggle ISLR dataset](https://www.kaggle.com/competitions/asl-signs), adapted for a 50-sign MVP.

---

## Pipeline Overview

```
Kaggle ASL Dataset (40 GB)
        │
        ▼
  preconvert.py          ← Parquet → NumPy conversion
        │
        ▼
  asl_converted/         ← 50-sign local database
        │
        ▼
  train_winner.py        ← Model training
        │
        ▼
  model_winner/          ← Saved model + metadata
```

---

## Step 0 — Download the Dataset

The raw dataset (~40 GB) is available on Kaggle:

> [https://www.kaggle.com/competitions/asl-signs/data](https://www.kaggle.com/competitions/asl-signs/data)

Download and extract into `./asl-signs/`. You need:
- `train.csv`
- `sign_to_prediction_index_map.json`
- `train_landmark_files/` (Parquet files, one per video)

---

## Step 1 — Pre-convert (`preconvert.py`)

> **Recommended: run Steps 1 and 2 on Linux or WSL2**, not on Windows natively.
> The dataset consists of ~94 000 individual Parquet files. Reading them sequentially on Windows (NTFS filesystem) is significantly slower than on a Linux filesystem (ext4). In practice, the conversion can be 3–5× faster on Linux/WSL2. TensorFlow GPU support is also more reliable under Linux.

Reads all Parquet files once and converts them to NumPy arrays for fast training.

```bash
python3 preconvert.py --data-dir ./asl-signs --output ./asl_converted --max-signs 50
```

**What it does:**
1. Loads `train.csv` and `sign_to_prediction_index_map.json`
2. Selects 50 priority ASL signs (predefined MVP list: *hello, bye, thankyou, please, yes, no…*)
3. For each video: reads the Parquet file → converts to array `(n_frames, 543, 3)` (x, y, z)
   - 543 landmarks = 468 face + 21 left hand + 12 pose + 21 right hand + extras
4. Splits train / val / test **by participant** to prevent data leakage
5. Saves `seq_00000.npy`, `seq_00001.npy`… + `index.json` into `asl_converted/`

---

## Step 2 — Train (`train_winner.py`)

```bash
python3 train_winner.py --data-dir ./asl_converted --output ./model_winner --epochs 100
```

**What it does:**
- Selects 118 key landmarks (lips, hands, nose, eyes) → 708-channel feature vector
- Augmentations: temporal resampling, left-right flip, spatial rotation, temporal masking
- Architecture: hybrid **Conv1D + Transformer** (3× Conv1DBlock + TransformerBlock, repeated twice)
- Optimizer: Adam + CosineDecay — loss: CrossEntropy with label smoothing 0.1
- Saves `best_model.keras`, `final_model.keras`, and `model_meta.json`

---

## Model Architecture Credit

> The model architecture is adapted from the 1st place solution of the [Kaggle ISLR 2023 competition](https://www.kaggle.com/competitions/asl-signs) by Hoyeol Sohn. We adapted it for an MVP context: reduced to 50 priority signs, local data pipeline (Parquet → NumPy), removed TPU/TFA dependencies, and lowered MAX_LEN for real-time inference.

Original notebook: [`winner_solution/ISLR_1st_place_Hoyeol_Sohn.ipynb`](winner_solution/ISLR_1st_place_Hoyeol_Sohn.ipynb)
