#!/usr/bin/env python3
"""Mandatory gate: load classifier.tflite on this machine and run one window.

Usage:
    python server/scripts/check_classifier.py \
        --model /opt/sign2speak/models/classifier.tflite \
        --meta /opt/sign2speak/models/model_meta.json

Exit code 0 only when the interpreter loads, shapes match, and a synthetic
window produces top-3 probabilities. Run this after installing
server/p1/requirements.txt and before restarting the service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT / "p1"))

from p2_recognizer import ClassifierService  # noqa: E402


def make_window(frames: int, points: int, coords: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    window = rng.random((frames, points, coords), dtype=np.float32)
    # Simulate a hand that appears halfway through and missing face rows.
    window[: frames // 2, points - 42 :, :] = np.nan
    return window


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runtime", default="auto", choices=["auto", "litert", "tf"])
    args = parser.parse_args()

    service = ClassifierService(
        model_path=args.model,
        meta_path=args.meta,
        threads=args.threads,
        runtime=args.runtime,
    )
    service.start()
    if not service.loaded:
        print(json.dumps({"ok": False, "error": service.load_error}, indent=2))
        return 1

    frames = make_window(service.max_len, service.num_nodes, 3)
    result = service.debug_classify(frames, has_hand=True)
    too_short = service.debug_classify(frames[:2], has_hand=True)

    payload = {
        "ok": True,
        "runtime": service.runtime_name,
        "input_shape": service.input_shape,
        "output_shape": service.output_shape,
        "warmup_ms": round(service.warmup_ms, 2),
        "model_version": service.model_version,
        "model_sha256": service.model_sha256,
        "synthetic_top3": result.get("top3"),
        "synthetic_preprocess_ms": round(result.get("preprocess_ms", 0.0), 2),
        "synthetic_inference_ms": round(result.get("inference_ms", 0.0), 2),
        "insufficient_frames_status": too_short.get("status"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.get("inferred") and too_short.get("inferred") is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
