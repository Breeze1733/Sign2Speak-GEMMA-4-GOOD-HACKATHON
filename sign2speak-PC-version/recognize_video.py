#!/usr/bin/env python3
"""Offline video recognition harness for Sign2Speak (PC, no camera).

Reads a video file, replays it through the *same* pipeline as the Android app
(MediaPipe landmarks -> 118-point preprocessing -> classifier.tflite -> voting ->
gloss acceptance -> 3 s no-hand flush) and optionally sends the resulting glosses
to the remote P1 service.

This deliberately mirrors the phone, not live_winner.py:
  * classifier.tflite + TFLite 2.16-compatible runtime (not the Keras model)
  * face detection every 3rd frame, hands every frame, full resolution
  * SignRecognizer: vote over 8 predictions, top-1 confidence >= 0.2 gate
  * MainViewModel: accept a gloss only when confidence >= 0.5
  * flush 3 s after the last frame with a hand

Usage:
  python recognize_video.py ../testdata/videos/clip.mp4 --pace real
  python recognize_video.py ../testdata/videos/clip.mp4 --pace fast --no-remote
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean over all-NaN slices

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
)
from ai_edge_litert.interpreter import Interpreter

# ---------------------------------------------------------------- constants --
# Copied from SignRecognizer.kt / MainViewModel.kt. Do not "improve" these.

ROWS_PER_FRAME = 543
NOSE = [1, 2, 98, 327]
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = 118
CHANNELS = 708
MAX_LEN = 64
PAD = -100.0
NOSE_HOLISTIC_IDX = 17
NOSE_NODE_IDX = POINT_LANDMARKS.index(NOSE_HOLISTIC_IDX)  # 15

FACE_EVERY_N = 3          # LandmarkDetector.kt
INFER_EVERY_N = 3         # SignRecognizer.kt
MIN_FRAMES_FOR_INFER = 10
PREDICTION_HISTORY_SIZE = 8
VOTE_THRESHOLD = 0.2      # SignRecognizer.kt DEFAULT_THRESHOLD
GLOSS_THRESHOLD = 0.5     # MainViewModel.kt
FLUSH_MS = 3_000

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "On_device" / "app" / "src" / "main" / "assets"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- landmarks --


class LandmarkRunner:
    """MediaPipe Hand + Face, matching LandmarkDetector.kt."""

    def __init__(self, assets: Path, detect_scale: float = 1.0):
        self.detect_scale = detect_scale
        self._face_cache = None
        self._face_counter = 0
        self._hand = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(assets / "hand_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
        )
        self._face = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(assets / "face_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )

    def detect(self, frame_bgr, timestamp_ms: int):
        t0 = time.perf_counter()
        if self.detect_scale != 1.0:
            h, w = frame_bgr.shape[:2]
            frame_bgr = cv2.resize(frame_bgr, (int(w * self.detect_scale), int(h * self.detect_scale)))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        holistic = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

        hand_result = self._hand.detect_for_video(mp_image, timestamp_ms)
        has_hand = bool(hand_result.hand_landmarks)
        if has_hand:
            for h_idx, handedness in enumerate(hand_result.handedness):
                label = handedness[0].category_name
                offset = 468 if label == "Left" else 522
                for i, lm in enumerate(hand_result.hand_landmarks[h_idx][:21]):
                    holistic[offset + i] = (lm.x, lm.y, lm.z)
        hand_ms = (time.perf_counter() - t0) * 1000.0

        should_run_face = self._face_counter == 0
        self._face_counter = (self._face_counter + 1) % FACE_EVERY_N
        face_ms = 0.0
        if should_run_face:
            tf = time.perf_counter()
            self._face_cache = self._face.detect_for_video(mp_image, timestamp_ms)
            face_ms = (time.perf_counter() - tf) * 1000.0
        if self._face_cache is not None and self._face_cache.face_landmarks:
            for i, lm in enumerate(self._face_cache.face_landmarks[0][:468]):
                holistic[i] = (lm.x, lm.y, lm.z)

        return holistic, has_hand, hand_ms, face_ms

    def close(self):
        for m in (self._hand, self._face):
            try:
                m.close()
            except Exception:
                pass


# -------------------------------------------------------------- preprocessing --


def preprocess(frames: list[np.ndarray]) -> np.ndarray | None:
    """Kotlin SignRecognizer.runInference + window padding, vectorised."""
    pts = np.stack([f[POINT_LANDMARKS] for f in frames])  # (T, 118, 3)
    valid = ~np.isnan(pts[:, :, 0]).all(axis=1)
    if int(valid.sum()) < 3:
        return None
    vp = pts[valid]                                        # (Tv, 118, 3)

    nose = vp[:, NOSE_NODE_IDX, :]                         # (Tv, 3)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(nose, axis=0)
    mean = np.where(np.isnan(mean), 0.5, mean).astype(np.float32)

    diff = vp - mean
    with np.errstate(invalid="ignore"):
        std = np.sqrt(np.nanmean(diff * diff, axis=(0, 1)))
    std = np.where(np.isnan(std) | (std < 1e-6), 1.0, std).astype(np.float32)

    x = (vp - mean) / std
    x = x[:, :, :2]                                        # (Tv, 118, 2)

    dx = np.zeros_like(x)
    if len(x) > 1:
        dx[:-1] = x[1:] - x[:-1]
    dx2 = np.zeros_like(x)
    if len(x) > 2:
        dx2[:-2] = x[2:] - x[:-2]

    feats = np.concatenate(
        [x.reshape(-1, 2 * NUM_NODES), dx.reshape(-1, 2 * NUM_NODES), dx2.reshape(-1, 2 * NUM_NODES)],
        axis=1,
    ).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    padded = np.full((MAX_LEN, CHANNELS), PAD, dtype=np.float32)
    padded[: len(feats)] = feats
    return padded


# --------------------------------------------------------------- recogniser --


class SignRecognizer:
    """Mirrors SignRecognizer.kt state machine."""

    def __init__(self, classifier_path: Path, labels: dict[int, str], threads: int = 4):
        self.interp = Interpreter(model_path=str(classifier_path), num_threads=threads)
        self.interp.allocate_tensors()
        self.input_index = self.interp.get_input_details()[0]["index"]
        self.output_index = self.interp.get_output_details()[0]["index"]
        self.labels = labels
        self.frames: deque[np.ndarray] = deque(maxlen=MAX_LEN)
        self.history: deque[int] = deque(maxlen=PREDICTION_HISTORY_SIZE)
        self.frame_index = 0
        self.current_label = ""
        self.current_conf = 0.0
        self.serial = 0

    def process(self, holistic, has_hand: bool):
        self.frames.append(holistic)
        inferred = False
        if self.frame_index % INFER_EVERY_N == 0 and len(self.frames) >= MIN_FRAMES_FOR_INFER:
            inferred = self._infer(has_hand)
        self.frame_index += 1
        return self.current_label, self.current_conf, inferred

    def _infer(self, has_hand: bool) -> bool:
        x = preprocess(list(self.frames))
        if x is None:
            return False
        self.interp.set_tensor(self.input_index, x[np.newaxis, ...])
        self.interp.invoke()
        logits = self.interp.get_tensor(self.output_index)[0]
        probs = _softmax(logits)
        top_idx = int(probs.argmax())
        top_conf = float(probs[top_idx])
        self.serial += 1

        if top_conf >= VOTE_THRESHOLD and has_hand:
            self.history.append(top_idx)
            votes = Counter(self.history)
            best_idx = max(votes.items(), key=lambda kv: kv[1])[0]
            self.current_label = self.labels.get(best_idx, "?")
            self.current_conf = top_conf
        else:
            if not has_hand:
                self.history.clear()
            self.current_label = ""
            self.current_conf = 0.0
        return True

    def close(self):
        try:
            self.interp.close()
        except Exception:
            pass


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


# ------------------------------------------------------------------- remote --


def call_remote(url: str, token: str, glosses: list[str], style: str, req_id: str, timeout: float):
    body = json.dumps(
        {
            "request_id": req_id,
            "session_id": "pc-video",
            "sentence_index": 0,
            "glosses": [g.lower() for g in glosses],
            "style": style,
            "model_version": "gemma-4-E2B-it.litertlm",
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        client_ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": True, "client_ms": round(client_ms, 1), **payload}
    except Exception as exc:  # noqa: BLE001
        client_ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "client_ms": round(client_ms, 1), "error": str(exc)}


def fallback_sentence(glosses: list[str]) -> str:
    return " ".join(g.lower().capitalize() for g in glosses)


# --------------------------------------------------------------------- main --


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a video through the Sign2Speak pipeline")
    ap.add_argument("video", type=Path)
    ap.add_argument("--assets", type=Path, default=ASSETS)
    ap.add_argument("--classifier", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pace", choices=["real", "fast"], default="real")
    ap.add_argument("--detect-scale", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--remote-url", default="http://121.41.19.165:8080/translate")
    ap.add_argument("--token", default=os.getenv("S2S_TOKEN", ""))
    ap.add_argument("--no-remote", action="store_true")
    ap.add_argument("--style", default="friendly")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    classifier = args.classifier or (args.assets / "classifier.tflite")
    meta = args.meta or (args.assets / "model_meta.json")
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    labels = {int(k): v for k, v in meta_obj["idx_to_label"].items()}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        log(f"ERROR: cannot open {args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames:
        total = min(total, args.max_frames)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or (REPO / "testdata" / "results" / f"{args.video.stem}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    events_fh = (out_dir / "events.ndjson").open("w", encoding="utf-8")

    log(f"video   : {args.video} ({width}x{height}, {fps:.2f} fps, {total} frames)")
    log(f"output  : {out_dir}")
    log(f"remote  : {'disabled' if args.no_remote else args.remote_url}")

    runner = LandmarkRunner(args.assets, args.detect_scale)
    recognizer = SignRecognizer(classifier, labels, args.threads)

    glosses: list[str] = []
    last_stable = ""
    last_hand_seen_ms: int | None = None
    sentences: list[dict] = []
    hand_times: list[float] = []
    face_times: list[float] = []
    hand_frames = 0
    last_conf = 0.0
    last_label = ""
    label_events: list[tuple[float, str, float]] = []

    def emit(kind: str, **kw):
        events_fh.write(json.dumps({"kind": kind, "t_ms": round(t_ms, 1), **kw}, ensure_ascii=False) + "\n")

    def flush(trigger: str):
        nonlocal glosses, last_stable, last_hand_seen_ms
        if not glosses:
            last_hand_seen_ms = None
            return
        words = list(glosses)
        req_id = f"{args.video.stem}-{len(sentences)}"
        if args.no_remote or not args.token:
            result = {"ok": False, "error": "remote disabled", "client_ms": 0.0}
            sentence = fallback_sentence(words)
            status = "fallback"
        else:
            result = call_remote(args.remote_url, args.token, words, args.style, req_id, args.timeout)
            sentence = result.get("sentence") or fallback_sentence(words)
            status = result.get("status", "error" if not result.get("ok") else "generated")
        entry = {
            "trigger": trigger,
            "glosses": words,
            "sentence": sentence,
            "status": status,
            "t_ms": round(t_ms, 1),
            "server": result,
        }
        sentences.append(entry)
        emit("sentence", **{k: v for k, v in entry.items() if k != "t_ms"})
        log(f"\n[SENTENCE @ {t_ms/1000:.2f}s] {' '.join(w.upper() for w in words)}")
        log(f"  -> {sentence}")
        if result.get("ok"):
            log(f"  queue={result.get('queue_ms')}ms inference={result.get('inference_ms')}ms client={result.get('client_ms')}ms")
        elif status != "fallback":
            log(f"  remote error: {result.get('error')}")
        glosses = []
        last_stable = ""
        last_hand_seen_ms = None

    start = time.perf_counter()
    i = 0
    while i < total:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = i * 1000.0 / fps

        if args.pace == "real":
            target = start + i / fps
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)

        holistic, has_hand, hand_ms, face_ms = runner.detect(frame, int(t_ms))
        hand_times.append(hand_ms)
        if face_ms:
            face_times.append(face_ms)
        if has_hand:
            hand_frames += 1

        label, conf, inferred = recognizer.process(holistic, has_hand)
        if inferred:
            last_label, last_conf = label, conf

        # MainViewModel: accept a stable sign only at >= 0.5, push the previous one.
        if label and conf >= GLOSS_THRESHOLD and label != last_stable:
            if last_stable:
                glosses.append(last_stable)
                emit("gloss", word=last_stable)
                log(f"[GLOSS @ {t_ms/1000:.2f}s] {last_stable.upper()} ({conf:.2f})")
            last_stable = label
            label_events.append((t_ms / 1000.0, label, conf))
        elif not label and last_stable:
            glosses.append(last_stable)
            emit("gloss", word=last_stable)
            log(f"[GLOSS @ {t_ms/1000:.2f}s] {last_stable.upper()} (hand gone)")
            last_stable = ""

        if has_hand:
            last_hand_seen_ms = int(t_ms)
        elif last_hand_seen_ms is not None and t_ms - last_hand_seen_ms >= FLUSH_MS:
            flush("auto")

        if inferred:
            emit("inference", label=label, conf=round(conf, 4), has_hand=has_hand)
        i += 1

    if last_stable:
        glosses.append(last_stable)
        emit("gloss", word=last_stable)
    # Clip ended: flush what we have without waiting for a hand gap.
    flush("end_of_clip")

    cap.release()
    runner.close()
    recognizer.close()
    events_fh.close()

    elapsed = time.perf_counter() - start
    summary = {
        "video": str(args.video),
        "resolution": f"{width}x{height}",
        "fps": fps,
        "frames_processed": i,
        "wall_s": round(elapsed, 2),
        "processing_fps": round(i / elapsed, 2) if elapsed else 0,
        "hand_frames": hand_frames,
        "hand_frame_pct": round(100.0 * hand_frames / i, 1) if i else 0,
        "hand_detect_ms_avg": round(float(np.mean(hand_times)), 2) if hand_times else 0,
        "face_detect_ms_avg": round(float(np.mean(face_times)), 2) if face_times else 0,
        "stable_labels": [{"t_s": round(t, 2), "label": l, "conf": round(c, 3)} for t, l, c in label_events],
        "sentences": sentences,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n=== SUMMARY ===")
    log(f"frames={i} wall={elapsed:.1f}s processing_fps={summary['processing_fps']}")
    log(f"hand frames={hand_frames} ({summary['hand_frame_pct']}%)")
    log(f"avg hand detect={summary['hand_detect_ms_avg']}ms  avg face detect={summary['face_detect_ms_avg']}ms")
    log("stable labels: " + ", ".join(f"{l}@{t:.2f}s" for t, l, _ in label_events))
    for s in sentences:
        log(f"sentence [{s['status']}]: {' '.join(w.upper() for w in s['glosses'])} -> {s['sentence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
