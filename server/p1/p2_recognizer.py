"""Server-side M3: preprocessing, TFLite classification and majority voting.

The preprocessing and voting rules are a line-by-line port of
``sign2speak-PC-version/recognize_video.py`` (itself a port of the Android
``SignRecognizer``). Do not "improve" the arithmetic: parity with the phone is
the point of P2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from p2_protocol import ProtocolError, Window

log = logging.getLogger("s2s.p2")

# Canonical layout from SignRecognizer.kt / model_meta.json.
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
PREDICTION_HISTORY_SIZE = 8
VOTE_THRESHOLD = 0.2


def _nanmean(values: np.ndarray, axis: Any) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def preprocess(
    frames: np.ndarray,
    *,
    point_landmarks: np.ndarray | None = None,
    num_nodes: int = NUM_NODES,
    channels: int = CHANNELS,
    max_len: int = MAX_LEN,
    pad: float = PAD,
) -> np.ndarray | None:
    """Return a padded ``(max_len, channels)`` feature tensor or ``None``.

    ``frames`` is ``(T, point_count, coord_count)`` with NaN for missing
    values. It may already contain the 118 selected points or the full 543
    holistic rows.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("frames must be (T, points, coords)")

    if frames.shape[1] == 543:
        indices = POINT_LANDMARKS if point_landmarks is None else point_landmarks
        pts = frames[:, np.asarray(indices, dtype=np.int64), :]
    elif frames.shape[1] == num_nodes:
        pts = frames
    else:
        raise ValueError(
            f"expected {num_nodes} or 543 points per frame, got {frames.shape[1]}"
        )

    valid = ~np.isnan(pts[:, :, 0]).all(axis=1)
    if int(valid.sum()) < 3:
        return None
    vp = pts[valid]  # (Tv, N, C)

    nose = vp[:, NOSE_NODE_IDX, :]
    mean = _nanmean(nose, axis=0)
    mean = np.where(np.isnan(mean), 0.5, mean).astype(np.float32)

    diff = vp - mean
    std = np.sqrt(_nanmean(diff * diff, axis=(0, 1)))
    std = np.where(np.isnan(std) | (std < 1e-6), 1.0, std).astype(np.float32)

    x = ((vp - mean) / std)[:, :, :2]  # only x/y reach the model

    dx = np.zeros_like(x)
    if len(x) > 1:
        dx[:-1] = x[1:] - x[:-1]
    dx2 = np.zeros_like(x)
    if len(x) > 2:
        dx2[:-2] = x[2:] - x[:-2]

    feats = np.concatenate(
        [
            x.reshape(-1, 2 * num_nodes),
            dx.reshape(-1, 2 * num_nodes),
            dx2.reshape(-1, 2 * num_nodes),
        ],
        axis=1,
    ).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    padded = np.full((max_len, channels), pad, dtype=np.float32)
    padded[: len(feats)] = feats
    return padded


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


@dataclass
class SessionState:
    episode_id: int = 0
    last_window_id: int = 0
    history: deque[int] = field(default_factory=lambda: deque(maxlen=PREDICTION_HISTORY_SIZE))
    label: str = ""
    conf: float = 0.0
    top3: list[tuple[str, float]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    requests: int = 0


@dataclass
class RecognitionOutcome:
    status: str = "ok"
    inferred: bool = True
    label: str = ""
    confidence: float = 0.0
    top3: list[tuple[str, float]] = field(default_factory=list)
    has_hand: bool = False
    frame_count: int = 0
    queue_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    vote_ms: float = 0.0
    vote_len: int = 0
    error: ProtocolError | None = None


def accept_window(state: SessionState, window: Window) -> None:
    """Apply the per-session episode/window ordering rules.

    Raises ``ProtocolError(409)`` for an older episode or a replayed window.
    A larger episode resets the vote history; gaps in ``window_id`` are allowed
    because the phone drops windows while a request is in flight.
    """
    if window.episode_id < state.episode_id:
        raise ProtocolError(409, "stale episode")
    if window.episode_id > state.episode_id:
        state.episode_id = window.episode_id
        state.history.clear()
        state.last_window_id = 0
    if window.window_id <= state.last_window_id:
        raise ProtocolError(409, "stale window")
    state.last_window_id = window.window_id
    if window.reset_vote:
        state.history.clear()
    state.last_seen = time.monotonic()
    state.requests += 1


def apply_vote(
    state: SessionState,
    probs: np.ndarray,
    labels: dict[int, str],
    has_hand: bool,
    *,
    threshold: float = VOTE_THRESHOLD,
) -> list[tuple[str, float]]:
    """Mirror SignRecognizer.runInference's vote block exactly."""
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    order = np.argsort(-probs, kind="stable")[:3]
    top3 = [(labels.get(int(i), "?"), float(probs[int(i)])) for i in order]

    if top_conf >= threshold and has_hand:
        state.history.append(top_idx)
        counts = Counter(state.history)
        # Counter preserves first-occurrence order; max() keeps the first max on ties.
        best_idx = max(counts.items(), key=lambda kv: kv[1])[0]
        state.label = labels.get(best_idx, "?")
        state.conf = top_conf
    else:
        if not has_hand:
            state.history.clear()
        state.label = ""
        state.conf = 0.0
    state.top3 = top3
    return top3


class ClassifierService:
    """Single-worker classifier with per-session vote state."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        meta_path: str | Path,
        threads: int = 1,
        runtime: str = "auto",
        queue_max: int = 64,
        timeout_s: float = 3.0,
        session_ttl_s: float = 600.0,
        session_max: int = 32,
        log_path: str | Path | None = None,
        model_version_override: str = "",
        strict_version: bool = False,
        dump_dir: str | Path | None = None,
        max_body_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.threads = int(threads)
        self.runtime = runtime
        self.queue_max = int(queue_max)
        self.timeout_s = float(timeout_s)
        self.session_ttl_s = float(session_ttl_s)
        self.session_max = int(session_max)
        self.log_path = Path(log_path) if log_path else None
        self.model_version_override = model_version_override
        self.strict_version = bool(strict_version)
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.max_body_bytes = int(max_body_bytes)

        self.loaded = False
        self.load_error: str | None = None
        self.runtime_name = "unknown"
        self.warmup_ms = 0.0
        self.model_version = "unknown"
        self.model_sha256 = ""
        self.input_shape: list[int] = []
        self.output_shape: list[int] = []
        self.num_classes = 0
        self.num_nodes = NUM_NODES
        self.channels = CHANNELS
        self.max_len = MAX_LEN
        self.point_landmarks = np.asarray(POINT_LANDMARKS, dtype=np.int64)
        self.idx_to_label: dict[int, str] = {}
        self.instance_id = uuid.uuid4().hex[:8]

        self._interp: Any = None
        self._input_index = 0
        self._output_index = 0
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._jobs: queue.Queue[tuple[Window, Any, float]] = queue.Queue(
            maxsize=self.queue_max
        )
        self._stop = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._sweeper_thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle --

    def start(self) -> None:
        try:
            self._load_meta()
            self._load_interpreter()
            self._warmup()
            self.loaded = True
            self._worker_thread = threading.Thread(
                target=self._worker, name="s2s-p2-worker", daemon=True
            )
            self._worker_thread.start()
            self._sweeper_thread = threading.Thread(
                target=self._sweeper, name="s2s-p2-sweeper", daemon=True
            )
            self._sweeper_thread.start()
            log.info(
                "classifier ready: runtime=%s version=%s threads=%d warmup=%.1fms input=%s output=%s",
                self.runtime_name,
                self.model_version,
                self.threads,
                self.warmup_ms,
                self.input_shape,
                self.output_shape,
            )
        except Exception as exc:  # noqa: BLE001 - must not break /translate
            self.loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("classifier load failed: %s", exc)

    def close(self) -> None:
        self._stop.set()
        if self._interp is not None:
            try:
                self._interp.close()
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- config --

    def _load_meta(self) -> None:
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.num_classes = int(meta["num_classes"])
        self.max_len = int(meta.get("max_len", MAX_LEN))
        self.channels = int(meta.get("channels", CHANNELS))
        self.num_nodes = int(meta.get("num_nodes", NUM_NODES))
        self.idx_to_label = {
            int(k): str(v) for k, v in meta["idx_to_label"].items()
        }

        raw_points = meta.get("point_landmarks")
        if raw_points:
            points = np.asarray(raw_points, dtype=np.int64)
            if points.shape != (self.num_nodes,):
                raise ValueError(
                    f"model_meta point_landmarks has {points.size} entries, "
                    f"expected {self.num_nodes}"
                )
            if not np.array_equal(points, np.asarray(POINT_LANDMARKS, dtype=np.int64)):
                log.warning("model_meta point_landmarks differ from canonical layout")
            self.point_landmarks = points

        self.model_version = (
            self.model_version_override
            or str(meta.get("model_version", ""))
            or self.model_path.name
        )
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    def _make_interpreter(self, candidate: str) -> tuple[Any, str]:
        if candidate == "litert":
            from ai_edge_litert.interpreter import Interpreter  # type: ignore

            return (
                Interpreter(model_path=str(self.model_path), num_threads=self.threads),
                "ai-edge-litert",
            )
        if candidate == "tf":
            import tensorflow as tf  # type: ignore

            return (
                tf.lite.Interpreter(
                    model_path=str(self.model_path), num_threads=self.threads
                ),
                "tensorflow",
            )
        raise ValueError(f"unknown classifier runtime: {candidate}")

    def _load_interpreter(self) -> None:
        if self.runtime == "auto":
            candidates = ["litert", "tf"]
        else:
            candidates = [self.runtime]
        errors: list[str] = []
        for candidate in candidates:
            try:
                interp, runtime_name = self._make_interpreter(candidate)
                interp.allocate_tensors()
                input_details = interp.get_input_details()
                output_details = interp.get_output_details()
                self.input_shape = [int(v) for v in input_details[0]["shape"]]
                self.output_shape = [int(v) for v in output_details[0]["shape"]]
                expected_in = [1, self.max_len, self.channels]
                if self.input_shape != expected_in:
                    raise ValueError(
                        f"input shape {self.input_shape} != {expected_in}"
                    )
                if self.output_shape != [1, self.num_classes]:
                    raise ValueError(
                        f"output shape {self.output_shape} != [1, {self.num_classes}]"
                    )
                self._interp = interp
                self._input_index = int(input_details[0]["index"])
                self._output_index = int(output_details[0]["index"])
                self.runtime_name = runtime_name
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                log.warning("classifier runtime %s failed: %s", candidate, exc)
                if candidate == "litert":
                    try:
                        interp.close()  # type: ignore[possibly-undefined]
                    except Exception:  # noqa: BLE001
                        pass
        raise RuntimeError("no usable TFLite runtime (" + "; ".join(errors) + ")")

    def _warmup(self) -> None:
        x = np.zeros((1, self.max_len, self.channels), dtype=np.float32)
        started = time.perf_counter()
        self._interp.set_tensor(self._input_index, x)
        self._interp.invoke()
        self._interp.get_tensor(self._output_index)
        self.warmup_ms = (time.perf_counter() - started) * 1000.0

    # ------------------------------------------------------------- requests --

    def submit(self, window: Window) -> Any:
        from concurrent.futures import Future

        future: Future = Future()
        self._jobs.put_nowait((window, future, time.perf_counter()))
        return future

    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def health(self) -> dict[str, Any]:
        with self._lock:
            sessions = len(self._sessions)
        return {
            "classifier_loaded": self.loaded,
            "classifier_error": self.load_error,
            "classifier_runtime": self.runtime_name,
            "classifier_version": self.model_version,
            "classifier_model_sha256": self.model_sha256,
            "classifier_model_path": str(self.model_path),
            "classifier_threads": self.threads,
            "classifier_queue_depth": self.queue_depth(),
            "classifier_queue_max": self.queue_max,
            "classifier_sessions": sessions,
            "classifier_warmup_ms": round(self.warmup_ms, 2),
            "classifier_input_shape": self.input_shape,
            "classifier_output_shape": self.output_shape,
            "server_instance": self.instance_id,
        }

    # --------------------------------------------------------------- worker --

    def _worker(self) -> None:
        log.info("classifier worker started (max=%d)", self.queue_max)
        while not self._stop.is_set():
            try:
                window, future, enqueued_at = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            queue_ms = (time.perf_counter() - enqueued_at) * 1000.0
            try:
                outcome = self._process(window, queue_ms)
            except ProtocolError as exc:
                if not future.cancelled():
                    future.set_exception(exc)
                self._jobs.task_done()
                continue
            except Exception as exc:  # noqa: BLE001
                if not future.cancelled():
                    future.set_exception(
                        ProtocolError(500, f"classifier failure: {type(exc).__name__}: {exc}")
                    )
                self._jobs.task_done()
                continue
            if not future.cancelled():
                future.set_result(outcome)
            self._jobs.task_done()
        log.info("classifier worker stopped")

    def _get_or_create_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                if len(self._sessions) >= self.session_max:
                    oldest = min(self._sessions, key=lambda key: self._sessions[key].last_seen)
                    self._sessions.pop(oldest, None)
                state = SessionState()
                self._sessions[session_id] = state
            return state

    def _process(self, window: Window, queue_ms: float) -> RecognitionOutcome:
        state = self._get_or_create_session(window.session_id)
        with self._lock:
            try:
                accept_window(state, window)
            except ProtocolError:
                if state.requests == 0 and state.last_window_id == 0:
                    self._sessions.pop(window.session_id, None)
                raise

        frames = window.values
        if window.point_count == 543:
            frames = frames[:, self.point_landmarks, :]

        started = time.perf_counter()
        features = preprocess(
            frames,
            point_landmarks=self.point_landmarks,
            num_nodes=self.num_nodes,
            channels=self.channels,
            max_len=self.max_len,
        )
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        if features is None:
            return RecognitionOutcome(
                status="insufficient_frames",
                inferred=False,
                has_hand=window.has_hand,
                frame_count=window.frame_count,
                queue_ms=queue_ms,
                preprocess_ms=preprocess_ms,
                vote_len=len(state.history),
            )

        started = time.perf_counter()
        self._interp.set_tensor(self._input_index, features[np.newaxis, ...])
        self._interp.invoke()
        logits = self._interp.get_tensor(self._output_index)[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        probs = softmax(logits)

        started = time.perf_counter()
        with self._lock:
            state = self._sessions.get(window.session_id)
            if state is None:
                state = SessionState(episode_id=window.episode_id)
                self._sessions[window.session_id] = state
            top3 = apply_vote(
                state, probs, self.idx_to_label, window.has_hand
            )
            label = state.label
            confidence = state.conf
            vote_len = len(state.history)
        vote_ms = (time.perf_counter() - started) * 1000.0

        return RecognitionOutcome(
            status="ok",
            inferred=True,
            label=label,
            confidence=confidence,
            top3=top3,
            has_hand=window.has_hand,
            frame_count=window.frame_count,
            queue_ms=queue_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            vote_ms=vote_ms,
            vote_len=vote_len,
        )

    def _sweeper(self) -> None:
        while not self._stop.wait(60.0):
            now = time.monotonic()
            with self._lock:
                expired = [
                    key
                    for key, state in self._sessions.items()
                    if now - state.last_seen > self.session_ttl_s
                ]
                for key in expired:
                    self._sessions.pop(key, None)
                while len(self._sessions) > self.session_max:
                    oldest = min(
                        self._sessions, key=lambda key: self._sessions[key].last_seen
                    )
                    self._sessions.pop(oldest, None)

    # --------------------------------------------------------------- helpers --

    def debug_classify(self, frames: np.ndarray, has_hand: bool = True) -> dict[str, Any]:
        """Run preprocess + inference without touching session/vote state."""
        if not self.loaded:
            raise RuntimeError(self.load_error or "classifier not loaded")
        selected = frames
        if frames.shape[1] == 543:
            selected = frames[:, self.point_landmarks, :]
        started = time.perf_counter()
        features = preprocess(
            selected,
            point_landmarks=self.point_landmarks,
            num_nodes=self.num_nodes,
            channels=self.channels,
            max_len=self.max_len,
        )
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        if features is None:
            return {"inferred": False, "status": "insufficient_frames"}
        started = time.perf_counter()
        self._interp.set_tensor(self._input_index, features[np.newaxis, ...])
        self._interp.invoke()
        logits = self._interp.get_tensor(self._output_index)[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        probs = softmax(logits)
        order = np.argsort(-probs, kind="stable")[:3]
        return {
            "inferred": True,
            "status": "ok",
            "top3": [
                {"label": self.idx_to_label.get(int(i), "?"), "confidence": float(probs[int(i)])}
                for i in order
            ],
            "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "runtime": self.runtime_name,
        }

    def log_record(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 - logging must not break requests
            log.exception("failed to write p2 jsonl")

    def dump_window(self, window: Window, body: bytes) -> None:
        if self.dump_dir is None:
            return
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            name = f"{window.session_id}-{window.window_id:08d}-{window.episode_id:08d}.bin"
            (self.dump_dir / name).write_bytes(body)
        except Exception:  # noqa: BLE001
            log.exception("failed to dump window")
