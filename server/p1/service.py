"""Sign2Speak P1 remote service: ASL glosses -> English sentence.

Implements the "卸载组句" (offload sentence generation) plan from the proposal:
the phone keeps M1-M4 and M6, sends an ordered gloss sequence over HTTP, and
receives a generated sentence plus server-side timings.

Concurrency model (matches proposal 7.1 "服务器按顺序排队执行"):
  * one preloaded Engine, one dedicated worker thread
  * bounded FIFO queue, requests executed strictly in arrival order
  * queue_ms = time spent waiting before inference started
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from auth import require_token
from litert_backend import GenerationResult, GlossTranslator
from p2_api import router as p2_router
from p2_recognizer import ClassifierService

# ---------------------------------------------------------------- config ----

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/sign2speak/models/gemma-4-E2B-it.litertlm")
MODEL_VERSION = os.getenv("MODEL_VERSION", Path(MODEL_PATH).name)
CACHE_DIR = os.getenv("CACHE_DIR", "/opt/sign2speak/cache")
LOG_PATH = os.getenv("LOG_PATH", "/opt/sign2speak/logs/p1.jsonl")
THREADS = int(os.getenv("THREADS", "4"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "64"))
MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", "2048"))
TIMEOUT_S = float(os.getenv("TIMEOUT_S", "60"))
QUEUE_MAX = int(os.getenv("QUEUE_MAX", "8"))

# ---------------------------------------------------------------- P2 config --

P2_ENABLED = os.getenv("P2_ENABLED", "1") == "1"
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH", "/opt/sign2speak/models/classifier.tflite"
)
CLASSIFIER_META = os.getenv(
    "CLASSIFIER_META", "/opt/sign2speak/models/model_meta.json"
)
CLASSIFIER_VERSION = os.getenv("CLASSIFIER_VERSION", "")
CLASSIFIER_THREADS = int(os.getenv("CLASSIFIER_THREADS", "1"))
CLASSIFIER_RUNTIME = os.getenv("CLASSIFIER_RUNTIME", "auto")
P2_LOG_PATH = os.getenv("P2_LOG_PATH", "/opt/sign2speak/logs/p2.jsonl")
P2_QUEUE_MAX = int(os.getenv("P2_QUEUE_MAX", "64"))
P2_TIMEOUT_S = float(os.getenv("P2_TIMEOUT_S", "3"))
P2_SESSION_TTL_S = float(os.getenv("P2_SESSION_TTL_S", "600"))
P2_SESSION_MAX = int(os.getenv("P2_SESSION_MAX", "32"))
P2_MAX_BODY_BYTES = int(os.getenv("P2_MAX_BODY_BYTES", str(2 * 1024 * 1024)))
P2_DUMP_WINDOWS = os.getenv("P2_DUMP_WINDOWS", "0") == "1"
P2_DUMP_DIR = os.getenv("P2_DUMP_DIR", "/opt/sign2speak/results/windows")
P2_STRICT_VERSION = os.getenv("P2_STRICT_VERSION", "0") == "1"

try:
    LITERT_LM_VERSION = pkg_version("litert-lm-api")
except PackageNotFoundError:  # pragma: no cover
    LITERT_LM_VERSION = "unknown"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("s2s.service")

# ---------------------------------------------------------------- state -----

translator: GlossTranslator | None = None
translator_error: str | None = None
jobs: "queue.Queue[tuple[dict, Future]]" = queue.Queue(maxsize=QUEUE_MAX)
_stop = threading.Event()
_log_lock = threading.Lock()


def _jsonl(record: dict) -> None:
    try:
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - logging must never break a request
        log.exception("failed to write jsonl log")


def _load_model() -> None:
    global translator, translator_error
    try:
        translator = GlossTranslator(
            MODEL_PATH,
            thread_count=THREADS,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            max_num_tokens=MAX_NUM_TOKENS or None,
            cache_dir=CACHE_DIR,
        )
        log.info("model ready: %s (load_ms=%.0f)", MODEL_PATH, translator.load_ms)
    except Exception as exc:  # noqa: BLE001
        translator_error = str(exc)
        log.exception("model load failed: %s", exc)


def _worker() -> None:
    log.info("queue worker started (max=%d)", QUEUE_MAX)
    while not _stop.is_set():
        try:
            payload, future = jobs.get(timeout=0.5)
        except queue.Empty:
            continue
        queue_ms = (time.perf_counter() - payload["enqueued_at"]) * 1000.0
        result = GenerationResult("", "model_unavailable", 0.0)
        if translator is not None:
            try:
                result = translator.generate(payload["glosses"], payload["style"])
            except Exception as exc:  # noqa: BLE001
                result = GenerationResult("", f"generation_error: {exc}", 0.0)
        if not future.cancelled():
            future.set_result((result, queue_ms))
        jobs.task_done()
    log.info("queue worker stopped")


def _load_classifier(service: ClassifierService) -> None:
    # Classifier failures must never prevent the Gemma loader from running.
    try:
        service.start()
    except Exception:  # noqa: BLE001
        service.loaded = False
        service.load_error = "classifier loader crashed"
        log.exception("classifier loader crashed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, name="s2s-worker", daemon=True).start()
    threading.Thread(target=_load_model, name="s2s-loader", daemon=True).start()

    if P2_ENABLED:
        classifier = ClassifierService(
            model_path=CLASSIFIER_PATH,
            meta_path=CLASSIFIER_META,
            threads=CLASSIFIER_THREADS,
            runtime=CLASSIFIER_RUNTIME,
            queue_max=P2_QUEUE_MAX,
            timeout_s=P2_TIMEOUT_S,
            session_ttl_s=P2_SESSION_TTL_S,
            session_max=P2_SESSION_MAX,
            log_path=P2_LOG_PATH,
            model_version_override=CLASSIFIER_VERSION,
            strict_version=P2_STRICT_VERSION,
            dump_dir=P2_DUMP_DIR if P2_DUMP_WINDOWS else None,
            max_body_bytes=P2_MAX_BODY_BYTES,
        )
        app.state.classifier = classifier
        threading.Thread(
            target=_load_classifier,
            args=(classifier,),
            name="s2s-p2-loader",
            daemon=True,
        ).start()
    else:
        app.state.classifier = None

    yield
    _stop.set()
    if translator is not None:
        translator.close()
    classifier = getattr(app.state, "classifier", None)
    if classifier is not None:
        classifier.close()


app = FastAPI(
    title="Sign2Speak P1/P2 remote service (Gemma G2T + TFLite M3)",
    version="1.1",
    lifespan=lifespan,
)
app.include_router(p2_router)

# ---------------------------------------------------------------- schemas ---


class TranslateRequest(BaseModel):
    request_id: str = Field(..., description="client-generated id, echoed in logs")
    session_id: str = ""
    sentence_index: int = 0
    glosses: list[str] = Field(default_factory=list)
    style: str = "friendly"
    model_version: str | None = None


class TranslateResponse(BaseModel):
    request_id: str
    session_id: str
    sentence_index: int
    sentence: str
    status: str
    queue_ms: float
    inference_ms: float
    total_ms: float
    model_version: str
    litert_lm_version: str


# ---------------------------------------------------------------- routes ----


@app.get("/healthz")
def healthz() -> dict:
    payload = {
        "status": "ok" if translator is not None else "loading",
        "model_loaded": translator is not None,
        "model_error": translator_error,
        "model_path": MODEL_PATH,
        "model_version": MODEL_VERSION,
        "backend": "cpu",
        "threads": THREADS,
        "litert_lm_version": LITERT_LM_VERSION,
        "queue_depth": jobs.qsize(),
        "queue_max": QUEUE_MAX,
    }
    classifier = getattr(app.state, "classifier", None)
    if classifier is not None:
        payload.update(classifier.health())
    else:
        payload.update(
            {
                "classifier_loaded": False,
                "classifier_error": "disabled" if not P2_ENABLED else "not initialized",
            }
        )
    return payload


@app.post("/translate", response_model=TranslateResponse)
async def translate(
    req: TranslateRequest,
    _: None = Depends(require_token),
) -> TranslateResponse:
    if translator is None:
        detail = translator_error or "model still loading"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    if not req.glosses:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="empty glosses")

    future: Future = Future()
    payload = {
        "enqueued_at": time.perf_counter(),
        "glosses": req.glosses,
        "style": req.style,
    }
    try:
        jobs.put_nowait((payload, future))
    except queue.Full:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="queue full"
        ) from None

    try:
        result, queue_ms = await asyncio.wait_for(asyncio.wrap_future(future), timeout=TIMEOUT_S)
    except asyncio.TimeoutError:
        future.cancel()
        _jsonl({**_log_base(req), "status": "timeout", "timeout_s": TIMEOUT_S})
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="inference timeout"
        ) from None

    total_ms = queue_ms + result.inference_ms
    record = {
        **_log_base(req),
        "status": result.status,
        "queue_ms": round(queue_ms, 2),
        "inference_ms": round(result.inference_ms, 2),
        "total_ms": round(total_ms, 2),
        "sentence": result.text,
    }
    _jsonl(record)

    if result.status.startswith("generation_error"):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=result.status
        )

    return TranslateResponse(
        request_id=req.request_id,
        session_id=req.session_id,
        sentence_index=req.sentence_index,
        sentence=result.text,
        status=result.status,
        queue_ms=round(queue_ms, 2),
        inference_ms=round(result.inference_ms, 2),
        total_ms=round(total_ms, 2),
        model_version=MODEL_VERSION,
        litert_lm_version=LITERT_LM_VERSION,
    )


def _log_base(req: TranslateRequest) -> dict:
    return {
        "ts": time.time(),
        "request_id": req.request_id,
        "session_id": req.session_id,
        "sentence_index": req.sentence_index,
        "style": req.style,
        "glosses": req.glosses,
    }
