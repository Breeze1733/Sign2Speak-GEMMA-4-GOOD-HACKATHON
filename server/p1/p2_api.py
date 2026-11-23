"""P2 ``POST /recognize`` route: S2LW window in, RecognitionEvent out."""

from __future__ import annotations

import asyncio
import hashlib
import queue
import time

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from auth import require_token
from p2_protocol import ProtocolError, decode_window
from p2_recognizer import ClassifierService, RecognitionOutcome

router = APIRouter()


class Top3Item(BaseModel):
    label: str
    confidence: float


class RecognizeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: int = Field(default=1, alias="schema")
    request_id: str
    session_id: str
    window_id: int
    episode_id: int
    status: str
    inferred: bool
    label: str
    confidence: float
    top3: list[Top3Item]
    has_hand: bool
    frame_count: int
    queue_ms: float
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    vote_ms: float
    total_ms: float
    model_version: str
    server_instance: str


def _service(request: Request) -> ClassifierService:
    service = getattr(request.app.state, "classifier", None)
    if service is None or not getattr(service, "loaded", False):
        detail = getattr(service, "load_error", None) or "classifier not loaded"
        raise HTTPException(status_code=503, detail=detail)
    return service


def _base_record(
    service: ClassifierService, session_id: str, request_id: str, body: bytes
) -> dict:
    return {
        "ts": time.time(),
        "request_id": request_id,
        "session_id": session_id,
        "bytes": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "model_version": service.model_version,
        "server_instance": service.instance_id,
    }


def _log_error(
    service: ClassifierService,
    record: dict,
    http_status: int,
    error: str,
) -> None:
    service.log_record(
        {**record, "status": "error", "http_status": http_status, "error": error}
    )


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    request: Request,
    authorization: str | None = Header(default=None),
) -> RecognizeResponse:
    require_token(authorization)
    service = _service(request)

    session_id = (request.headers.get("X-S2S-Session-Id") or "").strip()
    request_id = (request.headers.get("X-S2S-Request-Id") or "").strip()
    client_version = (request.headers.get("X-S2S-Model-Version") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="X-S2S-Session-Id required")

    body = await request.body()
    record = _base_record(service, session_id, request_id, body)
    if len(body) > service.max_body_bytes:
        _log_error(service, record, 413, "body too large")
        raise HTTPException(status_code=413, detail="body too large")

    started = time.perf_counter()
    try:
        window = decode_window(body, session_id=session_id, request_id=request_id)
    except ProtocolError as exc:
        _log_error(service, record, exc.http_status, exc.detail)
        raise HTTPException(status_code=exc.http_status, detail=exc.detail) from None
    decode_ms = (time.perf_counter() - started) * 1000.0
    record.update(
        {
            "window_id": window.window_id,
            "episode_id": window.episode_id,
            "frame_count": window.frame_count,
            "point_count": window.point_count,
            "coord_count": window.coord_count,
            "has_hand": window.has_hand,
        }
    )

    if service.strict_version and client_version and client_version != service.model_version:
        _log_error(service, record, 409, "model version mismatch")
        raise HTTPException(
            status_code=409,
            detail=(
                f"model version mismatch: client={client_version} "
                f"server={service.model_version}"
            ),
        )

    if service.dump_dir is not None:
        service.dump_window(window, body)

    try:
        future = service.submit(window)
    except queue.Full:
        _log_error(service, record, 503, "classifier queue full")
        raise HTTPException(status_code=503, detail="classifier queue full") from None

    try:
        outcome: RecognitionOutcome = await asyncio.wait_for(
            asyncio.wrap_future(future), timeout=service.timeout_s
        )
    except asyncio.TimeoutError:
        future.cancel()
        _log_error(service, record, 504, "classification timeout")
        raise HTTPException(
            status_code=504, detail="classification timeout"
        ) from None
    except ProtocolError as exc:
        _log_error(service, record, exc.http_status, exc.detail)
        raise HTTPException(status_code=exc.http_status, detail=exc.detail) from None
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _log_error(service, record, 500, f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500, detail=f"classifier failure: {type(exc).__name__}"
        ) from None

    total_ms = (
        outcome.queue_ms
        + decode_ms
        + outcome.preprocess_ms
        + outcome.inference_ms
        + outcome.vote_ms
    )
    service.log_record(
        {
            **record,
            "status": outcome.status,
            "http_status": 200,
            "inferred": outcome.inferred,
            "label": outcome.label,
            "confidence": round(outcome.confidence, 4),
            "vote_len": outcome.vote_len,
            "queue_ms": round(outcome.queue_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "preprocess_ms": round(outcome.preprocess_ms, 2),
            "inference_ms": round(outcome.inference_ms, 2),
            "vote_ms": round(outcome.vote_ms, 2),
            "total_ms": round(total_ms, 2),
        }
    )

    return RecognizeResponse(
        request_id=request_id,
        session_id=session_id,
        window_id=window.window_id,
        episode_id=window.episode_id,
        status=outcome.status,
        inferred=outcome.inferred,
        label=outcome.label,
        confidence=outcome.confidence,
        top3=[
            Top3Item(label=label, confidence=confidence)
            for label, confidence in outcome.top3
        ],
        has_hand=outcome.has_hand,
        frame_count=outcome.frame_count,
        queue_ms=round(outcome.queue_ms, 2),
        decode_ms=round(decode_ms, 2),
        preprocess_ms=round(outcome.preprocess_ms, 2),
        inference_ms=round(outcome.inference_ms, 2),
        vote_ms=round(outcome.vote_ms, 2),
        total_ms=round(total_ms, 2),
        model_version=service.model_version,
        server_instance=service.instance_id,
    )
