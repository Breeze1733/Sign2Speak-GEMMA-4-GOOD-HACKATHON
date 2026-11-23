from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth
from p2_api import router
from p2_protocol import (
    FLAG_HAS_HAND,
    ProtocolError,
    Window,
    bits_to_mask,
    body_bytes,
    encode_window,
)
from p2_recognizer import RecognitionOutcome


def make_body(frames: int = 10) -> bytes:
    points, coords = 118, 3
    values = np.zeros((frames, points, coords), dtype=np.float32)
    values[:, 76:, :] = np.nan
    masks = np.stack(
        [bits_to_mask(~np.isnan(values[i, :, 0]), points) for i in range(frames)]
    )
    ids = np.arange(1, frames + 1, dtype=np.uint64)
    window = Window(
        session_id="s",
        request_id="r",
        window_id=1,
        episode_id=1,
        first_frame_id=int(ids[0]),
        frame_count=frames,
        point_count=points,
        coord_count=coords,
        flags=FLAG_HAS_HAND,
        frame_ids=ids,
        timestamps_ms=ids * np.uint64(33),
        values=values,
        masks=masks,
        body_bytes=body_bytes(frames, points, coords),
        body_sha256="",
    )
    return encode_window(window)


class FakeService:
    loaded = True
    load_error = None
    model_version = "test-v1"
    instance_id = "srv-test"
    strict_version = False
    dump_dir = None
    max_body_bytes = 2 * 1024 * 1024
    timeout_s = 3.0

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.submit_error: Exception | None = None
        self.raise_queue_full = False

    def submit(self, window: Window) -> Future:
        if self.raise_queue_full:
            import queue

            raise queue.Full()
        if self.submit_error is not None:
            future: Future = Future()
            future.set_exception(self.submit_error)
            return future
        future = Future()
        future.set_result(
            RecognitionOutcome(
                status="ok",
                inferred=True,
                label="hello",
                confidence=0.9,
                top3=[("hello", 0.9), ("water", 0.05)],
                has_hand=True,
                frame_count=window.frame_count,
                queue_ms=1.0,
                preprocess_ms=2.0,
                inference_ms=3.0,
                vote_ms=0.1,
                vote_len=1,
            )
        )
        return future

    def log_record(self, record: dict) -> None:
        self.records.append(record)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth, "TOKEN", "")
    app = FastAPI()
    app.include_router(router)
    service = FakeService()
    app.state.classifier = service
    with TestClient(app) as test_client:
        yield test_client, service


HEADERS = {
    "X-S2S-Session-Id": "session-1",
    "X-S2S-Request-Id": "m3-1",
    "X-S2S-Model-Version": "test-v1",
}


def test_success(client):
    test_client, service = client
    response = test_client.post("/recognize", content=make_body(), headers=HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "hello"
    assert payload["inferred"] is True
    assert payload["window_id"] == 1
    assert payload["top3"][0] == {"label": "hello", "confidence": 0.9}
    assert service.records and service.records[-1]["http_status"] == 200


def test_missing_session_header(client):
    test_client, _ = client
    response = test_client.post("/recognize", content=make_body(), headers={})
    assert response.status_code == 400


def test_bad_magic_is_logged(client):
    test_client, service = client
    body = bytearray(make_body())
    body[0:4] = b"NOPE"
    response = test_client.post("/recognize", content=bytes(body), headers=HEADERS)
    assert response.status_code == 400
    assert service.records[-1]["http_status"] == 400


def test_body_too_large(client):
    test_client, service = client
    service.max_body_bytes = 10
    response = test_client.post("/recognize", content=make_body(), headers=HEADERS)
    assert response.status_code == 413


def test_classifier_not_loaded(client):
    test_client, service = client
    service.loaded = False
    service.load_error = "no model"
    response = test_client.post("/recognize", content=make_body(), headers=HEADERS)
    assert response.status_code == 503


def test_queue_full_maps_to_503(client):
    test_client, service = client
    service.raise_queue_full = True
    response = test_client.post("/recognize", content=make_body(), headers=HEADERS)
    assert response.status_code == 503


def test_stale_window_maps_to_409(client):
    test_client, service = client
    service.submit_error = ProtocolError(409, "stale window")
    response = test_client.post("/recognize", content=make_body(), headers=HEADERS)
    assert response.status_code == 409


def test_token_required(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(auth, "TOKEN", "secret")
    assert test_client.post(
        "/recognize", content=make_body(), headers=HEADERS
    ).status_code == 401
    response = test_client.post(
        "/recognize",
        content=make_body(),
        headers={**HEADERS, "Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
