from __future__ import annotations

import numpy as np
import pytest

from p2_protocol import (
    FLAG_HAS_HAND,
    HEADER_BYTES,
    ProtocolError,
    Window,
    bits_to_mask,
    body_bytes,
    decode_window,
    encode_window,
    frame_stride,
    mask_bytes,
)


def make_window(frames=10, points=118, coords=3, window_id=1, episode_id=1) -> Window:
    rng = np.random.default_rng(3)
    values = rng.random((frames, points, coords), dtype=np.float32)
    values[:2, points - 42 :, :] = np.nan
    masks = np.stack(
        [bits_to_mask(~np.isnan(values[i, :, 0]), points) for i in range(frames)]
    )
    frame_ids = np.arange(1, frames + 1, dtype=np.uint64)
    return Window(
        session_id="s",
        request_id="r",
        window_id=window_id,
        episode_id=episode_id,
        first_frame_id=int(frame_ids[0]),
        frame_count=frames,
        point_count=points,
        coord_count=coords,
        flags=FLAG_HAS_HAND,
        frame_ids=frame_ids,
        timestamps_ms=frame_ids * np.uint64(33),
        values=values,
        masks=masks,
        body_bytes=body_bytes(frames, points, coords),
        body_sha256="",
    )


def test_sizes():
    assert mask_bytes(118) == 16
    assert mask_bytes(543) == 68
    assert frame_stride(118, 3) == 16 + 16 + 118 * 3 * 4
    assert body_bytes(64, 118, 3) == HEADER_BYTES + 64 * 1448
    assert body_bytes(64, 118, 2) == HEADER_BYTES + 64 * 976


@pytest.mark.parametrize("points,coords", [(118, 3), (118, 2), (543, 3)])
def test_round_trip(points, coords):
    window = make_window(frames=12, points=points, coords=coords)
    body = encode_window(window)
    decoded = decode_window(body, session_id="s", request_id="r")

    assert len(body) == body_bytes(12, points, coords)
    assert decoded.window_id == window.window_id
    assert decoded.episode_id == window.episode_id
    assert decoded.frame_count == 12
    assert np.array_equal(decoded.frame_ids, window.frame_ids)
    assert np.array_equal(np.isnan(decoded.values), np.isnan(window.values))
    np.testing.assert_allclose(
        decoded.values[np.isfinite(window.values)],
        window.values[np.isfinite(window.values)],
        rtol=0,
        atol=0,
    )


def test_decode_restores_nan_for_absent_points():
    window = make_window(frames=4)
    decoded = decode_window(encode_window(window), session_id="s")
    assert np.isnan(decoded.values[0, -1, 0])
    assert np.isfinite(decoded.values[0, 0, 0])


def test_bad_magic():
    body = bytearray(encode_window(make_window()))
    body[0:4] = b"XXXX"
    with pytest.raises(ProtocolError) as exc:
        decode_window(bytes(body), session_id="s")
    assert exc.value.http_status == 400


def test_bad_length():
    body = encode_window(make_window())
    with pytest.raises(ProtocolError) as exc:
        decode_window(body[:-1], session_id="s")
    assert exc.value.http_status == 400


def test_bad_version():
    body = bytearray(encode_window(make_window()))
    body[4] = 9
    with pytest.raises(ProtocolError) as exc:
        decode_window(bytes(body), session_id="s")
    assert exc.value.http_status == 400


def test_mask_padding_must_be_zero():
    window = make_window(frames=1)
    body = bytearray(encode_window(window))
    # 118 points: byte 14 holds points 112..119; bits 6 and 7 are padding.
    mask_offset = HEADER_BYTES + 16
    body[mask_offset + 14] |= 1 << 6
    with pytest.raises(ProtocolError) as exc:
        decode_window(bytes(body), session_id="s")
    assert exc.value.http_status == 422


def test_first_frame_id_mismatch():
    body = bytearray(encode_window(make_window(frames=2)))
    body[32:40] = (99).to_bytes(8, "little")
    with pytest.raises(ProtocolError) as exc:
        decode_window(bytes(body), session_id="s")
    assert exc.value.http_status == 400
