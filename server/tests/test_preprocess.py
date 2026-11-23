from __future__ import annotations

import numpy as np
import pytest

from p2_recognizer import (
    CHANNELS,
    MAX_LEN,
    NOSE_NODE_IDX,
    NUM_NODES,
    PAD,
    POINT_LANDMARKS,
    preprocess,
)


def reference_preprocess(frames: np.ndarray) -> np.ndarray | None:
    """Frozen copy of sign2speak-PC-version/recognize_video.py::preprocess."""
    pts = frames[:, POINT_LANDMARKS]
    valid = ~np.isnan(pts[:, :, 0]).all(axis=1)
    if int(valid.sum()) < 3:
        return None
    vp = pts[valid]

    nose = vp[:, NOSE_NODE_IDX, :]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(nose, axis=0)
    mean = np.where(np.isnan(mean), 0.5, mean).astype(np.float32)

    diff = vp - mean
    with np.errstate(invalid="ignore"):
        std = np.sqrt(np.nanmean(diff * diff, axis=(0, 1)))
    std = np.where(np.isnan(std) | (std < 1e-6), 1.0, std).astype(np.float32)

    x = (vp - mean) / std
    x = x[:, :, :2]

    dx = np.zeros_like(x)
    if len(x) > 1:
        dx[:-1] = x[1:] - x[:-1]
    dx2 = np.zeros_like(x)
    if len(x) > 2:
        dx2[:-2] = x[2:] - x[:-2]

    feats = np.concatenate(
        [
            x.reshape(-1, 2 * NUM_NODES),
            dx.reshape(-1, 2 * NUM_NODES),
            dx2.reshape(-1, 2 * NUM_NODES),
        ],
        axis=1,
    ).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    padded = np.full((MAX_LEN, CHANNELS), PAD, dtype=np.float32)
    padded[: len(feats)] = feats
    return padded


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_543_matches_reference(seed):
    rng = np.random.default_rng(seed)
    frames = rng.random((rng.integers(10, 64), 543, 3), dtype=np.float32)
    # Drop a hand and some face points to exercise NaN propagation.
    frames[:, 468:489, :] = np.nan
    frames[::5, 30:50, :] = np.nan

    expected = reference_preprocess(frames)
    actual = preprocess(frames)
    assert expected is not None and actual is not None
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_118_selected_points_match_543():
    rng = np.random.default_rng(42)
    frames = rng.random((32, 543, 3), dtype=np.float32)
    frames[3:8, 468:, :] = np.nan
    selected = frames[:, np.asarray(POINT_LANDMARKS)]

    np.testing.assert_allclose(preprocess(selected), preprocess(frames), atol=0)


def test_insufficient_valid_frames():
    frames = np.full((8, 118, 3), np.nan, dtype=np.float32)
    assert preprocess(frames) is None


def test_padding_and_prefix():
    rng = np.random.default_rng(9)
    frames = rng.random((10, 118, 3), dtype=np.float32)
    out = preprocess(frames)
    assert out.shape == (64, 708)
    assert np.all(out[10:] == PAD)
    assert np.all(np.isfinite(out[:10]))


def test_two_coordinate_input():
    rng = np.random.default_rng(11)
    frames = rng.random((12, 118, 2), dtype=np.float32)
    out = preprocess(frames)
    assert out.shape == (64, 708)
    np.testing.assert_allclose(
        out, preprocess(np.concatenate([frames, rng.random((12, 118, 1), dtype=np.float32)], axis=2)),
        atol=0,
    )
