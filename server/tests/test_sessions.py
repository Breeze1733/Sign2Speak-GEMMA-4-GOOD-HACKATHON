from __future__ import annotations

import numpy as np
import pytest

from p2_protocol import FLAG_HAS_HAND, FLAG_RESET_VOTE, ProtocolError, Window
from p2_recognizer import SessionState, accept_window


def window(window_id: int, episode_id: int = 1, flags: int = FLAG_HAS_HAND) -> Window:
    return Window(
        session_id="s",
        request_id="r",
        window_id=window_id,
        episode_id=episode_id,
        first_frame_id=window_id,
        frame_count=1,
        point_count=118,
        coord_count=3,
        flags=flags,
        frame_ids=np.asarray([window_id], dtype=np.uint64),
        timestamps_ms=np.asarray([0], dtype=np.uint64),
        values=np.zeros((1, 118, 3), dtype=np.float32),
        masks=np.zeros((1, 16), dtype=np.uint8),
        body_bytes=0,
        body_sha256="",
    )


def test_monotonic_windows_and_gaps():
    state = SessionState()
    accept_window(state, window(1))
    accept_window(state, window(5))
    assert state.last_window_id == 5
    with pytest.raises(ProtocolError) as exc:
        accept_window(state, window(5))
    assert exc.value.http_status == 409


def test_new_episode_resets_history_and_window():
    state = SessionState()
    state.history.append(0)
    accept_window(state, window(10, episode_id=1))
    accept_window(state, window(1, episode_id=2))
    assert state.episode_id == 2
    assert state.last_window_id == 1
    assert len(state.history) == 0


def test_old_episode_is_rejected():
    state = SessionState()
    accept_window(state, window(1, episode_id=3))
    with pytest.raises(ProtocolError) as exc:
        accept_window(state, window(2, episode_id=2))
    assert exc.value.http_status == 409


def test_reset_flag_clears_history_without_changing_episode():
    state = SessionState(episode_id=1, last_window_id=1)
    state.history.append(0)
    accept_window(state, window(2, episode_id=1, flags=FLAG_HAS_HAND | FLAG_RESET_VOTE))
    assert state.episode_id == 1
    assert len(state.history) == 0
