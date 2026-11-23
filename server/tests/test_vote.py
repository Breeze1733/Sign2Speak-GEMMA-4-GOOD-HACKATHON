from __future__ import annotations

from collections import deque

import numpy as np

from p2_recognizer import SessionState, apply_vote

LABELS = {0: "hello", 1: "hungry", 2: "water", 3: "thanks"}


def probs(*values: float) -> np.ndarray:
    return np.asarray(values + (0.0,) * (4 - len(values)), dtype=np.float32)


def test_confident_prediction_enters_history():
    state = SessionState()
    top3 = apply_vote(state, probs(0.9, 0.05, 0.03, 0.02), LABELS, has_hand=True)
    assert state.history == deque([0])
    assert state.label == "hello"
    assert state.conf == np.float32(0.9)
    assert top3[0][0] == "hello"


def test_majority_vote_and_first_tie_wins():
    state = SessionState()
    for idx in (1, 2, 1):
        p = np.zeros(4, dtype=np.float32)
        p[idx] = 0.9
        apply_vote(state, p, LABELS, has_hand=True)
    assert state.label == "hungry"
    # Tie between 1 and 2 in first-occurrence order -> first inserted stays.
    state2 = SessionState()
    for idx in (1, 2):
        p = np.zeros(4, dtype=np.float32)
        p[idx] = 0.9
        apply_vote(state2, p, LABELS, has_hand=True)
    assert state2.label == "hungry"


def test_top3_is_stable_on_ties():
    state = SessionState()
    p = np.asarray([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    top3 = apply_vote(state, p, LABELS, has_hand=True)
    assert len(top3) == 3
    assert [label for label, _ in top3[:2]] == ["hello", "hungry"]


def test_low_confidence_clears_label_but_keeps_history_with_hand():
    state = SessionState()
    apply_vote(state, np.asarray([0.9, 0.1, 0.0, 0.0], np.float32), LABELS, True)
    apply_vote(state, np.asarray([0.1, 0.1, 0.1, 0.1], np.float32), LABELS, True)
    assert state.label == ""
    assert state.conf == 0.0
    assert len(state.history) == 1


def test_no_hand_clears_history():
    state = SessionState()
    apply_vote(state, np.asarray([0.9, 0.1, 0.0, 0.0], np.float32), LABELS, True)
    apply_vote(state, np.asarray([0.9, 0.1, 0.0, 0.0], np.float32), LABELS, False)
    assert state.label == ""
    assert len(state.history) == 0


def test_history_is_bounded_to_eight():
    state = SessionState()
    for i in range(12):
        p = np.zeros(4, dtype=np.float32)
        p[i % 4] = 0.9
        apply_vote(state, p, LABELS, True)
    assert len(state.history) == 8
