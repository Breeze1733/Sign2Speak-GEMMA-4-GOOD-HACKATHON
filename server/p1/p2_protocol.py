"""S2LW binary protocol for P2 landmark-window uploads.

Wire format (all integers/floats little-endian):

  40-byte header
    magic 4s "S2LW" | version u8 | header_bytes u8
    point_count u16 | coord_count u16 | frame_count u16
    flags u16 | reserved u16
    window_id u64 | episode_id u64 | first_frame_id u64

  per frame
    frame_id u64 | timestamp_ms u64
    validity mask, align4(ceil(point_count / 8)) bytes, LSB-first
    point_count * coord_count float32, point-major

A missing point has mask bit 0 and all coordinates written as 0.0; the
server restores NaN before preprocessing. A frame is valid when any mask bit
is set, which is equivalent to Android's "any x is not NaN".
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"S2LW"
VERSION = 1
HEADER_BYTES = 40
MAX_FRAMES = 64
ALLOWED_POINT_COUNTS = (118, 543)
ALLOWED_COORD_COUNTS = (2, 3)

FLAG_HAS_HAND = 1 << 0
FLAG_RESET_VOTE = 1 << 1
FLAG_FULL_543 = 1 << 2
_FLAG_KNOWN = FLAG_HAS_HAND | FLAG_RESET_VOTE | FLAG_FULL_543

_HEADER = struct.Struct("<4sBBHHHHHQQQ")
_FRAME_IDS = struct.Struct("<QQ")


class ProtocolError(Exception):
    """A request-level error with the HTTP status the API should return."""

    def __init__(self, http_status: int, detail: str):
        super().__init__(detail)
        self.http_status = int(http_status)
        self.detail = detail


@dataclass
class Window:
    session_id: str
    request_id: str
    window_id: int
    episode_id: int
    first_frame_id: int
    frame_count: int
    point_count: int
    coord_count: int
    flags: int
    frame_ids: np.ndarray       # (T,) uint64
    timestamps_ms: np.ndarray   # (T,) uint64
    values: np.ndarray          # (T, P, C) float32, NaN for absent points
    masks: np.ndarray           # (T, mask_bytes) uint8
    body_bytes: int
    body_sha256: str

    @property
    def has_hand(self) -> bool:
        return bool(self.flags & FLAG_HAS_HAND)

    @property
    def reset_vote(self) -> bool:
        return bool(self.flags & FLAG_RESET_VOTE)

    @property
    def full_543(self) -> bool:
        return bool(self.flags & FLAG_FULL_543)


def mask_bytes(point_count: int) -> int:
    """Bytes per frame validity mask, rounded up to a 4-byte boundary."""
    return ((point_count + 7) // 8 + 3) // 4 * 4


def frame_stride(point_count: int, coord_count: int) -> int:
    return 16 + mask_bytes(point_count) + point_count * coord_count * 4


def body_bytes(frame_count: int, point_count: int, coord_count: int) -> int:
    return HEADER_BYTES + frame_count * frame_stride(point_count, coord_count)


def bits_to_mask(bits: np.ndarray, point_count: int) -> np.ndarray:
    """Pack a (P,) boolean array into little-endian bytes, MSB zero-padded."""
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    out = np.zeros(mask_bytes(point_count), dtype=np.uint8)
    out[: packed.size] = packed
    return out


def mask_to_bits(mask: np.ndarray, point_count: int) -> np.ndarray:
    return np.unpackbits(mask, bitorder="little")[:point_count].astype(bool)


def _validate_mask_padding(mask: np.ndarray, point_count: int) -> None:
    used_in_last = point_count % 8
    last_used_byte = point_count // 8
    if used_in_last:
        allowed = (1 << used_in_last) - 1
        if int(mask[last_used_byte]) & ~allowed:
            raise ProtocolError(422, "validity mask has non-zero padding bits")
        if mask[last_used_byte + 1 :].any():
            raise ProtocolError(422, "validity mask has non-zero padding bytes")
    else:
        if mask[last_used_byte:].any():
            raise ProtocolError(422, "validity mask has non-zero padding bytes")


def decode_window(body: bytes, *, session_id: str, request_id: str = "") -> Window:
    if len(body) < HEADER_BYTES:
        raise ProtocolError(400, "body shorter than 40-byte header")
    (
        magic,
        version,
        header_bytes,
        point_count,
        coord_count,
        frame_count,
        flags,
        reserved,
        window_id,
        episode_id,
        first_frame_id,
    ) = _HEADER.unpack_from(body, 0)

    if magic != MAGIC:
        raise ProtocolError(400, "bad magic")
    if version != VERSION:
        raise ProtocolError(400, f"unsupported version {version}")
    if header_bytes != HEADER_BYTES:
        raise ProtocolError(400, f"unsupported header_bytes {header_bytes}")
    if point_count not in ALLOWED_POINT_COUNTS:
        raise ProtocolError(422, f"unsupported point_count {point_count}")
    if coord_count not in ALLOWED_COORD_COUNTS:
        raise ProtocolError(422, f"unsupported coord_count {coord_count}")
    if not 1 <= frame_count <= MAX_FRAMES:
        raise ProtocolError(422, f"frame_count out of range: {frame_count}")
    if flags & ~_FLAG_KNOWN:
        raise ProtocolError(422, "unknown flag bits set")
    if reserved != 0:
        raise ProtocolError(422, "reserved header field is non-zero")

    expected = body_bytes(frame_count, point_count, coord_count)
    if len(body) != expected:
        raise ProtocolError(
            400, f"body length {len(body)} does not match expected {expected}"
        )

    mb = mask_bytes(point_count)
    frame_ids = np.empty(frame_count, dtype=np.uint64)
    timestamps = np.empty(frame_count, dtype=np.uint64)
    values = np.empty((frame_count, point_count, coord_count), dtype=np.float32)
    masks = np.empty((frame_count, mb), dtype=np.uint8)

    offset = HEADER_BYTES
    for i in range(frame_count):
        fid, ts = _FRAME_IDS.unpack_from(body, offset)
        offset += _FRAME_IDS.size
        frame_ids[i] = fid
        timestamps[i] = ts

        masks[i] = np.frombuffer(body, dtype=np.uint8, count=mb, offset=offset)
        offset += mb
        _validate_mask_padding(masks[i], point_count)

        n_values = point_count * coord_count
        values[i] = np.frombuffer(
            body, dtype="<f4", count=n_values, offset=offset
        ).reshape(point_count, coord_count)
        offset += n_values * 4

    if frame_count and int(first_frame_id) != int(frame_ids[0]):
        raise ProtocolError(400, "first_frame_id does not match first frame record")

    present = np.stack(
        [mask_to_bits(masks[i], point_count) for i in range(frame_count)]
    )
    values[~present] = np.nan

    return Window(
        session_id=session_id,
        request_id=request_id,
        window_id=int(window_id),
        episode_id=int(episode_id),
        first_frame_id=int(first_frame_id),
        frame_count=int(frame_count),
        point_count=int(point_count),
        coord_count=int(coord_count),
        flags=int(flags),
        frame_ids=frame_ids,
        timestamps_ms=timestamps,
        values=values,
        masks=masks,
        body_bytes=len(body),
        body_sha256=hashlib.sha256(body).hexdigest(),
    )


def encode_window(window: Window) -> bytes:
    """Reference encoder used by tests and tools (not the production server path)."""
    t, p, c = window.values.shape
    if window.frame_count != t:
        raise ValueError("frame_count does not match values")
    if window.point_count != p or window.coord_count != c:
        raise ValueError("point/coord count does not match values")
    if window.masks.shape != (t, mask_bytes(p)):
        raise ValueError("mask shape does not match point_count")
    if t and int(window.first_frame_id) != int(window.frame_ids[0]):
        raise ValueError("first_frame_id does not match first frame")

    out = bytearray()
    out += _HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        p,
        c,
        t,
        window.flags,
        0,
        int(window.window_id),
        int(window.episode_id),
        int(window.first_frame_id),
    )

    for i in range(t):
        out += _FRAME_IDS.pack(int(window.frame_ids[i]), int(window.timestamps_ms[i]))
        mask = window.masks[i].astype(np.uint8).tobytes()
        out += mask
        vals = window.values[i].astype("<f4").copy()
        absent = ~mask_to_bits(window.masks[i], p)
        vals[absent] = 0.0
        out += vals.astype("<f4").tobytes()

    if len(out) != body_bytes(t, p, c):
        raise AssertionError("encoder produced unexpected length")
    return bytes(out)
