"""
pack_frames.py — Pre-decode a video into an uncompressed RGBA_8888 frame pack (.bin)
for the Sign2Speak Android local file import mode.

Binary layout (little-endian):
  Header (32 bytes):
    0..3   : magic       = b"S2SF"
    4..7   : version     = int32 (1)
    8..11  : width       = int32
    12..15 : height      = int32
    16..19 : frame_count = int32
    20..23 : fps         = int32
    24..27 : flags       = int32 (0)
    28..31 : reserved    = int32 (0)
  Body:
    frame_count * (width * height * 4) bytes of RGBA_8888 pixel data.

Usage:
    python pack_frames.py input.mp4
    python pack_frames.py input.mp4 -o test_480x640.bin --width 480 --height 640
    python pack_frames.py input.mp4 --max-frames 300 --mirror
"""

import argparse
import struct
from pathlib import Path

import cv2

MAGIC = b"S2SF"
VERSION = 1
HEADER_FMT = "<4s7i"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 32 bytes


def _compute_target_size(src_w: int, src_h: int, width: int, height: int, max_dim: int, native: bool) -> tuple[int, int]:
    if native:
        return src_w, src_h
    if width > 0 and height > 0:
        return width, height
    if width > 0:
        scale = width / src_w
        return width, max(2, int(round(src_h * scale)) & ~1)
    if height > 0:
        scale = height / src_h
        return max(2, int(round(src_w * scale)) & ~1), height

    longest = max(src_w, src_h)
    if longest <= max_dim:
        return src_w, src_h
    scale = max_dim / longest
    out_w = max(2, int(round(src_w * scale)) & ~1)
    out_h = max(2, int(round(src_h * scale)) & ~1)
    return out_w, out_h


def pack_video(
    input_path: Path,
    output_path: Path,
    width: int = 0,
    height: int = 0,
    max_dim: int = 640,
    native: bool = False,
    fps_override: int = 0,
    max_frames: int = 0,
    mirror: bool = False,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_override if fps_override > 0 else max(1, int(round(src_fps))) if src_fps and src_fps > 0 else 30

    out_w, out_h = _compute_target_size(src_w, src_h, width, height, max_dim, native)
    frame_bytes = out_w * out_h * 4

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "wb") as f:
        # Placeholder header; updated once exact frame_count is known
        f.write(b"\x00" * HEADER_SIZE)

        while True:
            if max_frames > 0 and written >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                interp = cv2.INTER_AREA if (out_w < frame.shape[1] or out_h < frame.shape[0]) else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (out_w, out_h), interpolation=interp)

            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            f.write(rgba.tobytes())
            written += 1

        # Rewrite header with actual written count
        f.seek(0)
        header = struct.pack(HEADER_FMT, MAGIC, VERSION, out_w, out_h, written, fps, 0, 0)
        f.write(header)

    cap.release()

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames decoded from: {input_path}")

    total_mb = (HEADER_SIZE + written * frame_bytes) / (1024 * 1024)
    print(f"Packed {written} frames ({out_w}x{out_h} @ {fps} fps, {total_mb:.2f} MB) -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack video frames into a Sign2Speak RGBA .bin file")
    parser.add_argument("input", type=Path, help="Input video file (.mp4, .mov, .avi, ...)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .bin path (default: <input>.bin)")
    parser.add_argument("--width", type=int, default=0, help="Target width (0 = auto)")
    parser.add_argument("--height", type=int, default=0, help="Target height (0 = auto)")
    parser.add_argument("--max-dim", type=int, default=640, help="Max width/height when auto-scaling (default: 640)")
    parser.add_argument("--native", action="store_true", help="Keep original video resolution without scaling")
    parser.add_argument("--fps", type=int, default=0, help="Override FPS stored in header (default: from video or 30)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to export (0 = all)")
    parser.add_argument("--mirror", action="store_true", help="Horizontally flip frames before packing")
    args = parser.parse_args()

    out = args.output if args.output is not None else args.input.with_suffix(".bin")
    pack_video(
        input_path=args.input,
        output_path=out,
        width=args.width,
        height=args.height,
        max_dim=args.max_dim,
        native=args.native,
        fps_override=args.fps,
        max_frames=args.max_frames,
        mirror=args.mirror,
    )


if __name__ == "__main__":
    main()
