#!/usr/bin/env python3
"""Drive repeated .bin replay trials on a connected phone while collecting logs.

What it automates:
  * one collect.mjs capture for the whole condition
  * per trial: tap "清空并开始新一轮测量" (start trial), tap REPLAY, wait
  * writing meta.json (condition, device, timings) into the run directory

What it cannot automate:
  * the first `.BIN` file selection (Android SAF picker)
  * the one-time CFG (URL + token) and REMOTE toggle

So before running: open the app, switch to G2T REMOTE, set CFG, tap .BIN and
choose the clip once. Then this script repeats start-trial + REPLAY.

Usage:
  uv run python eval/run_trials.py --condition p1-rtt0 --reps 30 \
      --clip-seconds 17 --settle-seconds 8 --mode remote \
      --url http://121.41.19.165:8080
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_COLLECT = REPO / "On_device" / "performance-tools" / "collect.mjs"


def run(cmd: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def adb(adb_path: str, device: str | None, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    base = [adb_path] + (["-s", device] if device else [])
    return run(base + list(args), timeout=timeout)


def one_device(adb_path: str) -> str | None:
    out = run([adb_path, "devices"]).stdout or ""
    devices = [l.split("\t")[0] for l in out.splitlines() if l.strip().endswith("\tdevice")]
    return devices[0] if len(devices) == 1 else None


def dump_ui(adb_path: str, device: str) -> str:
    adb(adb_path, device, "shell", "uiautomator", "dump", "/sdcard/s2s_ui.xml", timeout=20)
    return adb(adb_path, device, "shell", "cat", "/sdcard/s2s_ui.xml", timeout=20).stdout or ""


def find_center(xml_text: str, target: str) -> tuple[int, int] | None:
    if not xml_text.strip():
        return None
    start = xml_text.find("<")
    if start > 0:
        xml_text = xml_text[start:]
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for node in root.iter():
        text = node.get("text") or ""
        desc = node.get("content-desc") or ""
        if target.lower() in text.lower() or target.lower() in desc.lower():
            bounds = node.get("bounds") or ""
            # bounds="[x1,y1][x2,y2]"
            if bounds.count(",") == 2 and "][" in bounds:
                try:
                    (x1, y1), (x2, y2) = [
                        tuple(int(v) for v in part.strip("[]").split(","))
                        for part in bounds.split("][")
                    ]
                    return (x1 + x2) // 2, (y1 + y2) // 2
                except ValueError:
                    continue
    return None


def tap_text(adb_path: str, device: str, target: str, retries: int = 4) -> bool:
    for attempt in range(retries):
        center = find_center(dump_ui(adb_path, device), target)
        if center:
            adb(adb_path, device, "shell", "input", "tap", str(center[0]), str(center[1]))
            return True
        time.sleep(1.0)
    return False


def device_props(adb_path: str, device: str) -> dict:
    props = {}
    for key in ("ro.product.model", "ro.build.version.release", "ro.product.cpu.abi"):
        props[key] = adb(adb_path, device, "shell", "getprop", key).stdout.strip()
    return props


def main() -> int:
    ap = argparse.ArgumentParser(description="Run repeated Sign2Speak replay trials")
    ap.add_argument("--condition", required=True, help="label, e.g. p1-rtt0, p0-local")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--clip-seconds", type=float, default=17.0, help="video length")
    ap.add_argument("--settle-seconds", type=float, default=8.0,
                    help="extra wait for flush + UI settle after the clip")
    ap.add_argument("--package", default="com.sign2speak.app.perf")
    ap.add_argument("--activity", default=".MainActivity")
    ap.add_argument("--mode", default="remote", choices=["remote", "local"])
    ap.add_argument("--url", default="http://121.41.19.165:8080")
    ap.add_argument("--device", default=None)
    ap.add_argument("--adb", dest="adb_path", default=shutil.which("adb") or "adb")
    ap.add_argument("--node", default=shutil.which("node") or "node")
    ap.add_argument("--collect", type=Path, default=DEFAULT_COLLECT)
    ap.add_argument("--out-root", type=Path, default=REPO / "eval" / "runs")
    ap.add_argument("--start-text", default="清空并开始新一轮测量")
    ap.add_argument("--replay-text", default="REPLAY")
    ap.add_argument("--no-tap", action="store_true", help="do not tap; you tap manually")
    ap.add_argument("--shuffle", action="store_true", help="randomise a small jitter between trials")
    args = ap.parse_args()

    device = args.device or one_device(args.adb_path)
    if not device:
        print("Need exactly one authorized device (adb devices).")
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / f"{args.condition}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_trial = args.clip_seconds + args.settle_seconds
    total_seconds = int(per_trial * args.reps + 20)

    meta = {
        "condition": args.condition,
        "mode": args.mode,
        "url": args.url,
        "package": args.package,
        "reps": args.reps,
        "clip_seconds": args.clip_seconds,
        "settle_seconds": args.settle_seconds,
        "started_at": stamp,
        "device": device_props(args.adb_path, device),
        "adb": args.adb_path,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    focus = adb(args.adb_path, device, "shell", "dumpsys", "window").stdout or ""
    if args.package not in focus:
        print(f"launching {args.package}")
        adb(args.adb_path, device, "shell", "am", "start", "-n",
            f"{args.package}/{args.package}{args.activity}")
        time.sleep(4)

    collect_cmd = [args.node, str(args.collect), "--device", device,
                   "--package", args.package, "--seconds", str(total_seconds),
                   "--out", str(run_dir)]
    print("starting collector:", " ".join(collect_cmd))
    collector = subprocess.Popen(collect_cmd, cwd=str(REPO), text=True)
    time.sleep(3)

    if not args.no_tap:
        if not tap_text(args.adb_path, device, args.replay_text):
            print(f'Could not find "{args.replay_text}" on screen. '
                  'Open the .bin once manually, or rerun with --no-tap.')
            collector.terminate()
            return 3

    for rep in range(1, args.reps + 1):
        print(f"--- trial {rep}/{args.reps} ---", flush=True)
        if not args.no_tap:
            if not tap_text(args.adb_path, device, args.start_text):
                print(f'Could not find "{args.start_text}". Is this the timing build '
                      'with PerfTestPanel visible?')
                break
            time.sleep(1.5)
            if not tap_text(args.adb_path, device, args.replay_text):
                print(f'Could not find "{args.replay_text}".')
                break
        else:
            input(f"trial {rep}: start the replay, then press Enter... ")
        wait = per_trial
        if args.shuffle:
            wait += random.uniform(-1.0, 1.0)
        time.sleep(max(1.0, wait))

    print("waiting for collector to finish...")
    try:
        collector.wait(timeout=total_seconds + 60)
    except subprocess.TimeoutExpired:
        collector.terminate()

    print(f"\nrun directory: {run_dir}")
    print(f"analyze with: uv run python eval/stats.py {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
