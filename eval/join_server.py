#!/usr/bin/env python3
"""Join app-side S2S_PERF events with the cloud service's p1.jsonl.

They cannot be joined on session id: the app sends `appSessionId-epoch` while
PerfTrace logs its own UUID. `request_id` is the shared key, unique within one
process lifetime, so run this per collection run.

Usage:
  uv run python eval/join_server.py --events run/events.ndjson \
      --server server.jsonl --out run/joined.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def num(value):
    return value if isinstance(value, (int, float)) else None


def delta_ms(later, earlier):
    try:
        return (int(later) - int(earlier)) / 1e6
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("joined.csv"))
    args = ap.parse_args()

    events = load_jsonl(args.events)
    server = load_jsonl(args.server)

    server_by_req: dict[str, dict] = {}
    duplicates = defaultdict(int)
    for rec in server:
        rid = str(rec.get("request_id", ""))
        if not rid:
            continue
        if rid in server_by_req:
            duplicates[rid] += 1
        server_by_req[rid] = rec

    by_req: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        rid = e.get("request_id")
        if rid:
            by_req[str(rid)].append(e)

    rows = []
    unmatched_app = []
    for rid, rel in by_req.items():
        sub = next((x for x in rel if x.get("event") == "sentence_submitted"), None)
        if sub is None:
            continue
        ret = next((x for x in rel if x.get("event") == "sentence_returned"), None)
        srv = server_by_req.get(rid)
        if srv is None:
            unmatched_app.append(rid)

        client_ms = num((ret or {}).get("client_ms"))
        queue_ms = num((ret or {}).get("server_queue_ms"))
        infer_ms = num((ret or {}).get("server_inference_ms"))
        if srv:
            queue_ms = queue_ms if queue_ms is not None else num(srv.get("queue_ms"))
            infer_ms = infer_ms if infer_ms is not None else num(srv.get("inference_ms"))
        network_ms = None
        if client_ms is not None and queue_ms is not None and infer_ms is not None:
            network_ms = client_ms - queue_ms - infer_ms

        rows.append({
            "request_id": rid,
            "session_app": sub.get("session", ""),
            "trial_id": sub.get("trial_id", ""),
            "trigger": sub.get("trigger", ""),
            "glosses": sub.get("glosses", ""),
            "text": (ret or {}).get("text", ""),
            "app_status": (ret or {}).get("status", ""),
            "source": (ret or {}).get("source", ""),
            "submit_to_text_ms": delta_ms((ret or {}).get("t_ns"), sub.get("submitted_ns")),
            "client_ms": client_ms,
            "server_queue_ms": queue_ms,
            "server_inference_ms": infer_ms,
            "network_ms": network_ms,
            "server_status": (srv or {}).get("status", ""),
            "server_session_id": (srv or {}).get("session_id", ""),
            "server_matched": srv is not None,
        })

    rows.sort(key=lambda r: (str(r["session_app"]), str(r["trial_id"]), str(r["request_id"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["request_id", "session_app", "trial_id", "trigger", "glosses",
                            "text", "app_status", "source", "submit_to_text_ms", "client_ms",
                            "server_queue_ms", "server_inference_ms", "network_ms",
                            "server_status", "server_session_id", "server_matched"])
        w.writeheader()
        w.writerows(rows)

    matched = sum(1 for r in rows if r["server_matched"])
    print(f"requests: {len(rows)}, matched on server: {matched}")
    if unmatched_app:
        print(f"WARNING: {len(unmatched_app)} requests have no server row "
              f"(e.g. {unmatched_app[:3]}). Local-mode calls or a different run?")
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicated request_id on server "
              "(different app process reusing the counter?); last row wins.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
