#!/usr/bin/env python3
"""Aggregate Sign2Speak timing runs into report-ready tables.

Input: a root directory whose subdirectories are runs produced by
`On_device/performance-tools/collect.mjs`. Each run may contain:
  events.ndjson   (required)  S2S_PERF events
  meta.json       (optional)  {"condition","mode","rtt_ms","bw_mbps",...}
  server.jsonl    (optional)  copy of /opt/sign2speak/logs/p1.jsonl
  p1.jsonl        (optional)  same thing under another name

Output:
  report.md   markdown tables grouped by condition (drop into the report)
  summary.csv long-format metric rows

Only the standard library is used. Run with:  uv run python eval/stats.py <root>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

GENERATED_STATUSES = {"generated"}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, math.ceil(p / 100.0 * len(s)) - 1)
    return s[min(idx, len(s) - 1)]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ns_delta_ms(later: str | int | None, earlier: str | int | None) -> float | None:
    if later is None or earlier is None:
        return None
    try:
        return (int(later) - int(earlier)) / 1e6
    except (TypeError, ValueError):
        return None


def parse_run(run_dir: Path) -> dict:
    meta = {}
    mp = run_dir / "meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    events = load_jsonl(run_dir / "events.ndjson")
    events.sort(key=lambda e: int(e.get("t_ns", 0)))

    server_rows = load_jsonl(run_dir / "server.jsonl") + load_jsonl(run_dir / "p1.jsonl")
    server_by_req: dict[str, dict] = {}
    for rec in server_rows:
        rid = rec.get("request_id")
        if rid:
            server_by_req[str(rid)] = rec

    trials: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        trials[(e.get("session"), e.get("trial_id"))].append(e)

    sentence_rows: list[dict] = []
    word_latencies: list[float] = []
    fps_samples: list[float] = []
    statuses: Counter = Counter()
    sources: Counter = Counter()
    triggers: Counter = Counter()
    stale = 0

    for (session, trial_id), evs in trials.items():
        by_req: dict[str, list[dict]] = defaultdict(list)
        for e in evs:
            rid = e.get("request_id")
            if rid:
                by_req[str(rid)].append(e)

        for rid, rel in by_req.items():
            sub = next((x for x in rel if x.get("event") == "sentence_submitted"), None)
            if sub is None:
                continue
            ret = next((x for x in rel if x.get("event") == "sentence_returned"), None)
            gs = next((x for x in rel if x.get("event") == "generation_status"), None)
            rr = next((x for x in rel if x.get("event") == "remote_response"), None)
            fail = next((x for x in rel if x.get("event") in
                         ("remote_timeout", "remote_error", "remote_http_error")), None)
            if any(x.get("event") == "stale_response" for x in rel):
                stale += 1

            status = (ret or {}).get("status") or (gs or {}).get("status")
            if not status and fail:
                status = fail.get("event")
            source = (ret or {}).get("source") or ("remote" if rr else "")
            client_ms = (ret or {}).get("client_ms")
            if client_ms is None or (isinstance(client_ms, (int, float)) and client_ms < 0):
                client_ms = (rr or {}).get("client_ms")
            queue_ms = (ret or {}).get("server_queue_ms")
            if queue_ms is None or (isinstance(queue_ms, (int, float)) and queue_ms < 0):
                queue_ms = (rr or {}).get("queue_ms")
            infer_ms = (ret or {}).get("server_inference_ms")
            if infer_ms is None or (isinstance(infer_ms, (int, float)) and infer_ms < 0):
                infer_ms = (rr or {}).get("inference_ms")

            server = server_by_req.get(rid, {})
            if queue_ms is None:
                queue_ms = server.get("queue_ms")
            if infer_ms is None:
                infer_ms = server.get("inference_ms")
            if not status and server.get("status"):
                status = server["status"]

            network_ms = None
            if isinstance(client_ms, (int, float)) and isinstance(queue_ms, (int, float)) \
                    and isinstance(infer_ms, (int, float)):
                network_ms = client_ms - queue_ms - infer_ms

            sentence_rows.append({
                "session": session,
                "trial_id": trial_id,
                "request_id": rid,
                "trigger": sub.get("trigger", ""),
                "glosses": sub.get("glosses", ""),
                "submit_to_text_ms": ns_delta_ms((ret or {}).get("t_ns"), sub.get("submitted_ns")),
                "client_ms": client_ms if isinstance(client_ms, (int, float)) else None,
                "server_queue_ms": queue_ms if isinstance(queue_ms, (int, float)) else None,
                "server_inference_ms": infer_ms if isinstance(infer_ms, (int, float)) else None,
                "network_ms": network_ms,
                "status": status or "unknown",
                "source": source,
                "text": (ret or {}).get("text", ""),
                "server_reported": bool(server),
            })
            statuses[status or "unknown"] += 1
            if source:
                sources[source] += 1
            triggers[sub.get("trigger", "")] += 1

        word_starts = [x for x in evs if x.get("event") == "word_started"]
        for start in word_starts:
            same = [x for x in evs
                    if x.get("event") == "word_recognized"
                    and x.get("session") == start.get("session")
                    and x.get("trial_id") == start.get("trial_id")
                    and x.get("episode_id") == start.get("episode_id")]
            if same:
                lat = ns_delta_ms(same[0].get("t_ns"), start.get("hand_seen_ns"))
                if lat is not None:
                    word_latencies.append(lat)

        frame_ts = [int(x["frame_observed_ns"]) for x in evs
                    if x.get("event") == "recognition_result" and x.get("frame_observed_ns")]
        for a, b in zip(frame_ts, frame_ts[1:]):
            d = (b - a) / 1e9
            if 0 < d < 1.0:
                fps_samples.append(1.0 / d)

    m3_groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if str(e.get("event", "")).startswith("m3_window_"):
            m3_groups[(e.get("session"), e.get("trial_id"), e.get("window_id"))].append(e)

    m3_rows: list[dict] = []
    for (session, trial_id, window_id), rel in m3_groups.items():
        def find(name: str) -> dict | None:
            return next((x for x in rel if x.get("event") == name), None)

        sub = find("m3_window_submitted")
        ret = find("m3_window_returned")
        dropped = find("m3_window_dropped")
        timeout = find("m3_window_timeout")
        error = find("m3_window_error")
        stale_ev = find("m3_window_stale")
        base = sub or ret or dropped or timeout or error or stale_ev or rel[0]
        if ret is not None:
            status = ret.get("status", "ok")
        elif dropped is not None:
            status = "dropped"
        elif timeout is not None:
            status = "timeout"
        elif error is not None:
            status = "error"
        elif stale_ev is not None:
            status = "stale"
        else:
            status = "submitted_only"

        client_ms = ret.get("client_ms") if ret else None
        server_total = ret.get("server_total_ms") if ret else None
        network_ms = None
        if isinstance(client_ms, (int, float)) and isinstance(server_total, (int, float)):
            network_ms = client_ms - server_total

        m3_rows.append({
            "session": session,
            "trial_id": trial_id,
            "window_id": window_id,
            "episode_id": base.get("episode_id", ""),
            "frame_count": (sub or {}).get("frame_count", ""),
            "bytes": (sub or {}).get("bytes"),
            "body_sha256": (sub or {}).get("body_sha256", ""),
            "status": status,
            "label": (ret or {}).get("label", ""),
            "confidence": (ret or {}).get("confidence"),
            "client_ms": client_ms,
            "server_queue_ms": (ret or {}).get("server_queue_ms"),
            "server_preprocess_ms": (ret or {}).get("server_preprocess_ms"),
            "server_inference_ms": (ret or {}).get("server_inference_ms"),
            "server_vote_ms": (ret or {}).get("server_vote_ms"),
            "server_total_ms": server_total,
            "network_ms": network_ms,
            "bytes_down": (ret or {}).get("bytes_down"),
            "reason": (dropped or timeout or error or stale_ev or {}).get("reason", "")
                      or (timeout or error or {}).get("error", ""),
        })

    m3_rows.sort(key=lambda r: (str(r["session"]), r["trial_id"], int(r["window_id"] or 0)))
    m3_times = [int(e["t_ns"]) for e in events
                if str(e.get("event", "")).startswith("m3_window_") and e.get("t_ns")]
    m3_span_s = (max(m3_times) - min(m3_times)) / 1e9 if len(m3_times) > 1 else None
    m3_returned = [r for r in m3_rows
                   if r["status"] not in {"dropped", "timeout", "error", "stale", "submitted_only"}]

    return {
        "run": run_dir.name,
        "meta": meta,
        "condition": str(meta.get("condition") or run_dir.name),
        "sentences": sentence_rows,
        "words": word_latencies,
        "fps": fps_samples,
        "statuses": statuses,
        "sources": sources,
        "triggers": triggers,
        "stale": stale,
        "m3": m3_rows,
        "m3_span_s": m3_span_s,
        "m3_returned": len(m3_returned),
        "m3_bytes_up": sum(int(r["bytes"] or 0) for r in m3_rows),
        "m3_bytes_down": sum(int(r["bytes_down"] or 0) for r in m3_rows),
    }


def fmt(value, digits=1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_line(name: str, values: list[float]) -> str:
    if not values:
        return f"| {name} | 0 | — | — | — | — | — |"
    return (f"| {name} | {len(values)} | {fmt(mean(values))} | {fmt(pct(values,50))} "
            f"| {fmt(pct(values,95))} | {fmt(min(values))} | {fmt(max(values))} |")


def build_report(runs: list[dict]) -> str:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_cond[r["condition"]].append(r)

    lines = ["# Sign2Speak evaluation summary", ""]
    lines.append("Latency in milliseconds. P95 uses nearest-rank. "
                 "`network_ms = client_ms - server_queue_ms - server_inference_ms` "
                 "(includes HTTP + last-mile RTT).")
    lines.append("")

    for cond, group in sorted(by_cond.items()):
        sent = [s for r in group for s in r["sentences"]]
        words = [w for r in group for w in r["words"]]
        fps = [f for r in group for f in r["fps"]]
        statuses: Counter = Counter()
        sources: Counter = Counter()
        stale = 0
        for r in group:
            statuses.update(r["statuses"])
            sources.update(r["sources"])
            stale += r["stale"]

        lines.append(f"## {cond}")
        meta_keys = {}
        for r in group:
            for k, v in r["meta"].items():
                if k != "condition":
                    meta_keys.setdefault(k, set()).add(str(v))
        for k, vals in meta_keys.items():
            lines.append(f"- {k}: {', '.join(sorted(vals))}")
        lines.append(f"- runs: {len(group)}, sentences: {len(sent)}")
        lines.append("")
        lines.append("| metric | n | mean | p50 | p95 | min | max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        lines.append(metric_line("submit_to_text", [s["submit_to_text_ms"] for s in sent if s["submit_to_text_ms"] is not None]))
        lines.append(metric_line("client_total", [s["client_ms"] for s in sent if s["client_ms"] is not None]))
        lines.append(metric_line("server_queue", [s["server_queue_ms"] for s in sent if s["server_queue_ms"] is not None]))
        lines.append(metric_line("server_inference", [s["server_inference_ms"] for s in sent if s["server_inference_ms"] is not None]))
        lines.append(metric_line("network_overhead", [s["network_ms"] for s in sent if s["network_ms"] is not None]))
        lines.append(metric_line("word recognition", words))
        lines.append(metric_line("processing fps", fps))

        m3 = [x for r in group for x in r["m3"]]
        m3_ret = [x for x in m3
                  if x["status"] not in {"dropped", "timeout", "error", "stale", "submitted_only"}]
        m3_span = sum(r["m3_span_s"] or 0.0 for r in group)
        lines.append(metric_line("M3 submit_to_return", [x["client_ms"] for x in m3_ret if x["client_ms"] is not None]))
        lines.append(metric_line("M3 server_total", [x["server_total_ms"] for x in m3_ret if x["server_total_ms"] is not None]))
        lines.append(metric_line("M3 server_queue", [x["server_queue_ms"] for x in m3_ret if x["server_queue_ms"] is not None]))
        lines.append(metric_line("M3 server_preprocess", [x["server_preprocess_ms"] for x in m3_ret if x["server_preprocess_ms"] is not None]))
        lines.append(metric_line("M3 server_inference", [x["server_inference_ms"] for x in m3_ret if x["server_inference_ms"] is not None]))
        lines.append(metric_line("M3 server_vote", [x["server_vote_ms"] for x in m3_ret if x["server_vote_ms"] is not None]))
        lines.append(metric_line("M3 network_overhead", [x["network_ms"] for x in m3_ret if x["network_ms"] is not None]))
        lines.append(metric_line("M3 bytes_per_window", [x["bytes"] for x in m3 if x["bytes"] is not None]))
        lines.append("")
        m3_status = Counter(x["status"] for x in m3)
        m3_total = len(m3) or 1
        lines.append(f"- M3 windows: {len(m3)}, returned: {len(m3_ret)}")
        lines.append("- M3 status: " + ", ".join(f"{k}={v}" for k, v in m3_status.most_common()))
        lines.append(f"- M3 drop rate: {100 * m3_status.get('dropped', 0) / m3_total:.0f}%")
        lines.append(f"- M3 bytes up: {sum(int(r['m3_bytes_up']) for r in group)}; "
                     f"down: {sum(int(r['m3_bytes_down']) for r in group)}")
        if m3_span > 0:
            lines.append(f"- M3 returned window rate: {len(m3_ret) / m3_span:.2f}/s over {m3_span:.1f}s")
        lines.append("")
        total = sum(statuses.values()) or 1
        lines.append("Status: " + ", ".join(f"{k}={v} ({100*v/total:.0f}%)" for k, v in statuses.most_common()))
        if sources:
            lines.append("Source: " + ", ".join(f"{k}={v}" for k, v in sources.most_common()))
        if stale:
            lines.append(f"Stale responses dropped: {stale}")
        empty = [s for s in sent if s["status"] not in GENERATED_STATUSES]
        if empty:
            lines.append(f"Non-generated requests: {len(empty)}/{len(sent)}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Sign2Speak evaluation runs")
    ap.add_argument("root", type=Path, help="directory containing run subdirectories")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <root>/analysis)")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"no such directory: {root}")
        return 2

    if (root / "events.ndjson").exists():
        run_dirs = [root]
    else:
        run_dirs = [p for p in sorted(root.iterdir())
                    if p.is_dir() and (p / "events.ndjson").exists()]
    if not run_dirs:
        print(f"no runs (events.ndjson) found under {root}")
        return 2

    runs = [parse_run(d) for d in run_dirs]
    out_dir = args.out or (root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "report.md").write_text(build_report(runs), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "run", "session", "trial_id", "request_id", "trigger",
                    "glosses", "status", "source", "submit_to_text_ms", "client_ms",
                    "server_queue_ms", "server_inference_ms", "network_ms", "text"])
        for r in runs:
            for s in r["sentences"]:
                w.writerow([r["condition"], r["run"], s["session"], s["trial_id"],
                            s["request_id"], s["trigger"], s["glosses"], s["status"],
                            s["source"], s["submit_to_text_ms"], s["client_ms"],
                            s["server_queue_ms"], s["server_inference_ms"],
                            s["network_ms"], s["text"]])

    print(f"wrote {out_dir/'report.md'}")
    print(f"wrote {out_dir/'summary.csv'}")

    with (out_dir / "m3.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "run", "session", "trial_id", "window_id", "episode_id",
                    "frame_count", "bytes", "status", "label", "confidence", "client_ms",
                    "server_queue_ms", "server_preprocess_ms", "server_inference_ms",
                    "server_vote_ms", "server_total_ms", "network_ms", "bytes_down",
                    "reason", "body_sha256"])
        for r in runs:
            for x in r["m3"]:
                w.writerow([r["condition"], r["run"], x["session"], x["trial_id"],
                            x["window_id"], x["episode_id"], x["frame_count"], x["bytes"],
                            x["status"], x["label"], x["confidence"], x["client_ms"],
                            x["server_queue_ms"], x["server_preprocess_ms"],
                            x["server_inference_ms"], x["server_vote_ms"],
                            x["server_total_ms"], x["network_ms"], x["bytes_down"],
                            x["reason"], x["body_sha256"]])

    with (out_dir / "m3_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "run", "submitted", "returned", "dropped", "timeout",
                    "error", "stale", "bytes_up", "bytes_down", "span_s", "return_fps"])
        for r in runs:
            counts = Counter(x["status"] for x in r["m3"])
            span = r["m3_span_s"] or 0.0
            w.writerow([r["condition"], r["run"], len(r["m3"]), r["m3_returned"],
                        counts.get("dropped", 0), counts.get("timeout", 0),
                        counts.get("error", 0), counts.get("stale", 0),
                        r["m3_bytes_up"], r["m3_bytes_down"],
                        round(span, 3), round(r["m3_returned"] / span, 3) if span > 0 else ""])

    print(f"wrote {out_dir/'m3.csv'}")
    print(f"wrote {out_dir/'m3_summary.csv'}")
    print(build_report(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
