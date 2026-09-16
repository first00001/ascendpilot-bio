#!/usr/bin/env python3
"""Select a measured Qwen deployment policy under explicit constraints."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--max-memory-gb", type=float, default=12.0)
    parser.add_argument("--objective", choices=("balanced", "throughput", "memory"), default="balanced")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.benchmark.read_text(encoding="utf-8"))
    candidates = data["comparison"]
    feasible = [item for item in candidates if item["peak_memory_bytes"] / 1e9 <= args.max_memory_gb]
    if not feasible:
        result = {"status": "failed", "reason": "no candidate satisfies memory budget"}
    else:
        max_rate = max(item["mean_tokens_per_second"] for item in feasible)
        min_memory = min(item["peak_memory_bytes"] for item in feasible)
        for item in feasible:
            speed = item["mean_tokens_per_second"] / max_rate
            memory = min_memory / item["peak_memory_bytes"]
            weights = {"throughput": (0.9, 0.1), "memory": (0.1, 0.9), "balanced": (0.45, 0.55)}[args.objective]
            item["selection_score"] = weights[0] * speed + weights[1] * memory
        selected = max(feasible, key=lambda item: item["selection_score"])
        result = {
            "status": "passed",
            "selected_precision": selected["dtype"],
            "device": selected["device"],
            "objective": args.objective,
            "max_memory_gb": args.max_memory_gb,
            "measured": selected,
            "rejected": [item for item in candidates if item is not selected],
        }
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

