#!/usr/bin/env python3
"""Benchmark real PlantCell Agent inference endpoints with repeatable JSON output."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent)
    return ordered[max(0, min(len(ordered) - 1, index))]


def request_json(endpoint: str, token: str, path: str, payload=None, timeout=600):
    headers = {"x-api-key": token}
    data = None
    if payload is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint.rstrip("/") + path, data=data, headers=headers)
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, (time.perf_counter() - started) * 1000


def summarize(latencies: list[float], failures: int) -> dict:
    total = len(latencies) + failures
    return {
        "requests": total,
        "successes": len(latencies),
        "failures": failures,
        "error_rate": failures / total,
        "min_ms": min(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--api-token", default=os.getenv("PLANTCELL_API_TOKEN"))
    parser.add_argument("--mode", choices=("chat", "ricefm"), default="chat")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.api_token:
        parser.error("set PLANTCELL_API_TOKEN or pass --api-token")
    if min(args.runs, args.concurrency) < 1 or args.warmup < 0:
        parser.error("runs/concurrency must be positive and warmup non-negative")

    version, _ = request_json(args.endpoint, args.api_token, "/version")
    sample = None
    if args.mode == "ricefm":
        sample_set, _ = request_json(
            args.endpoint, args.api_token, "/ricefm/demo-samples"
        )
        sample = sample_set["samples"][0]

    def one():
        if args.mode == "chat":
            return request_json(
                args.endpoint,
                args.api_token,
                "/chat",
                {
                    "prompt": "用两点说明水稻单细胞标签迁移结果的可信度和限制。",
                    "max_new_tokens": args.max_new_tokens,
                },
            )
        return request_json(
            args.endpoint,
            args.api_token,
            "/ricefm/annotate",
            {"genes": sample["genes"], "counts": sample["counts"], "k": 5},
        )

    for _ in range(args.warmup):
        one()
    latencies, server_seconds, output_rates, failures = [], [], [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one) for _ in range(args.runs)]
        for future in concurrent.futures.as_completed(futures):
            try:
                body, elapsed = future.result()
                latencies.append(elapsed)
                if args.mode == "chat":
                    server_seconds.append(body["performance"]["generation_seconds"])
                    output_rates.append(body["performance"]["tokens_per_second"])
            except Exception as exc:
                failures += 1
                print(f"request failed: {exc}", file=os.sys.stderr)
    if not latencies:
        raise SystemExit("all benchmark requests failed")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup,
        "service": version,
        "latency": summarize(latencies, failures),
    }
    if output_rates:
        result["generation"] = {
            "mean_server_seconds": statistics.mean(server_seconds),
            "mean_tokens_per_second": statistics.mean(output_rates),
            "min_tokens_per_second": min(output_rates),
            "max_tokens_per_second": max(output_rates),
            "max_new_tokens": args.max_new_tokens,
            "note": "Endpoint is non-streaming; TTFT is not reported.",
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
