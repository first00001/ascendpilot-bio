#!/usr/bin/env python3
"""Evaluate the end-to-end Qwen planner/tool/evidence contract."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


TASKS = [
    "解释 ZH11 标签迁移结果的可信度和限制",
    "总结当前 Qwen 在昇腾上的性能证据",
    "对这个表达矩阵执行细胞类型注释",
    "生成一段不超过 100 字的评测结论",
    "说明为什么不能把当前结果称为独立外部验证",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, failures = [], 0
    for task in TASKS:
        payload = json.dumps({"query": task}, ensure_ascii=False).encode()
        request = Request(
            args.endpoint.rstrip("/") + "/agent/run",
            data=payload,
            headers={"x-api-key": args.api_token, "content-type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=600) as response:
                result = json.loads(response.read())
            elapsed = time.perf_counter() - started
            contract = result["agent_contract"]
            input_required = result.get("status") == "input_required"
            valid = (
                contract["tool_whitelist"]
                and contract["evidence_required"]
                and contract["planner_output_validated"]
                and result["evidence"]
                and result["report"]
                and result["verification"]["passed"]
                and (not input_required or result["plan"]["required_inputs"] == ["genes", "counts"])
            )
            rows.append({"task": task, "seconds": elapsed, "valid": bool(valid), "status": result.get("status"), "tools": result["plan"]["tools"], "planner": result["plan"]["planner"], "verification": result["verification"]})
            failures += not bool(valid)
        except Exception as exc:
            rows.append({"task": task, "valid": False, "error": str(exc)})
            failures += 1
    successful = [row for row in rows if row.get("valid")]
    payload = {
        "tasks": len(TASKS),
        "successful_tasks": len(successful),
        "task_completion_rate": len(successful) / len(TASKS),
        "mean_seconds": statistics.mean(row["seconds"] for row in successful) if successful else None,
        "failures": failures,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
