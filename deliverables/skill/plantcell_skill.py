"""Plant single-cell analysis Skill with a replaceable riceFM adapter."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen


class RiceFMAdapter(Protocol):
    def analyze(self, query: str, dataset: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class PendingRiceFMAdapter:
    """Result used when a query does not include an expression matrix."""

    def analyze(self, query: str, dataset: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "input-required",
            "query": query,
            "dataset": dataset,
            "message": (
                "未提供表达矩阵，因此没有对新细胞执行注释或差异分析。"
                "已部署的公开数据 riceFM embedding pilot 可通过 /ricefm/metrics 和 "
                "/ricefm/annotate 调用；它不输出细胞类型标签。"
            ),
        }


def load_ricefm(model_dir: str, repo_dir: str, device: str = "npu:0") -> RiceFMAdapter:
    """Load the public riceFM implementation and a separately downloaded checkpoint."""
    from .ricefm_adapter import RiceFMRuntime
    return RiceFMRuntime(model_dir, repo_dir, device)


def plan(query: str) -> dict[str, Any]:
    if any(k in query for k in ("解释", "总结", "报告", "限制", "可信度")):
        return {
            "domain": "plant_single_cell",
            "query": query,
            "steps": ["report"],
            "skills": ["report_generation"],
            "requires_riceFM": False,
        }
    steps: list[str] = []
    if any(k in query.lower() for k in ("qc", "质控", "过滤", "质量")):
        steps.append("qc")
    if any(k in query for k in ("细胞类型", "注释", "marker", "标记")):
        steps.append("annotation")
    if any(k in query for k in ("差异", "表达", "基因")):
        steps.append("differential_expression")
    if any(k in query for k in ("聚类", "亚群")):
        steps.append("clustering")
    if not steps:
        steps = ["qc", "annotation", "differential_expression"]
    return {
        "domain": "plant_single_cell",
        "query": query,
        "steps": steps,
        "skills": [
            "builtin_qc"
            if s == "qc"
            else "riceFM"
            if s == "annotation"
            else "statistical_analysis"
            for s in steps
        ],
        "requires_riceFM": any(s != "qc" for s in steps),
    }


def summarize_table(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        return {"rows": 0, "columns": 0, "numeric_features": 0}
    header, body = rows[0], rows[1:]
    features = []
    for idx, name in enumerate(header):
        values = []
        for row in body:
            if idx < len(row):
                try:
                    values.append(float(row[idx]))
                except ValueError:
                    pass
        if values:
            features.append({"name": name, "mean": sum(values) / len(values), "n": len(values)})
    return {"rows": len(body), "columns": len(header), "numeric_features": len(features), "features": features[:100]}


def call_qwen(
    endpoint: str,
    prompt: str,
    max_new_tokens: int = 256,
    api_token: str | None = None,
) -> str:
    payload = json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens}).encode()
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["x-api-key"] = api_token
    request = Request(endpoint.rstrip("/") + "/chat", data=payload, headers=headers)
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read()).get("text", "")


def run(
    query: str,
    endpoint: str | None = None,
    dataset: dict[str, Any] | None = None,
    adapter: RiceFMAdapter | None = None,
    api_token: str | None = None,
) -> dict[str, Any]:
    result = {"plan": plan(query), "dataset": dataset or {}}
    if result["plan"]["requires_riceFM"]:
        result["biological_analysis"] = (adapter or PendingRiceFMAdapter()).analyze(query, result["dataset"])
    if endpoint:
        result["narrative"] = call_qwen(endpoint, query, api_token=api_token)
    return result
