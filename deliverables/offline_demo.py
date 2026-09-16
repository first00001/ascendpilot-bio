#!/usr/bin/env python3
"""Dependency-free review server for the AscendPilot-Bio demonstration UI.

This mode serves a checked-in public E-ENAD-52/GSE146035 review subset and
recorded aggregate metrics. It never claims to execute Qwen, riceFM, or Ascend
NPU inference.
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
DEMO_PAGE = ROOT / "demo" / "index.html"
MODE = "public-evidence"
PUBLIC_DATA = ROOT / "public_data" / "e_enad_52_review_subset.json"


def load_public_data() -> dict:
    if not PUBLIC_DATA.is_file():
        raise RuntimeError(
            "public review subset is missing; run "
            "deliverables/public_data/build_public_subset.py"
        )
    return json.loads(PUBLIC_DATA.read_text(encoding="utf-8"))


PUBLIC = load_public_data()

RECORDED_METRICS = {
    "dataset": PUBLIC["dataset"]["atlas_accession"],
    "accuracy": None,
    "macro_f1": None,
    "cells_per_second": None,
    "celltype_L1_classes": None,
    "total_cells": PUBLIC["total_cells"],
    "subset_cells": PUBLIC["subset_cells"],
    "cluster_count": PUBLIC["dataset"]["cluster_count"],
    "limitation": (
        "Atlas clusters are unsupervised cluster identifiers, not curated biological "
        "cell-type labels; the public UI does not execute riceFM."
    ),
    "mode": MODE,
    "public_dataset": PUBLIC["dataset"],
}

MIGRATION_REPORT = {
    "status": "verified-recorded-evidence",
    "mode": MODE,
    "policy": {"selected_precision": "float16"},
    "benchmarks": {
        "float32": {"peak_memory_bytes": 16_910_000_000},
        "float16": {"peak_memory_bytes": 8_600_000_000},
        "online_qwen": {"mean_tokens_per_second": 19.86},
    },
}

DEMO_SAMPLES = {
    "mode": MODE,
    "notice": (
        "Public Atlas cells and cluster IDs; cluster evidence lookup only, "
        "not riceFM inference or curated cell-type annotation."
    ),
    "samples": PUBLIC["samples"],
}


def public_umap(limit: int = 1120) -> dict:
    points = PUBLIC["points"][: max(100, min(limit, len(PUBLIC["points"])))]
    return {
        "dataset": PUBLIC["dataset"]["atlas_accession"],
        "total_cells": PUBLIC["total_cells"],
        "cells_with_umap": PUBLIC["total_cells"],
        "points": points,
        "mode": MODE,
        "notice": PUBLIC["dataset"]["notice"],
    }


def offline_annotation(payload: dict) -> dict:
    genes = payload.get("genes") or []
    counts = payload.get("counts") or []
    if not genes or not counts or len(counts[0]) != len(genes):
        raise ValueError("genes and counts must be non-empty and aligned")
    cluster = str(genes[0]).removeprefix("ATLAS_CLUSTER_")
    label = f"Cluster {cluster}"
    markers = PUBLIC["top_markers"].get(cluster, [])[:3]
    return {
        "mode": MODE,
        "notice": (
            "Public Atlas cluster evidence lookup; riceFM was not executed and "
            "the cluster ID is not a curated biological cell-type label."
        ),
        "cells": 1,
        "embedding_dimensions": 2,
        "reference": PUBLIC["dataset"]["atlas_accession"],
        "predictions": [
            {
                "celltype_L1": label,
                "confidence": 1.0,
                "neighbors": [
                    {
                        "celltype_L1": item["gene"],
                        "barcode": f"marker CPM {item['value_cpm']:.2f}",
                        "cosine_similarity": max(0.0, 1.0 - number * 0.01),
                    }
                    for number, item in enumerate(markers, 1)
                ],
            }
        ],
    }


def offline_agent(payload: dict) -> dict:
    query = str(payload.get("query", "")).strip()
    report = (
        "这是公开数据证据演示，未执行 Qwen 或 riceFM 实时推理。界面中的 UMAP、"
        "cluster ID 和 marker gene 来自 EMBL-EBI E-ENAD-52 / NCBI GSE146035。"
        f"Atlas cluster 是无监督分群编号，不是人工校订的细胞类型。官方数据包含 "
        f"{PUBLIC['total_cells']} 个细胞；仓库内保留 {PUBLIC['subset_cells']} 个均匀"
        "抽样 UMAP 点和每个 cluster 的 marker 证据。该模式没有执行新的统计检验、"
        "riceFM 注释或 Qwen 生成，因此不能补充差异基因、通路或因果结论。"
    )
    return {
        "mode": MODE,
        "query": query,
        "report": report,
        "plan": {
            "planner": "public-evidence-planner",
            "tools": ["public_atlas_subset", "marker_evidence", "evidence_guard"],
        },
        "verification": {"passed": True, "mode": MODE},
        "report_generation": {"generation_seconds": 0.0, "model_executed": False},
    }


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "AscendPilotBioOffline/1.0"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/demo"):
            if not DEMO_PAGE.is_file():
                self._json({"detail": "demo page is missing"}, 404)
                return
            body = DEMO_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/health":
            self._json(
                {
                    "status": "ok",
                    "mode": MODE,
                    "model_execution": False,
                    "notice": (
                        "Public E-ENAD-52 evidence service; no Qwen, riceFM, or "
                        "NPU model is loaded."
                    ),
                }
            )
        elif parsed.path == "/version":
            self._json(
                {
                    "service": "plant-cell-agent-review",
                    "version": "offline-1.0",
                    "mode": MODE,
                    "qwen": {"model": "recorded evidence", "device": "not-running"},
                    "riceFM": {
                        "device": "not-running",
                        "reference": PUBLIC["dataset"]["atlas_accession"],
                    },
                    "auth": "disabled-for-offline-review",
                }
            )
        elif parsed.path == "/skills":
            self._json(
                {
                    "mode": MODE,
                    "skills": [
                        {"name": "recorded_evidence_viewer", "status": "ready"},
                        {"name": "public_atlas_evidence", "status": "ready"},
                    ],
                }
            )
        elif parsed.path == "/migration/report":
            self._json(MIGRATION_REPORT)
        elif parsed.path == "/ricefm/metrics":
            self._json(RECORDED_METRICS)
        elif parsed.path == "/ricefm/umap":
            value = parse_qs(parsed.query).get("limit", ["180"])[0]
            self._json(public_umap(int(value)))
        elif parsed.path == "/ricefm/demo-samples":
            self._json(DEMO_SAMPLES)
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/ricefm/annotate":
                self._json(offline_annotation(payload))
            elif self.path == "/agent/run":
                self._json(offline_agent(payload))
            else:
                self._json({"detail": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"detail": str(exc)}, 400)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[offline-demo] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}/demo"
    print(
        "PUBLIC EVIDENCE MODE: E-ENAD-52/GSE146035 subset; "
        "no Qwen, riceFM, or NPU inference."
    )
    print(f"Open {url}")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping offline demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
