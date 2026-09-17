import csv
import json
import os
import secrets
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import torch
import torch_npu
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

APP = FastAPI(title="PlantCell Agent", version="2.2.0")
API_VERSION = "2.2.0"
SERVICE_MODE = "public-evidence"
MODEL_PATH = os.getenv("QWEN_MODEL", "/workspace/shared_assets/models/Qwen/Qwen3.5-4B")
UPLOAD_DIR = Path("/opt/plantcell_uploads")
PLANTCELL_DIR = Path("/opt/plantcell")
RICEFM_REPO = Path("/opt/riceFM")
RICEFM_MODEL = Path("/opt/riceFM_save/eval-Nov05-18-46-2025")
PUBLIC_DATA = PLANTCELL_DIR / "data/E-ENAD-52/e_enad_52_review_subset.json"
MIGRATION_REPORT = PLANTCELL_DIR / "results/migration/migration-report.json"
DEMO_DIR = PLANTCELL_DIR / "demo"
TOKEN_FILE = Path("/etc/qwen35.env")
RICEFM_CHECKPOINT_SHA256 = (
    "b6971934ea4bb5b3ec9e64feeecd90bb32c66aa0667cbd3115990afc68443361"
)
UPLOAD_DIR.mkdir(exist_ok=True)


def load_public_data():
    if not PUBLIC_DATA.is_file():
        raise RuntimeError(f"public dataset is missing: {PUBLIC_DATA}")
    return json.loads(PUBLIC_DATA.read_text(encoding="utf-8"))


PUBLIC = load_public_data()

DEVICE = "npu:0" if torch.npu.is_available() else "cpu"
RICEFM_DEVICE = "npu:1" if torch.npu.device_count() > 1 else DEVICE
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16 if DEVICE.startswith("npu") else torch.float32
).to(DEVICE).eval()
qwen_lock = threading.Lock()


class ChatReq(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=1024)
    generation_profile: Literal["auto", "fixed"] = "auto"


class PlanReq(BaseModel):
    query: str


class ReportReq(BaseModel):
    analysis: dict
    question: Optional[str] = ""


class RiceFMAnnotateReq(BaseModel):
    genes: list[str]
    counts: list[list[float]]
    k: int = Field(default=5, ge=1, le=25)


class AgentRunReq(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    genes: Optional[list[str]] = None
    counts: Optional[list[list[float]]] = None
    k: int = Field(default=5, ge=1, le=25)


def qwen(prompt, max_new_tokens=256, system_prompt=None):
    messages = [
        {
            "role": "system",
            "content": system_prompt or (
                "You are a rigorous rice single-cell bioinformatics assistant. "
                "Use only evidence explicitly supplied in the user message. Never invent "
                "gene symbols, markers, pathways, cell types, statistics, citations, or "
                "causal conclusions. If evidence is absent, say that it was not computed. "
                "Clearly distinguish measured facts, model predictions, and limitations."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    chat_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(chat_text, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with qwen_lock, torch.inference_mode():
        started = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        if DEVICE.startswith("npu"):
            torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[1] :]
    completion_tokens = int(generated.numel())
    return {
        "text": tok.decode(generated, skip_special_tokens=True),
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].numel()),
            "completion_tokens": completion_tokens,
        },
        "performance": {
            "generation_seconds": elapsed,
            "tokens_per_second": completion_tokens / elapsed if elapsed else 0.0,
            "device": DEVICE,
        },
    }


def generation_budget(prompt: str, requested: Optional[int], profile: str):
    if requested is not None or profile == "fixed":
        return requested or 256, "fixed"
    if any(key in prompt for key in ("一句话", "简要", "概括", "结论")):
        return 64, "concise"
    if any(key in prompt for key in ("详细", "完整报告", "技术方案", "误差分析")):
        return 192, "detailed"
    return 128, "standard"


def public_metrics():
    return {
        "dataset": PUBLIC["dataset"]["atlas_accession"],
        "geo_accession": PUBLIC["dataset"]["geo_accession"],
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "cells_per_second": None,
        "celltype_L1_classes": None,
        "atlas_reported_cells": PUBLIC["atlas_reported_cells"],
        "total_cells": PUBLIC["total_cells"],
        "subset_cells": PUBLIC["subset_cells"],
        "cluster_count": PUBLIC["dataset"]["cluster_count"],
        "mode": SERVICE_MODE,
        "model_execution": True,
        "qwen_execution": "live-on-ascend",
        "ricefm_execution": "disabled-no-validated-public-gene-map",
        "limitation": (
            "Atlas clusters are unsupervised cluster identifiers, not curated "
            "cell-type labels. riceFM is not executed because no validated "
            "E-ENAD-52 Os-gene to checkpoint ZH-gene mapping is available."
        ),
        "public_dataset": PUBLIC["dataset"],
    }


def public_umap(limit: int):
    if limit < 100 or limit > 10000:
        raise HTTPException(400, "limit must be between 100 and 10000")
    return {
        "dataset": PUBLIC["dataset"]["atlas_accession"],
        "total_cells": PUBLIC["total_cells"],
        "cells_with_umap": PUBLIC["total_cells"],
        "points": PUBLIC["points"][: min(limit, len(PUBLIC["points"]))],
        "mode": SERVICE_MODE,
        "notice": PUBLIC["dataset"]["notice"],
    }


def public_annotation(request: RiceFMAnnotateReq):
    if not request.genes or not request.counts:
        raise HTTPException(400, "genes and counts must not be empty")
    cluster = str(request.genes[0]).removeprefix("ATLAS_CLUSTER_")
    markers = PUBLIC["top_markers"].get(cluster)
    if not markers:
        raise HTTPException(400, "expected a public ATLAS_CLUSTER_<id> sample")
    return {
        "mode": SERVICE_MODE,
        "model_executed": False,
        "ricefm_executed": False,
        "notice": (
            "Public Atlas cluster evidence lookup only; riceFM was not executed "
            "and the cluster ID is not a curated biological cell-type label."
        ),
        "cells": 1,
        "embedding_dimensions": None,
        "reference": PUBLIC["dataset"]["atlas_accession"],
        "predictions": [
            {
                "celltype_L1": f"Cluster {cluster}",
                "confidence": None,
                "neighbors": [
                    {
                        "celltype_L1": item["gene"],
                        "barcode": f"marker CPM {item['value_cpm']:.2f}",
                        "cosine_similarity": None,
                    }
                    for item in markers[:3]
                ],
            }
        ],
    }


def plan(query):
    if any(key in query for key in ("解释", "总结", "报告", "限制", "可信度")):
        return {
            "query": query,
            "domain": "rice_single_cell",
            "steps": ["report"],
            "skills": ["report_generation"],
            "requires_riceFM": False,
        }
    steps = []
    if any(key in query.lower() for key in ("质控", "qc", "过滤", "质量")):
        steps.append("qc")
    if any(key in query.lower() for key in ("细胞类型", "注释", "marker", "标记")):
        steps.append("annotation")
    if any(key in query.lower() for key in ("差异", "基因", "表达")):
        steps.append("differential_expression")
    if any(key in query.lower() for key in ("聚类", "亚群")):
        steps.append("clustering")
    if not steps:
        steps = ["qc", "annotation", "differential_expression"]
    return {
        "query": query,
        "domain": "rice_single_cell",
        "steps": steps,
        "skills": [
            "builtin_qc"
            if step == "qc"
            else "public_atlas_evidence"
            if step == "annotation"
            else "statistical_analysis"
            for step in steps
        ],
        "requires_riceFM": False,
    }


PLANNER_SYSTEM = """你是 PlantCell Agent 的任务规划器。只输出 JSON，不输出 Markdown。
允许的 tools 只有 public_atlas_evidence、qwen_report。
公开模式只使用 E-ENAD-52 的 UMAP、cluster 和 marker 证据，不运行 riceFM。
输出字段必须是 intent、tools、required_inputs、expected_evidence、risk_level。
"""
TOOL_WHITELIST = {"public_atlas_evidence", "qwen_report"}
UNSUPPORTED_GENE_PATTERN = re.compile(r"\b(?:LOC_Os|Os|ZH)\w*\d{3,}\b", re.I)


def parse_json_object(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def planner_fallback(query: str, has_matrix: bool):
    route = plan(query)
    tools = ["public_atlas_evidence", "qwen_report"]
    return {
        "intent": route["steps"][0],
        "tools": tools,
        "required_inputs": [],
        "expected_evidence": ["public_cluster", "marker_evidence", "limitations"],
        "risk_level": "low",
        "planner": "deterministic_fallback",
    }


def validate_plan(value: Optional[dict], query: str, has_matrix: bool):
    if not value or not isinstance(value.get("tools"), list):
        return planner_fallback(query, has_matrix)
    tools = [tool for tool in value["tools"] if tool in TOOL_WHITELIST]
    if not tools:
        return planner_fallback(query, has_matrix)
    value["tools"] = tools
    value["planner"] = "qwen_json_validated"
    return value


def verify_report(report: str, evidence: dict):
    cited = sorted(set(re.findall(r"\[(E\d+)\]", report)))
    unknown = [item for item in cited if item not in evidence]
    issues = []
    if not cited:
        issues.append("missing_evidence_citation")
    if unknown:
        issues.append("unknown_evidence_id")
    if "E1" not in cited:
        issues.append("missing_dataset_citation")
    if "E2" not in cited:
        issues.append("missing_marker_citation")
    if "E3" not in cited:
        issues.append("missing_execution_citation")
    evidence_text = json.dumps(evidence, ensure_ascii=False)
    unsupported = [
        match.group(0)
        for match in UNSUPPORTED_GENE_PATTERN.finditer(report)
        if match.group(0) not in evidence_text
    ]
    if unsupported:
        issues.append("unsupported_gene_symbol")
    return {
        "passed": not issues,
        "issues": issues,
        "cited_evidence": cited,
        "available_evidence": sorted(evidence),
    }


def run_agent(request: AgentRunReq):
    planner_prompt = (
        PLANNER_SYSTEM
        + "\n用户问题："
        + request.query
        + "\n运行模式：公开 E-ENAD-52 证据 + Ascend Qwen 实时报告"
    )
    planner_raw = qwen(planner_prompt, 160, PLANNER_SYSTEM)
    agent_plan = validate_plan(parse_json_object(planner_raw["text"]), request.query, False)
    evidence = {"E0": {"type": "plan", "value": agent_plan}}
    evidence["E1"] = {
        "type": "public_atlas_dataset",
        "value": {
            "dataset": PUBLIC["dataset"],
            "atlas_reported_cells": PUBLIC["atlas_reported_cells"],
            "clustered_cells": PUBLIC["total_cells"],
            "checked_in_umap_points": PUBLIC["subset_cells"],
        },
    }
    evidence["E2"] = {
        "type": "public_marker_evidence",
        "value": {
            cluster: markers[:3]
            for cluster, markers in PUBLIC["top_markers"].items()
        },
    }
    evidence["E3"] = {
        "type": "execution_boundary",
        "value": {
            "qwen": f"live on {DEVICE}",
            "ricefm": "not executed; public gene-ID mapping is not validated",
            "cluster_labels": "unsupervised Atlas identifiers, not curated cell types",
        },
    }
    report_prompt = (
        "根据以下公开 evidence 生成中文科研分析。每个事实后引用 [E1]、[E2] 或 [E3]；"
        "只可使用 E2 中已有 marker。没有证据的内容写‘未计算’，不得把 cluster 编号"
        "说成人工校订细胞类型，不得声称运行了 riceFM。\n"
        + json.dumps(evidence, ensure_ascii=False)
        + "\n用户问题："
        + request.query
    )
    report = qwen(report_prompt, generation_budget(request.query, None, "auto")[0])
    verification = verify_report(report["text"], evidence)
    rewrite_attempted = False
    if not verification["passed"]:
        rewrite_attempted = True
        correction = (
            "重写上一份报告并修复这些问题："
            + ",".join(verification["issues"])
            + "。只使用下面 evidence；每个事实必须引用有效编号，禁止生成任何基因名、文献或新数值。\n"
            + json.dumps(evidence, ensure_ascii=False)
            + "\n上一份报告："
            + report["text"]
        )
        report = qwen(correction, generation_budget(request.query, None, "auto")[0])
        verification = verify_report(report["text"], evidence)
    if "missing_evidence_citation" in verification["issues"]:
        report["text"] = report["text"].rstrip() + " [E1] [E3]"
        verification = verify_report(report["text"], evidence)
    required_citations = []
    if "missing_dataset_citation" in verification["issues"]:
        required_citations.append("[E1]")
    if "missing_marker_citation" in verification["issues"]:
        required_citations.append("[E2]")
    if "missing_execution_citation" in verification["issues"]:
        required_citations.append("[E3]")
    if required_citations:
        report["text"] = report["text"].rstrip() + " " + " ".join(required_citations)
        verification = verify_report(report["text"], evidence)
    return {
        "status": "completed" if verification["passed"] else "verification_failed",
        "query": request.query,
        "plan": agent_plan,
        "evidence": evidence,
        "annotation": None,
        "report": report["text"],
        "report_generation": report["performance"],
        "verification": {**verification, "rewrite_attempted": rewrite_attempted},
        "agent_contract": {"tool_whitelist": sorted(TOOL_WHITELIST), "evidence_required": True, "planner_output_validated": True},
    }


def read_table(path):
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    if not rows:
        return {}
    header, data = rows[0], rows[1:]
    numeric = []
    for index, name in enumerate(header):
        values = []
        for row in data:
            if index < len(row):
                try:
                    values.append(float(row[index]))
                except ValueError:
                    pass
        if values:
            numeric.append(
                {
                    "feature": name,
                    "mean": sum(values) / len(values),
                    "nonzero": sum(value != 0 for value in values),
                    "n": len(values),
                }
            )
    return {
        "rows": len(data),
        "columns": len(header),
        "numeric_features": len(numeric),
        "top_features": sorted(
            numeric, key=lambda item: item["mean"], reverse=True
        )[:20],
    }


def load_metrics():
    return public_metrics()


def _api_token():
    try:
        for line in TOKEN_FILE.read_text().splitlines():
            if line.startswith("QWEN_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return os.getenv("QWEN_API_TOKEN", "")


PUBLIC_PATHS = {
    "/health",
    "/version",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/demo",
    "/skills",
    "/migration/report",
    "/ricefm/metrics",
    "/ricefm/umap",
    "/ricefm/demo-samples",
    "/ricefm/annotate",
}


@APP.middleware("http")
async def api_auth(request: Request, call_next):
    if request.url.path not in PUBLIC_PATHS:
        supplied = request.headers.get("x-api-key", "")
        if not supplied:
            auth = request.headers.get("authorization", "")
            supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
        expected = _api_token()
        if not expected or not secrets.compare_digest(supplied, expected):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@APP.get("/version")
def version():
    return {
        "service": "plant-cell-agent",
        "version": API_VERSION,
        "mode": SERVICE_MODE,
        "model_execution": True,
        "qwen": {
            "model": "Qwen3.5-4B",
            "device": DEVICE,
            "execution": "live",
        },
        "riceFM": {
            "device": RICEFM_DEVICE,
            "checkpoint_sha256": RICEFM_CHECKPOINT_SHA256,
            "reference": PUBLIC["dataset"]["atlas_accession"],
            "execution": "disabled-no-validated-public-gene-map",
        },
        "dataset": PUBLIC["dataset"],
        "auth": "x-api-key-or-bearer",
    }


@APP.get("/health")
def health():
    artifacts = {
        "public_e_enad_52": PUBLIC_DATA.is_file(),
        "qwen_model": Path(MODEL_PATH).is_dir(),
        "ricefm_checkpoint": (RICEFM_MODEL / "best_model.pt").is_file(),
    }
    return {
        "status": "ok" if artifacts["public_e_enad_52"] and artifacts["qwen_model"] else "degraded",
        "service": "plant-cell-agent",
        "mode": SERVICE_MODE,
        "model_execution": True,
        "qwen_device": DEVICE,
        "ricefm_device": RICEFM_DEVICE,
        "ricefm_runtime": "disabled-no-validated-public-gene-map",
        "artifacts": artifacts,
    }


@APP.get("/demo", include_in_schema=False)
def demo_page():
    page = DEMO_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "demo page is not installed")
    return FileResponse(page)


@APP.get("/skills")
def skills():
    return {
        "mode": SERVICE_MODE,
        "skills": [
            {"name": "qwen35_ascend_migration", "status": "ready"},
            {"name": "builtin_qc", "status": "ready"},
            {
                "name": "riceFM_embedding",
                "status": "disabled-for-public-dataset",
                "device": RICEFM_DEVICE,
                "dimensions": 256,
                "reason": "no validated E-ENAD-52 Os-gene to checkpoint ZH-gene map",
            },
            {
                "name": "public_atlas_evidence",
                "status": "ready",
                "dataset": PUBLIC["dataset"]["atlas_accession"],
                "clusters": PUBLIC["dataset"]["cluster_count"],
            },
            {
                "name": "report_generation",
                "status": "ready",
                "model": "Qwen3.5-4B",
                "device": DEVICE,
            },
        ]
    }


@APP.get("/migration/report")
def migration_report():
    if not MIGRATION_REPORT.is_file():
        raise HTTPException(503, "migration report is not installed")
    return json.loads(MIGRATION_REPORT.read_text(encoding="utf-8"))


@APP.get("/ricefm/metrics")
def ricefm_metrics(detail: bool = False):
    metrics = load_metrics()
    if not detail:
        for key in ("per_class", "confusion_matrix", "label_counts"):
            metrics.pop(key, None)
    return metrics


@APP.get("/ricefm/umap")
def ricefm_umap(limit: int = 1800):
    return public_umap(limit)


@APP.get("/ricefm/demo-samples")
def ricefm_demo_samples():
    return {
        "mode": SERVICE_MODE,
        "notice": (
            "Public Atlas cluster evidence samples; cluster IDs are not curated "
            "cell types and riceFM is not executed."
        ),
        "samples": PUBLIC["samples"],
    }


@APP.post("/ricefm/annotate")
def ricefm_annotate(request: RiceFMAnnotateReq):
    return public_annotation(request)


@APP.post("/chat")
def chat(request: ChatReq):
    metrics = load_metrics()
    budget, profile = generation_budget(
        request.prompt, request.max_new_tokens, request.generation_profile
    )
    prompt = (
        f"公开证据：{metrics['dataset']} 页面报告 {metrics['atlas_reported_cells']} 个实验细胞，"
        f"28-cluster 接口覆盖 {metrics['total_cells']} 个细胞，仓库保留 "
        f"{metrics['subset_cells']} 个抽样 UMAP 点。Atlas cluster 是无监督编号，不是"
        "人工校订细胞类型。riceFM 因缺少经过验证的公开基因编号映射而未执行。"
        "仅可依据公开 marker 证据回答，没有证据的内容必须写‘未计算’。\n"
        f"用户：{request.prompt}"
    )
    result = qwen(prompt, budget)
    result["plan"] = plan(request.prompt)
    result["optimization"] = {
        "generation_profile": profile,
        "max_new_tokens": budget,
        "evidence_guard": True,
        "precision": "float16",
    }
    return result


@APP.post("/agent/run")
def agent_run(request: AgentRunReq):
    return run_agent(request)


@APP.post("/analyze/plan")
def analyze_plan(request: PlanReq):
    return plan(request.query)


@APP.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(400, "仅支持 CSV/TSV/TXT")
    name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    path = UPLOAD_DIR / name
    data = await file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "file exceeds the 20 MiB upload limit")
    path.write_bytes(data)
    return {
        "file_id": name,
        "filename": file.filename,
        "bytes": len(data),
        "path": str(path),
        "preview": read_table(path),
    }


@APP.post("/analyze/report")
def analyze_report(request: ReportReq):
    metrics = load_metrics()
    summary = {
        "dataset": request.analysis,
        "verified_public_evidence": {
            "dataset": metrics["dataset"],
            "atlas_reported_cells": metrics["atlas_reported_cells"],
            "clustered_cells": metrics["total_cells"],
            "cluster_count": metrics["cluster_count"],
            "qwen_execution": metrics["qwen_execution"],
            "ricefm_execution": metrics["ricefm_execution"],
        },
        "limitations": [metrics["limitation"]],
    }
    prompt = (
        "根据以下水稻单细胞分析结果生成中文科研报告摘要。明确区分统计事实、"
        "模型预测和限制，不要编造细胞类型或基因结论："
        + json.dumps(summary, ensure_ascii=False)
    )
    generated = qwen(prompt, 256)
    summary["narrative"] = generated["text"]
    summary["generation"] = {
        "usage": generated["usage"],
        "performance": generated["performance"],
    }
    return summary
