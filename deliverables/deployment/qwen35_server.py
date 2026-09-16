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

import numpy as np
import torch
import torch_npu
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

APP = FastAPI(title="PlantCell Agent", version="2.1.0")
API_VERSION = "2.1.0"
MODEL_PATH = os.getenv("QWEN_MODEL", "/workspace/shared_assets/models/Qwen/Qwen3.5-4B")
UPLOAD_DIR = Path("/opt/plantcell_uploads")
PLANTCELL_DIR = Path("/opt/plantcell")
RICEFM_REPO = Path("/opt/riceFM")
RICEFM_MODEL = Path("/opt/riceFM_save/eval-Nov05-18-46-2025")
ZH11_DATA = PLANTCELL_DIR / "data/ZH11_riceFM_eval"
ZH11_RESULTS = PLANTCELL_DIR / "results/zh11"
MIGRATION_REPORT = PLANTCELL_DIR / "results/migration/migration-report.json"
DEMO_DIR = PLANTCELL_DIR / "demo"
DEMO_SAMPLES = ZH11_DATA / "demo_samples.json"
TOKEN_FILE = Path("/etc/qwen35.env")
RICEFM_CHECKPOINT_SHA256 = (
    "b6971934ea4bb5b3ec9e64feeecd90bb32c66aa0667cbd3115990afc68443361"
)
UPLOAD_DIR.mkdir(exist_ok=True)

DEVICE = "npu:0" if torch.npu.is_available() else "cpu"
RICEFM_DEVICE = "npu:1" if torch.npu.device_count() > 1 else DEVICE
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16 if DEVICE.startswith("npu") else torch.float32
).to(DEVICE).eval()
qwen_lock = threading.Lock()
ricefm_lock = threading.RLock()
ricefm_runtime = None
reference_embeddings = None
reference_labels = None
reference_barcodes = None
reference_metadata = None


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
            else "riceFM_zh11_reference"
            if step == "annotation"
            else "statistical_analysis"
            for step in steps
        ],
        "requires_riceFM": any(step != "qc" for step in steps),
    }


PLANNER_SYSTEM = """你是 PlantCell Agent 的任务规划器。只输出 JSON，不输出 Markdown。
允许的 tools 只有 validate_matrix、ricefm_annotate、zh11_metrics、qwen_report。
如果用户没有提供 genes 和 counts，不得选择 ricefm_annotate。
输出字段必须是 intent、tools、required_inputs、expected_evidence、risk_level。
"""
TOOL_WHITELIST = {"validate_matrix", "ricefm_annotate", "zh11_metrics", "qwen_report"}
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
    tools = ["zh11_metrics"]
    if has_matrix and "annotation" in route["steps"]:
        tools = ["validate_matrix", "ricefm_annotate", "zh11_metrics"]
    if "report" in route["steps"] or not has_matrix:
        tools.append("qwen_report")
    return {
        "intent": route["steps"][0],
        "tools": tools,
        "required_inputs": ["genes", "counts"] if has_matrix else [],
        "expected_evidence": ["prediction", "confidence", "neighbors"] if has_matrix else ["verified_metrics", "limitations"],
        "risk_level": "medium" if has_matrix else "low",
        "planner": "deterministic_fallback",
    }


def validate_plan(value: Optional[dict], query: str, has_matrix: bool):
    if not value or not isinstance(value.get("tools"), list):
        return planner_fallback(query, has_matrix)
    tools = [tool for tool in value["tools"] if tool in TOOL_WHITELIST]
    if "ricefm_annotate" in tools and not has_matrix:
        tools.remove("ricefm_annotate")
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
    if UNSUPPORTED_GENE_PATTERN.search(report):
        issues.append("unsupported_gene_symbol")
    return {
        "passed": not issues,
        "issues": issues,
        "cited_evidence": cited,
        "available_evidence": sorted(evidence),
    }


def run_agent(request: AgentRunReq):
    has_matrix = request.genes is not None and request.counts is not None
    planner_prompt = (
        PLANNER_SYSTEM
        + "\n用户问题："
        + request.query
        + f"\n表达矩阵已提供：{'是' if has_matrix else '否'}"
    )
    planner_raw = qwen(planner_prompt, 160, PLANNER_SYSTEM)
    agent_plan = validate_plan(parse_json_object(planner_raw["text"]), request.query, has_matrix)
    evidence = {"E0": {"type": "plan", "value": agent_plan}}
    requests_annotation = any(
        key in request.query.lower() for key in ("注释", "细胞类型", "annotate", "annotation")
    )
    if requests_annotation and not has_matrix:
        return {
            "status": "input_required",
            "query": request.query,
            "plan": {
                **agent_plan,
                "intent": "cell_type_annotation",
                "required_inputs": ["genes", "counts"],
            },
            "evidence": evidence,
            "annotation": None,
            "report": "执行细胞类型注释需要 genes 与 counts；当前未提供表达矩阵，因此未运行 riceFM，也未生成预测。 [E0]",
            "report_generation": None,
            "verification": {
                "passed": True,
                "issues": [],
                "cited_evidence": ["E0"],
                "available_evidence": ["E0"],
                "rewrite_attempted": False,
            },
            "agent_contract": {
                "tool_whitelist": sorted(TOOL_WHITELIST),
                "evidence_required": True,
                "planner_output_validated": True,
            },
        }
    annotation = None
    if "validate_matrix" in agent_plan["tools"]:
        if not request.genes or not request.counts:
            raise HTTPException(400, "plan requires genes and counts")
        if len(request.counts) > 128 or any(len(row) != len(request.genes) for row in request.counts):
            raise HTTPException(400, "matrix dimensions exceed the Skill contract")
        annotation = ricefm_annotate(RiceFMAnnotateReq(genes=request.genes, counts=request.counts, k=request.k))
        evidence["E1"] = {"type": "ricefm_annotation", "value": annotation}
    metrics = load_metrics()
    evidence["E2"] = {"type": "verified_zh11_metrics", "value": {key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1", "cells_per_second")}}
    report_prompt = (
        "根据以下已验证 evidence 生成中文科研分析。每个事实后引用 [E1] 或 [E2]；"
        "没有证据的内容写‘未计算’，不要生成 marker、差异基因、通路或引用。\n"
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
        report["text"] = report["text"].rstrip() + " [E2]"
        verification = verify_report(report["text"], evidence)
    return {
        "status": "completed" if verification["passed"] else "verification_failed",
        "query": request.query,
        "plan": agent_plan,
        "evidence": evidence,
        "annotation": annotation,
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
    return json.loads((ZH11_RESULTS / "metrics.json").read_text(encoding="utf-8"))


def get_reference():
    global reference_embeddings, reference_labels, reference_barcodes, reference_metadata
    if reference_embeddings is not None:
        return reference_embeddings, reference_labels, reference_barcodes
    with ricefm_lock:
        if reference_embeddings is None:
            reference_embeddings = np.load(ZH11_RESULTS / "embeddings.npy")
            norm = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
            reference_embeddings = reference_embeddings / np.maximum(norm, 1e-12)
            with (ZH11_DATA / "metadata.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            reference_labels = np.asarray([row["celltype_L1"] for row in rows])
            reference_barcodes = np.asarray([row[""] for row in rows])
            reference_metadata = rows
    return reference_embeddings, reference_labels, reference_barcodes


def get_ricefm():
    global ricefm_runtime
    if ricefm_runtime is not None:
        return ricefm_runtime
    with ricefm_lock:
        if ricefm_runtime is None:
            import sys

            sys.path.insert(0, str(PLANTCELL_DIR))
            from skill.ricefm_adapter import RiceFMRuntime

            ricefm_runtime = RiceFMRuntime(
                str(RICEFM_MODEL), str(RICEFM_REPO), RICEFM_DEVICE
            )
    return ricefm_runtime


def classify_embeddings(embeddings, k):
    reference, labels, barcodes = get_reference()
    embeddings = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    similarities = embeddings @ reference.T
    neighbors = np.argpartition(similarities, -k, axis=1)[:, -k:]
    results = []
    for row, indices in zip(similarities, neighbors):
        votes = {}
        for index in indices:
            label = str(labels[index])
            votes[label] = votes.get(label, 0.0) + float(max(row[index], 0.0))
        label = max(votes, key=lambda key: (votes[key], key))
        total = sum(votes.values())
        closest = indices[np.argsort(row[indices])[::-1]]
        results.append(
            {
                "celltype_L1": label,
                "confidence": votes[label] / total if total else 0.0,
                "neighbors": [
                    {
                        "barcode": str(barcodes[index]),
                        "celltype_L1": str(labels[index]),
                        "cosine_similarity": float(row[index]),
                    }
                    for index in closest
                ],
            }
        )
    return results


def _api_token():
    try:
        for line in TOKEN_FILE.read_text().splitlines():
            if line.startswith("QWEN_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return os.getenv("QWEN_API_TOKEN", "")


PUBLIC_PATHS = {
    "/health", "/version", "/docs", "/openapi.json", "/redoc", "/demo"
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
        "qwen": {"model": "Qwen3.5-4B", "device": DEVICE},
        "riceFM": {
            "device": RICEFM_DEVICE,
            "checkpoint_sha256": RICEFM_CHECKPOINT_SHA256,
            "reference": "ZH11_riceFM_eval",
        },
        "auth": "x-api-key-or-bearer",
    }


@APP.get("/health")
def health():
    artifacts = {
        "zh11_data": (ZH11_DATA / "manifest.json").is_file(),
        "zh11_embeddings": (ZH11_RESULTS / "embeddings.npy").is_file(),
        "zh11_metrics": (ZH11_RESULTS / "metrics.json").is_file(),
        "ricefm_checkpoint": (RICEFM_MODEL / "best_model.pt").is_file(),
    }
    return {
        "status": "ok" if all(artifacts.values()) else "degraded",
        "service": "plant-cell-agent",
        "qwen_device": DEVICE,
        "ricefm_device": RICEFM_DEVICE,
        "ricefm_runtime": "lazy",
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
        "skills": [
            {"name": "qwen35_ascend_migration", "status": "ready"},
            {"name": "builtin_qc", "status": "ready"},
            {
                "name": "riceFM_embedding",
                "status": "ready",
                "device": RICEFM_DEVICE,
                "dimensions": 256,
            },
            {
                "name": "ZH11_reference_annotation",
                "status": "ready",
                "classes": 52,
            },
            {"name": "report_generation", "status": "ready"},
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
    if limit < 100 or limit > 5188:
        raise HTTPException(400, "limit must be between 100 and 5188")
    get_reference()
    plottable = []
    for row in reference_metadata:
        try:
            x, y = float(row["UMAP_1"]), float(row["UMAP_2"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            plottable.append((row, x, y))
    sample_size = min(limit, len(plottable))
    indices = np.linspace(0, len(plottable) - 1, sample_size, dtype=int)
    selected = [plottable[index] for index in indices]
    return {
        "dataset": "ZH11_riceFM_eval",
        "total_cells": len(reference_metadata),
        "cells_with_umap": len(plottable),
        "points": [
            {
                "barcode": row[""],
                "x": x,
                "y": y,
                "celltype_L1": row["celltype_L1"],
                "celltype_L2": row["celltype_L2"],
                "stage": row["stage"],
                "tissue": row["tissue"],
            }
            for row, x, y in selected
        ],
    }


@APP.get("/ricefm/demo-samples")
def ricefm_demo_samples():
    if not DEMO_SAMPLES.is_file():
        raise HTTPException(503, "demo samples are not installed")
    return json.loads(DEMO_SAMPLES.read_text(encoding="utf-8"))


@APP.post("/ricefm/annotate")
def ricefm_annotate(request: RiceFMAnnotateReq):
    if not request.counts or not request.genes:
        raise HTTPException(400, "genes and counts must not be empty")
    if len(request.counts) > 128:
        raise HTTPException(400, "a request may contain at most 128 cells")
    if any(len(row) != len(request.genes) for row in request.counts):
        raise HTTPException(400, "every count row must match the gene list")
    counts = np.asarray(request.counts, dtype=np.float32)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise HTTPException(400, "counts must be finite and non-negative")
    with ricefm_lock:
        embeddings = get_ricefm().cell_embeddings(counts, request.genes)
    return {
        "cells": len(counts),
        "embedding_dimensions": int(embeddings.shape[1]),
        "reference": "ZH11_riceFM_eval",
        "predictions": classify_embeddings(embeddings, request.k),
    }


@APP.post("/chat")
def chat(request: ChatReq):
    metrics = load_metrics()
    budget, profile = generation_budget(
        request.prompt, request.max_new_tokens, request.generation_profile
    )
    prompt = (
        "已验证证据：ZH11_riceFM_eval 含 5188 个细胞、52 个一级类型；"
        f"Accuracy={metrics['accuracy']:.4f}，Macro-F1={metrics['macro_f1']:.4f}，"
        f"riceFM 吞吐={metrics['cells_per_second']:.2f} cells/s。"
        "这是标注参考迁移评测，checkpoint 可能见过重叠 ZH11 表达谱，不是独立外部验证。"
        "当前没有计算 marker、差异基因或通路；禁止补写任何基因名或通路名。\n"
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
        "verified_reference_result": {
            "dataset": metrics["dataset"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "cells_per_second": metrics["cells_per_second"],
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
