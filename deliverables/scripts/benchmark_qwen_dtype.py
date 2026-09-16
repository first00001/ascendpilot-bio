#!/usr/bin/env python3
"""Compare Qwen FP32 and FP16 generation on the same Ascend NPU."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure(model_path: str, device: str, dtype, runs: int, max_new_tokens: int):
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    torch.npu.synchronize()
    load_seconds = time.perf_counter() - load_started
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a rigorous rice single-cell assistant."},
            {"role": "user", "content": "用三点说明水稻单细胞标签迁移的可信度和限制。"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = {key: value.to(device) for key, value in tokenizer(text, return_tensors="pt").items()}

    def generate():
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started
        tokens = int(output.shape[1] - inputs["input_ids"].shape[1])
        return elapsed, tokens

    generate()
    samples = [generate() for _ in range(runs)]
    result = {
        "dtype": str(dtype).removeprefix("torch."),
        "device": device,
        "load_seconds": load_seconds,
        "runs": runs,
        "max_new_tokens": max_new_tokens,
        "mean_generation_seconds": statistics.mean(value[0] for value in samples),
        "mean_tokens_per_second": statistics.mean(value[1] / value[0] for value in samples),
        "peak_memory_bytes": int(torch.npu.max_memory_allocated(device)),
        "completion_tokens": [value[1] for value in samples],
    }
    del model, tokenizer, inputs
    gc.collect()
    torch.npu.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/shared_assets/models/Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float16"), default=("float32", "float16"))
    args = parser.parse_args()
    results = [
        measure(args.model, args.device, getattr(torch, name), args.runs, args.max_new_tokens)
        for name in args.dtypes
    ]
    payload = {"model": "Qwen3.5-4B", "comparison": results}
    if len(results) == 2:
        baseline, optimized = results
        payload["improvement"] = {
            "throughput_ratio": optimized["mean_tokens_per_second"] / baseline["mean_tokens_per_second"],
            "peak_memory_reduction_percent": 100 * (1 - optimized["peak_memory_bytes"] / baseline["peak_memory_bytes"]),
        }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
