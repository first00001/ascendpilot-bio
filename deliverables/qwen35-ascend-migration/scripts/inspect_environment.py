#!/usr/bin/env python3
"""Create a machine-readable Ascend migration environment fingerprint."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def module_version(name):
    try:
        module = __import__(name.replace("-", "_"))
        return getattr(module, "__version__", "installed")
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = args.model / "config.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "software": {
            "torch": module_version("torch"),
            "torch_npu": module_version("torch_npu"),
            "transformers": module_version("transformers"),
            "cann_home": os.getenv("ASCEND_HOME_PATH", ""),
        },
        "npu_smi": command(["npu-smi", "info"]),
        "model": {
            "path": str(args.model.resolve()),
            "config_exists": config.is_file(),
            "tokenizer_exists": any((args.model / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model_ready": payload["model"]}, ensure_ascii=False))
    if not payload["model"]["config_exists"] or not payload["model"]["tokenizer_exists"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

