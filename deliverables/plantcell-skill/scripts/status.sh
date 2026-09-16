#!/usr/bin/env bash
set -euo pipefail
supervisorctl status qwen35
curl -fsS http://127.0.0.1:8000/health
echo

