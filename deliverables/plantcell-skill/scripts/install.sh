#!/usr/bin/env bash
set -euo pipefail

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/usr/local/python3.12.13/bin/python3}
QWEN_MODEL=${QWEN_MODEL:-/workspace/shared_assets/models/Qwen/Qwen3.5-4B}
RICEFM_REPO=${RICEFM_REPO:-/opt/riceFM}
RICEFM_MODEL=${RICEFM_MODEL:-/opt/riceFM_save/eval-Nov05-18-46-2025}

for path in "$QWEN_MODEL/config.json" "$RICEFM_REPO" "$RICEFM_MODEL/args.json" \
  "$RICEFM_MODEL/best_model.pt" "$RICEFM_MODEL/vocab.json"; do
  [[ -e "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done
$PYTHON - <<'PY'
import fastapi, numpy, pydantic, torch, torch_npu, transformers, uvicorn
print("Python dependencies OK")
PY

install -d -m 0755 /opt/plantcell/{skill,data/ZH11_riceFM_eval,results/zh11,results/migration,demo} /var/log/qwen35
install -m 0644 "$SOURCE/deployment/qwen35_server.py" /opt/qwen35_server.py
install -m 0644 "$SOURCE/skill/plantcell_skill.py" "$SOURCE/skill/ricefm_adapter.py" /opt/plantcell/skill/
install -m 0644 "$SOURCE/demo/index.html" /opt/plantcell/demo/index.html
cp -a "$SOURCE/data/ZH11_riceFM_eval/." /opt/plantcell/data/ZH11_riceFM_eval/
cp -a "$SOURCE/results/zh11/." /opt/plantcell/results/zh11/
cp -a "$SOURCE/results/migration/." /opt/plantcell/results/migration/

if [[ ! -s /etc/qwen35.env ]]; then
  umask 077
  printf 'QWEN_API_TOKEN=%s\n' "$(openssl rand -base64 36 | tr -d '\n')" > /etc/qwen35.env
fi
cat >/etc/supervisord.d/qwen35.ini <<EOF
[program:qwen35]
command=$PYTHON -m uvicorn qwen35_server:APP --host 127.0.0.1 --port 8000
directory=/opt
environment=QWEN_MODEL="$QWEN_MODEL"
autostart=true
autorestart=true
startsecs=10
startretries=5
stopsignal=TERM
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/qwen35/stdout.log
stderr_logfile=/var/log/qwen35/stderr.log
stdout_logfile_maxbytes=20MB
stderr_logfile_maxbytes=20MB
stdout_logfile_backups=5
stderr_logfile_backups=5
EOF
cat >/etc/logrotate.d/qwen35 <<'EOF'
/var/log/qwen35/*.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
  copytruncate
}
EOF
supervisorctl reread
supervisorctl update
supervisorctl restart qwen35
for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8000/health >/dev/null && break; sleep 2; done
"$(dirname "${BASH_SOURCE[0]}")/healthcheck.sh"
echo "Demo: http://127.0.0.1:8000/demo"
