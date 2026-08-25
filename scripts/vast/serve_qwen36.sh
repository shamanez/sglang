#!/usr/bin/env bash
# Host Qwen/Qwen3.6-35B-A3B on 4x RTX 5090 from the editable dev install.
set -euo pipefail
set -a; . /workspace/.env; set +a
# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate
exec python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-35B-A3B \
    --download-dir /workspace/models \
    --tp-size 4 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    "$@"
