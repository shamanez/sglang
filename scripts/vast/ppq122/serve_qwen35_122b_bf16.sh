#!/usr/bin/env bash
# Host Qwen/Qwen3.5-122B-A10B in native bf16 (~227 GiB weights) on 8x RTX 5090.
# Fits only with a starved KV budget: the model is hybrid-GDN (12 of 48 layers
# carry KV, ~24.6 KB/token model-wide) and the eval needs <=4k contexts at low
# concurrency, so a few hundred MB of KV suffices. Topology and any
# SGLANG_PP_LAYER_PARTITION come from the caller. Env: MEM_FRACTION (0.96),
# CONTEXT_LEN (4096), MAX_RUNNING (4).
set -euo pipefail
set -a; . /workspace/.env; set +a
# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate
exec python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-122B-A10B \
    --download-dir /workspace/models \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static "${MEM_FRACTION:-0.96}" \
    --context-length "${CONTEXT_LEN:-4096}" \
    --max-running-requests "${MAX_RUNNING:-4}" \
    --chunked-prefill-size 1024 \
    --disable-cuda-graph \
    "$@"
