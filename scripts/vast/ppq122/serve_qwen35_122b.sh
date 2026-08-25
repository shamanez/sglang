#!/usr/bin/env bash
# Host Qwen/Qwen3.5-122B-A10B-FP8 (FP8 weights, ~118 GiB) on 4x RTX 5090 for
# the PP activation wire-quant experiment. Mirrors /workspace/serve_qwen36.sh,
# but topology (--tp-size/--pp-size) must be passed by the caller and the
# memory fraction is an env knob. sglang auto-detects the checkpoint's FP8
# quantization from config.json; FP8 weights are compute-side quantization,
# orthogonal to the inter-stage activation wire codec (activations stay bf16).
# Env: MEM_FRACTION (default 0.94).
set -euo pipefail
set -a; . /workspace/.env; set +a
# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate
exec python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-122B-A10B-FP8 \
    --download-dir /workspace/models \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static "${MEM_FRACTION:-0.94}" \
    "$@"
