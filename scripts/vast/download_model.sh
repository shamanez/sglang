#!/usr/bin/env bash
set -euo pipefail
set -a; . /workspace/.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.6-35B-A3B --cache-dir /workspace/models
