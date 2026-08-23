#!/usr/bin/env bash
set -uo pipefail
cd /workspace/sglang-src/scripts/vast/ppq
echo "=== pp4_int4 ==="
bash run_config.sh pp4_int4 int4 --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED pp4_int4"
echo "=== restoring production tp4 server ==="
pkill -f "sglang\.launch_server" 2>/dev/null; sleep 8
pkill -9 -f "sglang::" 2>/dev/null; sleep 5
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
nohup /workspace/serve_qwen36.sh > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
echo "INT4_ALL_DONE"
