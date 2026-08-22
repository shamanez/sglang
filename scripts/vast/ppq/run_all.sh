#!/usr/bin/env bash
# Driver: all PP wire-quant configs through the single run_config.sh path,
# then restore the production tp4 server. Every config launches its own
# server; the harness asserts pp/tp against each label before writing results.
set -uo pipefail
cd /workspace/sglang-src/scripts/vast/ppq

echo "=== [1/6] pp2_bf16 (generates reference trajectories) ==="
MAKE_REFERENCE=1 bash run_config.sh pp2_bf16 none --tp-size 2 --pp-size 2 --attention-backend flashinfer || echo "FAILED pp2_bf16"
echo "=== [2/6] pp2_int8 ==="
bash run_config.sh pp2_int8 int8 --tp-size 2 --pp-size 2 --attention-backend flashinfer || echo "FAILED pp2_int8"
echo "=== [3/6] pp2_int4 ==="
bash run_config.sh pp2_int4 int4 --tp-size 2 --pp-size 2 --attention-backend flashinfer || echo "FAILED pp2_int4"
echo "=== [4/6] pp4_bf16 ==="
bash run_config.sh pp4_bf16 none --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED pp4_bf16"
echo "=== [5/6] pp4_int8 ==="
bash run_config.sh pp4_int8 int8 --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED pp4_int8"
echo "=== [6/6] pp4_int4 ==="
bash run_config.sh pp4_int4 int4 --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED pp4_int4"

echo "=== restoring production tp4 server ==="
pkill -f "sglang\.launch_server" 2>/dev/null; sleep 8
pkill -9 -f "sglang::" 2>/dev/null; sleep 5
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
nohup /workspace/serve_qwen36.sh > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
echo "ALL_DONE"
