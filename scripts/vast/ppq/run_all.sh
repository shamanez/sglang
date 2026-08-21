#!/usr/bin/env bash
# Driver: all four PP wire-quant configs, then restore the production tp4 server.
# Assumes the pp2 bf16 server is ALREADY running for config 1.
set -uo pipefail
cd /workspace/sglang-src/scripts/vast/ppq
# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate
RESULTS=/workspace/ppq/results

echo "=== [1/4] pp2_bf16 (server already up; generates reference trajectories) ==="
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
export WIRE_QUANT_LABEL=none
python3 eval_harness.py --label pp2_bf16 --make-reference 2>&1 | tee /workspace/ppq/logs/eval_pp2_bf16.log || { echo "FAILED pp2_bf16 harness"; exit 1; }
python3 -m sglang.bench_serving --backend sglang --host localhost --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --random-range-ratio 1.0 --num-prompts 48 --max-concurrency 16 \
    --output-file "$RESULTS/pp2_bf16/bench.jsonl" 2>&1 | tail -25 | tee /workspace/ppq/logs/bench_pp2_bf16.log

echo "=== [2/4] pp2_int8 ==="
bash run_config.sh pp2_int8 int8 --tp-size 2 --pp-size 2 --attention-backend flashinfer || echo "FAILED pp2_int8"
echo "=== [3/4] pp4_bf16 ==="
bash run_config.sh pp4_bf16 none --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED pp4_bf16"
echo "=== [4/4] pp4_int8 ==="
bash run_config.sh pp4_int8 int8 --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED pp4_int8"

echo "=== restoring production tp4 server ==="
pkill -f "sglang[.]launch_server" 2>/dev/null; sleep 8
pkill -9 -f "sglang::scheduler[_]" 2>/dev/null; sleep 5
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
nohup /workspace/serve_qwen36.sh > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
echo "ALL_DONE"
