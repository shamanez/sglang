#!/usr/bin/env bash
set -uo pipefail
cd /workspace/sglang-src/scripts/vast/ppq
for cfg in "pp2_fp8 fp8 2 2" "pp2_mxfp8 mxfp8 2 2" "pp2_mxfp4 mxfp4 2 2" "pp2_nvfp4 nvfp4 2 2" \
           "pp4_fp8 fp8 4 1" "pp4_mxfp8 mxfp8 4 1" "pp4_mxfp4 mxfp4 4 1" "pp4_nvfp4 nvfp4 4 1"; do
    set -- $cfg
    echo "=== $1 ==="
    bash run_config.sh "$1" "$2" --tp-size "$4" --pp-size "$3" --attention-backend flashinfer || echo "FAILED $1"
done
echo "=== restoring production tp4 server ==="
pkill -f "sglang\.launch_server" 2>/dev/null; sleep 8
pkill -9 -f "sglang::" 2>/dev/null; sleep 5
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
nohup /workspace/serve_qwen36.sh > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
echo "MX_ALL_DONE"
