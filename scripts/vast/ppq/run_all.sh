#!/usr/bin/env bash
# Driver: all PP wire-quant configs through the single run_config.sh path,
# then restore the production tp4 server. Every config launches its own
# server; the harness asserts pp/tp against each label before writing results.
set -uo pipefail
cd /workspace/sglang-src/scripts/vast/ppq

i=0
run() {
    i=$((i + 1))
    echo "=== [$i/14] $1 ==="
    shift 0
    local label="$1" wire="$2" pp="$3" tp="$4"
    local extra=()
    if [ "$label" = "pp2_bf16" ]; then extra=(MAKE_REFERENCE=1); fi
    env "${extra[@]}" bash run_config.sh "$label" "$wire" \
        --tp-size "$tp" --pp-size "$pp" --attention-backend flashinfer \
        || echo "FAILED $label"
}

# pp2 (tp2), then pp4 (tp1); bf16 first so it can regenerate references.
run pp2_bf16 none 2 2
run pp2_int8 int8 2 2
run pp2_fp8 fp8 2 2
run pp2_mxfp8 mxfp8 2 2
run pp2_int4 int4 2 2
run pp2_mxfp4 mxfp4 2 2
run pp2_nvfp4 nvfp4 2 2
run pp4_bf16 none 4 1
run pp4_int8 int8 4 1
run pp4_fp8 fp8 4 1
run pp4_mxfp8 mxfp8 4 1
run pp4_int4 int4 4 1
run pp4_mxfp4 mxfp4 4 1
run pp4_nvfp4 nvfp4 4 1

echo "=== restoring production tp4 server ==="
pkill -f "sglang\.launch_server" 2>/dev/null; sleep 8
pkill -9 -f "sglang::" 2>/dev/null; sleep 5
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
nohup /workspace/serve_qwen36.sh > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
echo "ALL_DONE"
