#!/usr/bin/env bash
# Post-queue extension: variance replicates + pp2 long-horizon probes.
set -uo pipefail
cd /workspace/ppq122
Q=/workspace/ppq122/logs/queue.log
until grep -q "QUEUE_ALL_DONE" "$Q"; do sleep 60; done
sleep 120
echo "=== EXTENSION_START === $(date -u +%FT%TZ)"

# Round 2 replicates of the headline pp4 cells (seed/variance bounds).
export PPQ_DIR=/workspace/ppq122
export PPQ_RESULTS_DIR=/workspace/ppq122/results_rep2
mkdir -p "$PPQ_RESULTS_DIR"
for cfg in "rep2_pp4_bf16 none" "rep2_pp4_int4 int4" "rep2_pp4_mxfp4 mxfp4" "rep2_pp4_nvfp4 nvfp4"; do
    set -- $cfg
    echo "=== $1 ==="
    STAGES=wikitext,gsm8k SKIP_BENCH=1 bash run_config.sh "$1" "$2" \
        --tp-size 1 --pp-size 4 --attention-backend flashinfer || echo "FAILED $1"
done

# pp2 long-horizon probes against the same 1024-token references.
export PPQ_RESULTS_DIR=/workspace/ppq122/results_longprobe
export PPQ_REFERENCE_PATH=/workspace/ppq122/reference_trajectories_1024.json
export PROBE_LEN=1024
for cfg in "lp_pp2_bf16 none" "lp_pp2_int4 int4" "lp_pp2_mxfp4 mxfp4" "lp_pp2_nvfp4 nvfp4"; do
    set -- $cfg
    echo "=== $1 ==="
    STAGES=probe SKIP_BENCH=1 bash run_config.sh "$1" "$2" \
        --tp-size 2 --pp-size 2 --attention-backend flashinfer || echo "FAILED $1"
done

echo "=== EXT_RESTORE_PROD ==="
pkill -f "sglang\.launch_server" 2>/dev/null; sleep 8
pkill -9 -f "sglang::" 2>/dev/null; sleep 5
unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
nohup /workspace/serve_qwen36.sh > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
echo "EXT_ALL_DONE"
