#!/usr/bin/env bash
# PP wire-quant experiment on 8x RTX 5090 (256 GB total VRAM).
#
# Stage 1 extends the depth axis on the already-validated 35B model: pp8/tp1 gives
# 7 boundaries, versus the 1 (pp2) and 3 (pp4) we have measured. Stages 2-3 replicate
# the granularity finding on a 3.5x larger model at two depths.
#
# Usage:  bash run_8gpu.sh [stage1|stage2|stage3|all]
# Env:    PPQ8=/workspace/ppq8 (results root), SKIP_RESTORE=1 to leave no server up.
#
# Reference trajectories: stage 1 MUST reuse the archived 256-step references from
# the 4-GPU campaign (reports/data/reference_trajectories_256.json in the repo) so
# pp8 numbers are comparable to the published pp2/pp4 rows. Copy it to
# $PPQ8/results/reference_trajectories.json before running. Stage 2/3 use a
# different model, so they generate their own references from their bf16 control.
set -uo pipefail
STAGE="${1:-all}"
PPQ8="${PPQ8:-/workspace/ppq8}"
SRC=/workspace/sglang-src/scripts/vast/ppq122
mkdir -p "$PPQ8"/{logs,results}
cd "$SRC"

export PPQ_DIR="$PPQ8"
export PPQ_RESULTS_DIR="$PPQ8/results"

stop_server() {
    pkill -f "sglang\.launch_server" 2>/dev/null || true
    sleep 8
    pkill -9 -f "sglang::" 2>/dev/null || true
    sleep 5
}

# run <label> <wire> <pp> <tp> [extra server args...]
run() {
    local label="$1" wire="$2" pp="$3" tp="$4"; shift 4
    echo "=== $label (wire=$wire pp=$pp tp=$tp) === $(date -u +%FT%TZ)"
    STAGES="${STAGES:-wikitext,gsm8k,probe}" bash run_config.sh "$label" "$wire" \
        --tp-size "$tp" --pp-size "$pp" --attention-backend flashinfer "$@" \
        || echo "FAILED $label"
}

# ---------------------------------------------------------------- stage 1
# 35B at 7 boundaries. Model and codecs already validated on 4 GPUs; the only new
# variable is depth. 5 layers per stage means far more KV headroom than pp2 had,
# so the 1024-step probe that OOMed at pp2 should also fit here.
stage1() {
    echo "=== STAGE1_35B_PP8 === $(date -u +%FT%TZ)"
    export MODEL_ID="Qwen/Qwen3.6-35B-A3B"
    export SERVE_SCRIPT=/workspace/serve_qwen36.sh
    unset PPQ_REFERENCE_PATH
    if [ ! -f "$PPQ8/results/reference_trajectories.json" ]; then
        echo "FATAL: copy reference_trajectories_256.json to $PPQ8/results/ first"
        echo "       (pp8 rows must score against the same references as pp2/pp4)"
        return 1
    fi
    for cfg in "pp8_bf16 none" "pp8_int8 int8" "pp8_mxfp8 mxfp8" \
               "pp8_int4 int4" "pp8_mxfp4 mxfp4" "pp8_nvfp4 nvfp4"; do
        set -- $cfg
        run "$1" "$2" 8 1
    done
    # Long horizon at pp8: the pp2 attempt died in the scoring path with 20 layers
    # on the last stage; 5 layers per stage should clear it.
    echo "=== STAGE1B_LONGPROBE_PP8 === $(date -u +%FT%TZ)"
    export PPQ_RESULTS_DIR="$PPQ8/results_longprobe"
    export PPQ_REFERENCE_PATH="$PPQ8/reference_trajectories_1024.json"
    export PROBE_LEN=1024
    mkdir -p "$PPQ_RESULTS_DIR"
    STAGES=probe MAKE_REFERENCE=1 SKIP_BENCH=1 run lp_pp8_bf16 none 8 1
    for cfg in "lp_pp8_int8 int8" "lp_pp8_int4 int4" "lp_pp8_mxfp4 mxfp4" "lp_pp8_nvfp4 nvfp4"; do
        set -- $cfg
        STAGES=probe SKIP_BENCH=1 run "$1" "$2" 8 1
    done
    unset PROBE_LEN PPQ_REFERENCE_PATH
    export PPQ_RESULTS_DIR="$PPQ8/results"
}

# ---------------------------------------------------------------- stage 2
# 122B-FP8 at 7 boundaries. ~118 GiB of weights over 8 GPUs is ~15 GiB/GPU, so this
# is comfortable where the 4-GPU attempt missed by 18 MB. hidden_size 3072 means
# 12,288 B/token/boundary, 1.5x the 35B wire payload.
stage2() {
    echo "=== STAGE2_122B_PP8 === $(date -u +%FT%TZ)"
    export MODEL_ID="Qwen/Qwen3.5-122B-A10B-FP8"
    export SERVE_SCRIPT="$SRC/serve_qwen35_122b.sh"
    export PPQ_RESULTS_DIR="$PPQ8/results_122b_pp8"
    export PPQ_REFERENCE_PATH="$PPQ8/results_122b_pp8/reference_trajectories.json"
    mkdir -p "$PPQ_RESULTS_DIR"
    MAKE_REFERENCE=1 run q122_pp8_bf16 none 8 1
    for cfg in "q122_pp8_int8 int8" "q122_pp8_int4 int4" \
               "q122_pp8_mxfp4 mxfp4" "q122_pp8_nvfp4 nvfp4"; do
        set -- $cfg
        run "$1" "$2" 8 1
    done
}

# ---------------------------------------------------------------- stage 3
# Same model at 3 boundaries (pp4/tp2 keeps all 8 GPUs busy), giving a within-model
# depth comparison against stage 2 rather than a cross-model one.
stage3() {
    echo "=== STAGE3_122B_PP4 === $(date -u +%FT%TZ)"
    export MODEL_ID="Qwen/Qwen3.5-122B-A10B-FP8"
    export SERVE_SCRIPT="$SRC/serve_qwen35_122b.sh"
    export PPQ_RESULTS_DIR="$PPQ8/results_122b_pp4"
    export PPQ_REFERENCE_PATH="$PPQ8/results_122b_pp8/reference_trajectories.json"
    mkdir -p "$PPQ_RESULTS_DIR"
    for cfg in "q122_pp4_bf16 none" "q122_pp4_int8 int8" "q122_pp4_int4 int4" \
               "q122_pp4_mxfp4 mxfp4" "q122_pp4_nvfp4 nvfp4"; do
        set -- $cfg
        run "$1" "$2" 4 2
    done
}

case "$STAGE" in
    stage1) stage1 ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    all)    stage1; stage2; stage3 ;;
    *)      echo "usage: run_8gpu.sh [stage1|stage2|stage3|all]"; exit 2 ;;
esac

stop_server
[ "${SKIP_RESTORE:-0}" = "1" ] || {
    unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
    nohup /workspace/serve_qwen36.sh > "$PPQ8/logs/serve_restore.log" 2>&1 < /dev/null &
    disown
}
echo "RUN_8GPU_DONE $(date -u +%FT%TZ)"
