#!/usr/bin/env bash
# Overnight queue for the PP wire-quant experiment:
#   gate on the running mx batch -> owed 35B pp2/mxfp4 re-bench ->
#   Qwen3.5-122B-A10B-FP8 pp4/tp1 fit test + wire series
#   (or 35B long-horizon probe fallback) -> restore production server.
# Deliberately NO set -e/-u: one failed step must never abort the chain;
# every step is || FAILED-guarded. All output goes to
# /workspace/ppq122/logs/queue.log via the launcher redirection.

PPQ122=/workspace/ppq122
MXLOG=/workspace/ppq/logs/run_mx.log
DL_LOG=$PPQ122/logs/download.log
Q35=/workspace/serve_qwen36.sh
Q122=$PPQ122/serve_qwen35_122b.sh
RUNCFG=$PPQ122/run_config.sh
MODEL122=Qwen/Qwen3.5-122B-A10B-FP8

set -a; . /workspace/.env 2>/dev/null; set +a
# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate

stop_server() {
    # Plain patterns are safe inside a script file (no self-match).
    pkill -f "sglang\.launch_server" 2>/dev/null || true
    sleep 8
    pkill -9 -f "sglang::" 2>/dev/null || true
    sleep 5
}

# wait_healthy <max_iters> <server_log>: 6s per iter; 0=healthy, 1=dead/timeout.
wait_healthy() {
    local iters="$1" log="$2" i
    for i in $(seq 1 "$iters"); do
        sleep 6
        if curl -s --max-time 4 localhost:30000/health_generate -o /dev/null \
                -w "%{http_code}" | grep -q 200; then
            echo "healthy after ~$((i * 6))s"
            return 0
        fi
        if ! pgrep -f "sglang\.launch_server" > /dev/null; then
            echo "SERVER DIED, log tail:"
            tail -40 "$log"
            return 1
        fi
    done
    echo "HEALTH TIMEOUT after ~$((iters * 6))s, log tail:"
    tail -40 "$log"
    return 1
}

restart_download() {
    echo "download process gone without completion marker; restarting (resume)"
    setsid nohup bash -c "set -a; . /workspace/.env; set +a; \
        . /workspace/venv-sgl/bin/activate; \
        hf download $MODEL122 --cache-dir /workspace/models \
        && echo DOWNLOAD_COMPLETE_MARKER" >> "$DL_LOG" 2>&1 < /dev/null &
    disown
}

echo "=== GATE_MX_ALL_DONE === $(date -u +%FT%TZ)"
until grep -q MX_ALL_DONE "$MXLOG" 2>/dev/null; do sleep 60; done
echo "MX_ALL_DONE seen at $(date -u +%FT%TZ); sleeping 120s"
sleep 120

# ---------------------------------------------------------------- STEP 1
# Owed re-bench: 35B pp2/tp2 mxfp4 (previous bench measured with a codec bug).
echo "=== STEP1_REBENCH === $(date -u +%FT%TZ)"
step1() {
    stop_server
    SGLANG_PP_ACTIVATION_WIRE_QUANT=mxfp4 nohup "$Q35" --tp-size 2 --pp-size 2 \
        --attention-backend flashinfer \
        > "$PPQ122/logs/serve_step1_pp2_mxfp4.log" 2>&1 < /dev/null &
    disown
    wait_healthy 150 "$PPQ122/logs/serve_step1_pp2_mxfp4.log" || return 1
    rm -f /workspace/ppq/results/pp2_mxfp4/bench.jsonl
    python3 -m sglang.bench_serving --backend sglang --host localhost --port 30000 \
        --dataset-name random --random-input-len 1024 --random-output-len 256 \
        --random-range-ratio 1.0 --num-prompts 48 --max-concurrency 16 \
        --output-file /workspace/ppq/results/pp2_mxfp4/bench.jsonl 2>&1 | tail -25
    stop_server
}
step1 || { echo "FAILED STEP1_REBENCH"; stop_server; }

# ---------------------------------------------------------------- STEP 2
# Wait for the 122B FP8 download to complete (cap 3h), restarting it if it died.
echo "=== STEP2_WAIT_DOWNLOAD === $(date -u +%FT%TZ)"
DL_OK=0
DL_RESTARTS=0
for i in $(seq 1 180); do
    if grep -q DOWNLOAD_COMPLETE_MARKER "$DL_LOG" 2>/dev/null; then
        DL_OK=1
        break
    fi
    if ! pgrep -f "bin/hf download" > /dev/null && [ "$DL_RESTARTS" -lt 3 ]; then
        DL_RESTARTS=$((DL_RESTARTS + 1))
        restart_download
    fi
    sleep 60
done
echo "DOWNLOAD_OK=$DL_OK restarts=$DL_RESTARTS ($(date -u +%FT%TZ))"
[ "$DL_OK" = 1 ] || echo "FAILED STEP2_DOWNLOAD (3h timeout); will fall back to 35B long probe"

# ---------------------------------------------------------------- STEP 3
# 122B fit test at pp4/tp1, no wire quant. Escalating levers.
FIT_LEVER=SKIPPED
FIT_MF=0.94
FIT_EXTRA=""
if [ "$DL_OK" = 1 ]; then
    echo "=== Q122_FIT_TEST === $(date -u +%FT%TZ)"
    try_fit() { # $1=name $2=mem_fraction, rest = extra server args
        local name="$1" mf="$2"
        shift 2
        local log=$PPQ122/logs/serve_fit_${name}.log
        echo "--- fit attempt: $name (MEM_FRACTION=$mf extra: $*) $(date -u +%FT%TZ)"
        stop_server
        MEM_FRACTION=$mf nohup "$Q122" --tp-size 1 --pp-size 4 \
            --attention-backend flashinfer "$@" > "$log" 2>&1 < /dev/null &
        disown
        wait_healthy 300 "$log"
    }
    if try_fit default 0.94; then
        FIT_LEVER=none; FIT_MF=0.94; FIT_EXTRA=""
    elif try_fit memfrac96_nocg 0.96 --disable-cuda-graph; then
        FIT_LEVER="memfrac0.96+disable-cuda-graph"
        FIT_MF=0.96; FIT_EXTRA="--disable-cuda-graph"
    elif try_fit offload 0.96 --disable-cuda-graph --cpu-offload-gb 4; then
        # --cpu-offload-gb is per rank: 4 GB x 4 pp ranks = ~16 GB total.
        # Quality metrics stay valid under offload; only latency is distorted.
        FIT_LEVER="cpu-offload-gb4+memfrac0.96+disable-cuda-graph"
        FIT_MF=0.96; FIT_EXTRA="--disable-cuda-graph --cpu-offload-gb 4"
    else
        FIT_LEVER=NONE_FAILED
    fi
    stop_server
fi
echo "FIT_VERDICT: $FIT_LEVER (DL_OK=$DL_OK)"
echo "FIT_VERDICT: $FIT_LEVER (DL_OK=$DL_OK) $(date -u +%FT%TZ)" \
    > "$PPQ122/results/fit_verdict.txt" || echo "FAILED FIT_VERDICT_WRITE"

# ---------------------------------------------------------------- STEP 4
if [ "$DL_OK" = 1 ] && [ "$FIT_LEVER" != NONE_FAILED ] && [ "$FIT_LEVER" != SKIPPED ]; then
    # -------- 4a: 122B wire series at pp4/tp1 (wikitext + gsm8k; probe skipped:
    # reference trajectories are model-specific and would eat the night).
    N_GSM=100
    SKIPB=0
    case "$FIT_LEVER" in
        *offload*) N_GSM=50; SKIPB=1 ;;  # bench meaningless + gsm8k slow under offload
    esac
    for pair in q122_pp4_bf16wire:none q122_pp4_int8:int8 q122_pp4_int4:int4 \
                q122_pp4_mxfp4:mxfp4 q122_pp4_nvfp4:nvfp4; do
        label=${pair%%:*}
        wire=${pair##*:}
        echo "=== $label === $(date -u +%FT%TZ)"
        MEM_FRACTION=$FIT_MF PPQ_DIR=$PPQ122 SERVE_SCRIPT=$Q122 \
            MODEL_ID=$MODEL122 PPQ_RESULTS_DIR=$PPQ122/results \
            STAGES=wikitext,gsm8k N_GSM8K=$N_GSM SKIP_BENCH=$SKIPB \
            HEALTH_ITERS=300 \
            bash "$RUNCFG" "$label" "$wire" --tp-size 1 --pp-size 4 \
            --attention-backend flashinfer $FIT_EXTRA \
            || echo "FAILED $label"
    done
else
    # -------- 4b fallback: 35B long-horizon probe (PROBE_LEN=1024) at pp4/tp1.
    echo "=== FALLBACK_LONGPROBE_35B === $(date -u +%FT%TZ)"
    mkdir -p "$PPQ122/results_longprobe"
    for pair in pp4_bf16:none pp4_int8:int8 pp4_int4:int4 \
                pp4_mxfp4:mxfp4 pp4_nvfp4:nvfp4; do
        label=${pair%%:*}
        wire=${pair##*:}
        MR=0
        [ "$label" = pp4_bf16 ] && MR=1  # baseline generates the 1024-token refs
        echo "=== longprobe_$label === $(date -u +%FT%TZ)"
        PPQ_DIR=$PPQ122 PPQ_RESULTS_DIR=$PPQ122/results_longprobe \
            PPQ_REFERENCE_PATH=$PPQ122/reference_trajectories_1024.json \
            PROBE_LEN=1024 STAGES=probe MAKE_REFERENCE=$MR SKIP_BENCH=1 \
            bash "$RUNCFG" "$label" "$wire" --tp-size 1 --pp-size 4 \
            --attention-backend flashinfer \
            || echo "FAILED longprobe_$label"
    done
fi

# ---------------------------------------------------------------- FINAL
# Always restore the production 35B tp4 server, wire quant off.
echo "=== RESTORE_PROD === $(date -u +%FT%TZ)"
stop_server
unset SGLANG_PP_ACTIVATION_WIRE_QUANT
nohup "$Q35" > /workspace/ppq/logs/serve_restore_tp4.log 2>&1 < /dev/null &
disown
wait_healthy 150 /workspace/ppq/logs/serve_restore_tp4.log || echo "FAILED RESTORE_PROD"
echo "QUEUE_ALL_DONE $(date -u +%FT%TZ)"
