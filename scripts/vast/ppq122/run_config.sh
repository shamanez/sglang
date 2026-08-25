#!/usr/bin/env bash
# Run one config of the PP wire-quant experiment end to end:
#   start server -> wait healthy -> eval harness -> bench_serving -> stop server.
# Usage: run_config.sh <label> <wire_quant:none|int8|fp8|int4|mxfp8|mxfp4|nvfp4> [extra server args...]
# Env knobs: STAGES (harness stages), MAKE_REFERENCE=1, SKIP_SERVER=1 (evaluate
# an already-running server instead of launching one; topology is still asserted),
# SERVE_SCRIPT (server launcher, default /workspace/serve_qwen36.sh),
# PPQ_DIR (logs/results root, default /workspace/ppq),
# PPQ_RESULTS_DIR (results dir, default $PPQ_DIR/results),
# SKIP_BENCH=1 (skip bench_serving; e.g. under cpu offload latency is distorted).
set -euo pipefail
LABEL="$1"; WIRE="$2"; shift 2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVE_SCRIPT="${SERVE_SCRIPT:-/workspace/serve_qwen36.sh}"
PPQ_DIR="${PPQ_DIR:-/workspace/ppq}"
LOG=$PPQ_DIR/logs/serve_${LABEL}.log
RESULTS="${PPQ_RESULTS_DIR:-$PPQ_DIR/results}"
export PPQ_RESULTS_DIR="$RESULTS"  # the harness reads this as its --out default
mkdir -p "$PPQ_DIR/logs" "$RESULTS/$LABEL"

# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate

# Expected topology, parsed from the server args (last occurrence wins, like argparse).
EXPECT_PP=1; EXPECT_TP=1; prev=""
for a in "$@"; do
    case "$prev" in
        --pp-size) EXPECT_PP="$a" ;;
        --tp-size) EXPECT_TP="$a" ;;
    esac
    prev="$a"
done

stop_server() {
    pkill -f "sglang\.launch_server" 2>/dev/null || true
    sleep 8
    # Catches scheduler, detokenizer, and tokenizer workers alike.
    pkill -9 -f "sglang::" 2>/dev/null || true
    sleep 5
}

if [ "$WIRE" != "none" ]; then
    export SGLANG_PP_ACTIVATION_WIRE_QUANT="$WIRE"
else
    unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
fi
export WIRE_QUANT_LABEL="$WIRE"

if [ "${SKIP_SERVER:-0}" != "1" ]; then
    trap stop_server EXIT
    stop_server
    echo "[$LABEL] launching: wire=$WIRE serve=$SERVE_SCRIPT args: $*"
    nohup "$SERVE_SCRIPT" "$@" > "$LOG" 2>&1 < /dev/null &
    disown
    HEALTH_ITERS="${HEALTH_ITERS:-150}"  # 6s each; 150 = 15 min
    for i in $(seq 1 "$HEALTH_ITERS"); do
        sleep 6
        if curl -s --max-time 4 localhost:30000/health_generate -o /dev/null -w "%{http_code}" | grep -q 200; then
            echo "[$LABEL] healthy after ~$((i*6))s"; break
        fi
        if ! pgrep -f "sglang\.launch_server" > /dev/null; then
            echo "[$LABEL] SERVER DIED, log tail:"; tail -30 "$LOG"; exit 1
        fi
        if [ "$i" = "$HEALTH_ITERS" ]; then echo "[$LABEL] TIMEOUT"; tail -30 "$LOG"; exit 1; fi
    done
else
    echo "[$LABEL] SKIP_SERVER=1: evaluating the already-running server"
fi

curl -s localhost:30000/get_server_info | python3 -c "import json,sys; d=json.load(sys.stdin); print('pp_size', d['pp_size'], 'tp_size', d['tp_size'])"

HARNESS_ARGS=(--label "$LABEL" --stages "${STAGES:-wikitext,gsm8k,probe}" --expect-pp "$EXPECT_PP" --expect-tp "$EXPECT_TP")
if [ "${MAKE_REFERENCE:-0}" = "1" ]; then HARNESS_ARGS+=(--make-reference); fi
python3 "$SCRIPT_DIR/eval_harness.py" "${HARNESS_ARGS[@]}" 2>&1 | tee "$PPQ_DIR/logs/eval_${LABEL}.log"

if [ "${SKIP_BENCH:-0}" != "1" ]; then
    echo "[$LABEL] bench_serving"
    python3 -m sglang.bench_serving --backend sglang --host localhost --port 30000 \
        --dataset-name random --random-input-len 1024 --random-output-len 256 \
        --random-range-ratio 1.0 --num-prompts 48 --max-concurrency 16 \
        --output-file "$RESULTS/$LABEL/bench.jsonl" 2>&1 | tail -30 | tee "$PPQ_DIR/logs/bench_${LABEL}.log"
else
    echo "[$LABEL] SKIP_BENCH=1: skipping bench_serving"
fi

echo "[$LABEL] COMPLETE"
