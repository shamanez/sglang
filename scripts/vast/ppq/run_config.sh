#!/usr/bin/env bash
# Run one config of the PP wire-quant experiment end to end:
#   start server -> wait healthy -> eval harness -> bench_serving -> stop server.
# Usage: run_config.sh <label> <wire_quant:none|int8|fp8> [extra server args...]
set -euo pipefail
LABEL="$1"; WIRE="$2"; shift 2
LOG=/workspace/ppq/logs/serve_${LABEL}.log
RESULTS=/workspace/ppq/results
mkdir -p /workspace/ppq/logs "$RESULTS/$LABEL"

# shellcheck disable=SC1091
. /workspace/venv-sgl/bin/activate

stop_server() {
    pkill -f "sglang[.]launch_server" 2>/dev/null || true
    sleep 8
    pkill -9 -f "sglang::scheduler[_]" 2>/dev/null || true
    sleep 5
}
trap stop_server EXIT

stop_server
ENVPREFIX=()
if [ "$WIRE" != "none" ]; then
    export SGLANG_PP_ACTIVATION_WIRE_QUANT="$WIRE"
else
    unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
fi
export WIRE_QUANT_LABEL="$WIRE"

echo "[$LABEL] launching: wire=$WIRE args: $*"
nohup /workspace/serve_qwen36.sh "$@" > "$LOG" 2>&1 < /dev/null &
disown

for i in $(seq 1 150); do
    sleep 6
    if curl -s --max-time 4 localhost:30000/health_generate -o /dev/null -w "%{http_code}" | grep -q 200; then
        echo "[$LABEL] healthy after ~$((i*6))s"; break
    fi
    if ! pgrep -f "sglang[.]launch_server" > /dev/null; then
        echo "[$LABEL] SERVER DIED, log tail:"; tail -30 "$LOG"; exit 1
    fi
    if [ "$i" = 150 ]; then echo "[$LABEL] TIMEOUT"; tail -30 "$LOG"; exit 1; fi
done

curl -s localhost:30000/get_server_info | python3 -c "import json,sys; d=json.load(sys.stdin); print('pp_size', d['pp_size'], 'tp_size', d['tp_size'])"

HARNESS_ARGS=(--label "$LABEL")
if [ "${MAKE_REFERENCE:-0}" = "1" ]; then HARNESS_ARGS+=(--make-reference); fi
python3 /workspace/sglang-src/scripts/vast/ppq/eval_harness.py "${HARNESS_ARGS[@]}" 2>&1 | tee /workspace/ppq/logs/eval_${LABEL}.log

echo "[$LABEL] bench_serving"
python3 -m sglang.bench_serving --backend sglang --host localhost --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --random-range-ratio 1.0 --num-prompts 48 --max-concurrency 16 \
    --output-file "$RESULTS/$LABEL/bench.jsonl" 2>&1 | tail -30 | tee /workspace/ppq/logs/bench_${LABEL}.log

echo "[$LABEL] COMPLETE"
