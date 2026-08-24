#!/usr/bin/env bash
# =============================================================================
# run_gapfill.sh — the cells the 4- and 8-GPU campaigns left empty.
#
# Three holes in the published 35B matrix, all involving the per-token fp8 and
# microscaled mxfp8 wires at depth:
#   1. fp8 at 7 boundaries (pp8/tp1) — the only empty cell in the main
#      1/3/7-boundary quality matrix. HIGHEST PRIORITY.
#   2. fp8 and mxfp8 in the 1024-step decode probe at 7 boundaries.
#   3. fp8, mxfp8 and int8 in the round-2 replication at 3 boundaries.
#
# Usage:  bash run_gapfill.sh [stage0|stage1|stage2|stage3|all]
# Env:    PPQ9=/workspace/ppq9 (results root)
#         RESTORE_PROD=1  leave a tp4 production server up at the end
#         SKIP_CONTROL=1  skip stage 0 (not recommended, see below)
#
# Run bootstrap_gapfill.sh first: it stages the FROZEN reference trajectories,
# without which none of these numbers are comparable to the published rows.
# =============================================================================
set -uo pipefail
STAGE="${1:-all}"
PPQ9="${PPQ9:-/workspace/ppq9}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/workspace/venv-sgl/bin/python3}"
mkdir -p "$PPQ9"/{logs,results,results_longprobe,results_replicates}
cd "$SRC" || exit 1

export PPQ_DIR="$PPQ9"
export MODEL_ID="Qwen/Qwen3.6-35B-A3B"
export SERVE_SCRIPT=/workspace/serve_qwen36.sh
# run_config.sh polls health 150 times at 6s = 15 min. On a fresh box the first
# launch reads 70 GiB of cold weights across 8 ranks; 25 min is the ceiling, not
# the expected time, and a spurious timeout would cost a whole stage.
export HEALTH_ITERS="${HEALTH_ITERS:-250}"

# The published anchors these runs must be read against. bf16 teacher-forced NLL
# is bit-deterministic on this stack: pp4 and pp8, four separate servers, all
# returned this same float to the last digit. So it doubles as a box check.
ANCHOR_BF16_NLL=1.8932072605675765
ANCHOR_BF16_ACC=0.96

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

# ---------------------------------------------------------------- stage 0
# Box reproducibility control. These are new runs on a NEW physical box, but the
# deltas they feed are computed against anchors measured on a box that no longer
# exists. Re-measuring bf16 is 20 minutes and turns that assumption into a fact:
# if the NLL comes back bit-identical, every new cell drops straight into the
# published tables; if it drifts, the new cells must be read against THIS
# control instead, and that has to be stated in the report rather than
# discovered later. A drift does not invalidate the runs, so it warns, records,
# and continues.
stage0() {
    [ "${SKIP_CONTROL:-0}" = "1" ] && { echo "=== STAGE0 SKIPPED ==="; return 0; }
    echo "=== STAGE0_CONTROL_PP8_BF16 === $(date -u +%FT%TZ)"
    export PPQ_RESULTS_DIR="$PPQ9/results"
    STAGES=wikitext,gsm8k run ctl_pp8_bf16 none 8 1
    "$PY" - "$PPQ9/results/ctl_pp8_bf16" "$ANCHOR_BF16_NLL" "$ANCHOR_BF16_ACC" <<'PY'
import json, sys
from pathlib import Path
d, anchor_nll, anchor_acc = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
try:
    nll = json.loads((d / "wikitext.json").read_text())["mean_nll"]
    acc = json.loads((d / "gsm8k.json").read_text())["accuracy"]
except (OSError, KeyError) as e:
    print(f"CONTROL_VERDICT: UNAVAILABLE ({e})"); sys.exit(0)
print(f"  control NLL {nll!r}  vs anchor {anchor_nll!r}")
print(f"  control GSM8K {acc:.2f}  vs anchor {anchor_acc:.2f}")
if nll == anchor_nll:
    verdict = "BIT_IDENTICAL - new cells are directly comparable to the published rows"
else:
    verdict = (f"DRIFT {(nll/anchor_nll - 1)*100:+.4f}% - read every new cell against "
               f"THIS control ({nll!r}), not the archived anchor, and say so in the report")
print(f"CONTROL_VERDICT: {verdict}")
(d / "control_verdict.json").write_text(json.dumps(
    {"nll": nll, "anchor_nll": anchor_nll, "bit_identical": nll == anchor_nll,
     "acc": acc, "anchor_acc": anchor_acc, "verdict": verdict}))
PY
}

# ---------------------------------------------------------------- stage 1
# THE priority run: fp8 wire at 7 boundaries. Every other 8-bit format has all
# three depths; fp8 has 1 and 3. Its 4-GPU rows were quality-free (-0.005% at 1
# boundary, +0.06% at 3), so the expected result is flat - which is the point:
# it either completes the "no 8-bit format cares about depth" claim or breaks it.
# Full stage set, including the 256-step probe and bench, so the row matches the
# other pp8 rows field for field.
stage1() {
    echo "=== STAGE1_PP8_FP8 === $(date -u +%FT%TZ)"
    export PPQ_RESULTS_DIR="$PPQ9/results"
    unset PPQ_REFERENCE_PATH
    [ -f "$PPQ9/results/reference_trajectories.json" ] || {
        echo "FATAL: 256-step references not staged; run bootstrap_gapfill.sh"; return 1; }
    # Spelled out rather than left to run()'s default: a `VAR=x somefunc` prefix
    # persists after the call when bash is in POSIX mode, and stage 0 sets
    # STAGES=wikitext,gsm8k just before this. Leaking it would silently drop the
    # probe from the priority run and we would not notice until the report.
    unset SKIP_BENCH || true
    STAGES=wikitext,gsm8k,probe run pp8_fp8 fp8 8 1
}

# ---------------------------------------------------------------- stage 2
# Long-horizon probe at 7 boundaries for the two 8-bit formats it never covered.
# The published lp2_pp8_* set is bf16/int8/int4/mxfp4/nvfp4. mem-fraction 0.75 +
# no cuda graph is not a codec workaround: scoring a ~1300-token sequence
# materializes a [T, 248k-vocab] fp32 logprob transient outside the static pool,
# and the first attempt at 0.94 OOMed in _row_logsumexp_topk_kernel.
stage2() {
    echo "=== STAGE2_LONGPROBE_PP8 === $(date -u +%FT%TZ)"
    export PPQ_RESULTS_DIR="$PPQ9/results_longprobe"
    export PPQ_REFERENCE_PATH="$PPQ9/reference_trajectories_1024.json"
    export PROBE_LEN=1024
    [ -f "$PPQ_REFERENCE_PATH" ] || {
        echo "FATAL: 1024-step references not staged; run bootstrap_gapfill.sh"; return 1; }
    local LP_ARGS=(--mem-fraction-static 0.75 --disable-cuda-graph)
    for cfg in "lp2_pp8_fp8 fp8" "lp2_pp8_mxfp8 mxfp8"; do
        set -- $cfg
        STAGES=probe SKIP_BENCH=1 run "$1" "$2" 8 1 "${LP_ARGS[@]}"
    done
    unset PROBE_LEN PPQ_REFERENCE_PATH
}

# ---------------------------------------------------------------- stage 3
# Completes round 2 of the independent replication at 3 boundaries. Round 2
# covered bf16/int4/mxfp4/nvfp4 - i.e. every format whose result was dramatic,
# and none of the three whose result was "no effect". A null result that has
# never been reproduced is exactly the kind that quietly turns out to be a
# harness artifact, so these are the replicates that matter most.
# pp4/tp1 leaves 4 of the 8 GPUs idle; running two configs side by side would
# need per-config ports and device masks in run_config.sh, which is not worth
# the risk of mislabeling. Quality only, no bench.
stage3() {
    echo "=== STAGE3_REPLICATES_PP4 === $(date -u +%FT%TZ)"
    export PPQ_RESULTS_DIR="$PPQ9/results_replicates"
    unset PPQ_REFERENCE_PATH
    for cfg in "rep2_pp4_int8 int8" "rep2_pp4_fp8 fp8" "rep2_pp4_mxfp8 mxfp8"; do
        set -- $cfg
        STAGES=wikitext,gsm8k SKIP_BENCH=1 run "$1" "$2" 4 1
    done
}

case "$STAGE" in
    stage0) stage0 ;;
    stage1) stage1 ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    all)    stage0; stage1; stage2; stage3 ;;
    *)      echo "usage: run_gapfill.sh [stage0|stage1|stage2|stage3|all]"; exit 2 ;;
esac

stop_server
if [ "${RESTORE_PROD:-0}" = "1" ]; then
    unset SGLANG_PP_ACTIVATION_WIRE_QUANT || true
    setsid nohup /workspace/serve_qwen36.sh > "$PPQ9/logs/serve_restore.log" 2>&1 < /dev/null &
    disown
fi

echo "=== SUMMARY ==="
"$PY" - "$PPQ9" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
want = [("results/ctl_pp8_bf16", "wikitext.json"), ("results/pp8_fp8", "wikitext.json"),
        ("results_longprobe/lp2_pp8_fp8", "probe.json"),
        ("results_longprobe/lp2_pp8_mxfp8", "probe.json"),
        ("results_replicates/rep2_pp4_int8", "wikitext.json"),
        ("results_replicates/rep2_pp4_fp8", "wikitext.json"),
        ("results_replicates/rep2_pp4_mxfp8", "wikitext.json")]
for rel, key in want:
    p = root / rel / key
    if not p.exists():
        print(f"  MISSING  {rel}"); continue
    if key == "wikitext.json":
        g = root / rel / "gsm8k.json"
        acc = json.loads(g.read_text())["accuracy"] if g.exists() else None
        nll = json.loads(p.read_text())["mean_nll"]
        print(f"  ok       {rel:38s} nll={nll!r} acc={acc}")
    else:
        n = len(json.loads(p.read_text()))
        print(f"  ok       {rel:38s} probe items={n}")
PY
echo "GAPFILL_ALL_DONE $(date -u +%FT%TZ)"
