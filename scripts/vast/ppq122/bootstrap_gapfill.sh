#!/usr/bin/env bash
# =============================================================================
# bootstrap_gapfill.sh — provision a fresh 8x RTX 5090 Vast box for the
# gap-fill campaign (the cells the 4- and 8-GPU campaigns left empty).
#
# Run this ON THE BOX, as root, from a clone of the research branch:
#
#   git clone -b research/pp-activation-int8 \
#       https://github.com/shamanez/sglang.git /workspace/sglang-src
#   bash /workspace/sglang-src/scripts/vast/ppq122/bootstrap_gapfill.sh
#
# Prerequisite you must satisfy by hand first (it carries secrets, so it is not
# in git): copy your /workspace/.env to the box. From your laptop:
#
#   scp -i ~/.ssh/vast_ai -P <port> ~/Documents/sglang/.env root@<host>:/workspace/.env
#
# It needs HF_TOKEN (gated Qwen weights). Everything else in it is unused here.
#
# Steps: preflight -> editable sglang install -> serve script -> model download
#        -> stage the FROZEN reference trajectories -> print the run command.
#
# Knobs: PPQ9 (results root, default /workspace/ppq9), REPO_DIR, SKIP_INSTALL=1,
#        SKIP_DOWNLOAD=1.
# =============================================================================
set -euo pipefail

PPQ9="${PPQ9:-/workspace/ppq9}"
REPO_DIR="${REPO_DIR:-/workspace/sglang-src}"
VENV_DIR="${VENV_DIR:-/workspace/venv-sgl}"
# The ref whose python/ tree produced the archived pp8 results. What matters is
# not that HEAD equals it, but that python/ has not moved since — that is what
# makes a re-measured bf16 control bit-comparable to the published anchors.
RUNTIME_REF="${RUNTIME_REF:-a62276c35}"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33mwarn\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mFATAL\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "preflight"
command -v nvidia-smi >/dev/null || die "no nvidia-smi"
NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/   /'
[ "$NGPU" -ge 8 ] || die "need 8 GPUs for pp8/tp1, found $NGPU"
ok "$NGPU GPUs"

[ -f /workspace/.env ] || die "/workspace/.env missing — scp it over (see header); HF_TOKEN is needed for the gated Qwen weights"
# shellcheck disable=SC1091
set -a; . /workspace/.env; set +a
[ -n "${HF_TOKEN:-}" ] || die "/workspace/.env has no HF_TOKEN"
ok "secrets loaded (HF_TOKEN present, value masked)"

FREE_GB="$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc '0-9')"
[ -n "$FREE_GB" ] || die "cannot read free space on /workspace — does it exist?"
[ "$FREE_GB" -ge 150 ] || die "only ${FREE_GB} GiB free on /workspace; the 35B checkout + weights + venv need ~150 GiB"
ok "${FREE_GB} GiB free"

[ -d "$REPO_DIR/.git" ] || die "$REPO_DIR is not a git checkout — clone the branch first (see header)"
HEAD_REF="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
if ! git -C "$REPO_DIR" cat-file -e "${RUNTIME_REF}^{commit}" 2>/dev/null; then
    warn "cannot resolve $RUNTIME_REF (shallow clone?); skipping the runtime-drift check"
elif git -C "$REPO_DIR" diff --quiet "$RUNTIME_REF" HEAD -- python/; then
    ok "HEAD $HEAD_REF: python/ is byte-identical to $RUNTIME_REF, which produced the archived results"
else
    warn "python/ HAS MOVED since $RUNTIME_REF:"
    git -C "$REPO_DIR" diff --stat "$RUNTIME_REF" HEAD -- python/ | sed 's/^/     /'
    warn "the archived anchors may no longer be reproducible — stage 0 of the run will measure this"
fi

# ---------------------------------------------------------------- install
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    say "install (skipped)"
else
    say "editable sglang install (~20-30 min: torch stack, kernels, flashinfer cache)"
    export REPO_DIR VENV_DIR
    bash "$REPO_DIR/scripts/vast/install_sglang_dev.sh"
    ok "install done"
fi

# ---------------------------------------------------------------- serve script
say "serve script"
install -m 0755 "$REPO_DIR/scripts/vast/serve_qwen36.sh" /workspace/serve_qwen36.sh
ok "/workspace/serve_qwen36.sh (its hardcoded --tp-size 4 / --mem-fraction 0.85 are overridden by the driver's later args; argparse takes the last occurrence)"

# ---------------------------------------------------------------- weights
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    say "model download (skipped)"
else
    say "model download: Qwen/Qwen3.6-35B-A3B (~70 GiB)"
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    if command -v hf >/dev/null; then
        hf download Qwen/Qwen3.6-35B-A3B --cache-dir /workspace/models
    else
        huggingface-cli download Qwen/Qwen3.6-35B-A3B --cache-dir /workspace/models
    fi
    ok "weights in /workspace/models"
fi

# ---------------------------------------------------------------- references
# THE critical step. Every new cell is only comparable to the published rows if
# it scores against the SAME frozen reference trajectories. Regenerating them
# here would silently produce a self-consistent but incomparable set of numbers.
say "staging frozen reference trajectories from the repo archive"
mkdir -p "$PPQ9"/{logs,results,results_longprobe,results_replicates}
SRC_256="$REPO_DIR/reports/data/reference_trajectories_256.json"
SRC_1024="$REPO_DIR/reports/data/reference_trajectories_1024_pp8.json"
[ -f "$SRC_256" ]  || die "missing $SRC_256"
[ -f "$SRC_1024" ] || die "missing $SRC_1024"
cp "$SRC_256"  "$PPQ9/results/reference_trajectories.json"
cp "$SRC_1024" "$PPQ9/reference_trajectories_1024.json"
ok "256-step refs  -> $PPQ9/results/reference_trajectories.json  ($(md5sum < "$SRC_256"  | cut -c1-12))"
ok "1024-step refs -> $PPQ9/reference_trajectories_1024.json     ($(md5sum < "$SRC_1024" | cut -c1-12))"

# ---------------------------------------------------------------- done
say "ready"
cat <<EOF
   Results root: $PPQ9

   Launch the queue detached (setsid, or it dies with the ssh session — this
   cost several reruns last campaign):

     setsid nohup bash $REPO_DIR/scripts/vast/ppq122/run_gapfill.sh all \\
         > $PPQ9/logs/queue.log 2>&1 < /dev/null &

   Watch:   tail -f $PPQ9/logs/queue.log
   Done at: GAPFILL_ALL_DONE
EOF
