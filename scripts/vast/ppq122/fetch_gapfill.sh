#!/usr/bin/env bash
# =============================================================================
# fetch_gapfill.sh — pull the gap-fill results off the box into reports/data/,
# at the exact paths build_8gpu_section.py already reads. Run this LOCALLY,
# from the repo root, before the box is destroyed.
#
#   bash scripts/vast/ppq122/fetch_gapfill.sh <ssh-port> <host> [key]
#   e.g. bash scripts/vast/ppq122/fetch_gapfill.sh 40442 203.75.167.56
#
# Layout note: the 256-step quality dirs keep aggregates in git and their raw
# probe.json in a tarball (that is how the earlier campaigns are archived), while
# the long-probe dirs keep probe.json inline because the report reads it per
# segment. This script reproduces that split rather than inventing a new one.
# =============================================================================
set -euo pipefail
PORT="${1:?usage: fetch_gapfill.sh <ssh-port> <host> [key]}"
HOST="${2:?usage: fetch_gapfill.sh <ssh-port> <host> [key]}"
KEY="${3:-$HOME/.ssh/vast_ai}"
PPQ9="${PPQ9:-/workspace/ppq9}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA="$REPO/reports/data"
RSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "quality cells -> ppq8/results/"
mkdir -p "$DATA/ppq8/results"
rsync -av -e "$RSH" \
    "root@$HOST:$PPQ9/results/pp8_fp8" \
    "root@$HOST:$PPQ9/results/ctl_pp8_bf16" \
    "$DATA/ppq8/results/"

say "long probes -> ppq8/results_longprobe/"
mkdir -p "$DATA/ppq8/results_longprobe"
rsync -av -e "$RSH" \
    "root@$HOST:$PPQ9/results_longprobe/lp2_pp8_fp8" \
    "root@$HOST:$PPQ9/results_longprobe/lp2_pp8_mxfp8" \
    "$DATA/ppq8/results_longprobe/"

# Only present if stage 3 was run. Under the default `html` target it is not, and
# rsync exits 23 on a missing source - which under set -e would abort the fetch
# before the logs and the report rebuild.
say "replicates -> replicates/ (stage 3 only)"
if $RSH "root@$HOST" "[ -d $PPQ9/results_replicates/rep2_pp4_int8 ]" 2>/dev/null; then
    mkdir -p "$DATA/replicates"
    rsync -av -e "$RSH" \
        "root@$HOST:$PPQ9/results_replicates/rep2_pp4_int8" \
        "root@$HOST:$PPQ9/results_replicates/rep2_pp4_fp8" \
        "root@$HOST:$PPQ9/results_replicates/rep2_pp4_mxfp8" \
        "$DATA/replicates/"
else
    echo "   none on the box (stage 3 not run) - skipping"
fi

say "logs"
$RSH "root@$HOST" "tar czf - -C $PPQ9 logs" > "$DATA/ppq9_logs.tar.gz"

say "split raw 256-step probe out of the quality dir (matches the earlier archives)"
RAW="$DATA/ppq9_probe_raw.tar.gz"
if [ -f "$DATA/ppq8/results/pp8_fp8/probe.json" ]; then
    tar czf "$RAW" -C "$DATA/ppq8/results" pp8_fp8/probe.json
    rm "$DATA/ppq8/results/pp8_fp8/probe.json"
    echo "   pp8_fp8/probe.json -> $(basename "$RAW")"
fi

say "regenerate the report section"
python3 "$REPO/scripts/vast/ppq122/build_8gpu_section.py" --data "$DATA"

say "landed"
for d in "$DATA/ppq8/results/pp8_fp8" "$DATA/ppq8/results/ctl_pp8_bf16" \
         "$DATA/ppq8/results_longprobe/lp2_pp8_fp8" "$DATA/ppq8/results_longprobe/lp2_pp8_mxfp8" \
         "$DATA/replicates/rep2_pp4_int8" "$DATA/replicates/rep2_pp4_fp8" \
         "$DATA/replicates/rep2_pp4_mxfp8"; do
    [ -d "$d" ] && find "$d" -type f | sed "s|$DATA/|   |"
done
