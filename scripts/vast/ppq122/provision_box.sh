#!/usr/bin/env bash
# =============================================================================
# provision_box.sh — take a bare Vast box to "queue running", from your laptop.
#
#   bash scripts/vast/ppq122/provision_box.sh <ssh-port> <host> [ssh-key]
#
# Replaces the manual "scp your .env to the box" step, and narrows it: the box
# is a rented third-party machine, and the only secret the experiment needs
# there is HF_TOKEN for the gated Qwen weights. Copying the whole .env would
# also hand over the Vast API keys, the R2 credentials and the WANDB key, none
# of which the run touches. This ships exactly one variable.
#
# The token is piped over stdin, never passed as an argument, so it does not
# appear in the remote process table or in your shell history.
#
# Secret source, first hit wins:  $SECRETS_SRC
#                                 ~/.config/verl-research/secrets.env
#                                 <repo>/.env
# =============================================================================
set -euo pipefail
PORT="${1:?usage: provision_box.sh <ssh-port> <host> [ssh-key]}"
HOST="${2:?usage: provision_box.sh <ssh-port> <host> [ssh-key]}"
KEY="${3:-$HOME/.ssh/vast_ai}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BRANCH="${BRANCH:-research/pp-activation-int8}"
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST")

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFATAL\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ secret
say "locating HF_TOKEN"
SRC=""
for cand in "${SECRETS_SRC:-}" "$HOME/.config/verl-research/secrets.env" "$REPO/.env"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { SRC="$cand"; break; }
done
[ -n "$SRC" ] || die "no secrets file found; set SECRETS_SRC=/path/to/env"

# Subshell so nothing leaks into this script's environment or its children.
# shellcheck disable=SC1090  # the secrets path is resolved at runtime by design
TOKEN="$(set -a; . "$SRC" >/dev/null 2>&1; set +a; printf '%s' "${HF_TOKEN:-}")"
[ -n "$TOKEN" ] || die "$SRC defines no HF_TOKEN"
printf '   source %s\n   HF_TOKEN present (%d chars, value not shown)\n' "$SRC" "${#TOKEN}"

# ------------------------------------------------------------------ ship it
say "writing /workspace/.env on the box (HF_TOKEN only, via stdin)"
printf 'export HF_TOKEN=%s\nexport HUGGING_FACE_HUB_TOKEN=%s\nexport HUGGINGFACE_HUB_TOKEN=%s\n' \
    "$TOKEN" "$TOKEN" "$TOKEN" \
  | "${SSH[@]}" 'mkdir -p /workspace && cat > /workspace/.env && chmod 600 /workspace/.env && echo "   wrote $(wc -c < /workspace/.env) bytes, mode $(stat -c %a /workspace/.env)"'
unset TOKEN

# ------------------------------------------------------------------ clone
say "cloning $BRANCH"
"${SSH[@]}" "
    set -e
    if [ -d /workspace/sglang-src/.git ]; then
        git -C /workspace/sglang-src fetch --quiet origin '$BRANCH'
        git -C /workspace/sglang-src checkout --quiet '$BRANCH'
        git -C /workspace/sglang-src reset --hard --quiet \"origin/$BRANCH\"
        echo '   updated existing checkout'
    else
        git clone --quiet -b '$BRANCH' https://github.com/shamanez/sglang.git /workspace/sglang-src
        echo '   fresh clone'
    fi
    git -C /workspace/sglang-src log --oneline -1 | sed 's/^/   HEAD /'
"

# ------------------------------------------------------------------ bootstrap
say "running bootstrap on the box (install + weights + refs; ~45 min)"
echo "   streaming below; it is also logged to /workspace/bootstrap.log"
"${SSH[@]}" "bash /workspace/sglang-src/scripts/vast/ppq122/bootstrap_gapfill.sh 2>&1 | tee /workspace/bootstrap.log"

say "next"
cat <<EOF
   Start the queue detached (it must outlive the ssh session):

     ssh -i $KEY -p $PORT root@$HOST \\
       'setsid nohup bash /workspace/sglang-src/scripts/vast/ppq122/run_gapfill.sh all \\
        > /workspace/ppq9/logs/queue.log 2>&1 < /dev/null & echo started'

   Watch:  ssh -i $KEY -p $PORT root@$HOST 'tail -f /workspace/ppq9/logs/queue.log'
   Pull:   bash scripts/vast/ppq122/fetch_gapfill.sh $PORT $HOST
EOF
