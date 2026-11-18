#!/bin/bash
# Git-driven job queue runner, on a Leonardo LOGIN node inside tmux (see ops/README.md):
#
#   tmux new -s sca_runner
#   cd /leonardo_work/AIFAC_S07_041/sca && bash scripts/ops_runner.sh     # Ctrl-b d to detach
#
# Every INTERVAL seconds: rebase the checkout onto origin/$BRANCH (never merge), then one
# queue tick (scripts/ops_queue.py): submit new requests from ops/queue/, poll jobs, run post
# steps, push results and experiments/results/ops/status.json through scripts/ops_collect.sh.
# State lives outside the repo ($SCA_OPS_BASE/ops_state/state.json), so a restart never
# re-submits anything.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
BRANCH=${OPS_BRANCH:-claude/implementation-plan-review-csn6m0}
INTERVAL=${INTERVAL:-300}

# shellcheck disable=SC1090
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}" || { echo "cannot source sca_env.rc"; exit 1; }
: "${WORK:?sca_env.rc did not set WORK}"
export SCA_OPS_BASE=${SCA_OPS_BASE:-$WORK/sca_ops}
export OPS_LOGS=${OPS_LOGS:-$REPO/slurm_scripts/logs}
export OPS_BRANCH=$BRANCH
STATE=${OPS_STATE:-$SCA_OPS_BASE/ops_state/state.json}
LOG=$OPS_LOGS/ops_runner.log
mkdir -p "$(dirname "$STATE")" "$OPS_LOGS"

# sbatch hands this shell's environment to every job, so the runner must run in the sca env
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "a virtualenv is active ($VIRTUAL_ENV): run 'deactivate', activate the sca env, start again"; exit 1
fi
if ! python3 -c "import torch, decord, easydict" 2>/dev/null; then
    echo "python3 ($(command -v python3)) cannot import torch/decord/easydict: activate the sca env first"; exit 1
fi
echo "runner: python3=$(command -v python3) CONDA_PREFIX=${CONDA_PREFIX:-none}"

cur=$(git rev-parse --abbrev-ref HEAD)
if [ "$cur" != "$BRANCH" ]; then
    echo "checkout is on '$cur', not '$BRANCH' - see ops/README.md (first-time setup)"; exit 1
fi
echo "runner: repo=$REPO branch=$BRANCH every ${INTERVAL}s state=$STATE log=$LOG"

while true; do
    # harvest scripts rewrite tracked files under experiments/results; commit them locally
    # first so the rebase can run (the next collector call pushes them)
    if [ -n "$(git status --porcelain --untracked-files=no -- experiments/results)" ]; then
        bash scripts/ops_collect.sh --commit-only "local results (runner)" >> "$LOG" 2>&1
    fi
    if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD; then
        python3 scripts/ops_queue.py tick --repo "$REPO" --state "$STATE" >> "$LOG" 2>&1 \
            || echo "$(date -Is) tick exited with an error" >> "$LOG"
    else
        git rebase --abort 2>/dev/null
        echo "$(date -Is) git update failed" >> "$LOG"
    fi
    tail -n 3 "$LOG"
    sleep "$INTERVAL"
done
