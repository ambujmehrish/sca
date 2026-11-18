#!/bin/bash
# Results collector for the job queue (called by scripts/ops_queue.py; usable by hand on a
# login node after sourcing sca_env.rc):
#
#   bash scripts/ops_collect.sh "<message>"
#
# Full collect:
# 1. rsync $SCA_OPS_BASE/results/ -> experiments/results/ (json/txt/md/csv < 20 MB; this
#    carries ops/status.json)
# 2. every Slurm log in $OPS_LOGS changed in the last 7 days -> experiments/results/logs/<name>.txt
# 3. every eval/training log under workdir/ and workdir_pretrain/ changed in the last 7 days
#    (*/log/log*.txt, */run.log) -> experiments/results/cells/<path>.txt
#    A log over 10 MB keeps its first 2,000 + last 20,000 lines, with a marker line.
# 4. git add experiments/results (same filter), commit "Results: <message>" (skipped if
#    nothing changed), then fetch + rebase onto origin/$BRANCH and push.
#
#   bash scripts/ops_collect.sh --commit-only "<message>"
#      step 4 without fetch/rebase/push (the runner calls this before each update).
set -uo pipefail
cd "$(dirname "$0")/.."
COMMIT_ONLY=""
if [ "${1:-}" = --commit-only ]; then COMMIT_ONLY=1; shift; fi
MSG=${1:?usage: ops_collect.sh [--commit-only] "<message>"}
BRANCH=${OPS_BRANCH:-claude/implementation-plan-review-csn6m0}
: "${SCA_OPS_BASE:?run through scripts/ops_runner.sh or export SCA_OPS_BASE}"
LOGS=${OPS_LOGS:-$PWD/slurm_scripts/logs}
DST=experiments/results
mkdir -p "$DST/logs" "$DST/cells"

copy_log() {                                   # copy_log <src> <dst>
    mkdir -p "$(dirname "$2")"
    if [ "$(stat -c %s "$1")" -le 10485760 ]; then
        cp "$1" "$2"
    else
        { head -n 2000 "$1"; echo "[ops_collect] log over 10 MB: lines between the first 2000 and the last 20000 omitted"; tail -n 20000 "$1"; } > "$2"
    fi
}

if [ -z "$COMMIT_ONLY" ]; then
    if [ -d "$SCA_OPS_BASE/results" ]; then
        rsync -a --prune-empty-dirs --max-size=20m \
            --include='*/' --include='*.json' --include='*.txt' --include='*.md' --include='*.csv' \
            --exclude='*' "$SCA_OPS_BASE/results/" "$DST/" || { echo "rsync failed"; exit 1; }
    fi
    if [ -d "$LOGS" ]; then
        find "$LOGS" -maxdepth 1 -type f -mtime -7 -print0 | while IFS= read -r -d '' f; do
            name=$(basename "$f"); copy_log "$f" "$DST/logs/${name%.*}.txt"
        done
    fi
    for root in workdir workdir_pretrain; do
        [ -d "$root" ] || continue
        find -L "$root" -maxdepth 6 -type f -mtime -7 \( -path '*/log/log*.txt' -o -name run.log \) -print0 \
          | while IFS= read -r -d '' f; do
              rel=${f%.*}; copy_log "$f" "$DST/cells/${rel}.txt"
            done
    done
fi

find "$DST" -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' -o -name '*.csv' \) \
    -size -20M -print0 | xargs -0 -r git add -f --
if git diff --cached --quiet; then
    echo "collector: nothing changed"; exit 0
fi
git commit -q -m "Results: $MSG" || { echo "commit failed"; exit 1; }
[ -n "$COMMIT_ONLY" ] && { echo "collector: committed locally ($MSG)"; exit 0; }
if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD && git push -q origin "HEAD:$BRANCH"; then
    echo "collector: pushed ($MSG)"
else
    git rebase --abort 2>/dev/null
    echo "collector: push failed; the commit stays local and goes out with the next push"; exit 1
fi
