#!/bin/bash
# Run every extractor, write the output into the repo, commit and push.
#
#   bash scripts/harvest_and_push.sh
#   bash scripts/harvest_and_push.sh --no-push        # write and commit only
#
# One command instead of five, and the results land as text files on the branch so they can
# be read without pasting terminal output around. Only text is committed -- the feature and
# rerank dumps are hundreds of MB and stay where they are.
#
# Every step runs even if an earlier one fails; a failure is recorded in the file and in
# INDEX.md rather than aborting the harvest, because a partial harvest is still worth having
# and a missing file is indistinguishable from a step that was never run.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/results/harvest
mkdir -p "$OUT"

run() {                      # run <name> <description> <command...>
  local name="$1" desc="$2"; shift 2
  local f="$OUT/$name.txt"
  { echo "\$ $*"; echo; } > "$f"
  "$@" >> "$f" 2>&1
  local rc=$?
  echo "  [$name] rc=$rc -> $f"
  # the command itself is already the first line of the output file; keeping it out of the
  # status record avoids a multi-line command breaking the index table
  printf '%s|%s|%s\n' "$name" "$rc" "$desc" >> "$OUT/.status"
  return 0
}

: > "$OUT/.status"
echo "harvesting into $OUT"

run raw_vs_itm "Aggregator vs ITM per cell, plus arm x benchmark pivots" \
    python3 scripts/raw_vs_itm.py --root workdir/e1_zs --pivot

run raw_vs_itm_final "Final-checkpoint eval (workdir/e1_final), arm x benchmark pivots" \
    python3 scripts/raw_vs_itm.py --root workdir/e1_final --pivot

run raw_vs_itm_frames "Query-weighted / frame-set arms (workdir/e1_frames), pivots" \
    python3 scripts/raw_vs_itm.py --root workdir/e1_frames --pivot

run raw_vs_itm_itmfrozen "Reranker on FROZEN weights (workdir/e1_itmfrozen), pivots" \
    python3 scripts/raw_vs_itm.py --root workdir/e1_itmfrozen --pivot

run itm_frozen_delta "Frozen-reranker vs adapted, paired per cell, with the stage-1 check" \
    python3 scripts/itm_frozen_delta.py

run raw_vs_itm_repro "RETIRED reimplementation baselines (workdir/e1_repro) -- superseded by the two below" \
    python3 scripts/raw_vs_itm.py --root workdir/e1_repro --pivot

run pmrl_released "PMRL from the authors' RELEASED checkpoint, our protocol (workdir/pmrl_released)" \
    bash -c 'shopt -s nullglob; f=(workdir/pmrl_released/*/run.log);
             [ ${#f[@]} -gt 0 ] || { echo "no logs under workdir/pmrl_released"; exit 3; }
             python3 scripts/parse_authors_eval.py "${f[@]}"'

run hypergram_authors "HyperGRAM trained from the authors' code at their recipe, our protocol (workdir/hgeval)" \
    bash -c 'shopt -s nullglob; f=(workdir/hgeval/*/run.log);
             [ ${#f[@]} -gt 0 ] || { echo "no logs under workdir/hgeval"; exit 3; }
             python3 scripts/parse_authors_eval.py "${f[@]}"'

run raw_vs_itm_missing "Missing-modality sweep (workdir/e1_missing), pivots" \
    python3 scripts/raw_vs_itm.py --root workdir/e1_missing --pivot

run score_fusion "Candidate recall@k (the ceiling) and the fusion-weight sweep" \
    bash -c 'shopt -s nullglob; f=(workdir/e1_fusion/*/dumps/rerank_*.pt);
             [ ${#f[@]} -gt 0 ] || { echo "no rerank dumps under workdir/e1_fusion"; exit 3; }
             python3 scripts/sweep_score_fusion.py "${f[@]}"'

run main_table "The main table, generated from measured cells (MISSING where unmeasured)" \
    python3 scripts/build_main_table.py --out experiments/results/tables_final/table1_main_all.tex

run paper_tables "Per-benchmark paper tables (both directions, R@1/R@10, sectioned by geometry)" \
    python3 scripts/build_paper_table.py --all

run transfer_table "Table 2: the four transfer benchmarks side by side (T->V R@1/R@10)" \
    python3 scripts/build_transfer_table.py

run missing_table "Table 3: R@1 under test-time missing modalities, per benchmark" \
    python3 scripts/build_missing_table.py --all

run loss_ablation_table "Table 4: T9 with one objective component removed at a time" \
    python3 scripts/build_loss_ablation_table.py

# after the pmrl_released / hypergram_authors / raw_vs_itm extractors above: it reads their files
run gain_table "The aggregation-gain table: the abstract's headline numbers, in the paper" \
    python3 scripts/build_gain_table.py

run significance "Seed CIs, exact sign tests, and the eval noise floor" \
    python3 scripts/significance.py

run eval_geometry "Was every cell scored with the geometry its arm was TRAINED with?" \
    python3 scripts/audit_eval_geometry.py

run training_gaps "Completeness, step continuity, cross-arm uniformity, config drift" \
    python3 scripts/audit_training_gaps.py --workdir_root workdir_pretrain

run modality_arity "Gallery arity at train vs at each benchmark" \
    python3 scripts/audit_modality_arity.py

run training_curve "Where each arm peaks during training (overtraining check)" \
    python3 scripts/training_curve.py --workdir_root workdir_pretrain

run query_centroid "Query-weighted centroid vs uniform, tau sweep on cached features" \
    bash -c 'shopt -s nullglob; f=(results/e4_transfer/feats/*.pt);
             [ ${#f[@]} -gt 0 ] || { echo "no feature dumps under results/e4_transfer/feats"; exit 3; }
             python3 scripts/try_query_centroid.py "${f[@]}"'

# Slurm tails: the exit line and the last error, which is where a failed cell shows up
{
  echo "Last lines of the most recent job logs"
  # ALL launchers, most recent first -- the old hardcoded pattern list silently omitted
  # every launcher written after it (missing_eval, fs_eval, hgeval, pmrl_released), which
  # is exactly where the open failures were when it mattered.
  for lg in $(ls -t slurm_scripts/logs/*.out 2>/dev/null | head -14); do
    echo; echo "===== $lg"; tail -25 "$lg"
  done
  echo; echo "===== cell-level status lines from the missing-modality sweep (all logs)"
  grep -H -E "^== \[|LOAD NOT VERIFIED|refused|EXIT=" slurm_scripts/logs/missing_*.out \
    2>/dev/null | tail -120 || echo "(no missing_*.out logs)"
} > "$OUT/job_logs.txt" 2>&1
echo "  [job_logs] -> $OUT/job_logs.txt"

{
  echo "# Harvest"
  echo
  echo "Generated by \`scripts/harvest_and_push.sh\`. Each row is one extractor; rc=0 means it"
  echo "ran, any other value means the file records a failure rather than a result."
  echo
  echo '| output | rc | what it answers |'
  echo '|---|---|---|'
  while IFS='|' read -r name rc desc; do
    printf '| [%s](%s.txt) | %s | %s |\n' "$name" "$name" "$rc" "$desc"
  done < "$OUT/.status"
  echo '| [job_logs](job_logs.txt) | - | Tails of the most recent slurm logs |'
  echo
  echo "## Reading order"
  echo
  echo "1. \`training_gaps\` -- if arms are incomplete or non-uniform, nothing below is comparable."
  echo "2. \`score_fusion\` -- \`w=0\` must equal the ITM number in the eval log, or the metric is"
  echo "   wrong. Then \`cand recall@k\`: where it is saturated, the aggregator cannot influence"
  echo "   the reported number at all and only fusion can."
  echo "3. \`itm_frozen_delta\` -- READ THIS ONE FIRST of the three. It pairs each frozen-"
  echo "   reranker cell with the same cell in \`e1_frames\` and checks that AGGREG is"
  echo "   identical before reporting anything: stage 1 is untouched by the flag, so if it"
  echo "   moved, the ITM column cannot be attributed to the reranker and the run is void."
  echo "   \`raw_vs_itm_itmfrozen\` is the same cells as a raw pivot, for reading the detail."
  echo "2b. \`pmrl_released\` and \`hypergram_authors\` -- the two authors'-code baseline rows."
  echo "   The REPORTED line is the table number; raw_vs_itm_repro is RETIRED and must not"
  echo "   be quoted for PMRL or HyperGRAM."
  echo "3b. \`raw_vs_itm_frames\` -- the query-weighted arms AND the R1-R4 reranker arms;"
  echo "   \`raw_vs_itm_final\` -- final-checkpoint numbers for the earlier arms."
  echo "4. \`raw_vs_itm\` -- the pivots give arm x benchmark for the aggregator score and for the"
  echo "   reported ITM metric. Compare an SCA arm against \`released\` down each column."
  echo "5. \`job_logs\` -- only if something above is missing or failed."
} > "$OUT/INDEX.md"
echo "  [INDEX] -> $OUT/INDEX.md"
rm -f "$OUT/.status"

if [ "${1:-}" = "--no-push" ]; then
  echo "done (not pushed)"; exit 0
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git add "$OUT" experiments/results/tables_final
if git diff --cached --quiet; then
  echo "nothing changed -- not committing"; exit 0
fi
git commit -q -m "Harvest: extractor output at $(date -u +%Y-%m-%dT%H:%MZ)"
# Two failures look identical to `git push` but need opposite responses: a REJECTION means
# the branch moved and a rebase fixes it, while a NETWORK error means nothing is wrong with
# the history and rebasing just fails again with a misleading message. Read the output and
# only rebase on the first.
for d in 2 4 8 16; do
  out=$(git push -u origin "$BRANCH" 2>&1) && { echo "pushed to $BRANCH"; exit 0; }
  echo "$out" | tail -2
  if echo "$out" | grep -qiE "could not resolve host|connection timed out|network is unreachable|failed to connect|operation timed out"; then
    echo "network is down, not a rejection -- retrying the push in ${d}s (history is fine)"
  elif echo "$out" | grep -qiE "rejected|fetch first|non-fast-forward"; then
    echo "push rejected -- rebasing onto origin/$BRANCH and retrying in ${d}s"
    git pull --rebase origin "$BRANCH" || {
      git rebase --abort 2>/dev/null
      echo "FATAL: rebase failed. The results are committed locally on $BRANCH; resolve by" >&2
      echo "       hand with: git pull --rebase origin $BRANCH" >&2
      exit 1; }
  else
    echo "push failed for an unrecognised reason -- retrying in ${d}s without rebasing"
  fi
  sleep "$d"
done
echo "push failed after retries -- the results are committed locally on $BRANCH" >&2
exit 1
