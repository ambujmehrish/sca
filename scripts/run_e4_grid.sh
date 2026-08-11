#!/usr/bin/env bash
# E4/E5/E6 grid orchestrator (P4): the full 2x2 train/test masking design.
#
# TRAIN axis = which checkpoint (train-full vs train-masked; one pair per method):
#   SCA   : sca_pretrain_nomask.json        vs  sca_pretrain.json
#   GRAM  : config/gram pretrain (default)  vs  gram_masked_pretrain.json
#   PMRL  : pmrl_pretrain.json              vs  pmrl_masked_pretrain.json
# TEST axis = the {0,25,50,75}% x which-modality grids of run_eval_grids.py, with every
# scorer (centroid / volume masked-(i) / mean-imputed-(ii) / pmrl raw + norm) applied to
# THE SAME dumped features -- one encoder pass per checkpoint, honest comparisons.
#
# Usage (cluster; DATA_ROOT/WORK_ROOT/MODELS_DIR exported, checkpoints trained):
#   CKPTS="sca_full=./workdir_pretrain/sca_nomask/ckpt/model_step_XXXX.pt \
#          sca_masked=./workdir_pretrain/sca/ckpt/model_step_XXXX.pt \
#          gram_full=... gram_masked=... pmrl_full=... pmrl_masked=..." \
#   EVAL_CFG=./benchmark_eval/configs_baselines/zs_msrvtt_tvas_maskedii.json \
#   S_STAR=$DATA_ROOT/... bash scripts/run_e4_grid.sh
#
# Each cell -> results/e4/<name>.json (features dumped once per ckpt to results/e4/feats/).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPTS:?export CKPTS=\"name=path [name=path ...]\" (train-axis checkpoints)}"
: "${EVAL_CFG:?export EVAL_CFG=<eval config whose val loader defines the test set>}"
if [ -n "${MODELS_DIR:-}" ]; then source "$MODELS_DIR/env.sh"; fi
mkdir -p results/e4/feats

for pair in $CKPTS; do
  name="${pair%%=*}"; ckpt="${pair#*=}"
  [ -f "$ckpt" ] || { echo "FATAL: checkpoint $ckpt ($name) not found" >&2; exit 1; }
  feats="results/e4/feats/${name}.pt"
  if [ ! -f "$feats" ]; then
    echo "== [$name] extracting features (one encoder pass)"
    EVAL_CKPT="$ckpt" python3 evaluation/run_eval_grids.py \
      --config "$EVAL_CFG" --dump_features "$feats"
  fi
  echo "== [$name] running E4/E5/E6 grids on cached features"
  python3 evaluation/run_eval_grids.py --features "$feats" \
    --out "results/e4/${name}.json" ${S_STAR:+--s_star "$S_STAR"}
done
echo "== E4 grid complete: results/e4/*.json"
