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
#   CKPTS="name=<ckpt.pt>=<dump_cfg.json> [ ... ]" bash scripts/run_e4_grid.sh
# The third field is the family's feature-dump config (benchmark_eval/configs_e4/*.json):
# it selects the RIGHT model class for the checkpoint -- loading a LoRA checkpoint into a
# plain-GRAM model silently drops the adapter deltas, so the cfg is per-checkpoint, not
# global. A two-field pair falls back to $EVAL_CFG. slurm_scripts/e4_grid.sh builds the
# CKPTS string automatically from workdir_pretrain/.
#
# Each cell -> results/e4/<name>.json (features dumped once per ckpt to results/e4/feats/).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPTS:?export CKPTS=\"name=ckpt[=cfg] [name=ckpt[=cfg] ...]\" (train-axis checkpoints)}"
if [ -n "${MODELS_DIR:-}" ]; then source "$MODELS_DIR/env.sh"; fi
mkdir -p results/e4/feats

for pair in $CKPTS; do
  name="${pair%%=*}"; rest="${pair#*=}"; ckpt="${rest%%=*}"
  cfg="${rest#*=}"; [ "$cfg" = "$ckpt" ] && cfg="${EVAL_CFG:?pair $name has no cfg field and EVAL_CFG is unset}"
  [ -f "$ckpt" ] || { echo "FATAL: checkpoint $ckpt ($name) not found" >&2; exit 1; }
  [ -f "$cfg" ] || { echo "FATAL: dump config $cfg ($name) not found" >&2; exit 1; }
  feats="results/e4/feats/${name}.pt"
  if [ ! -f "$feats" ]; then
    echo "== [$name] extracting features (one encoder pass, cfg=$cfg)"
    EVAL_CKPT="$ckpt" python3 evaluation/run_eval_grids.py \
      --config "$cfg" --dump_features "$feats"
  fi
  echo "== [$name] running E4/E5/E6 grids on cached features"
  python3 evaluation/run_eval_grids.py --features "$feats" \
    --out "results/e4/${name}.json" ${S_STAR:+--s_star "$S_STAR"}
done
echo "== E4 grid complete: results/e4/*.json"
