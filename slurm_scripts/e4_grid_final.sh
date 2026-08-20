#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=e4_final
#SBATCH -o ./slurm_scripts/logs/e4_final_%j.out
#SBATCH -e ./slurm_scripts/logs/e4_final_%j.out
# E4/E5/E6 grids over the FINAL (end-of-training) checkpoints -- companion to
# e4_grid.sh (best ckpts): tests whether raw-space advantages survive training, or only
# exist at an early-stopped peak of an unstable trajectory (GRAM-LoRA question). One GPU: feature extraction is a single bare-process encoder pass per
# checkpoint; the grids themselves are CPU post-processing. Feature dumps are cached in
# results/e4/feats/, so resubmitting after a timeout resumes where it stopped.
#
# Each arm maps to the feature-dump config that builds the RIGHT model class for its
# checkpoint (benchmark_eval/configs_e4/) -- a LoRA checkpoint loaded into a plain-GRAM
# model would silently drop the adapter deltas.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline

# S* over the TEST captions for the E6 calibration block (strict gather: the 150k
# TRAIN cache does not contain test ids, so E6 needs its own cache)
S_STAR="$SCA_CACHE_ROOT/s_star_msrvtt_test.pt"
if [ ! -f "$S_STAR" ]; then
  echo "== building E6 S* cache from msrvtt test annotations"
  python3 data/semantic_targets.py \
    --annotation_json datasets/annotations/msrvtt/descs_ret_test.json \
    --out_path "$S_STAR" || exit 1
fi

# arm -> (workdir, dump config); best_*.pt preferred, else the latest model_step
# FINAL checkpoint (highest model_step), never best_*: this variant measures the
# end-of-training space, so unstable arms are not flattered by an early-stopped peak.
best_ckpt() {
  ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1
}
CKPTS=""; MISSING=""
for spec in \
    "sca:workdir_pretrain/sca:benchmark_eval/configs_e4/sca.json" \
    "sca_nomask:workdir_pretrain/sca_nomask:benchmark_eval/configs_e4/sca.json" \
    "gram:workdir_pretrain/gram:benchmark_eval/configs_e4/gram.json" \
    "gram_masked:workdir_pretrain/gram_masked:benchmark_eval/configs_e4/gram.json" \
    "gram_lora:workdir_pretrain/gram_lora:benchmark_eval/configs_e4/gram_lora.json" \
    "pmrl:workdir_pretrain/pmrl:benchmark_eval/configs_e4/pmrl.json" \
    "pmrl_masked:workdir_pretrain/pmrl_masked:benchmark_eval/configs_e4/pmrl.json" \
    "pmrl_lora:workdir_pretrain/pmrl_lora:benchmark_eval/configs_e4/pmrl_lora.json"; do
  name="${spec%%:*}"; rest="${spec#*:}"; wd="${rest%%:*}"; cfg="${rest#*:}"
  ckpt="$(best_ckpt "$wd" | head -1)"
  if [ -n "$ckpt" ]; then
    echo "ARM $name -> $ckpt"
    CKPTS="$CKPTS ${name}_final=$ckpt=$cfg"
  else
    echo "WARN: no checkpoint under $wd -- arm $name SKIPPED" >&2
    MISSING="$MISSING $name"
  fi
done
[ -n "$CKPTS" ] || { echo "FATAL: no checkpoints found at all" >&2; exit 1; }

CKPTS="$CKPTS" S_STAR="$S_STAR" bash scripts/run_e4_grid.sh
rc=$?
[ -n "$MISSING" ] && { echo "INCOMPLETE: missing arms:$MISSING" >&2; [ $rc -eq 0 ] && rc=3; }
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
