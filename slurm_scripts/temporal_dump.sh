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
#SBATCH --job-name=temporal
#SBATCH -o ./slurm_scripts/logs/temporal_%j.out
#SBATCH -e ./slurm_scripts/logs/temporal_%j.out
# Dump PER-FRAME video features, then measure what mean-pooling the frame axis costs.
#
# general_module.py:426 averages the per-frame CLS tokens into one vector before the
# contrastive head, so every clip -- ten seconds or two minutes -- reaches every aggregator
# as a single point. That is upstream of the centroid, the volume and the cross-encoder
# alike, which is why no amount of adapter or fusion tuning could reach it.
#
# AudioCaps is absent on purpose: it is audio-only (T-A), so there is no frame axis to keep.
#
#   SCA_TEMPORAL_ARM=sca sbatch slurm_scripts/temporal_dump.sh
#
# One GPU: this is feature extraction, not reranking.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p slurm_scripts/logs results/temporal

ARM="${SCA_TEMPORAL_ARM:-sca}"
# the FINAL checkpoint, matching e1_final_ckpt.sh -- not the aggregator-selected best_*.pt
CKPT=$(ls "workdir_pretrain/$ARM"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "FATAL: no model_step_*.pt under workdir_pretrain/$ARM/ckpt" >&2; exit 2; }
echo "arm $ARM -> $CKPT"

rc_all=0
for bench in msrvtt didemo activitynet vatex; do
  cfg="benchmark_eval/configs_temporal/sca_${bench}.json"
  out="results/temporal/${ARM}_${bench}.pt"
  [ -f "$out" ] && { echo "== [$bench] $out exists, skip"; continue; }
  echo "== [$bench] START $(date +%T)"
  EVAL_CKPT="$CKPT" python3 evaluation/run_eval_grids.py --config "$cfg" --dump_features "$out" \
    || { echo "== [$bench] FAILED" >&2; rc_all=1; }
done

echo
echo "Now, no GPU needed:"
echo "  python3 scripts/try_temporal_centroid.py results/temporal/${ARM}_*.pt"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
