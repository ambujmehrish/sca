#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=sca_run
#SBATCH -o ./slurm_scripts/logs/run_%j.out
#SBATCH -e ./slurm_scripts/logs/run_%j.out
# Generic pretrain/finetune launcher: one script for every config in the campaign.
#
#   sbatch slurm_scripts/run_config.sh <config.json> <output_dir> [extra run.py args...]
#
# Auto-resumes from the newest optimizer checkpoint in <output_dir>/ckpt (the 6h window
# is shorter than a full 150k epoch-5 pretrain: RESUBMIT THE SAME COMMAND until the run
# reaches its num_train_steps -- each resubmission continues where the last stopped).
# Requires $WORK/sca_env.rc (DATA_ROOT/WORK_ROOT/SCA_CACHE_ROOT/MODELS_DIR).
set -uo pipefail
CONFIG="${1:?usage: run_config.sh <config.json> <output_dir> [extra args]}"
OUTDIR="${2:?usage: run_config.sh <config.json> <output_dir> [extra args]}"
shift 2
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: config $CONFIG not found" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p "$OUTDIR" slurm_scripts/logs
RESUME=""
ls "$OUTDIR"/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && { RESUME="--resume true"; echo "RESUME from $OUTDIR/ckpt"; } || echo "FRESH"
echo "START $(date +%T)  config=$CONFIG  out=$OUTDIR"
# filter the benign h264 decoder noise (mmco/*ref* warnings from mid-GOP YouTube clip
# cuts -- harmless, but hundreds of MB over a full pretrain). pipefail + `|| true` on the
# grep keeps the pipeline's exit code = srun's.
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
  --master_port $((9000 + RANDOM % 999)) \
  ./run.py --config "$CONFIG" --output_dir "$OUTDIR" --checkpointing true $RESUME "$@" 2>&1 \
  | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($" || true; }
echo "EXIT=$? DONE $(date +%T)"
