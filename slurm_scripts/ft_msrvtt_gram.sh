#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=ft_msrvtt_gram
#SBATCH -o ./slurm_scripts/logs/ft_msrvtt_gram_%j.out
#SBATCH -e ./slurm_scripts/logs/ft_msrvtt_gram_%j.out
# GRAM-repro finetune on MSR-VTT: the same-budget counterpart of ft_msrvtt_sca.sh.
# Starts from the Wave-1 GRAM-repro pretrain (150k, full-FT) and uses a data_cfg copied
# VERBATIM from the SCA finetune config -- identical data, batch size, epochs, lr -- so
# "SCA-ft vs GRAM-ft @150k" is exactly matched. Published GRAM ft rows (from the 27M
# pretrain) are quoted separately as the scale reference.
set -uo pipefail
: "${DATA_ROOT:?export DATA_ROOT first}"; : "${WORK_ROOT:?export WORK_ROOT first}"
MODELS_DIR="${MODELS_DIR:-$WORK_ROOT/sca_models}"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
FT_CONFIG="${FT_CONFIG:-./config/baselines/finetune_cfg/gram_ft_msrvtt.json}"
FT_OUTDIR="${FT_OUTDIR:-./workdir/gram_ft_msrvtt}"
[ -f "$FT_CONFIG" ] || { echo "FATAL: FT_CONFIG $FT_CONFIG not found" >&2; exit 1; }
mkdir -p slurm_scripts/logs "$FT_OUTDIR"
INIT="${GRAM_PRETRAIN_CKPT:-$(ls -t ./workdir_pretrain/gram/ckpt/best_*.pt 2>/dev/null | head -1)}"
[ -z "$INIT" ] && { echo "ERROR: no GRAM-repro pretrain ckpt -- expected under ./workdir_pretrain/gram/ckpt or set GRAM_PRETRAIN_CKPT"; exit 1; }
RESUME=""; ls "$FT_OUTDIR"/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true"
echo "START $(date +%T)  GRAM-repro finetune MSR-VTT  (init=$INIT)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9906 \
  ./run.py --config "$FT_CONFIG" \
  --output_dir "$FT_OUTDIR" --checkpoint "$INIT" --save_best true --checkpointing true $RESUME 2>&1
rc=$?
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
