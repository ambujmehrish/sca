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
#SBATCH --job-name=ft_msrvtt_depth_sca
#SBATCH -o ./slurm_scripts/logs/ft_msrvtt_depth_sca_%j.out
#SBATCH -e ./slurm_scripts/logs/ft_msrvtt_depth_sca_%j.out
# SCA finetune on MSR-VTT + depth (5-modal tvasd, E10) (P4 grid clone of ft_msrvtt_depth.sh): GRAM recipe, model=sca + LoRA.
# Set SCA_PRETRAIN_CKPT to the Stage-1 pretrain checkpoint (default: latest best in
# ./workdir_pretrain/sca/ckpt). Requires DATA_ROOT/WORK_ROOT (+ prefetched MODELS_DIR).
set -uo pipefail
: "${DATA_ROOT:?export DATA_ROOT first}"; : "${WORK_ROOT:?export WORK_ROOT first}"
MODELS_DIR="${MODELS_DIR:-$WORK_ROOT/sca_models}"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p slurm_scripts/logs workdir/sca_ft_msrvtt_depth
# depth STACKS on the finetuned msrvtt ckpt (run ft_msrvtt_sca.sh FIRST)
INIT="${SCA_MSRVTT_CKPT:-$(ls -t ./workdir/sca_ft_msrvtt/ckpt/best_*.pt 2>/dev/null | head -1)}"
[ -z "$INIT" ] && { echo "ERROR: msrvtt SCA finetune not done -- run ft_msrvtt_sca.sh FIRST (depth stacks on it)"; exit 1; }
RESUME=""; ls workdir/sca_ft_msrvtt_depth/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true"
echo "START $(date +%T)  SCA finetune MSR-VTT + depth (5-modal tvasd, E10)  (init=$INIT)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9906 \
  ./run.py --config ./config/sca/finetune_cfg/retrieval-msrvtt_depth.json \
  --output_dir ./workdir/sca_ft_msrvtt_depth --checkpoint "$INIT" --save_best true --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
