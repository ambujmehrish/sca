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
#SBATCH --job-name=ft_msrvtt_sca_fullft
#SBATCH -o ./slurm_scripts/logs/ft_msrvtt_sca_fullft_%j.out
#SBATCH -e ./slurm_scripts/logs/ft_msrvtt_sca_fullft_%j.out
# SCA-ft-v2b: FULL finetuning (no adapters) from a MERGED checkpoint -- exact capacity
# parity with GRAM-ft, which trains all ~1B params. REQUIRES SCA_MERGED_CKPT pointing to
# a scripts/merge_lora_ckpt.py output (a LoRA checkpoint will NOT load into the
# use_lora=false model -- fail loud, no silent adapter dropping).
set -uo pipefail
: "${DATA_ROOT:?export DATA_ROOT first}"; : "${WORK_ROOT:?export WORK_ROOT first}"
MODELS_DIR="${MODELS_DIR:-$WORK_ROOT/sca_models}"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p slurm_scripts/logs workdir/sca_ft_msrvtt_fullft
INIT="${SCA_MERGED_CKPT:?export SCA_MERGED_CKPT=<merged ckpt from scripts/merge_lora_ckpt.py>}"
[ -f "$INIT" ] || { echo "ERROR: SCA_MERGED_CKPT $INIT not found"; exit 1; }
RESUME=""; ls workdir/sca_ft_msrvtt_fullft/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true"
echo "START $(date +%T)  SCA finetune MSR-VTT  (init=$INIT)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9907 \
  ./run.py --config ./config/sca/finetune_cfg/retrieval-msrvtt_fullft.json \
  --output_dir ./workdir/sca_ft_msrvtt_fullft --checkpoint "$INIT" --save_best true --checkpointing true $RESUME 2>&1
rc=$?
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
