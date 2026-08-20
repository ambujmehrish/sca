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
#SBATCH --job-name=depth_zs
#SBATCH -o ./slurm_scripts/logs/depth_zs_%j.out
#SBATCH -e ./slurm_scripts/logs/depth_zs_%j.out
# Zero-shot depth (k=5) with a real baseline: the ARITY DELTA for both methods.
#
# GRAM reports the depth gain in their Tab. 4 (zero-shot MSR-VTT T2V R@1):
#     T-V 52.8 | T-V-A 54.1 | T-V-A-S 54.8 | T-V-A-S-D 55.3
# so adding depth buys them +0.5, and they released the 5-modality checkpoint that produced
# it (GRAM_pretrained_5modalities). That makes the comparison we actually want possible:
# score BOTH methods at 4 and at 5 modalities in one environment, and compare the deltas.
#
# This replaces a claim that could not survive review. The +1.6 depth gain in Tab. 3 came
# from a finetuned run that continued the row-1 finetune for four more epochs at a different
# learning rate, so it confounded arity with extra training, and it had no GRAM baseline at
# all. Here nothing is trained: one checkpoint per method, scored twice.
#
#   GRAM5_CKPT=/path/to/GRAM_pretrained_5modalities/ckpt/model_step_*.pt \
#   SCA_ZS_CKPT=/path/to/sca/best.pt \
#   sbatch slurm_scripts/depth_zeroshot.sh
#
# Markers are config-aware: edit a config and only that cell re-runs.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

best_ckpt() {
  local b; b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$b"
}
SCA_CKPT="${SCA_ZS_CKPT:-$(best_ckpt workdir_pretrain/sca)}"
[ -n "$SCA_CKPT" ] && [ -f "$SCA_CKPT" ] || { echo "FATAL: no SCA pretrain ckpt" >&2; exit 1; }

# No fallback for the GRAM 5-modality checkpoint: silently substituting the 4-modality one
# would score a model that has never seen depth and report it as GRAM's depth result.
if [ -z "${GRAM5_CKPT:-}" ]; then
  echo "FATAL: set GRAM5_CKPT to GRAM_pretrained_5modalities (their T-VASD release)." >&2
  echo "       It is a DIFFERENT checkpoint from the 4-modality one used in Tables 1-2." >&2
  exit 2
fi
[ -f "$GRAM5_CKPT" ] || { echo "FATAL: GRAM5_CKPT=$GRAM5_CKPT does not exist" >&2; exit 1; }
echo "SCA  ckpt: $SCA_CKPT"
echo "GRAM5 ckpt: $GRAM5_CKPT"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for cell in sca_msrvtt_tvas sca_msrvtt_tvasd gram5_msrvtt_tvas gram5_msrvtt_tvasd; do
  cfg="benchmark_eval/configs_depth/${cell}.json"
  out="workdir/depth_zs/${cell}"
  ckpt="$SCA_CKPT"; case "$cell" in gram5_*) ckpt="$GRAM5_CKPT" ;; esac
  cell_is_done "$out" "$cfg" && { echo "== [$cell] already done, skip"; continue; }
  mkdir -p "$out"
  echo "== [$cell] START $(date +%T)"
  EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
    --master_port $((9600 + ${SLURM_JOB_ID:-$$} % 200)) \
    ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
    | { grep -v --line-buffered -E "$NOISE" || true; }
  rc=$?
  if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$cell] OK $(date +%T)"
  else echo "== [$cell] FAILED rc=$rc" >&2; rc_all=$rc; fi
done
echo
echo "Compare the two deltas, not the absolutes:"
echo "  SCA   T-VASD minus T-VAS"
echo "  GRAM5 T-VASD minus T-VAS   (their published delta is +0.5, Tab. 4)"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
