#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=depth_ctrl
#SBATCH -o ./slurm_scripts/logs/depth_ctrl_%j.out
#SBATCH -e ./slurm_scripts/logs/depth_ctrl_%j.out
# Matched control for the k=5 depth row of Table 3. The 58.7 number came from a run that
# CONTINUES the MSR-VTT SCA finetune for 4 more epochs at lr 1e-4, so comparing it against
# the 57.1 row confounds "added a 5th modality" with "trained longer at a different lr".
# Here we score the SAME depth checkpoint twice through the same eval path -- once on
# T-VASD (5 modalities) and once on T-VAS (4) -- so the difference is the modality alone.
# Eval only, no training. Per-cell done markers: resubmit to resume.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
# Resolve from CODE_DIR, which every launcher has already cd'd into. Under sbatch, $0 is a
# COPY of the script in Slurm's spool directory (/var/spool/slurmd/job<N>/slurm_script), so
# "$(dirname "$0")/.." points at /var/spool/slurmd and there is no scripts/ there. This source
# has been failing in every Slurm job since it was written. Nothing surfaced because the
# callers use `cell_is_done ... && continue`: an undefined function returns 127, the && short-
# circuits, and the cell simply runs. So the resume-skip and the config fingerprinting have
# both been inert under Slurm, which is also why eval cells show up twice in the logs.
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced $HELPER but cell_is_done is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

CKPT="${SCA_DEPTH_CKPT:-$(ls -t ./workdir/sca_ft_msrvtt_depth/ckpt/best_*.pt 2>/dev/null | head -1)}"
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "FATAL: depth finetune ckpt not found -- set SCA_DEPTH_CKPT" >&2; exit 1; }
echo "ckpt=$CKPT"

rc_all=0
for cell in depth_k5_tvasd depth_k4_tvas; do
  out="workdir/depth_control/$cell"
  cfg="benchmark_eval/configs_depth/${cell}.json"
  cell_is_done "$out" "$cfg" && { echo "== [$cell] already done, skip"; continue; }
  mkdir -p "$out"
  echo "== [$cell] START $(date +%T)"
  EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 \
    --node_rank 0 --nproc_per_node 4 --master_port $((9500 + RANDOM % 200)) \
    ./benchmark_eval/run_eval.py --config "benchmark_eval/configs_depth/${cell}.json" \
    --output_dir "$out" 2>&1 \
    | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($" || true; }
  rc=$?
  if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$cell] OK $(date +%T)"
  else echo "== [$cell] FAILED rc=$rc" >&2; rc_all=$rc; fi
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
