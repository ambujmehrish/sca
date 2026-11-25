#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=tvaparity
#SBATCH --array=0-1
#SBATCH -o ./slurm_scripts/logs/tvaparity_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/tvaparity_%A_%a.out
# SCALAR (final T9 checkpoint) and GRAM scored T-VA on the two T-VAS benchmarks, matching the
# released BaryBind T-VA checkpoint: MSR-VTT at 8 frames, VATEX at 8 and at 16 frames.
# Same checkpoints and configs as the reported rows; only the task is T-VA (configs_tva).
#
#   GRAM_RELEASED_CKPT=/path/to/GRAM/ckpt/model_step_459.pt sbatch slurm_scripts/tva_parity.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
# shellcheck source=/dev/null
source "$CODE_DIR/scripts/cell_done.sh" || { echo "FATAL: cannot source scripts/cell_done.sh" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced scripts/cell_done.sh but cell_is_done is not defined." >&2; exit 2; }
# shellcheck source=/dev/null
source "$CODE_DIR/scripts/parity_common.sh" || exit 2
mkdir -p slurm_scripts/logs

MODELS=(sca gram)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-${1:-}}]:-}"
[ -n "$MODEL" ] || { echo "FATAL: array index must be 0 (sca) or 1 (gram)" >&2; exit 2; }
check_ckpt "$MODEL" || exit 2
CKPT=$(ckpt_for "$MODEL")

rc_all=0
for cell in msrvtt:fs8 vatex:fs8 vatex:fsbb; do
  bench=${cell%%:*}; fs=${cell##*:}
  run_cell "workdir/parity/${MODEL}_tva_${fs}_${bench}" "$(cfg_for "$MODEL" "$bench" "tva_$fs")" "$CKPT" || rc_all=1
done
echo "  python3 scripts/summarize_parity.py"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
