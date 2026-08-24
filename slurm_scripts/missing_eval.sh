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
#SBATCH --job-name=missing
#SBATCH --array=0-4
#SBATCH -o ./slurm_scripts/logs/missing_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/missing_%A_%a.out
# The missing-modality sweep: SCA (T9) and the released GRAM checkpoint, five mask rates.
#
# THE CLAIM THIS MEASURES. The volume family cannot consume incomplete data -- both released
# repositories' loaders hard-drop any clip missing audio (data/IndexAnno.py, 9% of VAST-150k)
# -- while SCA's masked centroid is defined on any modality subset. This sweep asks what that
# is worth at TEST time: a fraction r of gallery clips lose one modality (deterministic per
# clip, identical for both methods, nested across rates), and the reported metric is traced
# from r=0 to r=0.9. r=0 is byte-identical to the standard eval, so those cells must
# reproduce Table 1/2 -- a built-in control, checked by the harvest, not assumed.
#
# One array task per benchmark; each runs 2 models x 5 rates sequentially, skipping cells
# already done. SCA uses the T9 final checkpoint; the GRAM side requires the same released
# checkpoint file Table 1's GRAM* row used:
#
#   GRAM_RELEASED_CKPT=/path/to/released.pt sbatch slurm_scripts/missing_eval.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTHONUNBUFFERED=1
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced $HELPER but cell_is_done is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

BENCHES=(msrvtt didemo activitynet vatex audiocaps)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-4." >&2; exit 2; }
BENCH="${BENCHES[$IDX]:-}"
[ -n "$BENCH" ] || { echo "FATAL: index $IDX out of range (0-4)" >&2; exit 2; }

# SCA: the T9 final checkpoint -- the same weights behind the reported row.
SCA_CKPT="${SCA_MISSING_CKPT:-workdir_pretrain/t9_qweight_only/ckpt/model_step_5330.pt}"
[ -f "$SCA_CKPT" ] || { echo "FATAL: SCA checkpoint not at $SCA_CKPT (set SCA_MISSING_CKPT)" >&2
  exit 2; }
# GRAM: no default guess -- the released checkpoint is not in this repo, and silently
# substituting a reproduction would relabel it as the released model.
if [ -z "${GRAM_RELEASED_CKPT:-}" ]; then
  echo "FATAL: set GRAM_RELEASED_CKPT to the released GRAM checkpoint -- the same file the" >&2
  echo "       GRAM* rows of Table 1/2 used." >&2; exit 2
fi
[ -f "$GRAM_RELEASED_CKPT" ] || {
  echo "FATAL: GRAM_RELEASED_CKPT=$GRAM_RELEASED_CKPT does not exist" >&2; exit 1; }

# the gram configs are generated; refuse to run from a stale set
python3 scripts/make_missing_configs.py || exit 2

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for model in sca gram; do
  ckpt="$SCA_CKPT"; [ "$model" = gram ] && ckpt="$GRAM_RELEASED_CKPT"
  for rate in r00 r25 r50 r75 r90; do
    cfg="benchmark_eval/configs_missing/${rate}/${model}_${BENCH}.json"
    [ -f "$cfg" ] || { echo "== [$model/$BENCH/$rate] SKIP: no $cfg" >&2; rc_all=2; continue; }
    out="workdir/e1_missing/${model}_${BENCH}_${rate}"
    cell_is_done "$out" "$cfg" && { echo "== [$model/$BENCH/$rate] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$model/$BENCH/$rate] START $(date +%T)  cfg=$cfg  ckpt=$ckpt"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node 4 --master_port $((9500 + IDX)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$model/$BENCH/$rate] OK $(date +%T)"
    else echo "== [$model/$BENCH/$rate] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo
echo "CONTROL: the r00 cells must reproduce Table 1/2 for their model (masking off is"
echo "         byte-identical to the standard eval path). If they do not, the sweep is"
echo "         mis-wired and NO rate is reportable."
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
