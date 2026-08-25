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
#SBATCH --job-name=vgg_eval
#SBATCH --array=0-3
#SBATCH -o ./slurm_scripts/logs/vgg_eval_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/vgg_eval_%A_%a.out
# VGGSound-5K (T-A-V, 5000 clips, class labels as queries) at the reported configuration.
#
# Scope, honestly: only GRAM publishes a number here (40.6/78.1 Acc@1/@10; our environment
# measured its released checkpoint at 38.3/76.3 through this trunk in wave 1 -- a fourth
# environment-shift data point). PMRL's and HyperGRAM's papers report no VGGSound at all,
# so this is a GRAM-vs-SCA comparison for the audio-anchored discussion, not a Tables-1/2
# benchmark. Arms: T9's three seeds + the released GRAM checkpoint, e1 geometry byte-for-
# byte (configs generated from the didemo templates; model_cfg drift-checked in tests).
#
#   VGG5K_ROOT=/path/to/vggsound_5k \
#   GRAM_RELEASED_CKPT=/path/to/released.pt sbatch slurm_scripts/vggsound_eval.sh
#
# VGG5K_ROOT must hold videos/ and audios/. The wave-1 run read another user's scratch
# (/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vggsound_5k)
# -- scratch is purged, so VERIFY it still exists or point at a copy; no default is guessed.
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

: "${VGG5K_ROOT:?set VGG5K_ROOT to the vggsound_5k directory holding videos/ and audios/}"
export VGG5K_ROOT
for d in videos audios; do
  [ -d "$VGG5K_ROOT/$d" ] || { echo "FATAL: $VGG5K_ROOT/$d missing -- the wave-1 data" >&2
    echo "       lived in another user's scratch and may have been purged." >&2; exit 2; }
done
[ -f benchmark_eval/vgg5k_annotation_5000.json ] || {
  echo "FATAL: benchmark_eval/vgg5k_annotation_5000.json missing" >&2; exit 2; }

ARMS=(t9_qweight_only s1_t9_seed51 s2_t9_seed52 gram_released)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-3." >&2; exit 2; }
ARM="${ARMS[$IDX]:-}"
[ -n "$ARM" ] || { echo "FATAL: index $IDX out of range (0-3)" >&2; exit 2; }

if [ "$ARM" = gram_released ]; then
  if [ -z "${GRAM_RELEASED_CKPT:-}" ]; then
    echo "FATAL: set GRAM_RELEASED_CKPT -- same file as the GRAM* rows of Tables 1/2." >&2
    exit 2
  fi
  CKPT="$GRAM_RELEASED_CKPT"
  CFG="benchmark_eval/configs_e1/gram_vggsound.json"
  OUT="workdir/e1_vgg/released_vggsound"
else
  CKPT="workdir_pretrain/${ARM}/ckpt/model_step_5330.pt"
  CFG="benchmark_eval/configs_qweight/sca_vggsound.json"
  OUT="workdir/e1_vgg/${ARM}_vggsound"
fi
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not at $CKPT" >&2; exit 2; }
[ -f "$CFG" ] || { echo "FATAL: config not at $CFG" >&2; exit 2; }

cell_is_done "$OUT" "$CFG" && { echo "== [$ARM/vggsound] already done, skip"; exit 0; }
mkdir -p "$OUT"
NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
echo "== [$ARM/vggsound] START $(date +%T)  ckpt=$CKPT"
EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
  --nproc_per_node 4 --master_port $((9900 + IDX)) \
  ./benchmark_eval/run_eval.py --config "$CFG" --output_dir "$OUT" 2>&1 \
  | tee "$OUT/eval.log" | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ] && [ "$ARM" = gram_released ]; then
  python3 scripts/verify_ckpt_load.py "$OUT/eval.log" || {
    echo "== [$ARM/vggsound] LOAD NOT VERIFIED -- cell refused" >&2; rc=3; }
fi
if [ $rc -eq 0 ]; then cell_mark_done "$OUT" "$CFG"; echo "== [$ARM/vggsound] OK $(date +%T)"
else echo "== [$ARM/vggsound] FAILED rc=$rc" >&2; fi
echo "SELF-CHECK: released_vggsound should land near the wave-1 measurement 38.3/76.3"
echo "            (published 40.6/78.1); a large deviation means the protocol drifted."
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
