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
#SBATCH --job-name=hgeval
#SBATCH --array=0-4
#SBATCH -o ./slurm_scripts/logs/hgeval_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/hgeval_%A_%a.out
# The HyperGRAM checkpoint we trained from THEIR code, evaluated on OUR protocol.
#
# The PMRL authors publish their code (github.com/Xiaohao-Liu/PMRL) and their trained weights
# (huggingface.co/xhLiu/PMRL, model_ckpts/pmrl_base.pt, 5.6 GB), so this row needs no training
# and no reimplementation -- the same standing as the GRAM released-checkpoint row. It is
# strictly stronger than running HyperGram's PMRL reimplementation, which has no PMRL config
# and would inherit HyperGRAM's recipe.
#
# SETUP, once, on a LOGIN node (compute nodes have no internet):
#
#   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Xiaohao-Liu/PMRL \
#     "$WORK_ROOT/pmrl"
#   HF_HUB_OFFLINE=0 huggingface-cli download xhLiu/PMRL model_ckpts/pmrl_base.pt \
#     --local-dir "$WORK_ROOT/pmrl_weights"
#
#   HF_HUB_OFFLINE=0 is load-bearing: $MODELS_DIR/env.sh sets HF_HUB_OFFLINE=1 so compute
#   nodes never reach for the network, and with it set the download reports
#   LocalEntryNotFoundError ("check your connection") rather than an offline-mode error.
#
#   sbatch slurm_scripts/pmrl_released.sh                 # all five benchmarks
#   sbatch --array=0 slurm_scripts/pmrl_released.sh       # MSR-VTT alone
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
# PYTHONUNBUFFERED: their stdout is a PIPE here (the noise filter), so python block-
# buffers it and a running job looks dead -- `tail` shows nothing for many minutes
# while 4-8KB accumulates. Unbuffered output is the difference between monitoring a
# run and guessing at it.
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTHONUNBUFFERED=1
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v claim_outdir >/dev/null || {
  echo "FATAL: sourced $HELPER but claim_outdir is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

BENCHES=(msrvtt didemo activitynet vatex audiocaps)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-4." >&2; exit 2; }
BENCH="${BENCHES[$IDX]:-}"
[ -n "$BENCH" ] || { echo "FATAL: index $IDX out of range (0-4)" >&2; exit 2; }

HG_ROOT="${HYPERGRAM_ROOT:-$WORK_ROOT/hypergram}"
# The checkpoint their save_best wrote on the REPORTED metric. Overriding this with the
# aggregator-selected file, or with a model_step_*, changes what the row means -- the
# generator prints which one it got and records it in the config.
HG_CKPT="${HYPERGRAM_CKPT:-$CODE_DIR/workdir_pretrain/hgauth_hybrid/ckpt/best_ret%tvas--msrvtt_ret_ret_itm_area.pt}"
[ -d "$HG_ROOT/configs/pretrain" ] || {
  echo "FATAL: $HG_ROOT is not a HyperGram checkout." >&2; exit 2; }
[ -f "$HG_CKPT" ] || {
  echo "FATAL: no trained checkpoint at" >&2
  echo "         $HG_CKPT" >&2
  echo "       Train it first with slurm_scripts/hypergram_authors.sh, or set HYPERGRAM_CKPT." >&2
  echo "       Available:" >&2
  ls -1 "$CODE_DIR/workdir_pretrain/hgauth_hybrid/ckpt" 2>/dev/null | sed 's/^/         /' >&2
  exit 2; }

# evaluation_tools and pretrained_weights are already linked by hypergram_authors.sh; verify
# rather than assume, because this job can be run on a fresh checkout.
for dep in evaluation_tools pretrained_weights datasets; do
  if [ ! -e "$HG_ROOT/$dep" ]; then
    ln -sfn "$CODE_DIR/$dep" "$HG_ROOT/$dep" || {
      echo "FATAL: could not link $dep into $HG_ROOT" >&2; exit 2; }
    echo "linked $dep"
  fi
  [ -e "$HG_ROOT/$dep" ] || { echo "FATAL: $HG_ROOT/$dep does not resolve" >&2; exit 2; }
done
( cd "$HG_ROOT" && python3 -c "import evaluation_tools" ) 2>/dev/null || {
  echo "FATAL: evaluation_tools does not import from $HG_ROOT" >&2; exit 2; }
export PYTHONPATH="$HG_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DIRTY=$(git -C "$HG_ROOT" status --porcelain \
          -- ':!configs/pretrain/repro_*' ':!configs/pretrain/hgeval_*' ':!evaluation_tools' \
             ':!pretrained_weights' ':!datasets' ':!*__pycache__*' 2>/dev/null | head -5)
if [ -n "$DIRTY" ]; then
  echo "FATAL: $HG_ROOT has local modifications:" >&2; echo "$DIRTY" >&2; exit 2
fi
echo "authors' code : $HG_ROOT @ $(git -C "$HG_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"

CFG_REL="configs/pretrain/hgeval_${BENCH}.json"
python3 scripts/make_hypergram_eval_config.py --hypergram_root "$HG_ROOT" \
  --checkpoint "$HG_CKPT" --bench "$BENCH" || exit 2

OUT="workdir/hgeval/$BENCH"
mkdir -p "$OUT"
claim_outdir "$OUT" || exit 2
echo "bench  : $BENCH"
echo "START=$(date +%T)"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
( cd "$HG_ROOT" && srun --chdir="$HG_ROOT" python3 -m torch.distributed.launch --nnodes 1 \
    --node_rank 0 --nproc_per_node 4 --master_port $((9600 + IDX)) \
    "$CODE_DIR/scripts/run_with_forkserver.py" ./run.py \
    --config "$CFG_REL" --output_dir "$CODE_DIR/$OUT" 2>&1 ) \
  | tee "$OUT/run.log" | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}
echo "EXIT=$rc DONE $(date +%T)"
if [ $rc -ne 0 ]; then
  echo "[$BENCH] their run.py exited $rc -- the real error is above this line." >&2
  exit $rc
fi
python3 scripts/parse_authors_eval.py "$OUT/run.log"
