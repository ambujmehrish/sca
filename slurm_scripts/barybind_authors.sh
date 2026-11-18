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
#SBATCH --job-name=bbauth
#SBATCH --array=0-1
#SBATCH -o ./slurm_scripts/logs/bbauth_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/bbauth_%A_%a.out
# BaryBind's released checkpoint, run by BaryBind's own code, on our annotation files and media.
#
#   index 0  fs8   8 test frames on every benchmark (our protocol)
#   index 1  fsbb  BaryBind's test frames: MSR-VTT 8, DiDeMo 32, ActivityNet 32, VATEX 16, AudioCaps 8
#
# Setup, once, on a login node:
#   git clone --depth 1 https://github.com/xl-tang3/BaryBind "$WORK_ROOT/barybind"
#   pip install gdown && gdown --folder \
#     https://drive.google.com/drive/folders/1poY_-gGBng8XDNsUGXqntQbWaRC-PpWU \
#     -O "$WORK_ROOT/barybind_ckpt"       # must contain log/hps.json and ckpt/model_step_*.pt
#
#   BB_CKPT_DIR=$WORK_ROOT/barybind_ckpt/barybind_pretrained_4modalities sbatch slurm_scripts/barybind_authors.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTHONUNBUFFERED=1
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced scripts/cell_done.sh but cell_is_done is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

FRAMESETS=(fs8 fsbb)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
FS="${FRAMESETS[$IDX]:-}"
[ -n "$FS" ] || { echo "FATAL: array index '$IDX' is not 0 (fs8) or 1 (fsbb)" >&2; exit 2; }

BB_ROOT="${BARYBIND_ROOT:-$WORK_ROOT/barybind}"
[ -f "$BB_ROOT/run.py" ] && [ -f "$BB_ROOT/config/barybind/default_model_cfg.json" ] || {
  echo "FATAL: $BB_ROOT is not a BaryBind checkout (see the setup lines at the top)" >&2; exit 2; }
: "${BB_CKPT_DIR:?set BB_CKPT_DIR to the released model directory (holds log/ and ckpt/)}"
[ -f "$BB_CKPT_DIR/log/hps.json" ] || { echo "FATAL: $BB_CKPT_DIR/log/hps.json missing" >&2; exit 2; }
BB_CKPT=$(ls "$BB_CKPT_DIR"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$BB_CKPT" ] || { echo "FATAL: no ckpt/model_step_*.pt under $BB_CKPT_DIR" >&2; exit 2; }
echo "BaryBind code : $BB_ROOT @ $(git -C "$BB_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "BaryBind ckpt : $BB_CKPT  (their loader takes the highest model_step in that folder)"

# their repo omits the vendored caption-eval package and the encoder weights; link ours in
link_dep() {
  local dst="$BB_ROOT/$1" src="$CODE_DIR/$1"
  [ -e "$src" ] || { echo "FATAL: $src missing" >&2; return 1; }
  if [ -L "$dst" ] && [ ! -e "$dst" ]; then rm -f "$dst"; fi
  [ -e "$dst" ] || ln -s "$src" "$dst" || { echo "FATAL: cannot link $1 into $BB_ROOT" >&2; return 1; }
  [ -e "$dst" ] || { echo "FATAL: $dst does not resolve" >&2; return 1; }
}
link_dep evaluation_tools || exit 2
link_dep pretrained_weights || exit 2
for f in clip/EVA01_CLIP_g_14_psz14_s11B.pt beats/BEATs_iter3_plus_AS2M.pt bert/bert-base-uncased; do
  [ -e "$BB_ROOT/pretrained_weights/$f" ] || { echo "FATAL: encoder weight $f missing under pretrained_weights" >&2; exit 2; }
done
export PYTHONPATH="$BB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
( cd "$BB_ROOT" && python3 -c "import evaluation_tools; from model import model_registry; assert 'barybind' in model_registry" ) \
  || { echo "FATAL: BaryBind's model package does not import in this environment" >&2; exit 2; }

# their checkout must be pristine apart from the links and their own run outputs
DIRTY=$(git -C "$BB_ROOT" status --porcelain -- ':!evaluation_tools' ':!pretrained_weights' \
          ':!results' ':!wandb' ':!output' 2>/dev/null | head -5)
[ -z "$DIRTY" ] || { echo "FATAL: $BB_ROOT has local modifications:" >&2; echo "$DIRTY" >&2; exit 2; }

CFG_DIR="$CODE_DIR/workdir/barybind_cfg"
python3 scripts/make_parity_configs.py barybind --out "$CFG_DIR" >/dev/null || exit 2

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for bench in msrvtt didemo activitynet vatex audiocaps; do
  cfg="$CFG_DIR/$FS/$bench.json"
  out="$CODE_DIR/workdir/parity/barybind_${FS}_${bench}"
  cell_is_done "$out" "$cfg" && { echo "== [barybind/$FS/$bench] already done, skip"; continue; }
  mkdir -p "$out"
  echo "== [barybind/$FS/$bench] START $(date +%T)  cfg=$cfg"
  ( cd "$BB_ROOT" && srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" --master_port $((9400 + IDX * 10 + ${SLURM_JOB_ID:-$$} % 10)) \
      ./run.py --config "$cfg" --pretrain_dir "$BB_CKPT_DIR" --output_dir "$out" --mode testing ) 2>&1 \
    | { grep -v --line-buffered -E "$NOISE" || true; } | tee "$out/run.log"
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ] && grep -q "ret_itm_area" "$out/run.log"; then
    cell_mark_done "$out" "$cfg"; echo "== [barybind/$FS/$bench] OK $(date +%T)"
  else echo "== [barybind/$FS/$bench] FAILED rc=$rc (or no ret_itm_area block)" >&2; rc_all=1; fi
done
echo
echo "  python3 scripts/summarize_parity.py"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
