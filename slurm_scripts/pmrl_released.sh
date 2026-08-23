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
#SBATCH --job-name=pmrlrel
#SBATCH --array=0-4
#SBATCH -o ./slurm_scripts/logs/pmrlrel_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/pmrlrel_%A_%a.out
# PMRL's RELEASED checkpoint, evaluated in our environment on our protocol.
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

PM_ROOT="${PMRL_ROOT:-$WORK_ROOT/pmrl}"
PM_CKPT="${PMRL_CKPT:-$WORK_ROOT/pmrl_weights/model_ckpts/pmrl_base.pt}"
[ -d "$PM_ROOT/config/pmrl" ] || {
  echo "FATAL: $PM_ROOT is not a PMRL checkout. Clone it first:" >&2
  echo "  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Xiaohao-Liu/PMRL \\" >&2
  echo "    \"$PM_ROOT\"   (or set PMRL_ROOT)" >&2; exit 2; }
[ -f "$PM_CKPT" ] || {
  echo "FATAL: released checkpoint not at $PM_CKPT. Fetch it on a LOGIN node:" >&2
  echo "  HF_HUB_OFFLINE=0 huggingface-cli download xhLiu/PMRL model_ckpts/pmrl_base.pt \\" >&2
  echo "    --local-dir \"$WORK_ROOT/pmrl_weights\"   (5.6 GB; or set PMRL_CKPT)" >&2
  echo "  HF_HUB_OFFLINE=0 matters: \$MODELS_DIR/env.sh sets HF_HUB_OFFLINE=1, and the" >&2
  echo "  download then fails as LocalEntryNotFoundError, which reads like a network fault." >&2
  exit 2; }

# ---- the three things their release omits. Symlinks and one stub; no edit to their code.
link_dep() {                          # link_dep <path-in-their-tree> <target> <what it is>
  local dst="$PM_ROOT/$1" src="$2" what="$3"
  [ -e "$src" ] || {
    echo "FATAL: $src missing -- nothing to supply from ($what)" >&2; return 1; }
  # `ln -sfn` rather than a test-then-create. Five array tasks share one checkout, and with a
  # plain `ln -s` they all saw the link absent, all created it, and the losers found
  # config/vast already resolving to a DIRECTORY -- so ln put the link INSIDE it, producing
  # the stray config/pmrl/pmrl. -n treats an existing symlink as a file instead of descending
  # into it, and -f replaces it, so concurrent tasks converge on the same correct link.
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    echo "$1 already present as a real path -> $dst"
  else
    ln -sfn "$src" "$dst" || {
      echo "FATAL: could not link $1 into $PM_ROOT (read-only checkout?)" >&2; return 1; }
    echo "linked $1 -> $src ($what)"
  fi
  [ -e "$dst" ] || { echo "FATAL: $dst does not resolve after linking" >&2; return 1; }
}
# Their configs inherit from ./config/vast/, which the release does not ship. config/pmrl
# carries default_run_cfg.json and default_model_cfg.json under exactly those names, and the
# model default still reads "model_type": "vast" -- it IS the VAST default they forked. The
# link makes the inheritance resolve to the file they meant.
link_dep config/vast "$PM_ROOT/config/pmrl" "their config/vast/ inheritance target" || exit 2
link_dep pretrained_weights "$CODE_DIR/pretrained_weights" "encoder checkpoints" || exit 2
link_dep datasets "$CODE_DIR/datasets" "annotation files; their repo ships no datasets/" || exit 2

# model/__init__.py does `from .vast import VAST`, and model/vast.py is NOT in the release, so
# `import model` fails outright. PMRL subclasses MMGeneralModule only -- VAST is never
# constructed when model_type is pmrl -- so a stub satisfies the import. It RAISES if anything
# ever instantiates it: a stub that silently returned a model would be a fabricated baseline,
# which is worse than the ImportError it replaces.
STUB="$PM_ROOT/model/vast.py"
if [ ! -f "$STUB" ]; then
  # written via a per-task temp file and moved into place: five array tasks writing the same
  # file directly can interleave and leave a truncated module
  STUB_TMP="$STUB.$$.tmp"
  cat > "$STUB_TMP" <<'PYEOF'
"""NOT part of the PMRL release.

model/__init__.py does `from .vast import VAST`, but model/vast.py is not shipped, so
`import model` raises ImportError before anything runs. PMRL subclasses MMGeneralModule and
never touches VAST, so this stub exists solely to satisfy that import.

It refuses to be instantiated on purpose. Returning a working-looking model here would
fabricate a baseline out of nothing -- and utils/build_model.py loads checkpoints with
strict=False, so the fabrication would evaluate without a single error.
"""


class VAST:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'model/vast.py is not shipped with the PMRL release; this is a stub that exists '
            'only so `from .vast import VAST` resolves. Only model_type "pmrl" can be run '
            'from this checkout. If you need VAST, take it from github.com/TXH-mercury/VAST.')
PYEOF
  mv -f "$STUB_TMP" "$STUB" || {
    echo "FATAL: could not install the import stub at $STUB" >&2; rm -f "$STUB_TMP"; exit 2; }
  echo "wrote $STUB (import stub; raises if instantiated)"
else
  echo "model/vast.py already present in $PM_ROOT"
fi
( cd "$PM_ROOT" && python3 -c "import model; assert 'pmrl' in model.model_registry" ) 2>&1 \
  | tail -3 || { echo "FATAL: their model package still does not import from $PM_ROOT" >&2; exit 2; }
export PYTHONPATH="$PM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# __pycache__ is produced by importing their modules -- a byproduct of RUNNING their code,
# not an edit to it. config/pmrl/pmrl is the stray a previous racing launcher could leave.
DIRTY=$(git -C "$PM_ROOT" status --porcelain \
          -- ':!config/pmrl/finetune_cfg/repro_*' ':!config/vast' ':!pretrained_weights' \
             ':!datasets' ':!model/vast.py' ':!*__pycache__*' ':!config/pmrl/pmrl' \
          2>/dev/null | head -5)
if [ -n "$DIRTY" ]; then
  echo "FATAL: $PM_ROOT has local modifications beyond the generated config and the" >&2
  echo "       supplied dependencies:" >&2
  echo "$DIRTY" >&2
  echo "       These results would be OUR code under THEIR name. Reset the checkout." >&2
  exit 2
fi
echo "authors' code : $PM_ROOT @ $(git -C "$PM_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"

CFG_REL="config/pmrl/finetune_cfg/repro_released_${BENCH}.json"
python3 scripts/make_pmrl_config.py --pmrl_root "$PM_ROOT" --checkpoint "$PM_CKPT" \
  --bench "$BENCH" || exit 2

OUT="workdir/pmrl_released/$BENCH"
mkdir -p "$OUT"
claim_outdir "$OUT" || exit 2
echo "bench  : $BENCH"
echo "outdir : $OUT"
echo "START=$(date +%T)"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
( cd "$PM_ROOT" && srun --chdir="$PM_ROOT" python3 -m torch.distributed.launch --nnodes 1 \
    --node_rank 0 --nproc_per_node 4 --master_port $((9700 + IDX)) \
    "$CODE_DIR/scripts/run_with_forkserver.py" ./run.py \
    --config "$CFG_REL" --output_dir "$CODE_DIR/$OUT" 2>&1 ) \
  | tee "$OUT/run.log" | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}
echo "EXIT=$rc DONE $(date +%T)"

# utils/build_model.py loads with strict=False and only LOGS the mismatch, so a checkpoint that
# does not fit the model produces numbers rather than an error. Read its own log back and
# refuse the result if the released weights did not actually land in the model.
#
# Only when the run itself SUCCEEDED. A failed run has no keys to check either, and reporting
# "the log does not record missing/unexpected keys" on top of a crash buries the actual error
# under a message about a different problem.
if [ $rc -ne 0 ]; then
  echo "[$BENCH] their run.py exited $rc -- the checkpoint-load check is skipped because the" >&2
  echo "        run did not get far enough to produce one. The real error is above this line." >&2
  exit $rc
fi
python3 - "$OUT/run.log" <<'PYEOF'
import re, sys
log = open(sys.argv[1], errors='replace').read()
def grab(label):
    m = re.search(re.escape(label) + r'\s*\[(.*?)\]', log, re.S)
    if not m:
        return None
    body = m.group(1).strip()
    return [k.strip().strip("'\"") for k in body.split(',') if k.strip()] if body else []
missing, unexpected = grab('missing_keys'), grab('Unexpected keys')
if missing is None or unexpected is None:
    sys.exit('FATAL: the log does not record missing/unexpected keys, so it cannot be shown\n'
             '       that the released weights loaded. build_model uses strict=False -- an\n'
             '       unverified load is a number from a partly random model.')
print('checkpoint load: %d missing, %d unexpected' % (len(missing), len(unexpected)))
if missing or unexpected:
    for k in missing[:10]:
        print('  MISSING    %s' % k)
    for k in unexpected[:10]:
        print('  UNEXPECTED %s' % k)
    sys.exit('FATAL: the released checkpoint did not fully load. strict=False means those\n'
             '       MISSING parameters kept their random initialisation and were evaluated\n'
             '       as if they were PMRL. This number must not be reported.')
print('the released checkpoint loaded completely.')
PYEOF
vrc=$?
[ $rc -eq 0 ] && rc=$vrc
exit $rc
