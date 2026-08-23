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
#SBATCH --job-name=hgauth
#SBATCH --array=0-3
#SBATCH -o ./slurm_scripts/logs/hgauth_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/hgauth_%A_%a.out
# The baselines, run from the AUTHORS' code rather than our reimplementation of it.
#
# github.com/uta-smile/HyperGram is released, and it is the same VAST/GRAM fork we build on.
# Our own gram_hyp differs from it in six substantive ways -- no learnable curvature, no
# curvature learning-rate group, and above all no scale matching between the Euclidean and
# hyperbolic volumes before they are mixed (experiments/results/HYPERGRAM_STATUS.md). Numbers
# from our reimplementation are therefore not evidence about their method, and the 37.4 that
# circulated as "our HyperGRAM reproduction does not work" is retracted.
#
# Their repo also implements PMRL, so all four geometries come from one codebase:
#
#   0  hybrid        HyperGRAM as published (their paper config, unchanged)
#   1  pmrl          their PMRL
#   2  pmrl_volume   their PMRL volume variant
#   3  hybrid_pmrl   their PMRL with hybrid-space SVD
#
# RECIPE CAVEAT that must travel with rows 1-3. Only the hybrid config ships with the repo, so
# the pmrl* modes inherit HyperGRAM's recipe: lr 5e-05, one epoch, task ret%tvas%tv%ta, PMRL
# defaults lambda1 1.0 / lambda2 0.1 / tau 0.07. They are THEIR IMPLEMENTATION AT HYPERGRAM'S
# RECIPE, never PMRL's published setup, and must be labelled that way.
#
# Nothing here edits their code. scripts/make_hypergram_config.py rewrites dataset and
# checkpoint paths only, then asserts that learning rate, epochs, batch, task and every
# geometry hyperparameter survived the rewrite -- a path edit that quietly moved one of those
# would produce a number labelled "authors' code" that is not.
#
# SETUP, once, on the cluster:
#   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/uta-smile/HyperGram \
#     "$WORK_ROOT/hypergram"
#
#   sbatch slurm_scripts/hypergram_authors.sh              # all four in parallel
#   sbatch --array=0 slurm_scripts/hypergram_authors.sh    # HyperGRAM alone
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v claim_outdir >/dev/null || {
  echo "FATAL: sourced $HELPER but claim_outdir is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

MODES=(hybrid pmrl pmrl_volume hybrid_pmrl)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-3 for one mode." >&2; exit 2; }
MODE="${MODES[$IDX]:-}"
[ -n "$MODE" ] || { echo "FATAL: index $IDX out of range (0-3)" >&2; exit 2; }

HG_ROOT="${HYPERGRAM_ROOT:-$WORK_ROOT/hypergram}"
[ -d "$HG_ROOT/configs/pretrain" ] || {
  echo "FATAL: $HG_ROOT is not a HyperGram checkout. Clone it first:" >&2
  echo "  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/uta-smile/HyperGram \\" >&2
  echo "    \"$HG_ROOT\"" >&2
  echo "  (or set HYPERGRAM_ROOT)" >&2; exit 2; }

# Their repo omits evaluation_tools/, the vendored caption-eval package (pycocoevalcap and
# friends) that both forks inherit from VAST. evaluation/evaluation_mm.py imports it at module
# level, so their code will not import at all without it -- this is a packaging omission on
# their side, not a difference in method. Supplying ours as a SYMLINK adds a missing dependency
# without editing a line of their code.
[ -d "$CODE_DIR/evaluation_tools" ] || {
  echo "FATAL: $CODE_DIR/evaluation_tools missing -- nothing to supply from" >&2; exit 2; }
if [ ! -e "$HG_ROOT/evaluation_tools" ]; then
  ln -s "$CODE_DIR/evaluation_tools" "$HG_ROOT/evaluation_tools" || {
    echo "FATAL: could not link evaluation_tools into $HG_ROOT (read-only checkout?)" >&2
    exit 2; }
  echo "linked evaluation_tools -> $CODE_DIR/evaluation_tools (their repo omits it)"
else
  echo "evaluation_tools already present in $HG_ROOT"
fi
# Verify it actually IMPORTS from their root. The previous attempt created the link and still
# died on ModuleNotFoundError, so existence on disk is not the property that matters --
# importability from the directory their run.py executes in is. Checked here, loudly, rather
# than discovered four ranks deep in a torchrun traceback.
( cd "$HG_ROOT" && python3 -c "import evaluation_tools" ) 2>/dev/null || {
  echo "FATAL: evaluation_tools is on disk at $HG_ROOT but does not import from there." >&2
  echo "       ls -l $HG_ROOT/evaluation_tools" >&2
  ls -l "$HG_ROOT/evaluation_tools" >&2
  echo "       (a dangling symlink, or a package with no __init__.py, looks exactly like" >&2
  echo "        this from inside torchrun)" >&2
  exit 2; }
# srun does not necessarily inherit the subshell's cwd, and sys.path[0] under
# `python3 -m torch.distributed.launch` is whatever cwd the RANKS start in -- which is why the
# link on disk was not enough. Naming their root explicitly makes the import independent of
# that. HG_ROOT goes FIRST so their modules still shadow ours everywhere else.
export PYTHONPATH="$HG_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Their repo must otherwise be pristine. A local edit would make this our code again under
# their name, which is the whole thing this job exists to avoid. The two exclusions are the
# configs we generate and the dependency symlink above -- neither is a change to their method.
DIRTY=$(git -C "$HG_ROOT" status --porcelain \
          -- ':!configs/pretrain/repro_*' ':!evaluation_tools' 2>/dev/null | head -5)
if [ -n "$DIRTY" ]; then
  echo "FATAL: $HG_ROOT has local modifications outside the generated configs:" >&2
  echo "$DIRTY" >&2
  echo "       These results would be OUR code under THEIR name. Reset the checkout." >&2
  exit 2
fi
echo "authors' code : $HG_ROOT @ $(git -C "$HG_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"

CFG_REL="configs/pretrain/repro_${MODE}_ours_paths.json"
python3 scripts/make_hypergram_config.py --hypergram_root "$HG_ROOT" \
  --geometry_mode "$MODE" ${SCA_HG_ALLOW_ANNO:+--allow_annotation_mismatch} || exit 2

OUT="workdir_pretrain/hgauth_${MODE}"
mkdir -p "$OUT"
claim_outdir "$OUT" || exit 2

echo "mode   : $MODE"
echo "config : $HG_ROOT/$CFG_REL"
echo "outdir : $OUT"
python3 -c "
import json
c = json.load(open('$HG_ROOT/$CFG_REL'))
n = c['_repro_note']
print('lr     : %s' % c['run_cfg']['learning_rate'])
print('epochs : %s  batch %s' % (c['data_cfg']['train'][0]['epoch'],
                                 c['data_cfg']['train'][0]['batch_size']))
print('task   : %s' % c['data_cfg']['train'][0]['task'])
print('geom   : %s' % c['model_cfg']['geometry_mode'])
print('anno   : %s' % n['annotations'])"
echo "START=$(date +%T)"

# run THEIR run.py from THEIR root, with our output dir
NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
( cd "$HG_ROOT" && srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node 4 --master_port $((9800 + IDX)) \
    ./run.py --config "$CFG_REL" --output_dir "$CODE_DIR/$OUT" --checkpointing true 2>&1 ) \
  | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}
echo "EXIT=$rc DONE $(date +%T)"
if [ $rc -ne 0 ]; then
  echo "[$MODE] failed. Their trunk expects the same VAST checkpoint and encoder weights we" >&2
  echo "        use; if it cannot find a model file, check \$MODELS_DIR/env.sh covers it." >&2
fi
exit $rc
