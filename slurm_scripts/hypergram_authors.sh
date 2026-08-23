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

# Their repo omits two directories that both forks inherit from VAST, and neither omission is
# a difference in method -- they are things the published tarball simply does not carry:
#
#   evaluation_tools/    the vendored caption-eval package (pycocoevalcap and friends).
#                        evaluation/evaluation_mm.py imports it at module level, so their code
#                        does not import at all without it.
#   pretrained_weights/  the encoder checkpoints their own configs/default_model_cfg.json
#                        names. model/general_module.py and model/gram.py load them by
#                        RELATIVE path (./pretrained_weights/...), so they must exist under
#                        their root, not merely somewhere on the system.
#
# Supplying ours as SYMLINKS adds the missing dependencies without editing a line of their
# code. Copying files in, or patching their paths, would make the run our code under their
# name -- which is the one thing this job exists to avoid.
link_dep() {                          # link_dep <dirname> <what it is>
  local name="$1" what="$2"
  [ -e "$CODE_DIR/$name" ] || {
    echo "FATAL: $CODE_DIR/$name missing -- nothing to supply from ($what)" >&2; return 1; }
  if [ ! -e "$HG_ROOT/$name" ]; then
    ln -s "$CODE_DIR/$name" "$HG_ROOT/$name" || {
      echo "FATAL: could not link $name into $HG_ROOT (read-only checkout?)" >&2; return 1; }
    echo "linked $name -> $CODE_DIR/$name (their repo omits it)"
  else
    echo "$name already present in $HG_ROOT"
  fi
}
link_dep evaluation_tools "the vendored caption-eval package" || exit 2
link_dep pretrained_weights "the encoder checkpoints their configs name" || exit 2
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
          -- ':!configs/pretrain/repro_*' ':!evaluation_tools' ':!pretrained_weights' \
          2>/dev/null | head -5)
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

# The encoder checkpoints, checked HERE rather than 38 seconds into a four-way array. Their
# code loads them by relative path from inside model construction, so a missing file surfaces
# as a torchrun traceback on every rank after the data pipeline has already spun up. Which
# files are needed is read from the RESOLVED config (their defaults, then this run's
# model_cfg) rather than assumed: a config naming a different encoder must be checked against
# the file THAT encoder loads, and an encoder name with no known file is fatal, not skipped.
python3 - "$HG_ROOT" "$CFG_REL" <<'PREFLIGHT' || exit 2
import json, os, sys
root, cfg_rel = sys.argv[1], sys.argv[2]
VISION = {'evaclip01_giant':        'clip/EVA01_CLIP_g_14_psz14_s11B.pt',
          'evaclip02_base':         'clip/EVA02_CLIP_B_psz16_s8B.pt',
          'evaclip02_base_self':    'clip/EVA02_B_psz14to16.pt',
          'evaclip02_large':        'clip/EVA02_CLIP_L_psz14_s4B.pt',
          'evaclip02_bige':         'clip/EVA02_CLIP_E_psz14_plus_s9B.pt',
          'clip_vit_base_16':       'clip/ViT-B-16.pt',
          'clip_vit_base_32':       'clip/ViT-B-32.pt',
          'clip_vit_large_14_336px': 'clip/ViT-L-14-336px.pt',
          'videoswin_base_k600_22k': 'videoswin_base_k600_22k.pth'}
AUDIO = {'beat': 'beats/BEATs_iter3_plus_AS2M.pt', 'ast': 'audioset_10_10_0.4593.pth'}

mcfg = json.load(open(os.path.join(root, 'configs/default_model_cfg.json')))
mcfg.update(json.load(open(os.path.join(root, cfg_rel))).get('model_cfg', {}))
vt, at = mcfg.get('vision_encoder_type'), mcfg.get('audio_encoder_type')

need = {'text/multimodal (BertForMaskedLM + tokenizer)': 'bert/bert-base-uncased'}
if vt not in VISION:
    sys.exit('FATAL: vision_encoder_type %r is not one this pre-flight knows a weight file\n'
             '       for. Add it from their model/general_module.py::load_clip_model rather\n'
             '       than letting the run find out.' % vt)
need['vision (%s)' % vt] = VISION[vt]
audio = [f for p, f in AUDIO.items() if (at or '').startswith(p)]
if not audio:
    sys.exit('FATAL: audio_encoder_type %r matches neither "beat*" nor "ast*", the two\n'
             '       branches in their construct_audio_encoder.' % at)
need['audio (%s)' % at] = audio[0]

missing = []
for what, rel in sorted(need.items()):
    p = os.path.join(root, 'pretrained_weights', rel)
    print('  %-46s %s %s' % (what, 'OK  ' if os.path.exists(p) else 'MISS', rel))
    if not os.path.exists(p):
        missing.append(p)
if missing:
    sys.exit('FATAL: their config names encoders whose weights are not under\n'
             '       %s/pretrained_weights/ :\n         %s\n'
             '       That directory is a symlink to $CODE_DIR/pretrained_weights, so the file\n'
             '       has to be fetched there -- scripts/prefetch_models.py reports which of\n'
             '       these it can download and which must come from the official release.'
             % (root, '\n         '.join(missing)))
PREFLIGHT

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

# Run THEIR run.py from THEIR root, with our output dir. srun --chdir is what actually puts
# the RANKS in $HG_ROOT: the subshell's `cd` sets the cwd of srun itself, and every relative
# path their code hardcodes (./pretrained_weights/..., ./configs/...) is resolved inside the
# ranks. The `cd` stays because $CFG_REL is passed relative.
NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
( cd "$HG_ROOT" && srun --chdir="$HG_ROOT" python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
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
