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
#SBATCH --job-name=sca_run
#SBATCH -o ./slurm_scripts/logs/run_%j.out
#SBATCH -e ./slurm_scripts/logs/run_%j.out
# Generic pretrain/finetune launcher: one script for every config in the campaign.
#
#   sbatch slurm_scripts/run_config.sh <config.json> <output_dir> [extra run.py args...]
#
# Submit through scripts/submit_recipe_runs.sh rather than by hand: it passes -J/-o/-e so
# the job and its log carry the arm's name, and --dependency=singleton so a resubmission
# queues behind the running job instead of training a second process into the same workdir.
#
# Auto-resumes from the newest optimizer checkpoint in <output_dir>/ckpt. If the 24h wall
# clock is still shorter than the full 150k epoch-5 pretrain, RESUBMIT THE SAME COMMAND --
# each resubmission continues where the last stopped, and --dependency=singleton makes a
# resubmission queue behind the running job rather than racing it.
# Requires $WORK/sca_env.rc (DATA_ROOT/WORK_ROOT/SCA_CACHE_ROOT/MODELS_DIR).
set -uo pipefail
CONFIG="${1:?usage: run_config.sh <config.json> <output_dir> [extra args]}"
OUTDIR="${2:?usage: run_config.sh <config.json> <output_dir> [extra args]}"
shift 2
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: config $CONFIG not found" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p "$OUTDIR" slurm_scripts/logs

# --- provenance guard -------------------------------------------------------------------
# This script auto-resumes from whatever checkpoints sit in $OUTDIR. Without a check that
# those checkpoints came from THIS config, pointing a second config at an existing outdir
# silently continues the first run's weights and the resulting numbers belong to neither
# arm. Stamp the outdir on the first run; refuse to resume when the stamp disagrees.
FP=$(python3 - "$CONFIG" "$@" <<'PY'
import hashlib, json, os, sys
h = hashlib.sha256()
def feed(p, seen):
    p = p.lstrip('./')
    if p in seen or not os.path.exists(p):
        return
    seen.add(p)
    raw = open(p, 'rb').read()
    h.update(raw)
    try:
        d = json.loads(raw)
    except Exception:
        return
    for sec in ('run_cfg', 'model_cfg'):
        dflt = (d.get(sec) or {}).get('default')
        if dflt:
            feed(dflt, seen)          # the resolved chain matters, not just the leaf file
feed(sys.argv[1], set())
h.update(('\0'.join(sys.argv[2:])).encode())   # --seed and friends change the run too
print(h.hexdigest()[:16])
PY
)
# --- exclusive lock ----------------------------------------------------------------------
# --dependency=singleton only protects jobs submitted with the same -J. A hand-submitted
# sbatch, or a job name typo, would put a second process into these checkpoints. mkdir is
# atomic on POSIX and on Lustre, so it is a sound mutex here.
LOCK="$OUTDIR/.lock"
ME="${SLURM_JOB_ID:-$$}"
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER=$(cat "$LOCK/jobid" 2>/dev/null || echo '?')
  ALIVE=no
  if [ "$OWNER" != '?' ] && command -v squeue >/dev/null 2>&1; then
    squeue -h -j "$OWNER" -o '%T' 2>/dev/null \
      | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING' && ALIVE=yes
  fi
  if [ "$ALIVE" = yes ]; then
    echo "FATAL: $OUTDIR is locked by job $OWNER, which is still active." >&2
    echo "       Refusing to run a second process into the same checkpoints." >&2
    exit 4
  fi
  echo "WARN: stale lock from job $OWNER (not in the queue) -- reclaiming"
  rm -rf "$LOCK"
  mkdir "$LOCK" || { echo "FATAL: cannot acquire $LOCK" >&2; exit 4; }
fi
echo "$ME" > "$LOCK/jobid"
# EXIT covers normal and error exits; TERM/INT cover the signal slurm sends at the wall
# clock, so a timed-out run releases its lock instead of blocking the next resubmission.
trap 'rm -rf "$LOCK"' EXIT TERM INT
# ------------------------------------------------------------------------------------------

STAMP="$OUTDIR/.provenance"
RESUME=""
if ls "$OUTDIR"/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  if [ ! -f "$STAMP" ]; then
    echo "FATAL: $OUTDIR holds checkpoints but no .provenance stamp -- refusing to resume" >&2
    echo "       into weights of unknown origin. Point at a fresh outdir, or if you are" >&2
    echo "       certain these are this config's, write the stamp by hand:" >&2
    echo "         printf 'config=%s\\nfingerprint=%s\\nargs=%s\\n' '$CONFIG' '$FP' '$*' > '$STAMP'" >&2
    exit 3
  fi
  HAVE=$(sed -n 's/^fingerprint=//p' "$STAMP")
  if [ "$HAVE" != "$FP" ]; then
    echo "FATAL: $OUTDIR was created by a DIFFERENT config -- refusing to mix runs." >&2
    echo "       stamped: $(sed -n 's/^config=//p' "$STAMP") (fingerprint $HAVE)" >&2
    echo "       asked:   $CONFIG (fingerprint $FP)" >&2
    echo "       Use a different output_dir for this arm." >&2
    exit 3
  fi
  RESUME="--resume true"; echo "RESUME from $OUTDIR/ckpt (provenance ok: $FP)"
else
  printf 'config=%s\nfingerprint=%s\nargs=%s\ncreated=%s\n' \
    "$CONFIG" "$FP" "$*" "$(date -Is)" > "$STAMP"
  echo "FRESH (provenance stamped: $FP)"
fi
# ----------------------------------------------------------------------------------------
echo "START $(date +%T)  config=$CONFIG  out=$OUTDIR"
# filter the benign h264 decoder noise (mmco/*ref* warnings from mid-GOP YouTube clip
# cuts -- harmless, but hundreds of MB over a full pretrain). pipefail + `|| true` on the
# grep keeps the pipeline's exit code = srun's.
NPROC="${SLURM_GPUS_ON_NODE:-${SCA_GPUS:-4}}"
echo "nproc_per_node=$NPROC (global batch from the config is split across these)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node "$NPROC" \
  --master_port $((9000 + ${SLURM_JOB_ID:-$$} % 900)) \
  ./run.py --config "$CONFIG" --output_dir "$OUTDIR" --checkpointing true $RESUME "$@" 2>&1 \
  | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($" || true; }
rc=$?
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
