#!/bin/bash
# Config-aware per-cell resume markers for the eval launchers.
#
#   source scripts/cell_done.sh
#   cell_is_done "$out" "$cfg" && continue      # skip only if the CONFIG also matches
#   ... run the cell ...
#   cell_mark_done "$out" "$cfg"
#
# Why. The launchers resume by touching "$out/.done" and skipping any cell that has one.
# That is right for a timed-out job, but wrong after a config change: on 2026-08-20 the
# DiDeMo/ActivityNet eval configs were corrected (max_caption_len 40 -> 70) and the re-run
# skipped all eight cells, reproducing the previous numbers byte-for-byte and appearing to
# show the fix made no difference. The marker recorded that the cell had run, not what it
# had run.
#
# So the marker now stores a fingerprint of the config and everything it inherits from. A
# cell is skipped only when its config is unchanged; edit the config and it re-runs by
# itself, with no tags to remember and no directories to clear by hand.

# Fingerprint a config plus its resolved "default" chain.
cell_fingerprint() {
  python3 - "$1" <<'PY'
import hashlib, json, os, sys

h = hashlib.sha256()

def feed(path, seen):
    # NOT lstrip('./'): that strips CHARACTERS, so '/tmp/c.json' becomes
    # 'tmp/c.json', which does not exist -- the walk then hashes nothing and every
    # config fingerprints identically. Strip only a leading './' prefix.
    if path.startswith('./'):
        path = path[2:]
    if path in seen or not os.path.exists(path):
        return
    seen.add(path)
    raw = open(path, 'rb').read()
    h.update(raw)
    try:
        cfg = json.loads(raw)
    except ValueError:
        return
    for section in ('run_cfg', 'model_cfg'):
        parent = (cfg.get(section) or {}).get('default')
        if parent:
            feed(parent, seen)   # inherited defaults change behaviour too

feed(sys.argv[1], set())
print(h.hexdigest()[:16])
PY
}

# cell_is_done <outdir> <config> -> 0 when the cell ran under THIS config
cell_is_done() {
  local out="$1" cfg="$2" stamp="$1/.done"
  [ -f "$stamp" ] || return 1
  local have want
  have=$(cat "$stamp" 2>/dev/null)
  want=$(cell_fingerprint "$cfg")
  # An empty stamp is one written by the old scheme, before fingerprints existed. Treat it
  # as stale rather than trusting it: re-running an eval cell is cheap, and silently
  # reusing a result whose config we cannot identify is what this file exists to prevent.
  if [ -z "$have" ]; then
    echo "   (marker in $out predates config fingerprinting -- re-running)" >&2
    return 1
  fi
  [ "$have" = "$want" ] && return 0
  echo "   (config for $out changed since it last ran -- re-running)" >&2
  return 1
}

# cell_mark_done <outdir> <config>
cell_mark_done() {
  cell_fingerprint "$2" > "$1/.done"
}

# ---- Output-directory exclusivity -------------------------------------------------------
#
# Two jobs writing one workdir is silent and destructive. On 2026-08-22 the seed arms were
# submitted twice (53620809 and 53621049, both --array=17-18), so two four-GPU jobs wrote
# ckpt/model_step_*.pt, ckpt/optimizer_step_*.pt and log/ under
# workdir_pretrain/s1_t9_seed51 for 26 minutes, five minutes out of phase. torch.save is not
# atomic, and the second job's resume check found the first job's optimizer checkpoint and
# continued from it, so their optimizer states diverged into shared files. Nothing in the
# logs said so; both jobs reported normal progress. Both runs had to be discarded.
#
# The claim is a live SLURM job id, not a bare lockfile: a job killed by a timeout or by
# scancel never removes its marker, and a stale marker that blocks a legitimate resume would
# be worse than no marker at all. So the id is looked up, and only a job that is STILL in the
# queue counts as an owner.
#
#   claim_outdir "$OUT" || exit 2

claim_outdir() {
  local out="$1" mine="${SLURM_JOB_ID:-$$}" owner state
  mkdir -p "$out"
  owner=$(cat "$out/.owner" 2>/dev/null)
  if [ -n "$owner" ] && [ "$owner" != "$mine" ]; then
    # squeue prints nothing for a job that has left the queue; that marker is stale.
    state=$(squeue -h -j "$owner" -o '%T' 2>/dev/null | head -1)
    if [ -n "$state" ]; then
      echo "FATAL: $out is already being written by job $owner (state $state)." >&2
      echo "       Two jobs sharing one output directory corrupt each other's checkpoints" >&2
      echo "       without any error being logged. Refusing to start." >&2
      echo "       If $owner is a duplicate submission: scancel $owner, delete $out, resubmit." >&2
      return 1
    fi
    echo "   ($out was claimed by job $owner, which is no longer queued -- taking it over)"
  fi
  echo "$mine" > "$out/.owner"
  return 0
}
