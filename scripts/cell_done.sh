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
    path = path.lstrip('./')
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
