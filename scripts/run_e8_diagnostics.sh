#!/usr/bin/env bash
# E8 diagnostics + A2 measure comparison over the cached E4 feature dumps (CPU only --
# run on a login node). One JSON per arm; PCA projections for the by-concept scatter.
#   bash scripts/run_e8_diagnostics.sh [feats_dir] [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
FEATS="${1:-results/e4/feats}"
OUT="${2:-experiments/results/e8}"
mkdir -p "$OUT/proj"
shopt -s nullglob
found=0
for f in "$FEATS"/*.pt; do
  n=$(basename "$f" .pt); found=1
  echo "== [$n] diagnostics"
  python3 evaluation/diagnostics.py --features "$f" --out "$OUT/${n}_diag.json" \
      --projection pca --proj_out "$OUT/proj/${n}_pca.pt" || { echo "  FAILED" >&2; continue; }
  echo "== [$n] A2 measure comparison"
  python3 evaluation/measure_comparison.py --features "$f" --out "$OUT/${n}_a2.json" \
      || echo "  A2 FAILED" >&2
done
[ "$found" = 1 ] || { echo "FATAL: no feature dumps in $FEATS" >&2; exit 1; }
echo "E8/A2 done -> $OUT"
