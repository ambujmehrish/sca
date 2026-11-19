#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=diag_paths
#SBATCH -o ./slurm_scripts/logs/diag_paths_%j.out
#SBATCH -e ./slurm_scripts/logs/diag_paths_%j.out
# Read-only: where the T9 run's files are, what the data loader does with multi-caption test
# entries and unreadable test videos, and whether past eval logs show test-time resampling.
# Writes experiments/results/harvest/diag_paths.txt; changes nothing else.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
OUT=experiments/results/harvest/diag_paths.txt
mkdir -p "$(dirname "$OUT")"
T9=workdir_pretrain/t9_qweight_only
{
  echo "== symlinks"
  for p in workdir workdir_pretrain slurm_scripts/logs "$T9"; do
    printf '%-32s -> %s\n' "$p" "$(readlink -f "$p" 2>/dev/null || echo MISSING)"
  done
  echo; echo "== $T9 (top two levels)"
  ls -la "$T9" 2>&1 | head -20
  ls -la "$T9/log" 2>&1 | head -20
  ls -la "$T9/ckpt" 2>&1 | head -20
  echo; echo "== training arms that still have log/hps.json"
  find -L workdir_pretrain -maxdepth 3 -name hps.json 2>/dev/null | sort | head -80
  echo; echo "== eval cells of T9 that recorded their resolved config"
  find -L workdir -maxdepth 4 -path '*t9_qweight_only*' -name hps.json 2>/dev/null | sort | head -40
  echo; echo "== data loader: caption choice and test-time resampling (data/IndexAnno.py)"
  if [ -f data/IndexAnno.py ]; then
    grep -n -E "raw_captions|random.choice|resample|wrong (image|audio)|raise ValueError|self.training|annos_new|\.wav" data/IndexAnno.py
  else
    echo "data/IndexAnno.py MISSING"
  fi
  echo; echo "== eval logs reporting a replaced test item ('returns wrong'), per log"
  grep -r -c --include='log*.txt' "returns wrong" workdir 2>/dev/null | grep -v ':0$' | sort | head -60
  echo "(end)"
} > "$OUT.partial" 2>&1
mv "$OUT.partial" "$OUT"
cat "$OUT"
