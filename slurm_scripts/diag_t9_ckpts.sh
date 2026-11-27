#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=diag_t9_ckpts
#SBATCH -o ./slurm_scripts/logs/diag_t9_ckpts_%j.out
#SBATCH -e ./slurm_scripts/logs/diag_t9_ckpts_%j.out
# Read-only: the files the retrained T9 / seed runs left. Writes experiments/results/harvest/diag_t9_ckpts.txt.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
OUT=experiments/results/harvest/diag_t9_ckpts.txt
mkdir -p "$(dirname "$OUT")"
{
  for a in t9_qweight_only s1_t9_seed51 s2_t9_seed52; do
    echo "== $a"
    ls -la --time-style=+%Y-%m-%dT%H:%M "workdir_pretrain/$a/ckpt" "workdir_pretrain/$a/log" 2>&1 | grep -v '^total'
  done
  echo "(end)"
} > "$OUT.partial" 2>&1
mv "$OUT.partial" "$OUT"
cat "$OUT"
