#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=diag_t9
#SBATCH -o ./slurm_scripts/logs/diag_t9_%j.out
#SBATCH -e ./slurm_scripts/logs/diag_t9_%j.out
# Read-only, wide search for any surviving T9 checkpoint: every .pt over 1 GB we can read,
# every other workdir_pretrain folder or sca checkout, and the Slurm records of the T9 training
# jobs (where they ran). Writes experiments/results/harvest/diag_t9.txt; changes nothing else.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
OUT=experiments/results/harvest/diag_t9.txt
mkdir -p "$(dirname "$OUT")"
ROOTS=("$WORK" "$FAST" "$HOME" "/leonardo_scratch/large/userexternal/$USER" "/leonardo_scratch/fast/$USER")
{
  echo "== .pt files over 1 GB (modified, size, path)"
  for r in "${ROOTS[@]}"; do
    [ -d "$r" ] || { echo "(no $r)"; continue; }
    timeout 1500 find "$r" -xdev -type f -name '*.pt' -size +1G -printf '%TY-%Tm-%Td %12s %p\n' 2>/dev/null \
      | grep -v -E '/(HyperAlign/pretrained_weights|conda_envs|pip_cache|hf_cache)/' | sort | tail -60
  done
  echo; echo "== other workdir_pretrain folders / sca checkouts"
  for r in "${ROOTS[@]}"; do
    [ -d "$r" ] || continue
    timeout 600 find "$r" -maxdepth 5 \( -name workdir_pretrain -o -name 't9_qweight_only' -o -name 's1_t9_seed51' -o -name 's2_t9_seed52' \) \
      -printf '%TY-%Tm-%Td %y %p -> %l\n' 2>/dev/null | head -30
  done
  echo; echo "== Slurm records of training jobs since 2026-07-01 (b_grid / pretrain names)"
  sacct -u "$USER" -S 2026-07-01 -X -n -P --format=JobID,JobName%40,State,Start,End,WorkDir%80 2>&1 \
    | grep -i -E 'grid|pretrain|t9|seed' | head -60
  echo; echo "== scratch usage now"
  ls -la "/leonardo_scratch/large/userexternal/$USER" 2>&1 | head -40
  echo "(end)"
} > "$OUT.partial" 2>&1
mv "$OUT.partial" "$OUT"
cat "$OUT"
