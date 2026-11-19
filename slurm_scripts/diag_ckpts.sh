#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=diag_ckpts
#SBATCH -o ./slurm_scripts/logs/diag_ckpts_%j.out
#SBATCH -e ./slurm_scripts/logs/diag_ckpts_%j.out
# Read-only: what is left of the training/eval run folders, and where every checkpoint the
# parity jobs need (T9 and its seeds, GRAM, VAST, the S* cache) still exists.
# Writes experiments/results/harvest/diag_ckpts.txt; changes nothing else.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
OUT=experiments/results/harvest/diag_ckpts.txt
mkdir -p "$(dirname "$OUT")"
RUNS=$(readlink -f workdir_pretrain)
{
  echo "== $RUNS"
  ls -la "$RUNS" 2>&1 | head -80
  echo; echo "== $(readlink -f workdir) (first level, count)"
  ls "$(readlink -f workdir)" 2>&1 | wc -l
  ls -la "$(readlink -f workdir)" 2>&1 | head -40
  echo; echo "== sca_runs folder"
  ls -la "$(dirname "$RUNS")" 2>&1 | head -40
  echo; echo "== model checkpoints named like T9 (model_step_5330.pt) or under *t9*/*seed* folders"
  for root in "$WORK" "$FAST" "$SCRATCH_ROOT" /leonardo_scratch/large/userexternal/amehrish; do
    [ -d "$root" ] || { echo "(no $root)"; continue; }
    timeout 900 find "$root" -maxdepth 7 \( -name 'model_step_5330.pt' -o -path '*t9*/*.pt' -o -path '*seed5[12]*/*.pt' \) \
      -printf '%TY-%Tm-%Td %10s %p\n' 2>/dev/null | head -40
  done
  echo; echo "== GRAM, VAST and S* files the configs name"
  for f in "$WORK/HyperAlign/pretrained_weights/GRAM_pretrained_TVAS/ckpt/model_step_459.pt" \
           "$WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt" \
           "$SCA_CACHE_ROOT/s_star_150k.pt"; do
    ls -la "$f" 2>&1
  done
  echo; echo "== backups or archives of run folders"
  for root in "$WORK" "$FAST"; do
    timeout 600 find "$root" -maxdepth 4 -iname '*sca*' \( -iname '*backup*' -o -iname '*archive*' -o -iname '*ckpt*' -o -iname '*runs*' \) \
      -printf '%TY-%Tm-%Td %p\n' 2>/dev/null | head -20
  done
  echo "(end)"
} > "$OUT.partial" 2>&1
mv "$OUT.partial" "$OUT"
cat "$OUT"
