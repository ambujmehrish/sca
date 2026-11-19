#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=diag_jobs
#SBATCH -o ./slurm_scripts/logs/diag_jobs_%j.out
#SBATCH -e ./slurm_scripts/logs/diag_jobs_%j.out
# Read-only: scontrol details of the jobs named in DIAG_JOBS (space-separated ids).
# Writes experiments/results/harvest/diag_jobs.txt; changes nothing else.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
: "${DIAG_JOBS:?set DIAG_JOBS to the job ids to show}"
OUT=experiments/results/harvest/diag_jobs.txt
mkdir -p "$(dirname "$OUT")"
{
  for j in $DIAG_JOBS; do
    scontrol show job "$j" 2>&1 | grep -E 'JobId=|JobName|JobState|Reason|SubmitTime|WorkDir|Command|Account|Dependency|ArrayTaskId'
    echo
  done
} > "$OUT.partial" 2>&1
mv "$OUT.partial" "$OUT"
cat "$OUT"
