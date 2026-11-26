#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=diag_queue
#SBATCH -o ./slurm_scripts/logs/diag_queue_%j.out
#SBATCH -e ./slurm_scripts/logs/diag_queue_%j.out
# Read-only: why the account's GPU jobs are not starting -- budget, fair share, priority,
# partition load. Writes experiments/results/harvest/diag_queue.txt; changes nothing else.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
OUT=experiments/results/harvest/diag_queue.txt
mkdir -p "$(dirname "$OUT")"
{
  echo "== budget (saldo -b)"; (command -v saldo >/dev/null && saldo -b 2>&1 | head -30) || echo "saldo not available"
  echo; echo "== fair share (sshare)"; sshare -A aifac_s07_041 -l 2>&1 | head -20
  echo; echo "== priority of our pending GPU jobs (sprio)"; sprio -u "$USER" -l 2>&1 | head -25
  echo; echo "== our pending jobs and reasons"; squeue --me -t PD -h -o "%i %j %Q %r %S" 2>&1 | sort -k3 -n -r | head -25
  echo; echo "== boost_usr_prod load (sinfo)"; sinfo -p boost_usr_prod -o "%P %a %D %T" 2>&1 | head -15
  echo; echo "== partition queue depth"; squeue -p boost_usr_prod -h -t PD 2>/dev/null | wc -l; squeue -p boost_usr_prod -h -t R 2>/dev/null | wc -l
  echo; echo "== QOS limits"; sacctmgr -n show assoc user="$USER" format=account%20,qos%40,grptres%30,maxjobs,maxsubmit 2>&1 | head -10
  echo "(end)"
} > "$OUT.partial" 2>&1
mv "$OUT.partial" "$OUT"
cat "$OUT"
