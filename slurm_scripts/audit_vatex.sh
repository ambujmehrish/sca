#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --job-name=audit_vatex
#SBATCH -o ./slurm_scripts/logs/audit_vatex_%j.out
#SBATCH -e ./slurm_scripts/logs/audit_vatex_%j.out
# CPU-only VATEX audit (scripts/audit_vatex.py) into experiments/results/harvest/vatex_audit.txt.
# Adds the MSR-VTT gallery-size estimate when rerank_swap.sh's dump of SCALAR on MSR-VTT exists.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
OUT=experiments/results/harvest/vatex_audit.txt
DUMP=workdir/rerank_swap/dumps/sca_msrvtt
mkdir -p "$(dirname "$OUT")"
ARGS=()
if ls "$DUMP"/rerank_*.pt >/dev/null 2>&1; then ARGS=(--dump "$DUMP"); else echo "no $DUMP yet: gallery-size part skipped"; fi
python3 scripts/audit_vatex.py "${ARGS[@]}" > "$OUT.partial" 2>&1
rc=$?
{ echo "\$ python3 scripts/audit_vatex.py ${ARGS[*]}"; echo; cat "$OUT.partial"; } > "$OUT" && rm -f "$OUT.partial"
cat "$OUT"
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
