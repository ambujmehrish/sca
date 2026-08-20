#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=e4itm_tr
#SBATCH -o ./slurm_scripts/logs/e4itm_tr_%j.out
#SBATCH -e ./slurm_scripts/logs/e4itm_tr_%j.out
# Missing-modality retrieval on the TABLE metric (ITM-reranked) for the three benchmarks
# that were not used for checkpoint selection. 4 arms x 3 benchmarks x {50,90}% = 24
# cells; rate 0 is the existing zero-shot row. Per-cell done markers: resubmit to resume.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p slurm_scripts/logs
best_ckpt() { local b; b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1); echo "$b"; }
declare -A CKPT=( [sca]="$(best_ckpt workdir_pretrain/sca)" \
                  [sca_t1]="$(best_ckpt workdir_pretrain/t1_lr1e4)" \
                  [gram]="$(best_ckpt workdir_pretrain/gram)" \
                  [gram_lora]="$(best_ckpt workdir_pretrain/gram_lora)" )
for a in "${!CKPT[@]}"; do [ -f "${CKPT[$a]}" ] || { echo "FATAL: $a ckpt missing" >&2; exit 1; }
  echo "$a -> ${CKPT[$a]}"; done
rc_all=0
for bench in didemo activitynet audiocaps; do
  for arm in sca sca_t1 gram gram_lora; do
    for rate in 50 90; do
      cell="${arm}_${bench}_r${rate}"; out="workdir/e4_itm_tr/$cell"
      [ -f "$out/.done" ] && { echo "== [$cell] done, skip"; continue; }
      mkdir -p "$out"; echo "== [$cell] START $(date +%T)"
      EVAL_CKPT="${CKPT[$arm]}" srun python3 -m torch.distributed.launch --nnodes 1 \
        --node_rank 0 --nproc_per_node 4 --master_port $((9300 + RANDOM % 300)) \
        ./benchmark_eval/run_eval.py --config "benchmark_eval/configs_e4itm/${cell}.json" \
        --output_dir "$out" 2>&1 \
        | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|antialias|^  warnings.warn\($" || true; }
      rc=$?
      if [ $rc -eq 0 ]; then touch "$out/.done"; echo "== [$cell] OK $(date +%T)"
      else echo "== [$cell] FAILED rc=$rc" >&2; rc_all=$rc; fi
    done
  done
done
echo "EXIT=$rc_all DONE $(date +%T)"; exit $rc_all
