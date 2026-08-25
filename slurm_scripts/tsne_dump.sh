#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=tsne_dump
#SBATCH --array=0-2
#SBATCH -o ./slurm_scripts/logs/tsne_dump_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/tsne_dump_%A_%a.out
# Feature dump for the latent-space figure: SCA (T9) and the released GRAM checkpoint on
# the 3-class VGGSound subset (~51 clips, ~2 min each). Scores nothing for any table --
# it triggers the SCA_DUMP_FEATS side channel and saves per-modality unit vectors into
# experiments/results/tables_final/tsne_feats/ (small .pt files, committed by the harvest
# so the panels can be rendered anywhere).
#
#   VGG5K_ROOT=/path/to/vggsound_5k \
#   GRAM_RELEASED_CKPT=/path/to/released.pt sbatch slurm_scripts/tsne_dump.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTHONUNBUFFERED=1
mkdir -p slurm_scripts/logs

: "${VGG5K_ROOT:?set VGG5K_ROOT to the vggsound_5k directory holding videos/ and audios/}"
export VGG5K_ROOT
python3 scripts/make_tsne_setup.py || exit 2

MODELS=(sca gram vast)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
MODEL="${MODELS[$IDX]:-}"
[ -n "$MODEL" ] || { echo "FATAL: index out of range (0-2)" >&2; exit 2; }
if [ "$MODEL" = gram ]; then
  : "${GRAM_RELEASED_CKPT:?set GRAM_RELEASED_CKPT -- same file as the GRAM* rows}"
  CKPT="$GRAM_RELEASED_CKPT"
elif [ "$MODEL" = vast ]; then
  # the common initialization of BOTH adapted models (the checkpoint every training config
  # starts from) -- scored through the gram config/class, which IS VAST's architecture
  CKPT="${VAST_CKPT:-$WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt}"
else
  CKPT="workdir_pretrain/t9_qweight_only/ckpt/model_step_5330.pt"
fi
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not at $CKPT" >&2; exit 2; }

OUT="workdir/e1_tsne/${MODEL}"
DUMP="experiments/results/tables_final/tsne_feats/${MODEL}.pt"
mkdir -p "$OUT" "$(dirname "$DUMP")"
echo "== [tsne/$MODEL] START $(date +%T)  ckpt=$CKPT"
CFGMODEL="$MODEL"; [ "$MODEL" = vast ] && CFGMODEL=gram   # vast = gram class, different weights
SCA_DUMP_FEATS="$DUMP" EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch \
  --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port $((9970 + IDX)) \
  ./benchmark_eval/run_eval.py --config "benchmark_eval/configs_tsne/${CFGMODEL}_vggtsne.json" \
  --output_dir "$OUT" 2>&1 | tee "$OUT/eval.log" | tail -5
rc=${PIPESTATUS[0]}
[ -f "$DUMP" ] || { echo "FATAL: dump not written to $DUMP" >&2; rc=1; }
echo "EXIT=$rc DONE $(date +%T)"
exit $rc
