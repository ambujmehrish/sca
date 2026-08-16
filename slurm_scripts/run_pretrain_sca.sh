#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=sca_pretrain
#SBATCH -o ./slurm_scripts/logs/sca_pretrain_%j.out
#SBATCH -e ./slurm_scripts/logs/sca_pretrain_%j.out
# SCA VAST-150k pretraining launcher (P2): same data/recipe as run_pretrain.sh (epoch5/bs256/
# lr2e-5/frames2, VAST ckpt start), model=sca with LoRA backbones. Set DATA_ROOT/WORK_ROOT to
# your allocation before submitting -- all config/sca files are parameterized on them.
# Build the S* cache once BEFORE the first run (into the WRITABLE cache root --
# DATA_ROOT may be a read-only shared staging):
#   python3 data/semantic_targets.py \
#     --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json \
#     --out_path $SCA_CACHE_ROOT/s_star_150k.pt
: "${DATA_ROOT:?export DATA_ROOT=<dataset root> (e.g. /leonardo_scratch/.../Multimodal_HyperGraph_Dataset)}"
: "${WORK_ROOT:?export WORK_ROOT=<work root holding the VAST ckpt>}"
: "${SCA_CACHE_ROOT:?export SCA_CACHE_ROOT=<writable cache dir holding s_star_150k.pt>}"
# compute nodes are OFFLINE: source the prefetched-model env (scripts/prefetch_models.py,
# run once on a login node with --models_dir $WORK_ROOT/sca_models)
MODELS_DIR="${MODELS_DIR:-$WORK_ROOT/sca_models}"
if [ -f "$MODELS_DIR/env.sh" ]; then
  source "$MODELS_DIR/env.sh"; echo "OFFLINE env: $MODELS_DIR/env.sh (HF_HOME=$HF_HOME)"
else
  echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py on a login node first" >&2
  exit 1
fi
echo "START=$(date +%T) [SCA pretrain -> ./workdir_pretrain/sca]"
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
RESUME=""
if ls ./workdir_pretrain/sca/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  RESUME="--resume true"; echo "RESUME: checkpoint found -> continue"
else
  echo "FRESH"
fi
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9894 \
  ./run.py --config ./config/sca/pretrain_cfg/sca_pretrain.json \
  --output_dir ./workdir_pretrain/sca --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
