# CINECA / Leonardo setup runbook — account AIFAC_S07_041

Three-tier layout ($HOME is almost full — NOTHING goes there):

| tier | path | holds |
|---|---|---|
| WORK (4 TB) | `/leonardo_work/AIFAC_S07_041` | code checkout, VAST ckpt staging |
| FAST (1 TB) | `/leonardo_scratch/fast/AIFAC_S07_041` | conda env, pip/conda/HF caches, model weights |
| SCRATCH | `/leonardo_scratch/large/userexternal/$USER` | datasets, checkpoints, run logs, results |

Login nodes have internet, compute nodes do NOT — steps 1–6 (login node) do every
download; step 7+ runs offline.

## 1. Persistent environment file (sourced by every shell and job)

```bash
export WORK=/leonardo_work/AIFAC_S07_041
export FAST=/leonardo_scratch/fast/AIFAC_S07_041
export SCRATCH_ROOT=/leonardo_scratch/large/userexternal/$USER   # adjust if your large
                                                                 # scratch lives elsewhere
cat > $WORK/sca_env.rc <<EOF
export ACCOUNT=AIFAC_S07_041
export WORK=$WORK
export FAST=$FAST
export SCRATCH_ROOT=$SCRATCH_ROOT
export CODE_DIR=\$WORK/sca
# configs read these two:
export WORK_ROOT=\$WORK                                  # \${WORK_ROOT}/GRAM/... = VAST ckpt
export DATA_ROOT=\$SCRATCH_ROOT/Multimodal_HyperGraph_Dataset
export MODELS_DIR=\$FAST/models/sca_models               # prefetch target + offline env.sh
# every cache on FAST (reusing your existing dirs), never \$HOME
export PIP_CACHE_DIR=\$FAST/pip_cache
export CONDA_PKGS_DIRS=\$FAST/conda_pkgs
export HF_HOME=\$MODELS_DIR/hf
export TORCH_HOME=\$FAST/hf_cache/torch
export TRITON_CACHE_DIR=\$FAST/triton-cache
conda activate \$FAST/conda_envs/sca 2>/dev/null || true
EOF
source $WORK/sca_env.rc
```

## 2. Code into WORK

```bash
cd $WORK
git clone -b claude/implementation-plan-review-csn6m0 https://github.com/ambujmehrish/sca.git
cd sca
```

## 3. Symlink checkpoints/logs/results into SCRATCH and weights into FAST

The launchers write to repo-relative paths; symlinks route them without code changes:

```bash
mkdir -p $SCRATCH_ROOT/sca_runs/{workdir_pretrain,workdir,results,slurm_logs} \
         $FAST/models/sca_pretrained_weights
cd $CODE_DIR
ln -sfn $SCRATCH_ROOT/sca_runs/workdir_pretrain workdir_pretrain
ln -sfn $SCRATCH_ROOT/sca_runs/workdir          workdir
ln -sfn $SCRATCH_ROOT/sca_runs/results          results
ln -sfn $SCRATCH_ROOT/sca_runs/slurm_logs       slurm_scripts/logs
ln -sfn $FAST/models/sca_pretrained_weights     pretrained_weights
```

## 4. Bring in the pieces the repo does not carry

```bash
# (a) the upstream VAST/GRAM data-loader package (data/loader.py, data_registry,
#     annoindexed dataset ...) from the existing HyperAlign checkout -- the repo's own
#     data/*.py files keep different names, nothing is overwritten:
cp -n /path/to/HyperAlign/data/*.py $CODE_DIR/data/
# and the annotation files the val loaders read (datasets/annotations/...):
cp -rn /path/to/HyperAlign/datasets $CODE_DIR/ 2>/dev/null || true

# (b) BEATs weights (official microsoft/unilm release; not on HF -- copy from the
#     existing setup). Lands on FAST via the pretrained_weights symlink:
mkdir -p pretrained_weights/beats
cp /path/to/HyperAlign/pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt pretrained_weights/beats/

# (c) VAST starting ckpt -- configs read
#     $WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt
mkdir -p $WORK/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt
cp (or ln -s) /path/to/model_step_204994.pt \
   $WORK/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/

# (d) the dataset root on SCRATCH: $DATA_ROOT must hold vast27m_150k/, MSRVTT_full/, ...
#     (copy/rsync from the existing staging if this scratch does not have it yet)
```

## 5. Environment on FAST + pinned dependencies

```bash
conda create --prefix $FAST/conda_envs/sca python=3.10 -y
conda activate $FAST/conda_envs/sca
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" "transformers==4.31.0" "tokenizers<0.14" \
    sentence-transformers==2.2.2 "huggingface_hub<0.20" \
    easydict opencv-python-headless einops timm ftfy regex tqdm \
    decord soundfile audioread ffmpeg-python yacs pyyaml \
    wandb openpyxl pytest scikit-learn "datasets<2.16" pillow ipdb
# transformers PINNED <5: the trunk's bert.py needs the 4.x API. apex NOT needed
# (torch.cuda.amp). xformers optional (EVA prints a hint without it; runs fine).
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 6. Prefetch every network asset (login node, once)

```bash
cd $CODE_DIR
python3 scripts/prefetch_models.py --models_dir $MODELS_DIR --with-smoke-data
# FETCHED: mpnet + MiniLM (S*), CLIP-B/32 + Flickr8k (smoke stage 2), and
#          bert-base-uncased + EVA01-giant into pretrained_weights/ (-> FAST).
# PRESENT expected: beats (4b), VAST ckpt (4c). EXITS NON-ZERO while anything is
# missing -- fix before continuing. Writes $MODELS_DIR/env.sh (offline flags).
```

## 7. S* cache (offline-safe; short single-GPU debug job)

```bash
source $MODELS_DIR/env.sh
srun -A $ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg --time=00:30:00 \
     --gres=gpu:1 --mem=64G --pty \
  python3 data/semantic_targets.py \
    --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json \
    --out_path $DATA_ROOT/vast27m_150k/s_star_150k.pt
```

## 8. Smoke test — CPU stages on the login node

```bash
cd $CODE_DIR && source $WORK/sca_env.rc
bash scripts/smoke_test.sh     # stage 1: test suite; stage 2: real-data loss smoke;
                               # stage 3 reports SKIPPED here (needs GPUs)
```

## 9. Smoke test — the REAL k=4 GPU stage (debug queue, ~30 min, 4 GPUs)

```bash
cat > $WORK/smoke_gpu.sbatch <<EOF
#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=sca_smoke
#SBATCH -o $WORK/smoke_gpu_%j.out
#SBATCH -e $WORK/smoke_gpu_%j.out
source $WORK/sca_env.rc
cd \$CODE_DIR
bash scripts/smoke_test.sh
EOF
sbatch $WORK/smoke_gpu.sbatch
tail -f $WORK/smoke_gpu_*.out        # must end with: ALL STAGES PASSED
```

Stage 3 runs 24 real steps of the k=4 vast27m pretrain with LoRA + a validation pass.
It fails loudly on: missing S* cache, missing weights, unset env vars, LoRA naming
drift, non-finite losses. **Submit nothing long until it prints ALL STAGES PASSED.**

## 10. Only after the smoke is green — the real runs

```bash
cd $CODE_DIR
sbatch slurm_scripts/run_pretrain_sca.sh      # Stage-1 LoRA pretrain (P2); account is
                                              # already in the header. Checkpoints land in
                                              # $SCRATCH_ROOT/sca_runs via the symlinks.
# then: sca_pretrain_stage0.json / _nomask.json, config/baselines/pretrain_cfg/*.json
#       (P3 + 2x2 arms), slurm_scripts/ft_*_sca.sh (P4 grid, incl. depth E10),
#       scripts/run_e4_grid.sh (E4/E5/E6), config/sca/ablations/ (P5).
```

## Troubleshooting

- `EnvironmentError: unset environment variable` → `source $WORK/sca_env.rc`.
- `FATAL: $MODELS_DIR/env.sh not found` → step 6 was skipped.
- `S* cache has no rows for batch ids ...` → cache built from a different annotation
  file; rebuild (step 7) against the one the config trains on.
- `ModuleNotFoundError: data.loader` → step 4a was skipped.
- `FileNotFoundError ... model_step_204994.pt` → step 4c staging/symlink.
- $HOME quota errors → a cache is leaking: check `du -sh ~/.cache ~/.conda` and the
  exports of step 1 (PIP_CACHE_DIR / CONDA_PKGS_DIRS / HF_HOME / TORCH_HOME / TRITON_CACHE_DIR).
