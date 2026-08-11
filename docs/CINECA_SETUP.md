# CINECA / Leonardo setup runbook

Everything lives OUTSIDE $HOME (50 GB quota): code + env + model caches in `$WORK`, data
in `$SCRATCH`. Login nodes have internet, compute nodes do NOT — every download happens
in steps 1–6 (login node); step 7+ runs offline.

Placeholders to set once (adjust to your allocation):

```bash
export ACCOUNT=IscrC_CASPER-A_0                                   # your slurm account
export WORK_ROOT=/leonardo_work/$ACCOUNT/$USER                    # big, backed up
export DATA_ROOT=/leonardo_scratch/large/userexternal/$USER/Multimodal_HyperGraph_Dataset
# If you are reusing the existing GRAM/HyperAlign staging, point WORK_ROOT/DATA_ROOT at
# those locations instead -- the configs only care about the env vars.
```

## 1. Persistent environment file (sourced by every shell and job)

```bash
mkdir -p $WORK_ROOT
cat > $WORK_ROOT/sca_env.rc <<EOF
export ACCOUNT=$ACCOUNT
export WORK_ROOT=$WORK_ROOT
export DATA_ROOT=$DATA_ROOT
export MODELS_DIR=\$WORK_ROOT/sca_models
# keep EVERY cache out of \$HOME
export PIP_CACHE_DIR=\$WORK_ROOT/.cache/pip
export CONDA_PKGS_DIRS=\$WORK_ROOT/.cache/conda_pkgs
export HF_HOME=\$WORK_ROOT/sca_models/hf
export TORCH_HOME=\$WORK_ROOT/.cache/torch
module load anaconda3 2>/dev/null || module load python
[ -d \$WORK_ROOT/sca_venv ] && conda activate \$WORK_ROOT/sca_venv 2>/dev/null
EOF
source $WORK_ROOT/sca_env.rc
```

## 2. Clone the code into $WORK

```bash
cd $WORK_ROOT
git clone -b claude/implementation-plan-review-csn6m0 https://github.com/ambujmehrish/sca.git
cd sca
```

## 3. Bring in the two pieces the repo does not carry

```bash
# (a) the upstream VAST/GRAM data-loader package (data/loader.py, data_registry,
#     annoindexed dataset ...) from your existing HyperAlign checkout -- the repo's own
#     data/*.py files keep different names, nothing is overwritten:
cp -n /path/to/HyperAlign/data/*.py $WORK_ROOT/sca/data/
# also the msrvtt val annotations the pretrain val loader reads:
cp -rn /path/to/HyperAlign/datasets $WORK_ROOT/sca/ 2>/dev/null || true

# (b) trunk weights that cannot be fetched from HF: BEATs (official microsoft/unilm
#     release) -- copy from the existing setup; bert + EVA-giant are fetched in step 6:
mkdir -p pretrained_weights/beats
cp /path/to/HyperAlign/pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt pretrained_weights/beats/
# VAST starting ckpt: the configs read
#   $WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt
# -- keep that path, or symlink your staged copy to it.
```

## 4. Create the environment (conda prefix in $WORK, never $HOME)

```bash
conda create --prefix $WORK_ROOT/sca_venv python=3.10 -y
conda activate $WORK_ROOT/sca_venv
```

## 5. Install dependencies (order matters: torch first, pinned transformers)

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" "transformers==4.31.0" "tokenizers<0.14" \
    sentence-transformers==2.2.2 "huggingface_hub<0.20" \
    easydict opencv-python-headless einops timm ftfy regex tqdm \
    decord soundfile audioread ffmpeg-python yacs pyyaml \
    wandb openpyxl pytest scikit-learn "datasets<2.16" pillow ipdb
# NOTES: transformers is PINNED <5 -- the trunk's bert.py needs 4.x
#        (apply_chunking_to_forward). apex is NOT needed (torch.cuda.amp is used).
#        xformers is optional (EVA prints a hint without it; runs fine).
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 6. Prefetch every network asset (login node, once) + verify

```bash
cd $WORK_ROOT/sca
python3 scripts/prefetch_models.py --models_dir $MODELS_DIR --with-smoke-data
# FETCHED: mpnet + MiniLM (S*), CLIP-B/32 + Flickr8k (smoke stage 2),
#          bert-base-uncased + EVA01-giant into ./pretrained_weights/
# PRESENT expected for: beats (step 3b) and the VAST ckpt (WORK_ROOT set).
# The script EXITS NON-ZERO while anything is missing -- fix before continuing.
```

## 7. Build the S* cache (offline-safe; short single-GPU job)

```bash
source $MODELS_DIR/env.sh          # HF offline flags + cache -- compute nodes need this
srun -A $ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg --time=00:30:00 \
     --gres=gpu:1 --mem=64G --pty \
  python3 data/semantic_targets.py \
    --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json \
    --out_path $DATA_ROOT/vast27m_150k/s_star_150k.pt
```

## 8. Smoke test — CPU stages on the login node first

```bash
cd $WORK_ROOT/sca && source $WORK_ROOT/sca_env.rc
bash scripts/smoke_test.sh          # stage 1: 78-test suite; stage 2: real-data loss
                                    # smoke (Flickr8k, hard gates); stage 3 reports
                                    # SKIPPED here -- it needs GPUs
```

## 9. Smoke test — the REAL k=4 GPU stage (debug queue, ~30 min, 4 GPUs)

```bash
cat > smoke_gpu.sbatch <<'EOF'
#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=sca_smoke
#SBATCH -o smoke_gpu_%j.out
#SBATCH -e smoke_gpu_%j.out
source $WORK_ROOT/sca_env.rc
cd $WORK_ROOT/sca
bash scripts/smoke_test.sh
EOF
sbatch -A $ACCOUNT --export=ALL,WORK_ROOT=$WORK_ROOT smoke_gpu.sbatch
tail -f smoke_gpu_*.out              # must end with: ALL STAGES PASSED
```

Stage 3 runs 24 real steps of the k=4 vast27m pretrain with LoRA + a validation pass.
It fails loudly on: missing S* cache, missing weights, unset env vars, LoRA naming
drift, or non-finite losses. **Do not submit the long runs until it prints
`ALL STAGES PASSED`.**

## 10. Only after the smoke is green — the real runs

```bash
sbatch -A $ACCOUNT slurm_scripts/run_pretrain_sca.sh        # Stage-1 LoRA pretrain (P2)
# then: config/sca/pretrain_cfg/sca_pretrain_stage0.json / _nomask.json,
#       config/baselines/pretrain_cfg/*.json (P3 + 2x2 arms),
#       slurm_scripts/ft_*_sca.sh (P4 finetune grid),
#       scripts/run_e4_grid.sh (E4/E5/E6), config/sca/ablations/ (P5).
```

## Troubleshooting

- `EnvironmentError: unset environment variable` → `source $WORK_ROOT/sca_env.rc`.
- `FATAL: $MODELS_DIR/env.sh not found` → step 6 was skipped.
- `S* cache has no rows for batch ids ...` → cache built from a different annotation
  file; rebuild (step 7) against the one the config trains on.
- `ModuleNotFoundError: data.loader` → step 3a was skipped.
- Disk quota on $HOME → some cache is leaking; check `PIP_CACHE_DIR/HF_HOME/TORCH_HOME`
  are exported (step 1) and `du -sh ~/.cache`.
