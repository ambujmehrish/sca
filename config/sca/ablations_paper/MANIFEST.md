# SCA ablation grid (A1-A9) -- one config per table row

Base: `config/sca/pretrain_cfg/sca_pretrain.json` (the A10-decided E6 headline config). Regenerate with `python3 scripts/gen_ablation_configs.py`.

| arm | config | change vs base | feeds |
|---|---|---|---|
| A1_lmask_off | `./config/sca/ablations_paper/A1_lmask_off.json` | L_mask OFF (beta = 0; masked view still sampled) | E5 |
| A1_lmask_term2_off | `./config/sca/ablations_paper/A1_lmask_term2_off.json` | L_mask cross-cardinality score term OFF | E5 |
| A3_sem_off | `./config/sca/ablations_paper/A3_sem_off.json` | L_sem OFF (alpha = 0) | E6 |
| A3_sstar_identity | `./config/sca/ablations_paper/A3_sstar_identity.json` | S* = I (one-hot targets) | E6 |
| A4_concept_off | `./config/sca/ablations_paper/A4_concept_off.json` | L_concept OFF | E7 |
| A4_proto_batch | `./config/sca/ablations_paper/A4_proto_batch.json` | batch-only nu_c (no EMA memory) | E7 |
| A4_eta_0.9 | `./config/sca/ablations_paper/A4_eta_0.9.json` | EMA eta 0.99 -> 0.9 | E7 |
| A4_eps_floor_0 | `./config/sca/ablations_paper/A4_eps_floor_0.json` | eps-floor OFF (collapse check) | E7 |
| A5_mask_freq | `./config/sca/ablations_paper/A5_mask_freq.json` | freq-weighted m-dagger (S dropped most, then A) | E4 |
| A5_mask_2drop | `./config/sca/ablations_paper/A5_mask_2drop.json` | 2-drop masking | E4 |
| A5_pfull_const_0.5 | `./config/sca/ablations_paper/A5_pfull_const_0.5.json` | no schedule: p_full == 0.5 from step 0 | E4 |
| A5_pfull_end_0.3 | `./config/sca/ablations_paper/A5_pfull_end_0.3.json` | deeper schedule 1.0 -> 0.3 | E4 |
| A6_lora_r2 | `./config/sca/ablations_paper/A6_lora_r2.json` | LoRA r=2 (alpha 4) | E9 |
| A6_lora_r4 | `./config/sca/ablations_paper/A6_lora_r4.json` | LoRA r=4 (alpha 8) | E9 |
| A6_lora_r16 | `./config/sca/ablations_paper/A6_lora_r16.json` | LoRA r=16 (alpha 32) | E9 |
| A6_lora_r32 | `./config/sca/ablations_paper/A6_lora_r32.json` | LoRA r=32 (alpha 64) | E9 |
| A6_lora_r64 | `./config/sca/ablations_paper/A6_lora_r64.json` | LoRA r=64 (alpha 128) | E9 |
| A6_lora_asym | `./config/sca/ablations_paper/A6_lora_asym.json` | asymmetric ranks V16/A8/T4 (alpha 16, scalings 1/2/4) | E9 |
| A6_full_ft | `./config/sca/ablations_paper/A6_full_ft.json` | full finetune (no adapters, backbones free) | E9 |
| A6_stage0 | `./config/sca/pretrain_cfg/sca_pretrain_stage0.json` | Stage-0: heads only, vision/audio frozen | E9 |
| A7_centroid_gates | `./config/sca/ablations_paper/A7_centroid_gates.json` | learned per-modality centroid gates (zero-init == uniform) | E8 |
| A7_gatedhgnn | `./config/gram/pretrain_cfg/hyperalign.json` | GatedHGNN refinement (existing GRAM stage-B arm, semantic edges on) | E8 |
| A8_lambda_0 | `./config/sca/ablations_paper/A8_lambda_0.json` | L_unif OFF | E8 |
| A8_lambda_0.05 | `./config/sca/ablations_paper/A8_lambda_0.05.json` | lambda = 0.05 | E8 |
| A8_lambda_0.3 | `./config/sca/ablations_paper/A8_lambda_0.3.json` | lambda = 0.3 | E8 |
| A8_unif_weighted | `./config/sca/ablations_paper/A8_unif_weighted.json` | (1 - S*)-weighted repulsion | E8 |
| A9_sstar_minilm | `./config/sca/ablations_paper/A9_sstar_minilm.json` | S* from all-MiniLM-L6-v2 | E6 |
| A9_taustar_0.3 | `./config/sca/ablations_paper/A9_taustar_0.3.json` | tau* = 0.3 (sharper) | E6 |
| A9_taustar_1.0 | `./config/sca/ablations_paper/A9_taustar_1.0.json` | tau* = 1.0 (no sharpening) | E6 |
| A9_topk_32 | `./config/sca/ablations_paper/A9_topk_32.json` | top-32 sparsification | E6 |
| A9_topk_128 | `./config/sca/ablations_paper/A9_topk_128.json` | top-128 sparsification | E6 |

A2 (alignment-measure comparison) is an analysis pass: `evaluation/measure_comparison.py` on any feature dump; its PMRL training points are `config/baselines/pretrain_cfg/pmrl*.json`.

A9 cache builds (login node, once each):
```bash
python3 data/semantic_targets.py --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json --model_name sentence-transformers/all-MiniLM-L6-v2 --out_path $SCA_CACHE_ROOT/s_star_150k_minilm.pt   # A9_sstar_minilm
python3 data/semantic_targets.py --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json --tau_star 0.3 --out_path $SCA_CACHE_ROOT/s_star_150k_tau03.pt   # A9_taustar_0.3
python3 data/semantic_targets.py --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json --tau_star 1.0 --out_path $SCA_CACHE_ROOT/s_star_150k_tau10.pt   # A9_taustar_1.0
python3 data/semantic_targets.py --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json --topk 32 --out_path $SCA_CACHE_ROOT/s_star_150k_top32.pt   # A9_topk_32
python3 data/semantic_targets.py --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json --topk 128 --out_path $SCA_CACHE_ROOT/s_star_150k_top128.pt   # A9_topk_128
```
