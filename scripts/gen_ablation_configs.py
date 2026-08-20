#!/usr/bin/env python3
"""Generate the A1-A9 ablation config grid (plan §5) -- ONE config per table row (DoD 6),
each a delta on the Stage-1 base config config/sca/pretrain_cfg/sca_pretrain.json, written
to config/sca/ablations/ with a MANIFEST.md mapping arm -> config -> what changes -> which
experiment consumes it. Deterministic: re-running regenerates the identical grid.

A10 is not here: it was run first (experiments/results/a10_flickr8k_k2.md) and its winner
IS the base config. A2 is an analysis (evaluation/measure_comparison.py), not a training
arm -- only its PMRL-head training points are listed. A7's GatedHGNN arm reuses the
existing GRAM stage-B pretrain config unchanged.
"""
import os
import json
import copy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BASE = os.path.join(ROOT, 'config/sca/pretrain_cfg/sca_pretrain.json')
OUT = os.path.join(ROOT, 'config/sca/ablations')

# arm id -> (model_cfg deltas | ('EXTERNAL', note), manifest row: what / consumed-by)
GRID = {
    # A1 -- L_mask: whole term off, and the cross-cardinality term alone off. beta=0 still
    # SAMPLES the masked view (mu_M drives L_align); it only drops the agreement term, which
    # is what separates it from the "no masked training" arm (p_full == 1, no masked view).
    'A1_lmask_off': ({'sca_beta': 0.0},
                     'L_mask OFF (beta = 0; masked view still sampled)', 'E5'),
    'A1_lmask_term2_off': ({'l_mask_term2': False},
                           'L_mask cross-cardinality score term OFF', 'E5'),
    # A3 -- L_sem: whole term off, and graded targets replaced by one-hot
    'A3_sem_off': ({'sca_alpha': 0.0}, 'L_sem OFF (alpha = 0)', 'E6'),
    'A3_sstar_identity': ({'sca_s_star_identity': True, 's_star_path': ''},
                          'S* = I (one-hot targets)', 'E6'),
    # A4 -- Level-2 grouping arms
    'A4_concept_off': ({'sca_delta': 0.0}, 'L_concept OFF', 'E7'),
    'A4_proto_batch': ({'sca_proto_mode': 'batch'}, 'batch-only nu_c (no EMA memory)', 'E7'),
    'A4_eta_0.9': ({'sca_eta': 0.9}, 'EMA eta 0.99 -> 0.9', 'E7'),
    'A4_eps_floor_0': ({'sca_eps_floor': 0.0}, 'eps-floor OFF (collapse check)', 'E7'),
    # A5 -- mask distribution / schedule / 2-drop
    'A5_mask_freq': ({'mask_mode': 'freq', 'mask_freq': [1.0, 2.0, 4.0, 1.0]},
                     'freq-weighted m-dagger (S dropped most, then A)', 'E4'),
    'A5_mask_2drop': ({'mask_n_drop': 2}, '2-drop masking', 'E4'),
    'A5_pfull_const_0.5': ({'mask_p_full_start': 0.5, 'mask_p_full_end': 0.5},
                           'no schedule: p_full == 0.5 from step 0', 'E4'),
    'A5_pfull_end_0.3': ({'mask_p_full_end': 0.3}, 'deeper schedule 1.0 -> 0.3', 'E4'),
    # A6 -- adaptation regime (Stage-0 config already exists; listed for the table)
    'A6_lora_r4': ({'lora_r_vision': 4, 'lora_r_audio': 4, 'lora_r_text': 4},
                   'LoRA r=4 everywhere', 'E9'),
    'A6_lora_r16': ({'lora_r_vision': 16, 'lora_r_audio': 16, 'lora_r_text': 16},
                    'LoRA r=16 everywhere', 'E9'),
    'A6_lora_asym': ({'lora_r_vision': 16, 'lora_r_audio': 8, 'lora_r_text': 4},
                     'asymmetric ranks V16/A8/T4', 'E9'),
    'A6_full_ft': ({'use_lora': False}, 'full finetune (no adapters, backbones free)', 'E9'),
    'A6_stage0': (('EXTERNAL', './config/sca/pretrain_cfg/sca_pretrain_stage0.json'),
                  'Stage-0: heads only, vision/audio frozen', 'E9'),
    # A7 -- Level-3 aggregation comparison
    'A7_centroid_gates': ({'sca_centroid_gates': True},
                          'learned per-modality centroid gates (zero-init == uniform)', 'E8'),
    'A7_gatedhgnn': (('EXTERNAL', './config/gram/pretrain_cfg/hyperalign.json'),
                     'GatedHGNN refinement (existing GRAM stage-B arm, semantic edges on)', 'E8'),
    # A8 -- uniformity weight sweep + weighted repulsion
    'A8_lambda_0': ({'sca_lambda': 0.0}, 'L_unif OFF', 'E8'),
    'A8_lambda_0.05': ({'sca_lambda': 0.05}, 'lambda = 0.05', 'E8'),
    'A8_lambda_0.3': ({'sca_lambda': 0.3}, 'lambda = 0.3', 'E8'),
    'A8_unif_weighted': ({'l_unif_weighted': True}, '(1 - S*)-weighted repulsion', 'E8'),
    # A9 -- S* source / tau* / sparsification (config points at a differently-built cache;
    # the matching build commands are in the manifest)
    'A9_sstar_minilm': ({'s_star_path': '${SCA_CACHE_ROOT}/s_star_150k_minilm.pt'},
                        'S* from all-MiniLM-L6-v2', 'E6'),
    'A9_taustar_0.3': ({'s_star_path': '${SCA_CACHE_ROOT}/s_star_150k_tau03.pt',
                        'sca_tau_star': 0.3}, 'tau* = 0.3 (sharper)', 'E6'),
    'A9_taustar_1.0': ({'s_star_path': '${SCA_CACHE_ROOT}/s_star_150k_tau10.pt',
                        'sca_tau_star': 1.0}, 'tau* = 1.0 (no sharpening)', 'E6'),
    'A9_topk_32': ({'s_star_path': '${SCA_CACHE_ROOT}/s_star_150k_top32.pt'},
                   'top-32 sparsification', 'E6'),
    'A9_topk_128': ({'s_star_path': '${SCA_CACHE_ROOT}/s_star_150k_top128.pt'},
                    'top-128 sparsification', 'E6'),
}

A9_BUILDS = {
    'A9_sstar_minilm': '--model_name sentence-transformers/all-MiniLM-L6-v2 '
                       '--out_path $SCA_CACHE_ROOT/s_star_150k_minilm.pt',
    'A9_taustar_0.3': '--tau_star 0.3 --out_path $SCA_CACHE_ROOT/s_star_150k_tau03.pt',
    'A9_taustar_1.0': '--tau_star 1.0 --out_path $SCA_CACHE_ROOT/s_star_150k_tau10.pt',
    'A9_topk_32': '--topk 32 --out_path $SCA_CACHE_ROOT/s_star_150k_top32.pt',
    'A9_topk_128': '--topk 128 --out_path $SCA_CACHE_ROOT/s_star_150k_top128.pt',
}


def main():
    os.makedirs(OUT, exist_ok=True)
    base = json.load(open(BASE))
    rows = []
    for arm, (delta, what, consumer) in GRID.items():
        if isinstance(delta, tuple) and delta[0] == 'EXTERNAL':
            rows.append((arm, delta[1], what, consumer))
            continue
        cfg = copy.deepcopy(base)
        cfg['model_cfg'].update(delta)
        path = os.path.join(OUT, f'{arm}.json')
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=1)
        rows.append((arm, f'./config/sca/ablations/{arm}.json', what, consumer))

    with open(os.path.join(OUT, 'MANIFEST.md'), 'w') as f:
        f.write('# SCA ablation grid (A1-A9) -- one config per table row\n\n'
                'Base: `config/sca/pretrain_cfg/sca_pretrain.json` (the A10-decided E6 '
                'headline config). Regenerate with `python3 scripts/gen_ablation_configs.py`.\n\n'
                '| arm | config | change vs base | feeds |\n|---|---|---|---|\n')
        for arm, path, what, consumer in rows:
            f.write(f'| {arm} | `{path}` | {what} | {consumer} |\n')
        f.write('\nA2 (alignment-measure comparison) is an analysis pass: '
                '`evaluation/measure_comparison.py` on any feature dump; its PMRL training '
                'points are `config/baselines/pretrain_cfg/pmrl*.json`.\n'
                '\nA9 cache builds (login node, once each):\n```bash\n')
        for arm, extra in A9_BUILDS.items():
            f.write(f'python3 data/semantic_targets.py '
                    f'--annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json '
                    f'{extra}   # {arm}\n')
        f.write('```\n')
    print(f'{len(rows)} arms -> {OUT} (+ MANIFEST.md)')


if __name__ == '__main__':
    main()
