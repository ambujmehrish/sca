#!/usr/bin/env python3
"""GRAM-checkpoint configs for the missing-modality sweep.

    python3 scripts/make_missing_configs.py

benchmark_eval/configs_missing/r{00,25,50,75,90}/ already holds the SCA configs (T9 geometry
plus eval_mask_rate/eval_mask_seed). This writes the matching gram_<bench>.json into the same
directories, derived from benchmark_eval/configs_e1/gram_<bench>.json -- the exact config the
released-checkpoint row of Table 1/2 was measured with -- plus the two mask keys and nothing
else.

WHY THE COMPARISON IS FAIR BY CONSTRUCTION. eval_mask_rate is implemented once, in
model/gram.py::batch_get, which every model class in our trunk inherits. The drop decision is
md5(seed|clip_id): deterministic, identical across arms and directions, nested across rates
(a clip dropped at 0.25 is dropped at 0.5). So SCA and the GRAM checkpoint lose exactly the
same modality of exactly the same clips, and rate 0.0 is byte-identical to the standard eval
path -- the r00 cells double as a control that masking-off reproduces Table 1/2.
"""
import collections
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
RATES = {'r00': 0.0, 'r25': 0.25, 'r50': 0.5, 'r75': 0.75, 'r90': 0.9}


def main():
    wrote = 0
    for rdir, rate in sorted(RATES.items()):
        outdir = os.path.join(ROOT, 'benchmark_eval/configs_missing', rdir)
        if not os.path.isdir(outdir):
            sys.exit('FATAL: %s does not exist -- the SCA side of the sweep should have '
                     'created it. Refusing to invent the directory layout.' % outdir)
        for b in BENCHES:
            src = os.path.join(ROOT, 'benchmark_eval/configs_e1', 'gram_%s.json' % b)
            if not os.path.exists(src):
                sys.exit('FATAL: %s not found -- it is the config the released-checkpoint '
                         'row was measured with, and this sweep must inherit it exactly.' % src)
            cfg = json.load(open(src), object_pairs_hook=collections.OrderedDict)
            before = dict(cfg['model_cfg'])
            cfg['model_cfg']['eval_mask_rate'] = rate
            cfg['model_cfg']['eval_mask_seed'] = 0
            # nothing else may differ from the Table-1 config, or the curve's anchor breaks
            for k, v in before.items():
                if cfg['model_cfg'][k] != v:
                    sys.exit('FATAL: %s drifted while adding mask keys' % k)
            dst = os.path.join(outdir, 'gram_%s.json' % b)
            with open(dst, 'w') as f:
                json.dump(cfg, f, indent=1)
                f.write('\n')
            wrote += 1

            # PMRL: the authors' RELEASED checkpoint through OUR pmrl class (masking lives in
            # our trunk; theirs has none). Derived from the configs_repro pmrl config -- our
            # only config that names their scoring (model_type pmrl, score_mode pmrl_raw) --
            # with the LoRA keys STRIPPED: the released checkpoint is full-FT and carries no
            # adapter weights, and a LoRA-bearing class would report them as missing keys.
            # Whether the released weights actually fit this class is not assumed here: the
            # launcher verifies the load from the eval log and refuses the cells otherwise.
            psrc = os.path.join(ROOT, 'benchmark_eval/configs_repro', 'pmrl_%s.json' % b)
            if not os.path.exists(psrc):
                sys.exit('FATAL: %s not found -- the pmrl side of the sweep has no base '
                         'config.' % psrc)
            pcfg = json.load(open(psrc), object_pairs_hook=collections.OrderedDict)
            for k in ('use_lora', 'lora_r_vision', 'lora_r_audio', 'lora_r_text',
                      'lora_alpha'):
                pcfg['model_cfg'].pop(k, None)
            pcfg['model_cfg']['use_lora'] = False
            # pmrl_norm, not pmrl_raw: lambda_1 of a Gram matrix of m unit vectors lives in
            # [1, m], so at MIXED arity the raw score structurally penalises masked clips --
            # their ceiling is lower regardless of alignment. The norm variant divides by the
            # clip's own arity (their head's built-in option), and at r=0 every clip has the
            # same arity so the division is a constant: ranking, and therefore R@1, is
            # IDENTICAL to raw -- the r00 control still anchors to PMRL's Table rows exactly.
            pcfg['model_cfg']['score_mode'] = 'pmrl_norm'
            pcfg['model_cfg']['eval_mask_rate'] = rate
            pcfg['model_cfg']['eval_mask_seed'] = 0
            pdst = os.path.join(outdir, 'pmrl_%s.json' % b)
            with open(pdst, 'w') as f:
                json.dump(pcfg, f, indent=1)
                f.write('\n')
            wrote += 1
    print('wrote %d gram+pmrl configs across %s' % (wrote, ', '.join(sorted(RATES))))

    # sanity: the SCA config at each rate must carry the same mask keys
    for rdir, rate in sorted(RATES.items()):
        p = os.path.join(ROOT, 'benchmark_eval/configs_missing', rdir, 'sca_msrvtt.json')
        m = json.load(open(p))['model_cfg']
        if m.get('eval_mask_rate') != rate or m.get('eval_mask_seed') != 0:
            sys.exit('FATAL: %s has mask_rate=%s seed=%s -- the two sides of the sweep '
                     'disagree about the condition.' % (p, m.get('eval_mask_rate'),
                                                        m.get('eval_mask_seed')))
    print('sca/gram mask keys agree at every rate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
