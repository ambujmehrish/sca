#!/usr/bin/env python3
"""Evaluate the HyperGRAM checkpoint we trained from THEIR code, on OUR protocol.

    python3 scripts/make_hypergram_eval_config.py --hypergram_root <dir> \
        --checkpoint workdir_pretrain/hgauth_hybrid/ckpt/best_....pt --bench msrvtt

slurm_scripts/hypergram_authors.sh trains their unmodified repository at their published
recipe. This turns the resulting checkpoint into a table row, the same way
make_pmrl_config.py does for PMRL's released weights: their code, their model, our five
benchmarks at the protocol every other row uses.

WHICH CHECKPOINT. Their run writes three:

    best_ret%tvas--msrvtt_ret_ret_itm_area.pt      best by the REPORTED metric
    best_ret%tvas--msrvtt_ret_ret_area_forward.pt  best by the aggregator
    model_step_<N>.pt                              the last step

`save_best` is theirs and is in the frozen set, so selection-by-validation is their design,
not our choice -- but WHICH of the two "best" files is used changes what is being reported.
The reported row must come from the ITM one, because ret_itm is the metric in the table.
Passing the aggregator checkpoint is allowed for analysis and is recorded in the config.

The val blocks come from OUR benchmark_eval configs for the same reason they do for PMRL:
their own config evaluates VATEX at 16 frames on the full test list, and a row measured on a
different protocol than the rest of the table is not comparable to it.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import absolutise, expand, resolve_audio_dir      # noqa: E402

PAPER_CFG = 'configs/pretrain/pretrain_hybrid_vast150k_vatex_val_paper.json'
OUR_EVAL = 'benchmark_eval/configs_e1/gram_%s.json'
BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')

# The geometry must survive: it is what makes this HyperGRAM rather than GRAM.
FROZEN_MODEL = ('geometry_mode', 'curvature_init', 'learn_curvature',
                'initial_euclidean_weight', 'initial_hyperbolic_weight',
                'learn_hybrid_weights', 'gradient_clip_hyperbolic', 'itm_rerank_num',
                'evaluation_type', 'contra_dim', 'vision_encoder_type', 'audio_encoder_type')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hypergram_root', required=True)
    ap.add_argument('--checkpoint', required=True,
                    help='from workdir_pretrain/hgauth_<mode>/ckpt/')
    ap.add_argument('--bench', required=True, choices=BENCHES)
    ap.add_argument('--data_root', default=os.environ.get('DATA_ROOT', ''))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if not args.data_root:
        sys.exit('FATAL: pass --data_root or export DATA_ROOT')
    root = args.hypergram_root
    src = os.path.join(root, PAPER_CFG)
    if not os.path.exists(src):
        sys.exit('FATAL: %s not found -- is --hypergram_root a HyperGram checkout?' % src)
    if not os.path.exists(args.checkpoint):
        sys.exit('FATAL: checkpoint %s not found.' % args.checkpoint)

    name = os.path.basename(args.checkpoint)
    if 'ret_itm' in name:
        selection = ('selected by ret_itm_area on MSR-VTT -- their save_best, on the metric '
                     'this table reports')
    elif 'area_forward' in name:
        selection = ('selected by the AGGREGATOR (ret_area_forward), NOT by the reported '
                     'metric. Analysis only: reporting this as the HyperGRAM row would '
                     'compare a differently-selected checkpoint against ours.')
    else:
        selection = ('the final step, no validation selection -- their config sets '
                     'save_best true, so this is not the checkpoint their recipe reports')
    print('  checkpoint : %s\n  selection  : %s' % (name, selection))

    code_dir = os.environ.get('CODE_DIR') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..')
    cfg = json.load(open(src), object_pairs_hook=collections.OrderedDict)
    before_model = dict(cfg['model_cfg'])

    ours_path = os.path.join(code_dir, OUR_EVAL % args.bench)
    if not os.path.exists(ours_path):
        sys.exit('FATAL: %s not found -- it defines the protocol every other row uses.'
                 % ours_path)
    val = expand(json.load(open(ours_path))['data_cfg']['val'], args.data_root)
    for d in val:
        d['txt'] = absolutise(d['txt'], code_dir, 'val annotation')
        if 'audio' in d:
            d['audio'] = resolve_audio_dir(d['audio'], d['txt'], d.get('name', args.bench))
        if 'vision' in d and not os.path.isdir(d['vision']):
            sys.exit('FATAL: vision directory %s does not exist for %s.'
                     % (d['vision'], d.get('name')))
    cfg['data_cfg']['val'] = val
    cfg['data_cfg']['train'] = json.loads(json.dumps(val))   # testing mode never builds it

    cfg['run_cfg']['mode'] = 'testing'
    cfg['run_cfg']['checkpoint'] = os.path.abspath(args.checkpoint)
    cfg['run_cfg']['output_dir'] = 'output_hgeval_%s' % args.bench
    # pretrain_dir would load the VAST foundation weights OVER the trained checkpoint --
    # build_model applies pretrain_dir first and `checkpoint` second, but leaving both set
    # invites exactly the confusion this row cannot afford.
    cfg['run_cfg']['pretrain_dir'] = ''

    for k in FROZEN_MODEL:
        if k in before_model and before_model[k] != cfg['model_cfg'].get(k):
            sys.exit('FATAL: model_cfg.%s changed (%r -> %r). The geometry is what makes this '
                     'HyperGRAM.' % (k, before_model[k], cfg['model_cfg'].get(k)))

    cfg['_repro_note'] = {
        'source_config': PAPER_CFG,
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_selection': selection,
        'provenance': ('HyperGRAM trained from github.com/uta-smile/HyperGram unmodified, at '
                       'their published recipe, on our VAST-150k subset. Not a reimplementation.'),
        'evaluation_protocol': ('OUR %s -- same annotation file, frame count, task string and '
                                'rerank depth as every other row.' % (OUR_EVAL % args.bench)),
        'benchmark': args.bench,
    }

    out = args.out or 'configs/pretrain/hgeval_%s.json' % args.bench
    dst = os.path.join(root, out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')
    print('wrote %s' % dst)
    print('  geometry   : %s' % cfg['model_cfg'].get('geometry_mode'))
    for d in val:
        print('  val %-14s vsn=%s task=%s rerank=%s' % (d.get('name'),
              d.get('vision_sample_num'), d.get('task'),
              cfg['model_cfg'].get('itm_rerank_num')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
