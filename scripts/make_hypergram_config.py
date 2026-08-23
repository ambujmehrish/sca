#!/usr/bin/env python3
"""Point HyperGram's own config at our data, changing nothing else.

    python3 scripts/make_hypergram_config.py --hypergram_root /path/to/hypergram

Their repository (github.com/uta-smile/HyperGram) is the same VAST/GRAM fork we use, so the
reproduction runs THEIR code with THEIR config rather than our reimplementation of it -- which
differs from theirs in six substantive ways (experiments/results/HYPERGRAM_STATUS.md).

The only edits are dataset and checkpoint PATHS. Every hyperparameter is left exactly as they
ship it: lr 5e-05, one epoch, batch 128, task ret%tvas%tv%ta, curvature learnable at 10x the
base lr, hybrid weights learnable from 0.5/0.5. The script asserts that afterwards, because a
path rewrite that quietly changed a hyperparameter would produce a number labelled "authors'
code" that is not.

WHAT THIS CANNOT GUARANTEE. They train on `annotations150k_clean.json`; ours is
`annotations150k.json`. If those differ the comparison is not at equal data, so the script
refuses unless a file with their name exists or --allow_annotation_mismatch is passed, and the
substitution is recorded in the generated config for whoever reads the numbers later.
"""
import argparse
import collections
import json
import os
import sys

PAPER_CFG = 'configs/pretrain/pretrain_hybrid_vast150k_vatex_val_paper.json'

# What must survive the rewrite untouched. Checked after, not merely intended.
FROZEN_RUN = ('learning_rate', 'train_epoch', 'valid_freq', 'save_best', 'first_eval',
              'grad_norm', 'fp16')
FROZEN_MODEL = ('curvature_init', 'learn_curvature',
                'initial_euclidean_weight', 'initial_hyperbolic_weight',
                'learn_hybrid_weights', 'gradient_clip_hyperbolic')
FROZEN_TRAIN = ('batch_size', 'epoch', 'task', 'vision_sample_num', 'audio_sample_num')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hypergram_root', required=True,
                    help='checkout of github.com/uta-smile/HyperGram')
    ap.add_argument('--data_root', default=os.environ.get('DATA_ROOT', ''),
                    help='our DATA_ROOT (default: $DATA_ROOT)')
    ap.add_argument('--vast_ckpt_dir', default='',
                    help='directory holding the VAST foundation checkpoint '
                         '(default: $WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast)')
    ap.add_argument('--geometry_mode', default='hybrid',
                    choices=('hybrid', 'pmrl', 'pmrl_volume', 'hybrid_pmrl',
                             'hyperbolic', 'euclidean'),
                    help="hybrid = HyperGRAM as published. The pmrl* modes are THEIR PMRL "
                         "implementation, but no PMRL config ships with the repo, so they run "
                         "at HYPERGRAM's recipe -- their implementation, not PMRL's recipe.")
    ap.add_argument('--out', default=None,
                    help='written INSIDE --hypergram_root; default repro_<mode>_ours_paths.json')
    ap.add_argument('--allow_annotation_mismatch', action='store_true')
    args = ap.parse_args()

    if not args.data_root:
        sys.exit('FATAL: pass --data_root or export DATA_ROOT')
    root = args.hypergram_root
    src = os.path.join(root, PAPER_CFG)
    if not os.path.exists(src):
        sys.exit('FATAL: %s not found -- is --hypergram_root a HyperGram checkout?' % src)

    vast = args.vast_ckpt_dir or os.path.join(
        os.environ.get('WORK_ROOT', ''), 'GRAM/code/pretrained_models/VAST_foundation/pretrain_vast')
    if not os.path.isdir(vast):
        sys.exit('FATAL: VAST checkpoint dir not found at %s -- pass --vast_ckpt_dir' % vast)

    cfg = json.load(open(src), object_pairs_hook=collections.OrderedDict)
    before = (dict(cfg['run_cfg']), dict(cfg['model_cfg']), dict(cfg['data_cfg']['train'][0]))

    # ---- the annotation file, which is the one place their data may not be our data
    train = cfg['data_cfg']['train'][0]
    theirs = os.path.basename(train['txt'])                 # annotations150k_clean.json
    ours_dir = os.path.join(args.data_root, 'vast27m_150k')
    same_name = os.path.join(ours_dir, theirs)
    fallback = os.path.join(ours_dir, 'annotations150k.json')
    if os.path.exists(same_name):
        train['txt'] = same_name
        note = 'their annotation file, present in our tree'
    elif args.allow_annotation_mismatch and os.path.exists(fallback):
        train['txt'] = fallback
        note = ('SUBSTITUTED annotations150k.json for their %s -- the training sets may '
                'differ, so this is NOT an equal-data comparison' % theirs)
    else:
        sys.exit('FATAL: they train on %s; we have only %s.\n'
                 '       Those may be different subsets of VAST-150k, and swapping one for the\n'
                 '       other silently makes the comparison unequal in the training data --\n'
                 '       the exact class of error this reproduction exists to avoid.\n'
                 '       Fetch their file, or pass --allow_annotation_mismatch to proceed with\n'
                 '       the substitution recorded in the config.' % (theirs, fallback))

    train['vision'] = os.path.join(ours_dir, 'clips')
    train['audio'] = os.path.join(ours_dir, 'audios_wav')

    # Their repo ships no datasets/ directory, so the annotation JSONs its val block names
    # (datasets/annotations/<bench>/descs_ret_test.json) resolve nowhere inside their tree.
    # Point them at ours: these are the same VAST-family annotation files both forks read, and
    # using ours keeps the eval split identical to every other row in our table.
    code_dir = os.environ.get('CODE_DIR') or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..')
    for d in cfg['data_cfg'].get('val', []):
        name = d.get('name', '')
        if name.startswith('vatex'):
            d['vision'] = os.path.join(args.data_root, 'VATEX/videos')
            d['audio'] = os.path.join(args.data_root, 'VATEX/audios')
        if not os.path.isabs(d['txt']):
            ours = os.path.join(code_dir, d['txt'])
            if not os.path.exists(ours):
                sys.exit('FATAL: val annotation %s not found at %s -- their repo ships no '
                         'datasets/ directory, so it has to come from ours.' % (d['txt'], ours))
            d['txt'] = ours

    cfg['run_cfg']['pretrain_dir'] = vast
    cfg['run_cfg']['output_dir'] = 'output_repro_%s' % args.geometry_mode
    # geometry_mode is the ONE hyperparameter this script is allowed to set, because it names
    # which of their methods is being run. Everything else stays as shipped.
    cfg['model_cfg']['geometry_mode'] = args.geometry_mode

    # ---- nothing but paths may have moved
    after = (cfg['run_cfg'], cfg['model_cfg'], cfg['data_cfg']['train'][0])
    for keys, b, a, what in ((FROZEN_RUN, before[0], after[0], 'run_cfg'),
                             (FROZEN_MODEL, before[1], after[1], 'model_cfg'),
                             (FROZEN_TRAIN, before[2], after[2], 'train block')):
        for k in keys:
            if b.get(k) != a.get(k):
                sys.exit('FATAL: %s.%s changed (%r -> %r). Only paths may be rewritten.'
                         % (what, k, b.get(k), a.get(k)))

    cfg['_repro_note'] = {
        'source_config': PAPER_CFG,
        'edits': 'dataset and checkpoint paths only; every hyperparameter left as shipped',
        'annotations': note,
        'geometry_mode': args.geometry_mode,
        'recipe_caveat': (
            'hybrid is HyperGRAM as published. pmrl / pmrl_volume / hybrid_pmrl use THEIR '
            'PMRL implementation but no PMRL config ships with the repo, so they inherit '
            'HyperGRAM\'s recipe (lr 5e-05, 1 epoch, task ret%tvas%tv%ta). Report them as '
            "their implementation at HyperGRAM's recipe, never as PMRL's published setup."),
        'hyperparameters_as_shipped': {
            'learning_rate': cfg['run_cfg'].get('learning_rate'),
            'epoch': train.get('epoch'),
            'batch_size': train.get('batch_size'),
            'task': train.get('task'),
            'geometry_mode': cfg['model_cfg'].get('geometry_mode'),
        },
    }

    out = args.out or 'configs/pretrain/repro_%s_ours_paths.json' % args.geometry_mode
    dst = os.path.join(root, out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')

    print('wrote %s' % dst)
    print('  lr=%s  epoch=%s  batch=%s  task=%s  geometry=%s'
          % (cfg['run_cfg']['learning_rate'], train['epoch'], train['batch_size'],
             train['task'], cfg['model_cfg']['geometry_mode']))
    print('  train txt : %s' % train['txt'])
    print('  annotations: %s' % note)
    print('  VAST ckpt : %s' % vast)
    return 0


if __name__ == '__main__':
    sys.exit(main())
