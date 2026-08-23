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
import glob
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


def sca_train_txt(cfg_path, data_root):
    """The training annotation file OUR reported row uses, read from its config rather than
    assumed. This is the thing the substitution has to match."""
    if not os.path.exists(cfg_path):
        sys.exit('FATAL: %s not found -- it is what the substituted annotation file is checked '
                 'against, so there is nothing to verify equality with.' % cfg_path)
    c = json.load(open(cfg_path))
    tr = c.get('data_cfg', {}).get('train', [])
    if not tr or not tr[0].get('txt'):
        sys.exit('FATAL: %s names no training annotation file.' % cfg_path)
    return tr[0]['txt'].replace('${DATA_ROOT}', data_root)


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
    ap.add_argument('--allow_annotation_mismatch', action='store_true',
                    help='substitute our annotation file for theirs, AFTER verifying it is the '
                         'same one --sca_train_config trains on')
    ap.add_argument('--sca_train_config', default='config/sca/pretrain_cfg/sca_paper.json',
                    help='the config of OUR reported row; the substituted annotation file must '
                         'be the one it trains on, or the comparison is unequal both ways')
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
    #
    # There are TWO "equal data" properties here and only one is obtainable. Their
    # annotations150k_clean.json is the VAST-150k caption pool restricted to the videos THEY
    # managed to download (their scripts/ writes download_progress.json and drops the
    # failures); ours is the same pool restricted to the videos WE managed to download.
    # Neither is canonical, they do not ship theirs, and it would not match our video store
    # anyway. So:
    #
    #   equal to their published training set : unobtainable
    #   equal to OUR SCA row's training set   : obtainable, and VERIFIED below
    #
    # The second is the one the table needs -- a HyperGRAM row trained on a different subset
    # than the SCA row is exactly the confound the single-configuration rule exists to
    # prevent. So the substitution is not merely permitted, it is checked against the file
    # our own paper config trains on, and refused if it is a different file. A substitution
    # that is unequal in BOTH directions is worth nothing.
    train = cfg['data_cfg']['train'][0]
    theirs = os.path.basename(train['txt'])                 # annotations150k_clean.json
    ours_dir = os.path.join(args.data_root, 'vast27m_150k')
    same_name = os.path.join(ours_dir, theirs)
    fallback = os.path.join(ours_dir, 'annotations150k.json')
    if os.path.exists(same_name):
        train['txt'] = same_name
        note = 'their annotation file, present in our tree'
    elif args.allow_annotation_mismatch and os.path.exists(fallback):
        sca = sca_train_txt(args.sca_train_config, args.data_root)
        if os.path.realpath(sca) != os.path.realpath(fallback):
            sys.exit('FATAL: the substitute %s is NOT the file our SCA row trains on (%s says\n'
                     '       %s). Substituting a third training set would leave the row unequal\n'
                     '       to their published data AND unequal to ours, which is worse than\n'
                     '       not running it.' % (fallback, args.sca_train_config, sca))
        st = os.stat(fallback)
        train['txt'] = fallback
        note = ('SUBSTITUTED annotations150k.json for their %s. VERIFIED identical to the file '
                '%s trains on (%d bytes), so this row is equal-data with our SCA row. It is '
                'NOT equal-data with their published run: their %s is the caption pool '
                'restricted to the videos they downloaded, which we do not have. Report as '
                "\"authors' code, our environment, our 150k subset\" -- never as reproducing "
                'their published number.'
                % (theirs, args.sca_train_config, st.st_size, theirs))
    else:
        sys.exit('FATAL: they train on %s; we have only %s.\n'
                 '       Those are different subsets of VAST-150k -- each is the caption pool\n'
                 '       restricted to the videos that side managed to download -- so swapping\n'
                 '       one for the other silently makes the comparison unequal in the\n'
                 '       training data.\n'
                 '       Passing --allow_annotation_mismatch substitutes ours AFTER checking it\n'
                 '       is the same file our SCA row trains on, which is the equality the\n'
                 '       table actually needs. The substitution is recorded in the config.'
                 % (theirs, fallback))

    train['vision'] = os.path.join(ours_dir, 'clips')
    train['audio'] = os.path.join(ours_dir, 'audios_wav')

    # Their repo ships no datasets/ directory, so the annotation JSONs its val block names
    # (datasets/annotations/<bench>/descs_ret_test.json) resolve nowhere inside their tree.
    # Point them at ours: these are the same VAST-family annotation files both forks read, and
    # using ours keeps the eval split identical to every other row in our table.
    code_dir = os.environ.get('CODE_DIR') or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..')
    val_notes = []
    for d in cfg['data_cfg'].get('val', []):
        name = d.get('name', '')
        if name.startswith('vatex'):
            d['vision'] = os.path.join(args.data_root, 'VATEX/videos')
            d['audio'] = os.path.join(args.data_root, 'VATEX/audios')
        if not os.path.isabs(d['txt']):
            ours = os.path.join(code_dir, d['txt'])
            if not os.path.exists(ours):
                # Same shape as the training annotations: their description list names videos
                # from THEIR download, and d['vision'] above points at OUR video store, so a
                # list referencing videos we do not hold makes checkpoint selection meaningless
                # rather than merely different. Our subset file is the only coherent choice --
                # but it is a real deviation (save_best picks a different checkpoint), so it is
                # gated on the same flag and recorded, never swapped in quietly.
                stem = os.path.basename(ours).replace('.json', '')
                cands = sorted(glob.glob(os.path.join(os.path.dirname(ours), stem + '_*.json')))
                if args.allow_annotation_mismatch and len(cands) == 1:
                    val_notes.append('val %s: SUBSTITUTED %s for their %s (our VATEX subset -- '
                                     'their list names videos we did not download, and the '
                                     'vision root is ours). save_best therefore selects on our '
                                     'subset.' % (name, os.path.basename(cands[0]),
                                                  os.path.basename(ours)))
                    d['txt'] = cands[0]
                    continue
                sys.exit('FATAL: val annotation %s not found at %s -- their repo ships no\n'
                         '       datasets/ directory, so it has to come from ours.\n'
                         '       Sibling candidates: %s\n'
                         '       With --allow_annotation_mismatch a SINGLE candidate is used '
                         'and recorded; %s.'
                         % (d['txt'], ours, cands or 'none',
                            'none were found' if not cands else
                            'here there are %d, which is ambiguous' % len(cands)))
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
        'val_annotations': val_notes or 'their filenames, present in our tree',
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
