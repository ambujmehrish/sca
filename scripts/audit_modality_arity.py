#!/usr/bin/env python3
"""How many gallery modalities does the model train on, and how many is it tested on?

    python3 scripts/audit_modality_arity.py
    python3 scripts/audit_modality_arity.py --configs config/sca/pretrain_cfg/sca_pretrain.json

The centroid's arity is decided by the DATA, not by the config. model/sca.py:149 builds the
gallery as {v, a} and adds 's' only if the batch carries raw_subtitles, which happens only
if the annotation carries a subtitle field. So an arm can train at k=2 for its whole run and
then be evaluated on a task string that asks for k=3, and nothing in the config, the log or
the completion audit would say so.

That combination is the one thing that would explain VATEX specifically. VATEX is the only
T-VAS benchmark, and it is the only benchmark where SCA's aggregation tax is catastrophic:
-9.5 R@1 against its own best modality, where the released GRAM checkpoint loses 1.9 -- on
the benchmark where SCA's video features are the STRONGEST it has (81.4 vs GRAM's 77.5). A
uniform mean over three modalities, one of which retrieves at 15.1 while another retrieves
at 81.4, is exactly the failure that would produce that; a determinant discounts a
near-uninformative axis on its own, and a uniform mean cannot.

But it only counts as an explanation if the arity gap is real, so this measures it rather
than assuming it -- for every benchmark at once, since a fix aimed at one modality on one
dataset is not a fix.

Reads annotations only: no GPU, no model, no checkpoints.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# probed in order; the annotation dialect is not assumed, and whichever key is found is
# reported so a rename shows up as "no subtitle field" rather than silently as k=2
SUBTITLE_KEYS = ('subtitle', 'subtitles', 'raw_subtitles')


def expand(p):
    p = os.path.expandvars(str(p))
    if '$' in p:                       # an unset ${DATA_ROOT} would silently miss the file
        return None
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def probe(txt):
    """-> (n_entries, subtitle_key, n_with_subtitle) or a reason it could not be read."""
    path = expand(txt)
    if path is None:
        return None, 'unexpanded variable in %r -- source the env rc first' % txt
    if not os.path.exists(path):
        return None, 'not found: %s' % path
    try:
        data = json.load(open(path))
    except (ValueError, IOError) as e:
        return None, 'unreadable: %s' % e
    if not isinstance(data, list) or not data:
        return None, 'not a non-empty list of entries'
    key = next((k for k in SUBTITLE_KEYS
                if any(isinstance(e, dict) and e.get(k) for e in data[:2000])), None)
    n_sub = sum(1 for e in data if isinstance(e, dict) and key and e.get(key)) if key else 0
    return (len(data), key, n_sub), None


def entries(cfg_path):
    """-> [(split, name, task, txt)] for every train and val block in a config."""
    try:
        cfg = json.load(open(cfg_path))
    except (ValueError, IOError):
        return []
    out = []
    for split in ('train', 'val'):
        for block in (cfg.get('data_cfg', {}).get(split) or []):
            if block.get('txt'):
                out.append((split, block.get('name', '?'), block.get('task', '?'), block['txt']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', help='default: the pretrain config + all e1 evals')
    args = ap.parse_args()

    paths = args.configs or (
        [os.path.join(ROOT, 'config/sca/pretrain_cfg/sca_pretrain.json')]
        + sorted(glob.glob(os.path.join(ROOT, 'benchmark_eval/configs_e1/sca_*.json'))))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print('no configs found', file=sys.stderr)
        return 2

    print('%-22s %-6s %-14s %9s %9s %6s  %s'
          % ('config', 'split', 'task', 'entries', 'w/ subs', 'k', 'annotation'))
    print('-' * 104)
    seen, train_k, eval_k = {}, None, {}
    for p in paths:
        for split, name, task, txt in entries(p):
            got, err = probe(txt)
            label = os.path.basename(p).replace('.json', '')
            if err:
                print('%-22s %-6s %-14s %s' % (label, split, task, err))
                continue
            n, key, n_sub = got
            # k = gallery modalities the centroid will actually be built over
            k = 2 + (1 if n_sub else 0)
            print('%-22s %-6s %-14s %9d %9s %6d  %s'
                  % (label, split, task, n,
                     ('%d (%s)' % (n_sub, key)) if n_sub else '0', k, os.path.basename(expand(txt))))
            if split == 'train':
                train_k = k if train_k is None else max(train_k, k)
            else:
                eval_k[name] = (k, task)
            seen[(label, split, name)] = (k, task, n_sub)

    print()
    if train_k is None:
        print('No train block resolved, so the train/test comparison cannot be made. Run this')
        print('where the annotations live, with the env rc sourced.')
        return 2

    print('training gallery arity: k=%d' % train_k)
    gaps = {n: (k, t) for n, (k, t) in eval_k.items() if k > train_k}
    for n, (k, t) in sorted(eval_k.items()):
        flag = '  <- ARITY GAP: never trained at this k' if k > train_k else ''
        print('  %-18s k=%d  (task %s)%s' % (n, k, t, flag))

    if not gaps:
        print('\nNo arity gap: every benchmark is evaluated at an arity the model trained at.')
        print('Whatever is wrong with VATEX, it is not that the centroid meets an unseen k.')
        return 0
    print('\n%d benchmark(s) evaluated above the training arity. The centroid is arity-invariant'
          % len(gaps))
    print('by construction, so this is not a crash -- it is that the UNIFORM weighting was')
    print('never exercised at this k, and a uniform mean cannot discount a weak modality.')
    print('This is also the regime where a learned or confidence weighting is well defined:')
    print('at k=2 leave-one-out consensus is degenerate by symmetry (cos(z0,z1) == cos(z1,z0)')
    print('forces 0.5/0.5), and at k>=3 it is not.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
