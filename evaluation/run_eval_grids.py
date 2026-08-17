#!/usr/bin/env python3
"""P4 grid driver: E4 missingness grids + E5 cardinality stats + E6 calibration, one JSON
per (checkpoint x scorer-set) cell of the 2x2 train/test masking design.

Two modes:

FEATURES MODE (anywhere, incl. offline nodes -- grids are pure post-processing):
    python3 evaluation/run_eval_grids.py --features feats.pt --out results/e4/cell.json \
        [--methods centroid volume_masked volume_imputed pmrl_raw pmrl_norm] \
        [--s_star cache.pt] [--rates 0 0.25 0.5 0.75]
  feats.pt: {'feat_t': (T, d), 'gallery': [(G, d), ...] or {'v': ..., 'a': ...},
             'gt_cols': (T,) optional (identity when T == G), 'ids': [G ids] optional
             (required with --s_star)}
  Features are L2-normalised here; build them once with --dump_features below (or any
  script) and score every method/rate from the same tensors -- one encoder pass serves the
  whole grid, and every method sees identical inputs (the honest-baseline design).

MODEL MODE (cluster -- needs the data package + weights):
    python3 evaluation/run_eval_grids.py --config <eval_cfg.json> \
        --dump_features feats.pt [--out results/e4/cell.json ...]
  Runs the eval dataloader through the model exactly like evaluation_mm's collection loop,
  dumps the features, then (with --out) runs the grids on them.
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_missing import missing_grid, recall_at_k, MISS_RATES
from evaluation.eval_calibration import calibration_grid

DEFAULT_METHODS = ('centroid', 'volume_masked', 'volume_imputed', 'pmrl_raw', 'pmrl_norm')


def run_grids(feat_t, gallery_feats, gt_cols=None, s_star=None,
              methods=DEFAULT_METHODS, rates=MISS_RATES, seed=0):
    """The full grid bundle on one feature set. Returns a JSON-ready dict:
    e4: per method x rate x which-modality R@k + per-cardinality score stats +
        rank-displacement bias + per-cardinality affine calibration (E5);
    e6: per method x rate S-vs-S* regression + graded nDCG (only when s_star given);
    full: rate-0 sanity recalls per method."""
    feat_t = F.normalize(feat_t.float(), dim=-1)
    gallery_feats = [F.normalize(g.float(), dim=-1) for g in gallery_feats]
    if gt_cols is None:
        assert feat_t.shape[0] == gallery_feats[0].shape[0], \
            'gt_cols required when text and gallery counts differ'
        gt_cols = torch.arange(feat_t.shape[0])
    out = {'setup': {'n_text': feat_t.shape[0], 'n_gallery': gallery_feats[0].shape[0],
                     'k_gallery_modalities': len(gallery_feats), 'methods': list(methods),
                     'rates': list(rates), 'seed': seed}}
    out['e4'] = missing_grid(feat_t, gallery_feats, gt_cols, methods=methods,
                             rates=rates, seed=seed, calibrate=True)
    if s_star is not None:
        out['e6'] = calibration_grid(feat_t, gallery_feats, s_star, methods=methods,
                                     rates=[r for r in rates if r < 1.0], seed=seed)
    return out


def _load_features(path):
    d = torch.load(path, map_location='cpu')
    if 'gallery' in d:
        gallery = d['gallery']
        if isinstance(gallery, dict):
            gallery = [gallery[k] for k in sorted(gallery)]
    elif 'img' in d and 'txt' in d:               # a10_prepare_flickr8k format (k=2)
        gallery = [d['img']]
        d = {'feat_t': d['txt'][:, 0], 'gallery': gallery, 'ids': d.get('ids')}
    else:
        raise KeyError(f'{path}: expected keys feat_t+gallery (or img+txt)')
    return d['feat_t'], gallery, d.get('gt_cols'), d.get('ids')


@torch.no_grad()
def collect_features(config_path, dump_path):
    """Cluster-side feature extraction: mirrors evaluation_mm's collection loop (model
    eval branch -> feat_t / feat_v / feat_a / feat_s / feat_d + ids), dumps one tensor
    file the grids (and any later re-scoring) run from."""
    # the trunk's args/initialize assume a torch.distributed launcher; default a
    # single-process group so this also runs bare inside an sbatch/srun allocation
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29511')

    from easydict import EasyDict as edict                       # heavy imports stay lazy
    from utils.args import get_args
    from utils.initialize import initialize
    from utils.build_model import build_model
    from utils.build_dataloader import create_val_dataloaders

    sys.argv = ['run_eval_grids', '--config', config_path, '--mode', 'testing']
    args = get_args()
    initialize(args)
    model, _, _ = build_model(args)
    model.eval()
    loaders = create_val_dataloaders(args)
    assert len(loaders) == 1, f'grid extraction expects ONE val loader, got {list(loaders)}'
    (task_name, loader), = loaders.items()
    task = task_name.split('--')[0]

    feats = {'feat_t': [], 'ids': [], 'gallery': {}}
    for batch in loader:
        ev = model(batch, task, compute_loss=False)
        feats['feat_t'].append(ev['feat_t'].float().cpu())
        feats['ids'] += list(batch['ids'])
        for m in ('v', 'a', 's', 'd'):
            if f'feat_{m}' in ev:
                feats['gallery'].setdefault(m, []).append(ev[f'feat_{m}'].float().cpu())
    out = {'feat_t': torch.cat(feats['feat_t']),
           'gallery': {m: torch.cat(v) for m, v in feats['gallery'].items()},
           'ids': feats['ids'],
           'meta': {'config': os.path.abspath(config_path), 'task': task}}
    os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
    torch.save(out, dump_path)
    print(f'[grids] dumped {out["feat_t"].shape[0]} texts x '
          f'{list(out["gallery"])} gallery feats -> {dump_path}')
    return dump_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', help='cached feature file (features mode)')
    ap.add_argument('--config', help='eval config json (model mode, cluster)')
    ap.add_argument('--dump_features', help='where model mode writes the feature dump')
    ap.add_argument('--out', help='grid results json')
    ap.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS))
    ap.add_argument('--rates', type=float, nargs='+', default=list(MISS_RATES))
    ap.add_argument('--s_star', help='SemanticTargets cache for the E6 block')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.config:
        assert args.dump_features, '--config (model mode) requires --dump_features'
        args.features = collect_features(args.config, args.dump_features)
    assert args.features, 'pass --features (or --config with --dump_features)'
    if not args.out:
        return                                                    # dump-only invocation

    feat_t, gallery, gt_cols, ids = _load_features(args.features)
    s_star = None
    if args.s_star:
        from data.semantic_targets import SemanticTargets
        assert ids is not None, '--s_star needs ids in the feature file'
        s_star = SemanticTargets(args.s_star).gather(ids)
    res = run_grids(feat_t, gallery, gt_cols, s_star=s_star,
                    methods=tuple(args.methods), rates=tuple(args.rates), seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    for m in args.methods:
        r0 = res['e4'][m][f'{int(args.rates[0] * 100)}%|rand']
        print(f"[grids] {m}: R@1={r0['R@1']:.2f} at rate {args.rates[0]}")
    print(f'[grids] -> {args.out}')


if __name__ == '__main__':
    main()
