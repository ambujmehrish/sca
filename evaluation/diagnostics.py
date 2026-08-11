#!/usr/bin/env python3
"""E8 diagnostics (plan P5): representation-quality metrics + embedding projections, all
computed from cached features (the run_eval_grids.py dumps), so one encoder pass serves
retrieval grids AND diagnostics.

Metrics:
  modality_gap : per modality pair, ||mean(X_a) - mean(X_b)|| of the L2-normalised
                 embeddings + the gap direction's cosine with the centroid difference
                 (Liang et al.'s modality-gap phenomenon).
  rankme       : RankMe effective rank = exp(entropy of normalised singular values)
                 (Garrido et al.); also reports the 99%-energy rank.
  align_unif   : Wang-Isola alignment (mean ||x - y||^2 over positive pairs) and
                 uniformity (log mean exp(-2||x-x'||^2)) per modality.
  tsne / pca   : 2-D projection of all modality embeddings (+ optional concept labels)
                 for the by-concept scatter. t-SNE requires scikit-learn -- a missing
                 dependency is a hard error when tsne is requested, never a silent PCA
                 substitution (PCA is its own explicit choice).

CLI:
  python3 evaluation/diagnostics.py --features feats.pt --out diag.json \
      [--projection pca|tsne|none] [--proj_out proj.pt] [--labels labels.pt]
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@torch.no_grad()
def modality_gap(feats_by_mod):
    """feats_by_mod: {name: (N, d)} L2-normalised. Returns per-pair gap stats."""
    means = {m: F.normalize(x.float(), dim=-1).mean(0) for m, x in feats_by_mod.items()}
    out = {}
    names = sorted(means)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f'{a}-{b}'] = {'gap': (means[a] - means[b]).norm().item(),
                               'norm_a': means[a].norm().item(),
                               'norm_b': means[b].norm().item()}
    return out


@torch.no_grad()
def rankme(x, energy=0.99):
    """RankMe effective rank of (N, d) embeddings: exp(H(p)) over p = sigma / ||sigma||_1,
    plus the smallest rank capturing `energy` of the spectral energy."""
    x = x.float() - x.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp(min=1e-12)
    eff = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    e = (s ** 2) / (s ** 2).sum().clamp(min=1e-12)
    k99 = int((e.cumsum(0) < energy).sum().item()) + 1
    return {'rankme': eff, f'rank@{energy:.0%}': k99, 'dim': x.shape[1]}


@torch.no_grad()
def align_unif(x, y=None, t=2.0):
    """Wang-Isola metrics on L2-normalised embeddings. With y: alignment over the positive
    pairs (x_i, y_i) + uniformity of each side; alone: uniformity of x."""
    x = F.normalize(x.float(), dim=-1)
    out = {}
    if y is not None:
        y = F.normalize(y.float(), dim=-1)
        out['align'] = ((x - y) ** 2).sum(-1).mean().item()
    n = min(x.shape[0], 2048)                                  # pairwise term, bounded
    xs = x[:n]
    sq = torch.cdist(xs, xs) ** 2
    off = ~torch.eye(n, dtype=torch.bool)
    out['uniformity'] = torch.log(torch.exp(-t * sq[off]).mean() + 1e-12).item()
    return out


@torch.no_grad()
def project_2d(feats_by_mod, method='pca', seed=0, max_per_mod=2000):
    """2-D projection of the concatenated modality embeddings. Returns
    (coords (N, 2), mod_names (N,), slices {mod: (start, end)}).
    method 'tsne' REQUIRES scikit-learn (hard error if absent); 'pca' is torch-native."""
    names, chunks = [], []
    for m in sorted(feats_by_mod):
        x = F.normalize(feats_by_mod[m].float(), dim=-1)[:max_per_mod]
        chunks.append(x)
        names += [m] * x.shape[0]
    X = torch.cat(chunks)
    if method == 'pca':
        Xc = X - X.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(Xc, q=2)
        coords = Xc @ V[:, :2]
    elif method == 'tsne':
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError('t-SNE projection requires scikit-learn (pip install '
                              'scikit-learn); PCA is available via --projection pca -- '
                              'refusing to substitute it silently.') from e
        coords = torch.from_numpy(
            TSNE(n_components=2, random_state=seed, init='pca',
                 perplexity=min(30, max(5, X.shape[0] // 100))).fit_transform(X.numpy()))
    else:
        raise ValueError(method)
    slices, start = {}, 0
    for m, c in zip(sorted(feats_by_mod), chunks):
        slices[m] = (start, start + c.shape[0])
        start += c.shape[0]
    return coords, names, slices


@torch.no_grad()
def run_diagnostics(feat_t, gallery_by_mod, gt_cols=None):
    """The E8 metric bundle on one feature dump. gallery_by_mod: {name: (G, d)}."""
    mods = {'t': feat_t, **gallery_by_mod}
    out = {'modality_gap': modality_gap(mods),
           'rankme': {m: rankme(x) for m, x in mods.items()},
           'align_unif': {}}
    if gt_cols is None and all(x.shape[0] == feat_t.shape[0] for x in gallery_by_mod.values()):
        gt_cols = torch.arange(feat_t.shape[0])
    for m, x in gallery_by_mod.items():
        y = x[gt_cols] if gt_cols is not None else None
        out['align_unif'][f't-{m}'] = align_unif(feat_t, y)
    out['align_unif']['t'] = align_unif(feat_t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='run_eval_grids feature dump')
    ap.add_argument('--out', required=True)
    ap.add_argument('--projection', default='none', choices=['none', 'pca', 'tsne'])
    ap.add_argument('--proj_out', help='where to save the 2-D coords (.pt)')
    ap.add_argument('--labels', help='optional concept labels .pt aligned with the gallery')
    args = ap.parse_args()

    from evaluation.run_eval_grids import _load_features
    feat_t, gallery, gt_cols, ids = _load_features(args.features)
    d = torch.load(args.features, map_location='cpu')
    by_mod = (d['gallery'] if isinstance(d.get('gallery'), dict)
              else {f'm{i}': g for i, g in enumerate(gallery)})

    res = run_diagnostics(feat_t, by_mod, gt_cols)
    if args.projection != 'none':
        coords, names, slices = project_2d({'t': feat_t, **by_mod}, method=args.projection)
        res['projection'] = {'method': args.projection, 'slices': slices}
        if args.proj_out:
            payload = {'coords': coords, 'mod': names, 'slices': slices}
            if args.labels:
                payload['labels'] = torch.load(args.labels, map_location='cpu')
            torch.save(payload, args.proj_out)
            res['projection']['proj_out'] = os.path.abspath(args.proj_out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ('modality_gap', 'rankme')}, indent=1)[:800])
    print(f'[diagnostics] -> {args.out}')


if __name__ == '__main__':
    main()
