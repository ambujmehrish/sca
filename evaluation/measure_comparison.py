#!/usr/bin/env python3
"""A2: is the FIRST-ORDER alignment measure A(M) sufficient, or does the second-order
volume/spectrum carry extra signal? Compares three per-clip set-alignment measures and the
retrieval rankings they induce, on the same cached features (plan: "A2 answers first-order
sufficiency either way -- both outcomes shape a publishable framing").

Per-clip measures over the PRESENT modality set M (text excluded -- set properties):
  A(M)             mean resultant length  ||1/|M| sum z||          (SCA, first-order)
  logdetG(M)       (1/|M|) log det G_M via the phantom-axis trick  (GRAM geometry)
  lambda1(M)       lambda_1(G_M) / |M|                             (PMRL geometry)

Text-conditional rankings (the E4 scorers): centroid cos vs -volume vs lambda_1 --
reported as pairwise Spearman correlations of the induced gallery rankings + R@1 each.

  python3 evaluation/measure_comparison.py --features feats.pt --out a2.json
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.centroid import masked_spherical_mean
from evaluation.eval_missing import score, recall_at_k


@torch.no_grad()
def set_measures(z, present=None):
    """z: (B, L, d) L2-normalised stack; present: (B, L). Returns the three per-clip
    measures {A, logdetG, lambda1_norm} (each (B,))."""
    B, L, _ = z.shape
    if present is None:
        present = z.new_ones(B, L)
    present = present.float()
    _, A, n = masked_spherical_mean(z.float(), present)

    G = torch.einsum('bld,bmd->blm', z.float(), z.float())          # (B, L, L)
    keep = present.unsqueeze(2) * present.unsqueeze(1)
    G = G * keep + torch.diag_embed(1.0 - present)                  # phantom identity
    n_safe = n.clamp(min=1.0)
    logdet = torch.logdet(G + 1e-6 * torch.eye(L)) / n_safe

    Gz = torch.einsum('bld,bmd->blm', z.float(), z.float()) * keep  # zero-masked for lambda1
    lam1 = torch.linalg.eigvalsh(Gz)[..., -1] / n_safe
    return {'A': A, 'logdetG': logdet, 'lambda1_norm': lam1}


def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    va, vb = ra - ra.mean(), rb - rb.mean()
    return (va * vb).sum() / (va.norm() * vb.norm()).clamp(min=1e-12)


@torch.no_grad()
def compare_measures(feat_t, gallery_feats, gt_cols=None, present=None):
    feat_t = F.normalize(feat_t.float(), dim=-1)
    gallery_feats = [F.normalize(g.float(), dim=-1) for g in gallery_feats]
    G = gallery_feats[0].shape[0]
    if gt_cols is None:
        gt_cols = torch.arange(feat_t.shape[0])
    if present is None:
        present = torch.ones(G, len(gallery_feats))
    z = torch.stack(gallery_feats, dim=1)

    out = {'set_measures': {}, 'ranking': {}}
    meas = set_measures(z, present)
    stats = {k: {'mean': v.mean().item(), 'std': v.std().item()} for k, v in meas.items()}
    corr = {}
    keys = list(meas)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            corr[f'{a}~{b}'] = {'pearson': torch.corrcoef(
                torch.stack([meas[a], meas[b]]))[0, 1].item(),
                'spearman': _spearman(meas[a], meas[b]).item()}
    out['set_measures'] = {'stats': stats, 'pairwise_corr': corr}

    dists = {m: score(feat_t, gallery_feats, present, m)
             for m in ('centroid', 'volume_masked', 'pmrl_norm')}
    for m, d in dists.items():
        rec, _ = recall_at_k(d, gt_cols)
        out['ranking'][m] = rec
    rank_corr = {}
    ms = list(dists)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            per_row = torch.stack([_spearman(dists[a][r], dists[b][r])
                                   for r in range(min(feat_t.shape[0], 500))])
            rank_corr[f'{a}~{b}'] = per_row.mean().item()
    out['ranking']['pairwise_spearman'] = rank_corr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from evaluation.run_eval_grids import _load_features
    feat_t, gallery, gt_cols, _ = _load_features(args.features)
    res = compare_measures(feat_t, gallery, gt_cols)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res['set_measures']['pairwise_corr'], indent=1))
    print(f"[A2] ranking Spearman: {res['ranking']['pairwise_spearman']}")
    print(f'[A2] -> {args.out}')


if __name__ == '__main__':
    main()
