#!/usr/bin/env python3
"""Does reliability weighting fix the deficit? Answered from cached features, no training.

    python3 scripts/try_reliability_centroid.py results/e4_transfer/feats/sca_activitynet.pt
    python3 scripts/try_reliability_centroid.py <dump> --rates 0 0.5 0.9

The question. SCA's margin over the released GRAM checkpoint tracks how informative the
non-video modalities are, almost monotonically:

    AudioCaps   T->A 25.1 R@1   +3.0        VATEX  T->A 15.1   +0.3
    DiDeMo      T->A  4.1       -0.7        ActivityNet 3.0    -3.7

Two explanations fit that, and they imply different fixes:

  A. the text-video pathway is under-trained -- the mask schedule shows a video-only view on
     20.8% of clip-steps against GRAM's 100% (scripts/diag_mask_schedule.py).
  B. a UNIFORM mean cannot discount a modality that carries no signal, while a Gramian
     volume implicitly can. On ActivityNet audio retrieves at 3.0 R@1 and still supplies
     half the centroid.

B is the one that acts at inference: evaluation always scores with the full blended
centroid, so the video-only pathway of (A) is never exercised there. This script tests B
directly by re-scoring a cached feature dump with weights instead of uniform -- no
retraining, no new checkpoint, minutes not GPU-days.

The weighted centroid is still a centroid: mu = normalise(sum_m w_m z_m), arity-invariant,
no determinant, uniform recovered exactly as tau -> infinity. Nothing that motivates the
method changes; only the equal-reliability assumption is dropped.

Read the output as: if R@1 rises materially at rate 0, B is the mechanism and the fix
belongs at inference (and then, for the paper, in the objective too). If it does not move,
B is wrong -- and since A cannot act at inference either, the deficit is in the video
representation itself, which is a different problem.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from model.centroid import masked_spherical_mean  # noqa: E402
from evaluation.run_eval_grids import _load_features  # noqa: E402


def recall_at_k(scores, gt_cols, ks=(1, 5, 10)):
    """scores: (Nt, Ng) higher is better; gt_cols: (Nt,) index of each text's true clip."""
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == gt_cols.unsqueeze(1)).float().argmax(dim=1)
    return {k: 100.0 * (ranks < k).float().mean().item() for k in ks}


def drop_modalities(present, rate, seed):
    """Zero out `rate` of the present (clip, modality) slots, never emptying a clip."""
    if rate <= 0:
        return present.clone()
    g = torch.Generator().manual_seed(seed)
    keep = present.clone()
    for j in range(present.shape[0]):
        idx = present[j].nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        n_drop = int(round(rate * idx.numel()))
        n_drop = min(n_drop, idx.numel() - 1)      # always leave one modality
        if n_drop <= 0:
            continue
        perm = torch.randperm(idx.numel(), generator=g)[:n_drop]
        keep[j, idx[perm]] = 0.0
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump', help='feature file from run_eval_grids.py --dump_features')
    ap.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument('--taus', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.dump):
        print('FATAL: %s not found -- run this where the dumps live' % args.dump, file=sys.stderr)
        return 2

    feat_t, gallery, gt_cols, _ids = _load_features(args.dump)
    feat_t = feat_t.float()
    z = torch.stack([g.float() for g in gallery], dim=1)            # (N, L, d)
    z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    present = (torch.stack([g.float() for g in gallery], dim=1).norm(dim=-1) > 0.5).float()
    if gt_cols is None:
        gt_cols = torch.arange(feat_t.shape[0])
    gt_cols = gt_cols.long()

    print('dump            : %s' % args.dump)
    print('  texts         : %d' % feat_t.shape[0])
    print('  clips         : %d   modalities: %d' % (z.shape[0], z.shape[1]))
    print('  mean present  : %.2f of %d' % (present.sum(1).mean().item(), z.shape[1]))

    # per-modality solo strength, to show WHICH modality the weighting should discount
    print('\nper-modality text retrieval (R@1, each modality alone):')
    for m in range(z.shape[1]):
        solo = feat_t @ z[:, m, :].T
        r = recall_at_k(solo, gt_cols, ks=(1,))
        print('    modality %d : %5.1f' % (m, r[1]))

    print('\n%-7s %-22s %s' % ('rate', 'uniform R@1/R@5/R@10', 'reliability-weighted by tau'))
    print('-' * 78)
    for rate in args.rates:
        pres = drop_modalities(present, rate, args.seed)
        mu_u, _, _ = masked_spherical_mean(z, pres)
        r_u = recall_at_k(feat_t @ mu_u.T, gt_cols)
        cells = []
        for tau in args.taus:
            mu_w, _, _ = masked_spherical_mean(z, pres, weighting='reliability', tau_w=tau)
            r_w = recall_at_k(feat_t @ mu_w.T, gt_cols)
            cells.append('t=%.2f %5.1f (%+.1f)' % (tau, r_w[1], r_w[1] - r_u[1]))
        print('%-7.2f %5.1f/%5.1f/%5.1f      %s'
              % (rate, r_u[1], r_u[5], r_u[10], '  '.join(cells)))

    print('\nRaw-scorer R@1, not the ITM-reranked table metric -- read the DELTA, not the level.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
