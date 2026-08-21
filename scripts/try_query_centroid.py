#!/usr/bin/env python3
"""Does query-conditioned weighting recover the aggregation tax? Measured, no training.

    python3 scripts/try_query_centroid.py results/e4_transfer/feats/sca_activitynet.pt
    python3 scripts/try_query_centroid.py results/e4_transfer/feats/*.pt --taus 0.02 0.05 0.1

The tax. Every aggregator we have scores WORSE than the best single modality it fused:

    benchmark      best 1-mod   SCA centroid   tax        GRAM volume   tax
    VATEX             81.4          71.9      -9.5           75.6      -1.9
    ActivityNet       38.4          33.3      -5.1           31.0      -5.8
    DiDeMo            37.1          32.8      -4.3           28.2      -3.8
    AudioCaps         25.1          26.1      +1.0           22.9      -6.1

VATEX is the clearest: SCA holds the better video features (81.4 against 77.5) and still
produces the worse aggregate, because a uniform mean gives a subtitle stream retrieving at
15.1 the same share as video retrieving at 81.4. That is 9.5 points of pure loss, and it is
the largest single number anywhere in our results.

The fix weights modalities by the query -- w_m proportional to exp(<t, z_m>/tau) -- which is
parameter-free, so it applies to checkpoints that are already trained. tau interpolates
between the uniform centroid (tau -> inf, reproduced exactly) and max-over-modalities
(tau -> 0). The useful setting is in between: enough to discount a modality the query does
not care about, not so much that fusion collapses to a single stream.

Read the output as: the uniform row is what the paper currently reports; 'best 1-mod' is the
ceiling that a pure max would reach; anything above 'best 1-mod' is fusion adding value
rather than destroying it. If the peak sits strictly between the two, the weighting is doing
real work and belongs in the objective, not just at inference.

This measures the AGGREGATOR score. That is the quantity the tax is defined on, and it feeds
the shortlist the cross-encoder reranks -- shortlist recall is 80.6 on DiDeMo, so it is not
saturated and improving it can still move the reported metric.
"""
import argparse
import glob
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from model.centroid import masked_spherical_mean, query_centroid_scores  # noqa: E402
from evaluation.run_eval_grids import _load_features  # noqa: E402


def recall(scores, gt_cols, ks=(1, 5, 10)):
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == gt_cols.unsqueeze(1)).float().argmax(dim=1)
    return [100.0 * (ranks < k).float().mean().item() for k in ks]


def report(path, taus):
    feat_t, gallery, gt_cols, _ids = _load_features(path)
    feat_t = feat_t.float()
    feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z = torch.stack([g.float() for g in gallery], dim=1)             # (Ng, L, d)
    present = (z.norm(dim=-1) > 0.5).float()                         # zero-filled = absent
    z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z = z * present.unsqueeze(-1)
    if gt_cols is None:
        gt_cols = torch.arange(feat_t.shape[0])
    gt_cols = gt_cols.long()

    print('\n== %s' % os.path.basename(path))
    print('   texts %d   clips %d   modalities %d   mean present %.2f'
          % (feat_t.shape[0], z.shape[0], z.shape[1], present.sum(1).mean().item()))

    solo = []
    for m in range(z.shape[1]):
        r = recall(feat_t @ z[:, m, :].T, gt_cols, ks=(1,))[0]
        solo.append(r)
    best_solo = max(solo)
    print('   per-modality R@1 : %s   -> best single modality %.1f'
          % ('  '.join('m%d %5.1f' % (m, v) for m, v in enumerate(solo)), best_solo))

    mu_u, _, _ = masked_spherical_mean(z, present)
    r_u = recall(feat_t @ mu_u.T, gt_cols)
    print('   uniform centroid : %5.1f/%5.1f/%5.1f      tax vs best single modality %+.1f'
          % (r_u[0], r_u[1], r_u[2], r_u[0] - best_solo))

    print('   %-8s %-24s %-10s %s' % ('tau', 'R@1/R@5/R@10', 'vs uniform', 'vs best 1-mod'))
    best = (None, -1.0)
    for tau in taus:
        r = recall(query_centroid_scores(feat_t, z, present, tau=tau), gt_cols)
        print('   %-8.3f %5.1f/%5.1f/%5.1f            %+6.1f     %+6.1f'
              % (tau, r[0], r[1], r[2], r[0] - r_u[0], r[0] - best_solo))
        if r[0] > best[1]:
            best = (tau, r[0])
    print('   best tau %.3f at R@1 %.1f  (uniform %.1f, best single modality %.1f)'
          % (best[0], best[1], r_u[0], best_solo))
    return best[1] - r_u[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+', help='feature dumps from run_eval_grids --dump_features')
    ap.add_argument('--taus', type=float, nargs='+',
                    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    args = ap.parse_args()
    paths = sorted(p for pat in args.dumps for p in glob.glob(pat))
    if not paths:
        print('no dumps matched %r' % args.dumps, file=sys.stderr)
        return 2
    gains = []
    for p in paths:
        gains.append((os.path.basename(p), report(p, args.taus)))
    print('\ngain over the uniform centroid, at each dump\'s best tau:')
    for name, g in gains:
        print('  %-40s %+.1f' % (name, g))
    print('\nA tau that helps on one benchmark and hurts on another is a tuned constant, not a')
    print('method. What the paper can claim is a SINGLE tau that is positive everywhere.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
