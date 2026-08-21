#!/usr/bin/env python3
"""Is mean-pooling the frame axis what costs us the long-video benchmarks?

    python3 scripts/try_temporal_centroid.py results/temporal/sca_activitynet.pt
    python3 scripts/try_temporal_centroid.py results/temporal/*.pt --taus 0.02 0.05 0.1

The bottleneck. model/general_module.py:426 takes the per-frame CLS tokens and averages
them into ONE vector before the contrastive head:

    feature = feature[:, :, 0]          # (B, frames, patches, C) -> (B, frames, C)
    feature = torch.mean(feature, dim=1)                          # -> (B, C)

That average is the last operation before the head, so a two-minute ActivityNet clip and a
ten-second VATEX clip both reach every aggregator -- ours, GRAM's, PMRL's -- as a single
point. It is upstream of everything the method does, which is why tuning rank, batch size,
adapters and fusion weights could not reach it.

What this measures, on an ALREADY-TRAINED checkpoint, no retraining:

    pooled        the current representation: cos(t, mean_f z_f)          <- what we report
    max           cos(t, z_f) maximised over frames                       <- pure late interaction
    softmax(tau)  sum_f sigma_f cos(t, z_f), sigma = softmax(cos/tau)     <- the set centroid

The third is the proposal: a clip is a SET of centroids, one per frame, scored by
query-conditioned aggregation over the set. tau -> inf recovers pooled exactly (up to the
renormalisation), tau -> 0 recovers max. It is a strict generalisation of SCA -- one frame
gives back the current method -- stays a spherical mean, keeps arity invariance and needs no
determinant. GRAM cannot adopt it: a Gramian volume requires one vector per modality, so it
would have to pool first, which is the operation being removed.

Read it as a CEILING, not a result. These features were trained under mean pooling, so the
frames were never optimised to be individually discriminative. A gain here is a lower bound
on what training with the set representation would give; no gain here kills the idea for the
cost of one eval.

Video pathway only (cos(t, video)), because that is the pathway the pooling damages.
"""
import argparse
import glob
import os
import sys

import torch


def recall(scores, gt_cols, ks=(1, 5, 10)):
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == gt_cols.unsqueeze(1)).float().argmax(dim=1)
    return [100.0 * (ranks < k).float().mean().item() for k in ks]


def softmax_over_frames(feat_t, zf, tau, chunk=64):
    """sum_f sigma_f <t, z_f> with sigma = softmax(<t, z_f>/tau), per (text, clip) pair.

    Never materialises a per-pair vector: the score only needs the (chunk, Ng, F)
    similarities, so peak memory is independent of the embedding dimension.
    """
    out = []
    for i in range(0, feat_t.shape[0], chunk):
        sims = torch.einsum('cd,gfd->cgf', feat_t[i:i + chunk], zf)      # (c, Ng, F)
        w = torch.softmax(sims / max(tau, 1e-6), dim=-1)
        out.append((w * sims).sum(-1))
    return torch.cat(out, dim=0)


def report(path, taus):
    d = torch.load(path, map_location='cpu')
    if 'v_frames' not in d:
        print('\n== %s\n   NO per-frame features in this dump. Re-extract with '
              'model_cfg.dump_frame_feats=true.' % os.path.basename(path))
        return None
    feat_t = d['feat_t'].float()
    feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    zf = d['v_frames'].float()                                            # (Ng, F, d)
    zf = zf / zf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gt_cols = torch.arange(feat_t.shape[0])
    if feat_t.shape[0] != zf.shape[0]:
        # more captions than clips: map each caption to its clip by id
        ids = d['ids']
        col = {}
        for j, c in enumerate(ids):
            col.setdefault(c, j)
        gt_cols = torch.tensor([col[c] for c in ids[:feat_t.shape[0]]], dtype=torch.long)

    print('\n== %s' % os.path.basename(path))
    print('   texts %d   clips %d   frames %d' % (feat_t.shape[0], zf.shape[0], zf.shape[1]))

    pooled = zf.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    r_p = recall(feat_t @ pooled.T, gt_cols)
    print('   pooled (current) : %5.1f/%5.1f/%5.1f' % tuple(r_p))

    sims_max = torch.stack([torch.einsum('cd,gfd->cgf', feat_t[i:i + 64], zf).max(-1).values
                            for i in range(0, feat_t.shape[0], 64)])
    r_m = recall(torch.cat(list(sims_max), dim=0)[:feat_t.shape[0]], gt_cols)
    print('   max over frames  : %5.1f/%5.1f/%5.1f   %+.1f' % (*r_m, r_m[0] - r_p[0]))

    best = (None, -1.0)
    print('   %-8s %-22s %s' % ('tau', 'R@1/R@5/R@10', 'vs pooled'))
    for tau in taus:
        r = recall(softmax_over_frames(feat_t, zf, tau), gt_cols)
        print('   %-8.3f %5.1f/%5.1f/%5.1f          %+6.1f' % (tau, *r, r[0] - r_p[0]))
        if r[0] > best[1]:
            best = (tau, r[0])
    print('   best tau %.3f -> %.1f  (pooled %.1f, gain %+.1f)'
          % (best[0], best[1], r_p[0], best[1] - r_p[0]))
    return best[1] - r_p[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--taus', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    args = ap.parse_args()
    paths = sorted(p for pat in args.dumps for p in glob.glob(pat))
    if not paths:
        print('no dumps matched %r' % args.dumps, file=sys.stderr)
        return 2
    gains = [(os.path.basename(p), report(p, args.taus)) for p in paths]
    print('\ngain over mean-pooled video, at each dump\'s best tau:')
    for name, g in gains:
        print('  %-40s %s' % (name, '--' if g is None else '%+.1f' % g))
    print('\nThe prediction is that the gain tracks video LENGTH: large on ActivityNet and')
    print('DiDeMo, small on VATEX and MSR-VTT. If it is flat across all four, the frame axis')
    print('is not where the missing information is and the idea is wrong.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
