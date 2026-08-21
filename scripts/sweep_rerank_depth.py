#!/usr/bin/env python3
"""How deep should the cross-encoder rerank? Answered exactly, from dumps already on disk.

    python3 scripts/sweep_rerank_depth.py workdir/e1_fusion/*/dumps/rerank_*.pt

itm_rerank_num = 50 is GRAM's setting, chosen for GRAM's first stage. Ours is stronger:
shortlist recall@50 is 85.2 against GRAM's 80.6 on DiDeMo and 94.2 against 93.4 on
ActivityNet. A better first stage should need a SHALLOWER rerank -- fewer distractors for
the cross-encoder, and the ranking it inherits is already closer to right. If that holds,
the claim is not only accuracy: reranking 10 candidates instead of 50 is 5x less
cross-encoder compute at inference, and GRAM cannot follow because its recall falls away
faster as k shrinks.

Exact, not simulated. refine_score_matrix writes the ITM probability into the top-50 cells
of a zero matrix, and the top-k set for any k <= 50 is a SUBSET of that, so re-slicing the
dump reproduces what an eval at that k would have computed, cell for cell. No GPU, no
re-running the cross-encoder.

Read it as: where does each arm peak? If SCA peaks below 50 and GRAM at 50, the depth is a
property of first-stage quality and both the accuracy and the cost argument follow. If both
peak at 50, the idea is wrong -- the cross-encoder is simply better than the aggregator at
every depth and there is nothing here.
"""
import argparse
import glob
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_score_fusion import gt_maps, recall_forward, recall_backward  # noqa: E402


def at_depth(dual, itm, k, direction):
    """The score matrix an eval with itm_rerank_num=k would have produced.

    Zeros everywhere, ITM probability in the top-k cells by dual score -- exactly
    refine_score_matrix, restricted to a k that the dump already covers.
    """
    mask = torch.zeros_like(dual, dtype=torch.bool)
    if direction == 'forward':
        mask.scatter_(1, dual.topk(k, dim=1)[1], True)
    else:
        mask.scatter_(0, dual.topk(k, dim=0)[1], True)
    return torch.where(mask, itm, torch.zeros_like(itm)), mask


def report(path, depths):
    d = torch.load(path, map_location='cpu')
    dual, itm_f, itm_b = d['dual'].float(), d['itm_fwd'].float(), d['itm_bwd'].float()
    ids, ids_txt, k_dump = d['ids'], d['ids_txt'], d['itm_rerank_num']
    gt_cols, gt_rows = gt_maps(ids, ids_txt)
    depths = [k for k in depths if k <= k_dump]

    label = os.path.join(os.path.basename(os.path.dirname(os.path.dirname(path))),
                         os.path.basename(path))
    print('\n== %s   (dump reranked to k=%d)' % (label, k_dump))
    print('   %-6s %-24s %-24s %s' % ('k', 'T2V R@1/R@5/R@10', 'V2T R@1/R@5/R@10', 'cand recall@k'))
    best_f = best_k = None
    for k in depths:
        s_f, m_f = at_depth(dual, itm_f, k, 'forward')
        s_b, m_b = at_depth(dual, itm_b, k, 'backward')
        f = recall_forward(s_f, gt_cols)
        b = recall_backward(s_b, gt_rows)
        hit = 100.0 * m_f.gather(1, gt_cols.unsqueeze(1)).float().mean().item()
        print('   %-6d %5.1f/%5.1f/%5.1f          %5.1f/%5.1f/%5.1f          %5.1f'
              % (k, f[0], f[1], f[2], b[0], b[1], b[2], hit))
        if best_f is None or f[0] > best_f:
            best_f, best_k = f[0], k
    ref = recall_forward(at_depth(dual, itm_f, k_dump, 'forward')[0], gt_cols)[0]
    print('   best k=%d at T2V R@1 %.1f   (k=%d, the reported setting, gives %.1f -> %+.1f)'
          % (best_k, best_f, k_dump, ref, best_f - ref))
    return label, best_k, best_f, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--depths', type=int, nargs='+',
                    default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50])
    args = ap.parse_args()
    paths = sorted(p for pat in args.dumps for p in glob.glob(pat))
    if not paths:
        print('no dumps matched %r -- run slurm_scripts/e1_fusion_dump.sh first' % args.dumps,
              file=sys.stderr)
        return 2
    rows = [report(p, args.depths) for p in paths]
    print('\n%-42s %6s %8s %8s %s' % ('dump', 'best k', 'R@1', 'at k=50', 'gain'))
    for label, bk, bf, ref in rows:
        print('%-42s %6d %8.1f %8.1f %+.1f' % (label, bk, bf, ref, bf - ref))
    print('\nCompare an SCA row against the released-GRAM row for the SAME benchmark. A')
    print('shallower peak for SCA is first-stage quality showing up as both accuracy and')
    print('inference cost; the same peak for both means the depth is not ours to exploit.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
