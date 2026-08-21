#!/usr/bin/env python3
"""Can the aggregator's advantage reach the reported metric at all? Answered post hoc.

    python3 scripts/sweep_score_fusion.py dumps/sca_didemo/rerank_*.pt
    python3 scripts/sweep_score_fusion.py dumps/*/rerank_*.pt --weights 0 0.25 0.5 1 2

The problem this measures. The reported number is produced by refine_score_matrix, which
allocates a zero matrix and writes the ITM probability into the top-k cells only. The
dual-encoder score -- the centroid for SCA, the Gramian volume for GRAM -- never enters the
final ranking. It only decides WHICH k clips the cross-encoder scores. So an aggregator that
is better at retrieval can improve the reported number through exactly one channel: putting
the ground truth inside the candidate set more often. Once recall@k is saturated, that
channel is closed and a better aggregator is worth precisely zero.

That is measurable, and this script measures it: `cand recall@k` is the ceiling on how much
any aggregator improvement can matter. If it reads 99.5 for both methods, the +4.6 R@1 the
centroid wins on DiDeMo cannot appear in the table no matter how large it grows.

The second half sweeps score fusion, which reopens the channel:

    fused = itm + w * z(dual)          z = per-query standardisation over the candidate set

BLIP does this (`score_matrix_i2t[start+i,topk_idx] = score + topk_sim`, verified in
salesforce/BLIP train_retrieval.py); ALBEF does not (`= score`), and GRAM follows ALBEF. So
fusion is an established variant of the protocol rather than a new one, but it is not the
universal convention and adopting it is a protocol change that has to be declared and
applied to every arm, baselines included.

Standardising per query rather than adding the raw score (as BLIP does) is what makes one w
mean the same thing for a volume and for a centroid; the two live on different scales, and a
raw sum would hand the method with the larger spread a free advantage.

w=0 must reproduce the number in the eval log exactly. It is printed for that reason -- if it
disagrees, this script's metric is wrong and nothing below it should be believed.
"""
import argparse
import glob
import os
import sys

import torch


def recall_forward(score, gt_cols, ks=(1, 5, 10)):
    """Text-to-clip, matching compute_metric_ret(direction='forward')."""
    order = score.argsort(dim=1, descending=True)
    ranks = (order == gt_cols.unsqueeze(1)).float().argmax(dim=1)
    return [100.0 * (ranks < k).float().mean().item() for k in ks]


def recall_backward(score, gt_rows, ks=(1, 5, 10)):
    """Clip-to-text. compute_metric_ret(direction='backward') sorts each COLUMN and takes the
    BEST-ranked of that clip's captions, because a clip may have several."""
    order = score.argsort(dim=0, descending=True)          # (Nt, Ng) row = rank position
    ranks = []
    for j, rows in enumerate(gt_rows):
        col = order[:, j].tolist()
        ranks.append(min(col.index(r) for r in rows))
    ranks = torch.tensor(ranks, dtype=torch.float)
    return [100.0 * (ranks < k).float().mean().item() for k in ks]


def gt_maps(ids, ids_txt):
    """(gt_cols, gt_rows): the gallery column of each text, and the text rows of each clip."""
    col_of = {cid: j for j, cid in enumerate(ids)}
    missing = [t for t in ids_txt if t not in col_of]
    if missing:
        raise KeyError('%d caption ids are absent from the gallery id list (e.g. %r) -- the '
                       'dump is inconsistent and no metric computed from it is meaningful'
                       % (len(missing), missing[0]))
    gt_cols = torch.tensor([col_of[t] for t in ids_txt], dtype=torch.long)
    gt_rows = [[] for _ in ids]
    for i, t in enumerate(ids_txt):
        gt_rows[col_of[t]].append(i)
    empty = sum(1 for r in gt_rows if not r)
    if empty:
        raise ValueError('%d gallery clips have no caption -- backward retrieval is undefined '
                         'for them; refusing to report a number over a partial gallery' % empty)
    return gt_cols, gt_rows


def candidate_mask(dual, k, direction):
    """The exact set refine_score_matrix scored: topk of the dual score, per query."""
    mask = torch.zeros_like(dual, dtype=torch.bool)
    if direction == 'forward':
        mask.scatter_(1, dual.topk(k, dim=1)[1], True)
    else:
        mask.scatter_(0, dual.topk(k, dim=0)[1], True)
    return mask


def fuse(itm, dual, mask, w, direction):
    """itm + w * z(dual), computed within each query's candidate set.

    Non-candidates keep the 0 that refine_score_matrix left them, so at w=0 this is the
    reported matrix untouched. For w>0 the fused candidate scores are shifted to stay
    strictly positive, which keeps every scored clip ranked above every unscored one --
    the ordering the current protocol already has, not a new one.
    """
    if w == 0:
        return itm.clone()
    dim = 1 if direction == 'forward' else 0
    m = mask.float()
    n = m.sum(dim=dim, keepdim=True).clamp(min=1)
    mean = (dual * m).sum(dim=dim, keepdim=True) / n
    var = (((dual - mean) * m) ** 2).sum(dim=dim, keepdim=True) / n
    z = (dual - mean) / var.sqrt().clamp(min=1e-6)
    fused = itm + w * z
    # push the lowest-scoring candidate just above zero, per query
    lo = torch.where(mask, fused, torch.full_like(fused, float('inf'))).min(dim=dim, keepdim=True)[0]
    fused = fused + (1e-3 - lo).clamp(min=0)
    return torch.where(mask, fused, torch.zeros_like(fused))


def report(path, weights):
    d = torch.load(path, map_location='cpu')
    dual, itm_f, itm_b = d['dual'].float(), d['itm_fwd'].float(), d['itm_bwd'].float()
    ids, ids_txt, k = d['ids'], d['ids_txt'], d['itm_rerank_num']
    gt_cols, gt_rows = gt_maps(ids, ids_txt)

    m_f = candidate_mask(dual, k, 'forward')
    m_b = candidate_mask(dual, k, 'backward')
    # the ceiling: how often the aggregator put the answer in front of the cross-encoder
    hit_f = 100.0 * m_f.gather(1, gt_cols.unsqueeze(1)).float().mean().item()
    hit_b = 100.0 * sum(bool(m_b[rows, j].any()) for j, rows in enumerate(gt_rows)) / len(gt_rows)

    print('\n== %s' % os.path.basename(path))
    print('   task %s   texts %d   clips %d   rerank k=%d'
          % (d['task'], dual.shape[0], dual.shape[1], k))
    print('   dual-encoder R@1      T2V %5.1f   V2T %5.1f'
          % (recall_forward(dual, gt_cols)[0], recall_backward(dual, gt_rows)[0]))
    print('   cand recall@%-3d       T2V %5.1f   V2T %5.1f   <- ceiling on any aggregator gain'
          % (k, hit_f, hit_b))

    print('   %-6s %-22s %-22s' % ('w', 'T2V R@1/R@5/R@10', 'V2T R@1/R@5/R@10'))
    base = None
    for w in weights:
        f = recall_forward(fuse(itm_f, dual, m_f, w, 'forward'), gt_cols)
        b = recall_backward(fuse(itm_b, dual, m_b, w, 'backward'), gt_rows)
        if base is None:
            base = (f[0], b[0])
        print('   %-6.2f %5.1f/%5.1f/%5.1f  (%+5.1f)  %5.1f/%5.1f/%5.1f  (%+5.1f)'
              % (w, f[0], f[1], f[2], f[0] - base[0], b[0], b[1], b[2], b[0] - base[1]))
    print('   w=0 must equal the ITM number in the eval log; if it does not, this metric is wrong.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dumps', nargs='+', help='rerank_*.pt files from SCA_DUMP_RERANK')
    ap.add_argument('--weights', type=float, nargs='+',
                    default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    paths = sorted(p for pat in args.dumps for p in glob.glob(pat))
    if not paths:
        print('no dumps matched %r -- run an eval with SCA_DUMP_RERANK set' % args.dumps,
              file=sys.stderr)
        return 2
    for p in paths:
        report(p, args.weights)
    print('\nRead cand recall@k first. Where it is saturated for both methods, the reported')
    print('metric cannot reflect the aggregator at all and fusion is the only channel left.')
    print('Any w chosen here must then be fixed once and applied to every arm, GRAM included.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
