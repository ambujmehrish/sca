#!/usr/bin/env python3
"""SCA finds more of the right clips. Why does the reranker not report them?

    python3 scripts/rerank_conversion.py --a workdir/e1_fusion/released_didemo/dumps/rerank_*.pt \
                                         --b workdir/e1_fusion/sca_didemo/dumps/rerank_*.pt

The reported metric is two stages. The dual encoder picks 50 candidates per query; a frozen
cross-encoder, shared by GRAM, HyperGRAM, PMRL and us and retrained by none of them, reranks
them. SCA's contribution lives entirely in stage 1, so it can reach the reported number
through exactly one channel: putting the ground truth into the candidate set more often.

It does. On DiDeMo, candidate recall@50 is 80.6 for GRAM's released checkpoint and 85.2 for
SCA. The reported R@1 is 50.7 and 50.4. Four and a half points of extra recall convert to
minus three tenths. On AudioCaps the same channel works: +2.4 recall, +3.0 R@1.

Two explanations survive that, and they demand opposite responses:

  RERANKER-LIMITED   The recovered clips are fine; the cross-encoder simply cannot
                     rank them first. Then no amount of stage-1 work will show up, and the
                     effort belongs in the reranking stage or in a protocol that does not
                     discard the dual score.

  CANDIDATE-QUALITY  The recovered clips enter the set at the bottom, on weak evidence, and
                     are genuinely hard. Then the recall number overstates what stage 1 did,
                     and the aggregator has less headroom than it looks.

They are separable from the dumps alone, with no GPU. Partition the queries by whether each
method's candidate set contains the ground truth, and ask what the reranker does with each
group -- in particular with the queries only SCA recovers, which are exactly the ones its
advantage is made of.

This measures. It does not fix. What it decides is where the next run should go.
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_score_fusion import candidate_mask, gt_maps  # noqa: E402


def gt_rank_forward(score, gt_cols):
    """Rank (0-based) of each text's ground-truth clip under a text-to-clip score matrix.

    Same ordering rule as compute_metric_ret(direction='forward'): argsort descending, then
    locate the ground-truth column. Non-candidates carry the 0 that refine_score_matrix left
    them, so they land below every scored clip exactly as in the reported metric.
    """
    order = score.argsort(dim=1, descending=True)
    return (order == gt_cols.unsqueeze(1)).float().argmax(dim=1).long()


def load(path):
    d = torch.load(path, map_location='cpu')
    gt_cols, _gt_rows = gt_maps(d['ids'], d['ids_txt'])
    dual, itm = d['dual'].float(), d['itm_fwd'].float()
    k = int(d['itm_rerank_num'])
    mask = candidate_mask(dual, k, 'forward')
    return {'path': path, 'ids': d['ids'], 'ids_txt': d['ids_txt'], 'k': k,
            'gt_cols': gt_cols, 'dual': dual, 'itm': itm, 'mask': mask,
            # per query: is the ground truth inside the 50 the cross-encoder was given?
            'hit': mask.gather(1, gt_cols.unsqueeze(1)).squeeze(1),
            'rank': gt_rank_forward(itm, gt_cols)}


def pct(x):
    return 100.0 * float(x.float().mean()) if x.numel() else float('nan')


def one(d, label, out=sys.stdout):
    hit, rank = d['hit'], d['rank']
    n = hit.numel()
    print('\n%s   (%d queries, k=%d)' % (label, n, d['k']), file=out)
    print('  candidate recall@%-3d      %5.1f' % (d['k'], pct(hit)), file=out)
    print('  reported R@1              %5.1f' % pct(rank == 0), file=out)
    if hit.any():
        print('  R@1 GIVEN the GT is a candidate   %5.1f   <- the reranker\'s own accuracy'
              % pct((rank == 0)[hit]), file=out)
        print('  median rank of the GT when it IS a candidate: %d'
              % int(rank[hit].median()), file=out)


def paired(a, b, name_a, name_b, out=sys.stdout):
    """The queries B recovers and A does not: what does B's reranking stage do with them?"""
    if a['ids'] != b['ids'] or a['ids_txt'] != b['ids_txt']:
        print('\nREFUSING to pair %s with %s: the two dumps do not enumerate the same gallery '
              'in the same order, so query i is not the same query in both. Comparing them '
              'positionally would invent a result.' % (name_a, name_b), file=out)
        return
    both = a['hit'] & b['hit']
    only_b = (~a['hit']) & b['hit']
    only_a = a['hit'] & (~b['hit'])
    neither = (~a['hit']) & (~b['hit'])

    print('\nWHERE THE EXTRA RECALL GOES   (%s vs %s, same queries, same reranker)'
          % (name_b, name_a), file=out)
    print('  %-34s %6s   %s R@1   %s R@1' % ('query group', 'count', name_a[:6], name_b[:6]),
          file=out)
    for label, sel in (('GT a candidate for both', both),
                       ('GT recovered by %s ONLY' % name_b, only_b),
                       ('GT a candidate for %s ONLY' % name_a, only_a),
                       ('GT missed by both (unreachable)', neither)):
        if not sel.any():
            print('  %-34s %6d' % (label, int(sel.sum())), file=out)
            continue
        print('  %-34s %6d   %8.1f   %8.1f'
              % (label, int(sel.sum()), pct((a['rank'] == 0)[sel]), pct((b['rank'] == 0)[sel])),
              file=out)

    if not only_b.any():
        print('\n  %s recovered no query that %s missed -- there is no extra recall here to '
              'convert, and this benchmark cannot answer the question.' % (name_b, name_a),
              file=out)
        return

    conv = pct((b['rank'] == 0)[only_b])
    base = pct((b['rank'] == 0)[both])
    med = int(b['rank'][only_b].median())
    print('\n  On the %d queries only %s puts in reach, the reranker ranks the ground truth'
          % (int(only_b.sum()), name_b), file=out)
    print('  first %.1f%% of the time, against %.1f%% on queries both methods reach.'
          % (conv, base), file=out)
    print('  Median rank of those recovered clips: %d of %d.' % (med + 1, b['k']), file=out)
    if base > 0 and conv < 0.5 * base:
        print('\n  RERANKER-LIMITED. The recovered clips are reranked far worse than the ones', file=out)
        print('  both methods already had, under the same run\'s own cross-encoder. Stage-1 recall is', file=out)
        print('  not the binding constraint, and more of it will not move the reported number.', file=out)
    elif conv >= 0.8 * base:
        print('\n  The recovered clips rerank about as well as any other. The extra recall IS', file=out)
        print('  usable, so if the reported number did not move, look at what the same change', file=out)
        print('  cost elsewhere -- recall gained on one query is often lost on another.', file=out)
    else:
        print('\n  Partial conversion: harder than average but not unusable.', file=out)


def saturation(d, label, out=sys.stdout):
    """How tightly the ITM probabilities bunch inside a candidate set.

    The reranker scores with softmax(...)[:, 1], a probability. If many candidates sit within
    a whisker of each other the ordering among them is decided by noise, and no stage-1 signal
    survives into the reported number because none of it is consulted at that point.
    """
    itm, mask = d['itm'], d['mask']
    big = torch.where(mask, itm, torch.full_like(itm, float('-inf')))
    top = big.max(dim=1, keepdim=True).values
    near = ((big > top - 0.01) & mask).sum(dim=1).float()
    print('  candidates within 0.01 of the top ITM score: median %d, mean %.1f (of %d)'
          % (int(near.median()), float(near.mean()), d['k']), file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='baseline dump (e.g. the released GRAM cell)')
    ap.add_argument('--b', required=True, help='the arm being tested (e.g. an SCA cell)')
    ap.add_argument('--name_a', default='GRAM')
    ap.add_argument('--name_b', default='SCA')
    args = ap.parse_args()

    def resolve(p):
        hits = sorted(glob.glob(p))
        if not hits:
            print('no dump matched %r' % p, file=sys.stderr)
            sys.exit(2)
        if len(hits) > 1:
            print('%r matched %d files; pass one dump, not a set: %s'
                  % (p, len(hits), ' '.join(os.path.basename(h) for h in hits)), file=sys.stderr)
            sys.exit(2)
        return hits[0]

    a, b = load(resolve(args.a)), load(resolve(args.b))
    one(a, args.name_a)
    saturation(a, args.name_a)
    one(b, args.name_b)
    saturation(b, args.name_b)
    paired(a, b, args.name_a, args.name_b)
    print('\nThe two stages answer to different things. Stage-1 recall is what SCA moves;')
    print('R@1-given-candidate is the reranking stage -- a per-method cross-encoder, all of')
    print('them descended from the same VAST checkpoint. Whichever of the two is the smaller')
    print('number is the one worth a GPU night.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
