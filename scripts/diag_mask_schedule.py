#!/usr/bin/env python3
"""How much text-video-only training signal does the mask schedule actually deliver?

    python3 scripts/diag_mask_schedule.py
    python3 scripts/diag_mask_schedule.py --config config/sca/ablations/A5_pfull_end_0.3.json

CORRECTION (this file previously argued the opposite). It used to claim GRAM's `forward_ret`
loops over the sub-tasks in `ret%tv%ta` and therefore trains a text-video volume every step,
against SCA's 20%. That is wrong. In model/gram.py the body of `for task in subtasks:`
(line 683) is a single statement, `loss_itc.append(torch.tensor(0))`; everything else in it
is commented out. `loss_area` and `loss_itm` are each computed ONCE, before that loop, over
whatever modalities the batch carries. GRAM trains one joint volume per step exactly as SCA
trains one joint centroid per step, and it never trains a text-video-only objective at all.
So there is no 100%-vs-20% asymmetry, and the ActivityNet deficit cannot be explained by it.

What the number below still means. The share of clip-steps whose centroid sees video alone
is a property of SCA's mask schedule and is worth knowing -- it is how much of the run
exercises a degenerate (single-modality) centroid. But it is a description of SCA's own
curriculum, NOT a deficit relative to GRAM, and it must not be reported as one.

No GPU, no data: it only replays the sampler.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from data.mask_sampler import MaskSampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/sca/pretrain_cfg/sca_pretrain.json')
    # MEASURED, not assumed: annotations150k.json carries a subtitle on all 150,154 entries
    # (scripts/audit_modality_arity.py), so the pretraining gallery is {v,a,s} = 3. The old
    # default of 2 here came from a comment claiming the 150k had no subtitles, and reporting
    # a k=2 replay of a k=3 run overstated the single-modality share by 3x.
    ap.add_argument('--modalities', type=int, default=3,
                    help='gallery size during pretraining: {v,a,s} = 3 for the 150k')
    ap.add_argument('--batch', type=int, default=None, help='override the config batch size')
    ap.add_argument('--clips', type=int, default=150000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    cfg = json.load(open(path))
    mcfg = cfg.get('model_cfg', {})
    train = cfg['data_cfg']['train'][0]
    batch = args.batch or int(train['batch_size'])
    epochs = int(train.get('epoch', 1))
    steps = (args.clips // batch) * epochs

    sampler = MaskSampler(
        num_modalities=args.modalities,
        p_full_start=float(mcfg.get('mask_p_full_start', 1.0)),
        p_full_end=float(mcfg.get('mask_p_full_end', 0.5)),
        schedule_steps=int(mcfg.get('mask_schedule_steps', 1000)),
        mode=mcfg.get('mask_mode', 'uniform'),
        freq=mcfg.get('mask_freq'),
        n_drop=int(mcfg.get('mask_n_drop', 1)),
    )

    print('config          : %s' % args.config)
    print('  p_full        : %.2f -> %.2f over %d steps'
          % (sampler.p_full_start, sampler.p_full_end, sampler.schedule_steps))
    print('  n_drop        : %d   gallery modalities: %d' % (sampler.n_drop, args.modalities))
    print('  run           : %d steps of batch %d (%d clips x %d epochs)'
          % (steps, batch, args.clips, epochs))

    gen = torch.Generator().manual_seed(args.seed)
    present = torch.ones(batch, args.modalities)
    # Counted by ARITY -- how many modalities the clip's centroid is taken over. The old
    # code counted 'video only' and 'audio only' from columns 0 and 1, which silently
    # ignores every modality beyond the second and mislabels a k=3 run as if it were k=2.
    by_arity = [0] * (args.modalities + 1)
    milestones = [int(steps * f) for f in (0.05, 0.25, 0.5, 1.0)]
    rows = []

    for step in range(steps):
        vmask = sampler.sample(batch, step, torch.device('cpu'), present=present, generator=gen)
        counts = (vmask > 0).sum(dim=1)
        for k in range(args.modalities + 1):
            by_arity[k] += int((counts == k).sum())
        if step + 1 in milestones:
            seen = (step + 1) * batch
            rows.append((step + 1, [100.0 * c / seen for c in by_arity]))

    total = steps * batch
    print('\nclip-steps by the ARITY of the centroid (how many modalities it averages):')
    for k in range(args.modalities, -1, -1):
        if not by_arity[k]:
            continue
        note = ''
        if k == args.modalities:
            note = '   <- full gallery'
        elif k == 1:
            note = '   <- degenerate: the spherical mean of one vector is that vector'
        elif k == 0:
            note = '   <- EMPTY: no modality at all, this should never happen'
        print('  |M| = %d       : %11d  %5.1f%%%s' % (k, by_arity[k], 100.0 * by_arity[k] / total, note))

    print('\ncumulative share by point in the run:')
    print('  %-10s %s' % ('step', ' '.join('|M|=%d' % k for k in range(args.modalities + 1))))
    for step, shares in rows:
        print('  %-10d %s' % (step, ' '.join('%5.1f%%' % s for s in shares)))

    if by_arity[0]:
        print('\n%d clip-steps had NO modality present. That is a bug in the sampler, not a'
              % by_arity[0])
        print('curriculum choice -- the centroid is undefined there.')
        return 1
    solo = 100.0 * by_arity[1] / total
    fusion = 100.0 - solo
    print('\n%.1f%% of clip-steps train a centroid over 2 or more modalities; %.1f%% train a'
          % (fusion, solo))
    print('single-modality centroid, where the spherical mean is the identity and no fusion')
    print('behaviour is exercised at all. GRAM has no comparable number -- it trains one joint')
    print('volume per step and no single-modality objective -- so this describes SCA\'s own')
    print('curriculum and is not a gap against the baseline. The earlier "100% vs 20%" framing')
    print('was a misreading of gram.py:683, whose subtask loop body is a single no-op line.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
