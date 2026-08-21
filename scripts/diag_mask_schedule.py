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
    ap.add_argument('--modalities', type=int, default=2,
                    help='gallery size during pretraining: {v,a} = 2 (no subtitles in the 150k)')
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
    # counts over the modality-set a clip's centroid actually sees
    kept_full = kept_v_only = kept_a_only = kept_none = 0
    first_v_only_step = None
    milestones = [int(steps * f) for f in (0.05, 0.25, 0.5, 1.0)]
    rows = []

    for step in range(steps):
        vmask = sampler.sample(batch, step, torch.device('cpu'), present=present, generator=gen)
        v_on = vmask[:, 0] > 0
        a_on = vmask[:, 1] > 0 if args.modalities > 1 else torch.zeros_like(v_on)
        n_full = int((v_on & a_on).sum())
        n_v = int((v_on & ~a_on).sum())
        n_a = int((~v_on & a_on).sum())
        kept_full += n_full
        kept_v_only += n_v
        kept_a_only += n_a
        kept_none += batch - n_full - n_v - n_a
        if n_v and first_v_only_step is None:
            first_v_only_step = step
        if step + 1 in milestones:
            seen = (step + 1) * batch
            rows.append((step + 1, 100.0 * kept_v_only / seen, 100.0 * kept_a_only / seen,
                         100.0 * kept_full / seen))

    total = steps * batch
    print('\nclip-steps by the modality set the centroid sees:')
    print('  full {v,a}    : %11d  %5.1f%%' % (kept_full, 100.0 * kept_full / total))
    print('  video only    : %11d  %5.1f%%   <- centroid over one modality (degenerate)'
          % (kept_v_only, 100.0 * kept_v_only / total))
    print('  audio only    : %11d  %5.1f%%' % (kept_a_only, 100.0 * kept_a_only / total))
    if kept_none:
        print('  empty         : %11d  %5.1f%%' % (kept_none, 100.0 * kept_none / total))

    print('\ncumulative share by point in the run:')
    print('  %-10s %10s %10s %10s' % ('step', 'video-only', 'audio-only', 'full'))
    for step, v, a, f in rows:
        print('  %-10d %9.1f%% %9.1f%% %9.1f%%' % (step, v, a, f))

    print('\nfirst video-only view at step %s of %d'
          % (first_v_only_step if first_v_only_step is not None else 'NEVER', steps))
    share = 100.0 * (kept_v_only + kept_a_only) / total
    print('\n%.1f%% of clip-steps train a SINGLE-modality centroid, where the spherical mean is'
          % share)
    print('the identity and none of the fusion behaviour is exercised. GRAM has no comparable')
    print('number -- it trains one joint volume per step and no single-modality objective --')
    print('so this is SCA\'s own curriculum, not a gap against the baseline. Do not report it')
    print('as one; the earlier "100% vs 20%" framing was based on a misreading of gram.py:683.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
