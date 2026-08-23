#!/usr/bin/env python3
"""Was every result produced by scoring a checkpoint with the geometry it was TRAINED with?

    python3 scripts/audit_eval_geometry.py
    python3 scripts/audit_eval_geometry.py --roots workdir/e1_frames workdir/e1_repro

This is the failure that keeps recurring, and it is silent every time. The launchers pick an
eval config, and for a long while they picked it from the arm's NAME or from a hardcoded path:

  frameset_eval.sh   matched '*qweight*', so g1_r16_qw / s1_t9_seed51 / x3_xenc_clean_lr2e5
                     would all have gone to configs_frames
  e1_fusion_dump.sh  hardcoded configs_e1 for the SCA arm
  b_grid_eval.sh     hardcoded configs_e1/sca_<bench>.json for any arm
  e1_final_ckpt.sh   the same

Routing a query-weighted checkpoint through the uniform-centroid config does not raise. The
tensors are the same shape, the run completes, the log looks normal, and the number that comes
out belongs to a model nobody trained. Only the frame-slot path has a guard, and that is
because it needs a tensor that is absent rather than merely wrong.

So rather than reading launchers, this compares what each result directory RECORDS about
itself. run_eval writes the resolved config to <outdir>/log/hps.json, and every training arm
has one too. The geometry keys have to agree.

Exit codes: 0 all cells verified, 3 nothing found to check, 4 at least one mismatch.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# What decides how a checkpoint is scored. A disagreement in any of these means the number
# describes a different model from the one that was trained.
GEOMETRY = ('model_type', 'score_mode', 'sca_query_weighting', 'sca_frame_slots',
            'sca_tau_w', 'use_lora')

# Differences that are deliberate rather than defects, keyed by the workdir root.
#   e1_itmfrozen  runs the reranker on the frozen backbone -- itm_lora_off is the experiment
#   e1_itmqw      adds the query-weighted reranker on top -- sca_itm_qw_gamma is the experiment
#   e1_missing    drops modalities at eval -- eval_mask_rate is the experiment
INTENDED = {'e1_itmfrozen': ('itm_lora_off',),
            'e1_itmqw': ('sca_itm_qw_gamma',),
            'e1_missing': ('eval_mask_rate', 'eval_mask_seed')}

BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')


def load_cfg(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def arm_of(cell, pretrain_root=None):
    """'t9_qweight_only_didemo' -> 't9_qweight_only'; strips a gamma/rate segment only if it
    is really one.

    The suffix rule alone is wrong: 'b2_bs128_r32_msrvtt' ends in '_r32', which matches
    r<digits> exactly as a missing-rate suffix does, so the arm came back as 'b2_bs128' and
    twenty-odd cells were reported unverifiable when their arms were on disk all along. An
    audit that quietly stops checking things is worse than one that fails.

    So the full head wins whenever a training arm exists for it, and the suffix is stripped
    only when it does not.
    """
    for b in BENCHES:
        if cell.endswith('_' + b):
            head = cell[:-(len(b) + 1)]
            if pretrain_root and os.path.isdir(os.path.join(pretrain_root, head)):
                return head
            arm, _, tail = head.rpartition('_')
            if arm and len(tail) in (3, 4) and tail[0] in 'gr' and tail[1:].isdigit():
                if not pretrain_root or os.path.isdir(os.path.join(pretrain_root, arm)):
                    return arm
            return head
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='*', default=None,
                    help='default: every workdir/e1_* directory that exists')
    ap.add_argument('--pretrain_root', default='workdir_pretrain')
    args = ap.parse_args()

    base = ROOT
    roots = args.roots or sorted(glob.glob(os.path.join(base, 'workdir', 'e1_*')))
    roots = [r if os.path.isabs(r) else os.path.join(base, r) for r in roots]
    pre = args.pretrain_root if os.path.isabs(args.pretrain_root) \
        else os.path.join(base, args.pretrain_root)

    checked = mismatched = unverifiable = 0
    problems, unknown = [], []

    for root in roots:
        rootname = os.path.basename(root)
        intended = INTENDED.get(rootname, ())
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            if not os.path.isdir(d):
                continue
            cell = os.path.basename(d)
            arm = arm_of(cell, pre)
            eval_hps = load_cfg(os.path.join(d, 'log', 'hps.json'))
            if eval_hps is None or arm is None:
                unknown.append((rootname, cell, 'no log/hps.json' if eval_hps is None
                                else 'cell name does not end in a known benchmark'))
                unverifiable += 1
                continue
            train_hps = load_cfg(os.path.join(pre, arm, 'log', 'hps.json'))
            if train_hps is None:
                # released-checkpoint cells have no arm in workdir_pretrain by construction
                unknown.append((rootname, cell, 'no training arm at %s/%s' % (args.pretrain_root, arm)))
                unverifiable += 1
                continue
            e, t = eval_hps.get('model_cfg', {}), train_hps.get('model_cfg', {})
            checked += 1
            bad = [(k, t.get(k), e.get(k)) for k in GEOMETRY
                   if k not in intended and _norm(t.get(k)) != _norm(e.get(k))]
            if bad:
                mismatched += 1
                problems.append((rootname, cell, arm, bad))

    print('checked %d cell(s) across %d root(s); %d could not be verified'
          % (checked, len(roots), unverifiable))

    if problems:
        print('\nGEOMETRY MISMATCH -- these numbers describe a model that was never trained:\n')
        print('%-16s %-34s %-22s %s' % ('root', 'cell', 'key', 'trained -> scored'))
        print('-' * 100)
        for rootname, cell, arm, bad in problems:
            for k, tv, ev in bad:
                print('%-16s %-34s %-22s %r -> %r' % (rootname, cell, k, tv, ev))
        print('\nEach line is a checkpoint scored under a geometry it was not trained with.')
        print('Nothing raises when this happens: the shapes match, the run completes, and the')
        print('number is wrong in a way no reader can detect. Re-run these cells against the')
        print('config that matches their arm before any of them reaches a table.')

    if unknown:
        print('\nNOT VERIFIED (%d) -- absence of a mismatch here is not evidence of one:' % len(unknown))
        for rootname, cell, why in unknown[:20]:
            print('  %-16s %-34s %s' % (rootname, cell, why))
        if len(unknown) > 20:
            print('  ... and %d more' % (len(unknown) - 20))

    if not checked and not unknown:
        print('\nno eval cells found -- run this on the cluster, where the workdirs live')
        return 3
    if problems:
        return 4
    if checked:
        print('\nEvery verifiable cell was scored with the geometry its arm was trained with.')
    return 0


def _norm(v):
    """False, 0 and absent all mean 'off' for these flags; only real disagreement counts."""
    if v is None or v is False or v == 0:
        return None
    return v


if __name__ == '__main__':
    sys.exit(main())
