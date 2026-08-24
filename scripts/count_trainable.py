#!/usr/bin/env python3
"""Measure trainable-parameter counts for the tables' Params column.

    # SCA (T9): trainable = the parameters the optimizer actually updated
    python3 scripts/count_trainable.py --set sca_t9 \
        --optimizer workdir_pretrain/t9_qweight_only/ckpt/optimizer_step_5330.pt
    # full-FT baselines: trainable = every parameter in the released checkpoint
    python3 scripts/count_trainable.py --set gram_released      --checkpoint /path/released_gram.pt
    python3 scripts/count_trainable.py --set pmrl_released      --checkpoint $WORK_ROOT/pmrl_weights/model_ckpts/pmrl_base.pt
    python3 scripts/count_trainable.py --set hypergram_trained  --checkpoint workdir_pretrain/hgauth_hybrid/ckpt/best_ret%tvas--msrvtt_ret_ret_itm_area.pt

Counts are MEASURED from files, never quoted from a paper:

  --optimizer   sums the elements of every parameter carrying optimizer state (exp_avg) --
                exactly the set that received updates, which for a LoRA run is the honest
                trainable count (adapters plus whatever heads trained), not a guess from the
                config.
  --checkpoint  sums every floating-point tensor in the model state dict. For a fully
                fine-tuned model, trainable = all of them.

Results accumulate in experiments/results/tables_final/trainable_params.json (inside the
directory the harvest commits), and the table generators read that file -- a row whose key
is absent prints MISSING, never a number from memory.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'experiments/results/tables_final/trainable_params.json')


def load_sd(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    return sd


def count_optimizer(path):
    sd = load_sd(path)
    for key in ('state',):
        if isinstance(sd, dict) and key in sd:
            state = sd[key]
            break
    else:
        state = sd.get('optimizer', {}).get('state')
    if not state:
        sys.exit('FATAL: %s has no optimizer state to count -- is it a model checkpoint? '
                 'Use --checkpoint for those.' % path)
    total = 0
    for entry in state.values():
        for k in ('exp_avg', 'momentum_buffer'):
            if k in entry and hasattr(entry[k], 'numel'):
                total += entry[k].numel()
                break
    if total == 0:
        sys.exit('FATAL: optimizer state in %s carries no exp_avg/momentum tensors; cannot '
                 'count what trained.' % path)
    return total


def count_checkpoint(path):
    import torch
    sd = load_sd(path)
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    total = sum(v.numel() for v in sd.values()
                if hasattr(v, 'dtype') and v.dtype.is_floating_point)
    if total == 0:
        sys.exit('FATAL: no floating-point tensors in %s' % path)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, dest='key',
                    choices=('sca_t9', 'sca_fullft', 'gram_released', 'pmrl_released',
                             'hypergram_trained'))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--optimizer')
    src.add_argument('--checkpoint')
    args = ap.parse_args()

    path = args.optimizer or args.checkpoint
    if not os.path.exists(path):
        sys.exit('FATAL: %s not found' % path)
    n = count_optimizer(path) if args.optimizer else count_checkpoint(path)

    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data[args.key] = {'trainable': n,
                      'source': os.path.relpath(path, ROOT) if path.startswith(ROOT) else path,
                      'method': 'optimizer state (exp_avg)' if args.optimizer
                                else 'all floating tensors in checkpoint (full-FT)'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write('\n')
    print('%s: %,d trainable parameters  (%s)'.replace(',', '') % (args.key, n, path))
    print('recorded in %s' % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
