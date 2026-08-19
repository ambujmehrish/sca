#!/usr/bin/env python3
"""Fold LoRA deltas of a trained checkpoint into the base weights (offline, state-dict
surgery only -- no model build), producing a plain-GRAM-shaped checkpoint that a
use_lora=false model loads directly. Enables the full-FT finetune arms from a
LoRA-pretrained checkpoint (SCA-ft-v2b: parity with GRAM-ft's full finetuning).

  python3 scripts/merge_lora_ckpt.py --ckpt <lora_ckpt.pt> --out <merged.pt> [--lora_alpha 16]

Handles both wrapper layouts (model/lora.py):
  LoRALinear    : <p>.base.weight [+ .base.bias], <p>.lora_A, <p>.lora_B
  LoRAQKVLinear : <p>.base.weight [+ .base.bias], <p>.lora_A_q/B_q, <p>.lora_A_v/B_v
                  (delta on the q rows [0:d) and v rows [2d:3d) of the fused qkv weight)
Merged: <p>.weight = base + delta (computed in fp32, cast back), <p>.bias = base bias;
all lora_* keys dropped. FAILS LOUD if no LoRA keys are found (wrong checkpoint) or a
wrapper is missing its counterpart keys (corrupt/partial save).
"""
import argparse
import torch


def merge_state_dict(sd, lora_alpha):
    out = {}
    lora_keys = {k for k in sd if '.lora_' in k}
    base_paths = sorted({k[:k.index('.lora_')] for k in lora_keys})
    if not base_paths:
        raise SystemExit('FATAL: no .lora_* keys in this checkpoint -- nothing to merge '
                         '(is this already a plain checkpoint?)')
    merged = 0
    for k, v in sd.items():
        if '.lora_' in k:
            continue
        for p in base_paths:
            if k == f'{p}.base.weight':
                w = v.float()
                if f'{p}.lora_A' in sd:                                   # plain LoRALinear
                    A, B = sd[f'{p}.lora_A'].float(), sd[f'{p}.lora_B'].float()
                    w = w + (B @ A) * (lora_alpha / A.shape[0])
                elif f'{p}.lora_A_q' in sd:                               # fused qkv
                    d = sd[f'{p}.lora_B_q'].shape[0]
                    Aq, Bq = sd[f'{p}.lora_A_q'].float(), sd[f'{p}.lora_B_q'].float()
                    Av, Bv = sd[f'{p}.lora_A_v'].float(), sd[f'{p}.lora_B_v'].float()
                    w[:d] += (Bq @ Aq) * (lora_alpha / Aq.shape[0])
                    w[2 * d:3 * d] += (Bv @ Av) * (lora_alpha / Av.shape[0])
                else:
                    raise SystemExit(f'FATAL: {p} has lora keys but neither layout matches')
                out[f'{p}.weight'] = w.to(v.dtype)
                merged += 1
                break
            if k == f'{p}.base.bias':
                out[f'{p}.bias'] = v
                break
        else:
            out[k] = v
    print(f'[merge] folded {merged} LoRA-wrapped weights '
          f'({len(base_paths)} wrapped modules) into plain keys')
    if merged != len(base_paths):
        raise SystemExit(f'FATAL: {len(base_paths)} wrapped modules but only {merged} '
                         'base weights merged -- partial checkpoint?')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lora_alpha', type=float, default=16.0,
                    help='must match the training config (lora_alpha)')
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt['model'] = merge_state_dict(ckpt['model'], args.lora_alpha)
    else:
        ckpt = merge_state_dict(ckpt, args.lora_alpha)
    torch.save(ckpt, args.out)
    print(f'[merge] -> {args.out}')


if __name__ == '__main__':
    main()
