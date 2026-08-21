"""The eval configs hardcode LoRA rank 8; arms trained at another rank must not be
evaluated against it. A rank mismatch raises on load, but an ALPHA mismatch does not --
LoRALinear scales by alpha/r, so a rank-32/alpha-64 arm loads cleanly into a rank-32/alpha-16
config and reports numbers from a model that was never trained."""
import json
import os

import pytest
import torch
from easydict import EasyDict as edict

from utils.lora_geometry import ranks_from_checkpoint, sync_lora_geometry


def _arm(tmp_path, name, r_text, r_vis, alpha, hps=True):
    d = tmp_path / name
    (d / 'ckpt').mkdir(parents=True)
    (d / 'log').mkdir(parents=True)
    torch.save({'module.multimodal_encoder.bert.encoder.layer.11.attention.self.query.lora_A':
                torch.zeros(r_text, 768),
                'vision_encoder.blocks.0.attn.qkv.lora_A': torch.zeros(r_vis, 1408)},
               d / 'ckpt' / 'model_step_2649.pt')
    if hps:
        json.dump({'model_cfg': {'use_lora': True, 'lora_r_vision': r_vis,
                                 'lora_r_audio': r_vis, 'lora_r_text': r_text,
                                 'lora_alpha': alpha}}, open(d / 'log' / 'hps.json', 'w'))
    return str(d / 'ckpt' / 'model_step_2649.pt')


def _cfg(ckpt):
    return edict({'run_cfg': edict({'checkpoint': ckpt}),
                  'model_cfg': edict({'use_lora': True, 'lora_r_vision': 8, 'lora_r_audio': 8,
                                      'lora_r_text': 8, 'lora_alpha': 16})})


def test_rank_read_from_checkpoint_tensors(tmp_path):
    ckpt = _arm(tmp_path, 'x2', 64, 8, 16)
    assert ranks_from_checkpoint(ckpt) == {'lora_r_text': 64, 'lora_r_vision': 8}


def test_rank_mismatch_is_corrected(tmp_path):
    """x2_xenc_r64 against a rank-8 eval config -- the crash that killed the xenc job."""
    args = _cfg(_arm(tmp_path, 'x2', 64, 8, 16))
    sync_lora_geometry(args)
    assert args.model_cfg.lora_r_text == 64


def test_alpha_mismatch_is_corrected(tmp_path):
    """The silent case: r=32/alpha=64 would otherwise run at a quarter adapter strength."""
    args = _cfg(_arm(tmp_path, 'b2', 32, 32, 64))
    sync_lora_geometry(args)
    assert args.model_cfg.lora_alpha == 64
    assert args.model_cfg.lora_alpha / args.model_cfg.lora_r_text == 2.0


def test_matching_arm_is_untouched(tmp_path):
    args = _cfg(_arm(tmp_path, 'r8', 8, 8, 16))
    before = dict(args.model_cfg)
    sync_lora_geometry(args)
    assert dict(args.model_cfg) == before


def test_unresolvable_geometry_raises_rather_than_guessing_alpha(tmp_path):
    args = _cfg(_arm(tmp_path, 'orphan', 32, 32, 64, hps=False))
    with pytest.raises(RuntimeError, match='alpha'):
        sync_lora_geometry(args)


def test_non_lora_checkpoint_is_a_noop(tmp_path):
    p = tmp_path / 'plain.pt'
    torch.save({'multimodal_encoder.bert.x.weight': torch.zeros(4, 4)}, p)
    args = _cfg(str(p))
    before = dict(args.model_cfg)
    sync_lora_geometry(args)
    assert dict(args.model_cfg) == before


def test_missing_checkpoint_is_a_noop(tmp_path):
    args = _cfg(str(tmp_path / 'nope.pt'))
    before = dict(args.model_cfg)
    sync_lora_geometry(args)
    assert dict(args.model_cfg) == before
