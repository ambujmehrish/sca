"""Training the cross-encoder: one parameterization, not two.

`multimodal_encoder` is both the text tower and the ITM cross-encoder, and GRAM/HyperGRAM
full-finetune it. Our attempts to do the same set `lora_freeze_multimodal=false` and left
`lora_r_text=8`, so its W_q and W_v received the base update at learning_rate AND a rank-8
update at 0.1*learning_rate scaled by alpha/r=2, while every other projection in the layer got
only the base update.

Three arms ran that way. x1_xenc_full_lr2e5 validated 52.6 / 45.8 / 50.3 / 48.7 / 40.7 / 40.3 /
43.7 / 49.5 / 49.8 / 49.2 -- oscillation, read at the time as overtraining decay and used to
declare a trainable cross-encoder unworkable. These tests pin the defect and the guard, so the
conclusion cannot be drawn from it again.
"""
import pytest
import torch
import torch.nn as nn

from model.lora import inject_lora, lora_modules


# production passes easydicts, and build_optimizer does `'key' in args.model_cfg` -- a plain
# namespace is not iterable and would fail for a reason unrelated to what is being tested
from easydict import EasyDict as _NS  # noqa: E402


def _args(freeze_mm, r_text, lr=2e-5):
    return _NS(model_cfg=_NS(vision_encoder_type='evaclip01_giant', use_lora=True,
                             lora_freeze_multimodal=freeze_mm, lora_r_text=r_text,
                             lora_r_vision=8, lora_r_audio=8, lora_alpha=16),
               run_cfg=_NS(learning_rate=lr, lora_lr=None, new_lr=0.0, new_params_name=[],
                           weight_decay=0.01, clip_lr=lr, optim='adamw', betas=[0.9, 0.98]))


class _Model(nn.Module):
    """The three named backbones, so build_optimizer's prefix matching is exercised for real."""
    def __init__(self):
        super().__init__()
        self.multimodal_encoder = nn.Module()
        self.multimodal_encoder.bert = nn.Module()
        self.multimodal_encoder.bert.attention = nn.Module()
        self.multimodal_encoder.bert.attention.query = nn.Linear(8, 8)
        self.multimodal_encoder.bert.attention.value = nn.Linear(8, 8)
        self.multimodal_encoder.bert.dense = nn.Linear(8, 8)
        self.vision_encoder = nn.Module()
        self.vision_encoder.qkv = nn.Linear(8, 24)
        self.audio_encoder = nn.Module()
        self.audio_encoder.q_proj = nn.Linear(8, 8)
        self.audio_encoder.v_proj = nn.Linear(8, 8)
        self.itm_head = nn.Linear(8, 2)


def _built(freeze_mm, r_text):
    from utils.build_optimizer import build_optimizer
    m = _Model()
    inject_lora(m.vision_encoder, r=8, alpha=16, prefix='vision_encoder')
    inject_lora(m.audio_encoder, r=8, alpha=16, prefix='audio_encoder')
    if r_text > 0:
        inject_lora(m.multimodal_encoder, r=r_text, alpha=16, prefix='multimodal_encoder')
    return m, build_optimizer(m, _args(freeze_mm, r_text), None)


def test_the_broken_combination_is_refused():
    """The exact setting B5/B6/X1/X2 used. It must not run at all now."""
    with pytest.raises(ValueError, match='trained by its base weights AND by a LoRA adapter'):
        _built(freeze_mm=False, r_text=8)


def test_the_error_names_the_fix():
    try:
        _built(freeze_mm=False, r_text=8)
    except ValueError as e:
        assert 'lora_r_text=0' in str(e), 'the message must say what to do, not just what is wrong'


def test_clean_full_finetune_of_the_cross_encoder_is_allowed_and_has_no_adapter_on_it():
    m, opt = _built(freeze_mm=False, r_text=0)
    mm_lora = [k for k, _ in m.named_parameters()
               if k.startswith('multimodal_encoder') and 'lora_' in k]
    assert not mm_lora, 'an adapter survived on the cross-encoder: %s' % mm_lora
    trainable = {k for k, v in m.named_parameters() if v.requires_grad}
    assert any(k.startswith('multimodal_encoder') for k in trainable), \
        'the cross-encoder is meant to be TRAINABLE here'
    assert not any(k.startswith('vision_encoder') and 'lora_' not in k for k in trainable), \
        'vision must stay frozen apart from its adapter -- this is still a LoRA recipe'


def test_every_cross_encoder_weight_appears_in_exactly_one_optimizer_group():
    """The defect restated as the property that was violated: a parameter tensor reached the
    optimizer once, but W_q's UPDATE arrived twice -- once through the base tensor and once
    through the adapter that adds to it. With r_text=0 that second path does not exist."""
    m, opt = _built(freeze_mm=False, r_text=0)
    seen = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g['params']:
            key = id(p)
            assert key not in seen, 'a parameter is in two optimizer groups'
            seen[key] = gi
    by_id = {id(p): k for k, p in m.named_parameters()}
    mm = [by_id[i] for i in seen if by_id.get(i, '').startswith('multimodal_encoder')]
    assert mm, 'no cross-encoder parameter reached the optimizer'


def test_frozen_cross_encoder_still_takes_an_adapter():
    """The default recipe (T9 and every reported arm) is unchanged by the guard."""
    m, opt = _built(freeze_mm=True, r_text=8)
    trainable = {k for k, v in m.named_parameters() if v.requires_grad}
    mm_base = [k for k in trainable if k.startswith('multimodal_encoder') and 'lora_' not in k]
    assert not mm_base, 'the cross-encoder base weights must stay frozen in the LoRA recipe'
    assert any(k.startswith('multimodal_encoder') and 'lora_' in k for k in trainable)


def test_the_shipped_xenc_configs_are_the_clean_kind():
    import glob
    import json
    for p in sorted(glob.glob('config/sca/ablations/X[3-9]_xenc_clean*.json')):
        mc = json.load(open(p))['model_cfg']
        assert mc.get('lora_freeze_multimodal') is False, p
        assert mc.get('lora_r_text') == 0, '%s would double-parameterize the cross-encoder' % p
        assert mc.get('lora_r_vision', 0) > 0, '%s dropped the vision adapter too' % p


def test_the_old_broken_configs_are_still_on_disk_and_now_refuse_to_run():
    """B5/B6/X1/X2 are kept as the record of what was run, not deleted. The guard is what
    stops them, so that pairing is asserted rather than assumed."""
    import glob
    import json
    broken = []
    for p in sorted(glob.glob('config/sca/ablations/*.json')):
        mc = json.load(open(p))['model_cfg']
        if mc.get('lora_freeze_multimodal') is False and int(mc.get('lora_r_text', 8)) > 0:
            broken.append(p)
    assert broken, 'expected the historical xenc configs to still be present'
    for p in broken:
        mc = json.load(open(p))['model_cfg']
        with pytest.raises(ValueError):
            _built(freeze_mm=False, r_text=int(mc.get('lora_r_text', 8)))
