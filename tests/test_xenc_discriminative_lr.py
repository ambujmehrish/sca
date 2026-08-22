"""How much to move the cross-encoder, not whether to.

Fine-tuning it at the base rate for the full schedule destroys it: X3/X4/X5 read 51.4, 51.1
and 50.9 against 54.8 frozen, and every one is HIGHEST at its first validation and falling.
Two things drive that. GRAM's released weights are model_step_459 on the same VAST foundation
we start from, against our 5330 steps on 150k clips. And the ITM loss trains on
condition_feats_va with no subtitles anywhere in our training set, while MSR-VTT and VATEX are
scored with tvas -- so fine-tuning erases a pathway no gradient here can restore.

xenc_lr gives the component its own, smaller step. xenc_train_layers moves only the top K BERT
layers. These tests cover the wiring, and above all that neither can silently do nothing.
"""
import pytest
import torch.nn as nn
from easydict import EasyDict as _NS

from model.lora import inject_lora


def _args(freeze_mm=False, r_text=0, lr=2e-5, xenc_lr=None, xenc_layers=0):
    return _NS(model_cfg=_NS(vision_encoder_type='evaclip01_giant', use_lora=True,
                             lora_freeze_multimodal=freeze_mm, lora_r_text=r_text,
                             lora_r_vision=8, lora_r_audio=8, lora_alpha=16,
                             xenc_train_layers=xenc_layers),
               run_cfg=_NS(learning_rate=lr, lora_lr=None, new_lr=0.0, new_params_name=[],
                           weight_decay=0.01, clip_lr=lr, optim='adamw', betas=[0.9, 0.98],
                           xenc_lr=xenc_lr))


class _Model(nn.Module):
    """A 4-layer stand-in for the cross-encoder, named as the real BERT is."""
    def __init__(self, n_layers=4):
        super().__init__()
        self.multimodal_encoder = nn.Module()
        self.multimodal_encoder.bert = nn.Module()
        self.multimodal_encoder.bert.encoder = nn.Module()
        self.multimodal_encoder.bert.encoder.layer = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.Module()
            blk.attention = nn.Module()
            blk.attention.query = nn.Linear(8, 8)
            blk.attention.value = nn.Linear(8, 8)
            blk.dense = nn.Linear(8, 8)
            self.multimodal_encoder.bert.encoder.layer.append(blk)
        self.vision_encoder = nn.Module()
        self.vision_encoder.qkv = nn.Linear(8, 24)
        self.audio_encoder = nn.Module()
        self.audio_encoder.q_proj = nn.Linear(8, 8)
        self.itm_head = nn.Linear(8, 2)


def _built(**kw):
    from utils.build_optimizer import build_optimizer
    m = _Model()
    inject_lora(m.vision_encoder, r=8, alpha=16, prefix='vision_encoder')
    inject_lora(m.audio_encoder, r=8, alpha=16, prefix='audio_encoder')
    return m, build_optimizer(m, _args(**kw), None)


def _lr_of(model, opt, name_startswith):
    by_id = {id(p): k for k, p in model.named_parameters()}
    out = set()
    for g in opt.param_groups:
        for p in g['params']:
            if by_id.get(id(p), '').startswith(name_startswith):
                out.add(g['lr'])
    return out


def test_the_cross_encoder_gets_its_own_rate():
    m, opt = _built(xenc_lr=2e-6)
    assert _lr_of(m, opt, 'multimodal_encoder') == {2e-6}
    assert _lr_of(m, opt, 'itm_head') == {2e-5}, \
        'the heads must keep the base rate -- only the pretrained trunk is held back'


def test_without_the_flag_the_cross_encoder_is_at_the_base_rate():
    """The setting X3/X4/X5 ran, kept as the reference point the new arms are measured against."""
    m, opt = _built(xenc_lr=None)
    assert _lr_of(m, opt, 'multimodal_encoder') == {2e-5}


def test_top_k_freezes_everything_below_the_cutoff():
    m, _opt = _built(xenc_lr=2e-6, xenc_layers=2)
    trainable = {k for k, v in m.named_parameters() if v.requires_grad}
    assert not any('.layer.0.' in k or '.layer.1.' in k for k in trainable), \
        'a layer below the cutoff is still training'
    assert any('.layer.2.' in k for k in trainable) and any('.layer.3.' in k for k in trainable), \
        'the top layers must still train, or the arm trains nothing at all'


def test_top_k_larger_than_the_model_trains_everything_rather_than_nothing():
    m, _opt = _built(xenc_lr=2e-6, xenc_layers=99)
    trainable = {k for k, v in m.named_parameters() if v.requires_grad}
    assert sum('.layer.' in k for k in trainable) > 0


def test_the_flags_are_refused_when_the_cross_encoder_is_frozen():
    """Set alongside a frozen cross-encoder they do nothing at all, and an arm that quietly
    reduces to the baseline would be reported as a failed idea rather than a config error."""
    with pytest.raises(ValueError, match='cross-encoder is frozen'):
        _built(freeze_mm=True, r_text=8, xenc_lr=2e-6)
    with pytest.raises(ValueError, match='cross-encoder is frozen'):
        _built(freeze_mm=True, r_text=8, xenc_layers=2)


def test_a_rate_that_reaches_no_parameter_is_an_error():
    """If the naming ever drifts so that no multimodal_encoder tensor is collected, the
    cross-encoder would train at no rate while the config claims a discriminative one."""
    from utils.build_optimizer import build_optimizer
    m = _Model()
    for _k, v in m.multimodal_encoder.named_parameters():
        v.requires_grad = False
    inject_lora(m.vision_encoder, r=8, alpha=16, prefix='vision_encoder')
    inject_lora(m.audio_encoder, r=8, alpha=16, prefix='audio_encoder')
    with pytest.raises(RuntimeError, match='no trainable multimodal_encoder parameter'):
        build_optimizer(m, _args(xenc_lr=2e-6), None)


def test_the_shipped_arms_all_hold_the_cross_encoder_back_somehow():
    import glob
    import json
    arms = sorted(glob.glob('config/sca/ablations/X[6-9]_*.json') +
                  glob.glob('config/sca/ablations/X1[0-3]_*.json'))
    assert len(arms) == 8, 'expected the eight cross-encoder arms, found %d' % len(arms)
    for p in arms:
        c = json.load(open(p))
        mc, rc = c['model_cfg'], c['run_cfg']
        assert mc['lora_freeze_multimodal'] is False and mc['lora_r_text'] == 0, p
        # every arm must differ from X3 (base rate, 5 epochs, 3 validations) in a way that
        # actually limits how far the cross-encoder travels
        short = c['data_cfg']['train'][0]['epoch'] < 5
        slow = rc.get('xenc_lr') is not None or rc['learning_rate'] < 2e-5
        shallow = mc.get('xenc_train_layers', 0) > 0
        assert short or slow or shallow, '%s does not restrain the cross-encoder at all' % p
        assert rc['valid_freq'] == 10, \
            "%s validates too sparsely to locate a peak near GRAM's 459 steps" % p
