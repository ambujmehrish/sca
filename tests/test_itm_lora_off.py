"""Adapters on stage 1, frozen backbone on stage 2.

The reported metric comes from two stages with different training histories. The dual encoder
is what LoRA was trained for. The reranker is a pretrained cross-encoder plus a frozen ITM
head that was never trained here -- but the retrieval loss reaches its BERT through the same
`multimodal_encoder` adapters, so every LoRA step moves it away from the calibration its own
head expects, for a gradient that is not the ITM objective.

`itm_lora_off` gives that stage its original weights back. These tests cover the switch
itself: that it really removes the delta everywhere the delta can be read (module call AND
the raw `.weight` reads EVA-CLIP and BEATs do), that it restores exactly, that it refuses the
one state where it cannot work, and that leaving it off changes nothing.
"""
import pytest
import torch
import torch.nn as nn

from model.lora import (LoRALinear, LoRAQKVLinear, inject_lora, lora_disabled,
                        lora_modules, merge_all, unmerge_all)


class _Attn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)


class _FusedAttn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)


def _trained(module, seed=0):
    """LoRA is identity at init (B is zero-init), so an untouched adapter would make every
    test below vacuous. Give A and B real values first."""
    g = torch.Generator().manual_seed(seed)
    for m in lora_modules(module):
        for name, p in m.named_parameters():
            if 'lora_' in name:
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return module


def test_disabling_removes_the_delta_from_the_module_call():
    mod = _Attn()
    inject_lora(mod, r=4)
    _trained(mod)
    x = torch.randn(3, 8)
    adapted = mod.q_proj(x)
    with lora_disabled(mod):
        frozen = mod.q_proj(x)
    base = mod.q_proj.base(x)
    assert torch.allclose(frozen, base, atol=1e-6)
    assert (adapted - frozen).abs().max() > 1e-3, 'adapter had no effect -- test is vacuous'


def test_disabling_removes_the_delta_from_the_raw_weight_read():
    """EVA-CLIP does F.linear(x, self.qkv.weight, ...) and BEATs cat's q/k/v_proj.weight in
    its fast path -- neither goes through forward(). A switch that only covered forward()
    would leave the vision and audio encoders adapted while claiming they were frozen."""
    mod = _FusedAttn()
    inject_lora(mod, r=4)
    _trained(mod)
    adapted = mod.qkv.weight
    with lora_disabled(mod):
        frozen = mod.qkv.weight
    assert torch.allclose(frozen, mod.qkv.base.weight, atol=1e-6)
    assert (adapted - frozen).abs().max() > 1e-3


def test_the_context_restores_the_previous_state_exactly():
    mod = _Attn()
    inject_lora(mod, r=4)
    _trained(mod)
    x = torch.randn(3, 8)
    before = mod.q_proj(x).clone()
    with lora_disabled(mod):
        pass
    assert torch.allclose(before, mod.q_proj(x), atol=1e-7)
    assert all(m.enabled for m in lora_modules(mod))


def test_nesting_restores_to_disabled_not_to_enabled():
    """The reranker calls compute_slice_scores inside a pass that may already be inside the
    context. A restore that put `enabled` back to True unconditionally would re-adapt the
    outer scope halfway through."""
    mod = _Attn()
    inject_lora(mod, r=4)
    _trained(mod)
    with lora_disabled(mod):
        with lora_disabled(mod):
            pass
        assert not any(m.enabled for m in lora_modules(mod)), \
            'inner exit re-enabled the adapters inside the outer context'
    assert all(m.enabled for m in lora_modules(mod))


def test_merged_adapters_are_refused_rather_than_silently_ignored():
    """merge() folds the delta into the base weight. There is then nothing to switch off, so
    the pass would be adapted while the flag claims it is frozen -- the exact silent fallback
    that would make the experiment meaningless and unfalsifiable."""
    mod = _Attn()
    inject_lora(mod, r=4)
    _trained(mod)
    merge_all(mod)
    with pytest.raises(RuntimeError, match='merged'):
        with lora_disabled(mod):
            pass
    unmerge_all(mod)
    with lora_disabled(mod) as n:                 # fine again once unmerged
        assert n == 2


def test_no_gradient_reaches_the_adapters_through_a_disabled_pass():
    """With itm_lora_off the ITM loss must train itm_head only. If gradient still reached
    lora_A/B, the branch would be fitted on weights it is not scored with."""
    mod = _Attn()
    inject_lora(mod, r=4)
    _trained(mod)
    head = nn.Linear(8, 2)                        # stands in for itm_head: trainable, downstream
    x = torch.randn(3, 8)

    with lora_disabled(mod):
        head(mod.q_proj(x)).sum().backward()
    assert mod.q_proj.lora_A.grad is None, 'the adapters were updated by the ITM branch'
    assert head.weight.grad is not None and head.weight.grad.abs().max() > 0, \
        'itm_head got no gradient either -- the branch would not train at all'

    head.zero_grad()
    head(mod.q_proj(x)).sum().backward()          # and with the adapters on, they do learn
    assert mod.q_proj.lora_A.grad is not None and mod.q_proj.lora_A.grad.abs().max() > 0


def test_default_path_is_unchanged():
    """The flag defaults off, and every arm already measured was measured without it. If the
    plain call were not bit-identical to before, every existing number would move."""
    mod = _Attn()
    inject_lora(mod, r=4)
    _trained(mod)
    x = torch.randn(5, 8)
    expected = mod.q_proj.base(x) + (x @ mod.q_proj.lora_A.T @ mod.q_proj.lora_B.T) \
        * mod.q_proj.scaling
    assert torch.allclose(mod.q_proj(x), expected, atol=1e-6)


def test_the_itm_eval_configs_actually_set_the_flag():
    """A config that lost the key would run the ordinary eval path and write its numbers into
    a directory named for the experiment -- an unfalsifiable null result. The slurm script
    checks this too; this catches it before a job is queued."""
    import glob
    import json
    cfgs = sorted(glob.glob('benchmark_eval/configs_qweight_itmfrozen/sca_*.json'))
    assert len(cfgs) == 5, 'expected one config per benchmark, found %d' % len(cfgs)
    for p in cfgs:
        mc = json.load(open(p))['model_cfg']
        assert mc.get('itm_lora_off') is True, '%s does not set itm_lora_off' % p
        assert mc.get('use_lora') is True, '%s has no adapters to switch off' % p
        # everything else must match the arm it evaluates, or this measures two changes
        base = json.load(open(p.replace('_itmfrozen', '')))['model_cfg']
        assert {k: v for k, v in mc.items() if k != 'itm_lora_off'} == base, \
            '%s differs from configs_qweight by more than itm_lora_off' % p


def test_the_readout_script_imports_and_covers_every_benchmark():
    """itm_frozen_delta.py is read once, on cluster output, after a job has already run. An
    import error or a benchmark missing from BAR would surface only then."""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'itm_frozen_delta.py')
    spec = importlib.util.spec_from_file_location('itm_frozen_delta', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.BAR) == set(mod.BENCHES), 'BAR and BENCHES disagree: %s' % (
        set(mod.BAR) ^ set(mod.BENCHES))
    for bench, (bar, who) in mod.BAR.items():
        assert isinstance(bar, float) and bar > 0 and who, bench
    # an absent root must say so and fail, never print an empty table as a negative result
    assert mod.collect('workdir/definitely-not-a-real-root') == {}


class _MaskingHost:
    """GRAM.batch_get / _eval_mask_drop lifted onto a stub, so the masking interaction can be
    tested without constructing three real encoders."""

    from model.gram import GRAM
    batch_get = GRAM.batch_get
    _eval_mask_drop = GRAM._eval_mask_drop

    def __init__(self, rate=1.0):
        self.eval_mask_rate = rate
        self.eval_mask_seed = 0
        self.training = False
        self._EVAL_MASK_KEYS = ('vision_output', 'audio_output', 'subtitle_output',
                                'depth_output')

    def _batch_get_impl(self, batch, key):
        return batch[key]


def test_the_itm_encoder_pass_is_masked_as_the_same_modality():
    """Test-time modality dropping (eval_mask_rate) zeroes an encoder output so the modality
    is gone from BOTH stages. `vision_output_itm` is a different key for the same modality; if
    the masking did not recognise it, the E4-ITM arm would drop a modality from the scorer and
    leave it in the reranker, and the missing-modality numbers would be measuring nothing."""
    host = _MaskingHost(rate=1.0)                 # every clip loses one modality
    ids = ['clip-%d' % i for i in range(8)]
    def fresh():
        return {'ids': ids,
                'vision_output': torch.ones(8, 3),
                'audio_output': torch.ones(8, 3),
                'vision_output_itm': torch.ones(8, 3),
                'audio_output_itm': torch.ones(8, 3)}

    b = fresh()
    plain_v = host.batch_get(b, 'vision_output')
    b2 = fresh()
    itm_v = host.batch_get(b2, 'vision_output_itm')
    dropped = (plain_v.sum(1) == 0)
    assert dropped.any(), 'nothing was dropped -- the test is vacuous'
    assert torch.equal(dropped, (itm_v.sum(1) == 0)), \
        'the _itm pass was masked differently from the stage-1 pass'


def test_both_wrapper_types_are_covered():
    root = nn.Module()
    root.a = _Attn()
    root.b = _FusedAttn()
    inject_lora(root, r=4)
    kinds = {type(m) for m in lora_modules(root)}
    assert kinds == {LoRALinear, LoRAQKVLinear}
    with lora_disabled(root) as n:
        assert n == 3                              # q_proj, v_proj, qkv
        assert not any(m.enabled for m in lora_modules(root))


def test_the_noise_floor_reports_the_worst_disagreement():
    """The repeated cells are free replicates -- same checkpoint, same config -- so their
    spread bounds eval jitter. The number that matters is the WORST pair, not the mean: a
    margin is only safe if it clears the largest disagreement observed, and averaging would
    quietly license claims the data does not support."""
    import importlib.util
    import io
    import os
    p = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'raw_vs_itm.py')
    spec = importlib.util.spec_from_file_location('raw_vs_itm_nf', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    buf = io.StringIO()
    mod.noise_floor({
        # a true replicate: same checkpoint, so the encoder scores reproduce exactly
        'a_didemo': {'cosine_TV': [35.3, 35.3], 'cosine_TA': [4.4, 4.4],
                     'ret_area_forward': [34.2, 34.2], 'ret_itm_area': [51.3, 51.5]},
        # NOT a replicate: the arm trained further between the two evals, so the encoder
        # outputs moved. Its 0.7 must not become the floor.
        'b_msrvtt': {'cosine_TV': [41.0, 40.4], 'cosine_TA': [5.2, 5.3],
                     'ret_itm_area': [54.8, 54.1]},
    }, out=buf)
    text = buf.getvalue()
    assert 'largest disagreement between two runs of one checkpoint: 0.2' in text, text
    assert 'b_msrvtt' in text.split('EXCLUDED')[1], 'the drifted cell must be named, not dropped'
    assert '51.3 51.5' in text, 'every run must be shown, not a summary'
    assert 'LOWER bound' in text, 'must not be read as a full error bar'

    # a floor built ONLY from drifted cells is no floor at all, and must not be reported as one
    only_drift = io.StringIO()
    mod.noise_floor({'b_msrvtt': {'cosine_TV': [41.0, 40.4], 'ret_itm_area': [54.8, 54.1]}},
                    out=only_drift)
    assert 'largest disagreement' not in only_drift.getvalue()
    assert 'not a floor of zero' in only_drift.getvalue()

    empty = io.StringIO()
    mod.noise_floor({}, out=empty)
    assert empty.getvalue() == '', 'no replicates must print nothing, not a floor of zero'
