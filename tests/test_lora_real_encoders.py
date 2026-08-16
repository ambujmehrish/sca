"""LoRA injection verified against the REAL encoder classes shipped in this repo -- not
stand-ins. P2 gate: sca.py raises on naming drift (zero wrapped layers), so these tests prove
the target names on the actual classes the pretrain will touch:

  vision  EVA-CLIP `Attention`  : fused `qkv` (evaclip01_giant, subln off) and q/v_proj (subln)
  audio   BEATs `MultiheadAttention` : q_proj / v_proj
  text    repo BERT `BertSelfAttention` : query / value  (functional when the repo's bert.py
          imports -- it targets the cluster's transformers 4.x -- else pinned at source level)
"""
import os
import re
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.lora import (LoRALinear, LoRAQKVLinear, inject_lora, merge_all, unmerge_all,
                        remap_lora_checkpoint, setup_lora_backbones)

torch.manual_seed(0)


class TestEvaClipAttention:
    def _attn(self, subln):
        eva = pytest.importorskip('model.vision_encoders.evaclip.eva_vit_model')
        return eva.Attention(dim=64, num_heads=4, subln=subln)

    def test_fused_qkv_wrapped(self):
        # evaclip01_giant (EVA01-CLIP-g-14) runs subln=False -> fused qkv: the production path
        attn = self._attn(subln=False)
        wrapped = inject_lora(attn, r=4)
        assert wrapped == ['qkv']
        assert isinstance(attn.qkv, LoRAQKVLinear)
        assert not attn.qkv.base.weight.requires_grad and attn.qkv.lora_A_q.requires_grad

    def test_subln_q_v_wrapped_k_frozen_untouched(self):
        attn = self._attn(subln=True)
        wrapped = inject_lora(attn, r=4)
        assert sorted(wrapped) == ['q_proj', 'v_proj']
        assert isinstance(attn.q_proj, LoRALinear) and isinstance(attn.v_proj, LoRALinear)
        assert not isinstance(attn.k_proj, (LoRALinear, LoRAQKVLinear))

    def test_zero_init_identity_and_merge_roundtrip(self):
        attn = self._attn(subln=False)
        x = torch.randn(2, 16, 64)
        ref = attn.qkv(x)
        inject_lora(attn, r=4)
        assert torch.allclose(attn.qkv(x), ref, atol=1e-6)     # B zero-init: step-0 identity
        with torch.no_grad():
            attn.qkv.lora_B_q.normal_(); attn.qkv.lora_B_v.normal_()
        y = attn.qkv(x)
        merge_all(attn)
        assert torch.allclose(attn.qkv(x), y, atol=1e-5)
        unmerge_all(attn)
        assert torch.allclose(attn.qkv(x), y, atol=1e-5)
        assert torch.allclose(attn.qkv.base(x), ref, atol=1e-5)  # base restored exactly


class TestRealForwardPaths:
    """The trunk reads raw .weight tensors instead of calling the wrapped module:
    eva_vit_model.py:310 does F.linear(x, self.qkv.weight, qkv_bias) and BEATs' MHA
    torch.cat's proj weights. These tests run the REAL encoder forwards end-to-end after
    injection -- the exact failure the smoke gate caught on the cluster
    (AttributeError: 'LoRAQKVLinear' object has no attribute 'weight')."""

    def test_eva_fused_forward_end_to_end(self):
        eva = pytest.importorskip('model.vision_encoders.evaclip.eva_vit_model')
        torch.manual_seed(0)
        attn = eva.Attention(dim=64, num_heads=4, subln=False).eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            ref = attn(x)
        inject_lora(attn, r=4)
        with torch.no_grad():
            out0 = attn(x)
        assert torch.allclose(out0, ref, atol=1e-5)          # zero-init == baseline forward
        with torch.no_grad():
            attn.qkv.lora_B_q.normal_(); attn.qkv.lora_B_v.normal_()
            out1 = attn(x)
        assert not torch.allclose(out1, ref, atol=1e-3)      # adapters reach the output
        merge_all(attn)
        with torch.no_grad():
            assert torch.allclose(attn(x), out1, atol=1e-4)  # merged == unmerged forward
        unmerge_all(attn)

    def test_eva_subln_forward_end_to_end(self):
        eva = pytest.importorskip('model.vision_encoders.evaclip.eva_vit_model')
        torch.manual_seed(0)
        attn = eva.Attention(dim=64, num_heads=4, subln=True).eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            ref = attn(x)
        inject_lora(attn, r=4)
        with torch.no_grad():
            assert torch.allclose(attn(x), ref, atol=1e-5)
        with torch.no_grad():
            attn.q_proj.lora_B.normal_()
            assert not torch.allclose(attn(x), ref, atol=1e-3)

    def test_beats_forward_end_to_end(self):
        beats = pytest.importorskip('model.audio_encoders.beats.beats')
        torch.manual_seed(0)
        m = beats.MultiheadAttention(64, 4, self_attention=True).eval()
        x = torch.randn(10, 2, 64)                            # (T, B, C) fairseq layout
        with torch.no_grad():
            ref, _, _ = m(query=x, key=x, value=x, position_bias=None)
        inject_lora(m, r=4)
        with torch.no_grad():
            out0, _, _ = m(query=x, key=x, value=x, position_bias=None)
        assert torch.allclose(out0, ref, atol=1e-5)
        with torch.no_grad():
            m.q_proj.lora_B.normal_()
            out1, _, _ = m(query=x, key=x, value=x, position_bias=None)
        assert not torch.allclose(out1, ref, atol=1e-3)

    def test_effective_weight_gradients_flow_to_lora(self):
        # F.linear against the .weight property must train A/B, never the frozen base
        base = torch.nn.Linear(16, 48, bias=False)
        lora = LoRAQKVLinear(base, r=4)
        x = torch.randn(3, 16)
        torch.nn.functional.linear(x, lora.weight).sum().backward()
        assert lora.lora_A_q.grad is not None and torch.isfinite(lora.lora_A_q.grad).all()
        assert lora.base.weight.grad is None                  # frozen base untouched


class TestBeatsAttention:
    def test_q_v_wrapped(self):
        beats = pytest.importorskip('model.audio_encoders.beats.beats')
        attn = beats.MultiheadAttention(64, 4, self_attention=True)
        wrapped = inject_lora(attn, r=4)
        assert sorted(wrapped) == ['q_proj', 'v_proj']
        assert isinstance(attn.q_proj, LoRALinear) and isinstance(attn.v_proj, LoRALinear)
        assert not isinstance(attn.k_proj, (LoRALinear, LoRAQKVLinear))
        assert not isinstance(attn.out_proj, (LoRALinear, LoRAQKVLinear))


class TestBertAttention:
    def test_query_value_wrapped(self):
        try:
            from model.text_encoders.bert.bert import BertSelfAttention, BertConfig
        except ImportError:
            # the repo's bert.py targets the cluster's transformers 4.x; under a newer local
            # transformers pin the naming contract at source level instead
            src = open(os.path.join(os.path.dirname(__file__), '..',
                                    'model/text_encoders/bert/bert.py')).read()
            m = re.search(r'class BertSelfAttention.*?(?=\nclass )', src, re.S)
            assert m, 'BertSelfAttention not found in repo bert.py'
            body = m.group(0)
            assert re.search(r'self\.query\s*=\s*nn\.Linear', body)
            assert re.search(r'self\.value\s*=\s*nn\.Linear', body)
            pytest.skip('repo bert.py needs transformers<5 (cluster env); '
                        'query/value naming verified at source level')
        cfg = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2,
                         vocab_size=100)
        attn = BertSelfAttention(cfg)
        wrapped = inject_lora(attn, r=4)
        assert sorted(wrapped) == ['query', 'value']
        assert not isinstance(attn.key, (LoRALinear, LoRAQKVLinear))


class TestSetupLoraBackbones:
    """The shared wiring used by SCA, GRAMLoRA and PMRL (model/baselines.py)."""

    def _model(self):
        eva = pytest.importorskip('model.vision_encoders.evaclip.eva_vit_model')
        beats = pytest.importorskip('model.audio_encoders.beats.beats')
        m = torch.nn.Module()
        m.vision_encoder = torch.nn.Sequential(eva.Attention(dim=32, num_heads=4, subln=False))
        m.audio_encoder = torch.nn.Sequential(beats.MultiheadAttention(32, 4, self_attention=True))
        return m

    def test_wraps_all_backbones_with_per_modality_ranks(self):
        from easydict import EasyDict as edict
        m = self._model()
        cfg = edict(lora_r_vision=4, lora_r_audio=16, lora_r_text=8, lora_alpha=16)
        wrapped = setup_lora_backbones(m, cfg)
        assert sorted(wrapped) == ['audio_encoder.0.q_proj', 'audio_encoder.0.v_proj',
                                   'vision_encoder.0.qkv']
        assert m.vision_encoder[0].qkv.r == 4                      # per-modality ranks land
        assert m.audio_encoder[0].q_proj.r == 16

    def test_rank_zero_skips_encoder(self):
        from easydict import EasyDict as edict
        m = self._model()
        wrapped = setup_lora_backbones(m, edict(lora_r_vision=0, lora_r_audio=8))
        assert sorted(wrapped) == ['audio_encoder.0.q_proj', 'audio_encoder.0.v_proj']
        assert not isinstance(m.vision_encoder[0].qkv, LoRAQKVLinear)

    def test_zero_wrapped_layers_is_hard_error(self):
        from easydict import EasyDict as edict
        m = torch.nn.Module()
        m.vision_encoder = torch.nn.Sequential(torch.nn.Linear(8, 8))   # no attention inside
        with pytest.raises(RuntimeError, match='naming drift'):
            setup_lora_backbones(m, edict(lora_r_vision=8))


class TestPublishedRows:
    def test_json_in_sync_and_schema_valid(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark_eval'))
        from import_published_rows import (load_published_rows, gram_rows_from_xlsx_source,
                                           OTHER_BASELINES)
        data = load_published_rows()                     # validates the [R@1, R@10] schema
        assert data['GRAM (paper)'] == gram_rows_from_xlsx_source(), \
            'published_rows.json out of sync with make_results_xlsx.py -- run --regen'
        for b in OTHER_BASELINES:
            assert b in data, f'{b} slot missing from published_rows.json'


class TestCheckpointRemap:
    def test_wrapped_keys_remapped_others_untouched(self):
        ckpt = {'vision_encoder.blocks.0.attn.qkv.weight': torch.randn(12, 4),
                'audio_encoder.layers.0.self_attn.q_proj.weight': torch.randn(4, 4),
                'audio_encoder.layers.0.self_attn.q_proj.bias': torch.randn(4),
                'audio_encoder.layers.0.self_attn.k_proj.weight': torch.randn(4, 4),
                'contra_head_t.linear.weight': torch.randn(4, 4)}
        wrapped = ['vision_encoder.blocks.0.attn.qkv',
                   'audio_encoder.layers.0.self_attn.q_proj']
        out = remap_lora_checkpoint(ckpt, wrapped)
        assert 'vision_encoder.blocks.0.attn.qkv.base.weight' in out
        assert 'audio_encoder.layers.0.self_attn.q_proj.base.weight' in out
        assert 'audio_encoder.layers.0.self_attn.q_proj.base.bias' in out
        assert 'audio_encoder.layers.0.self_attn.k_proj.weight' in out      # untouched
        assert 'contra_head_t.linear.weight' in out                          # untouched
        assert len(out) == len(ckpt)

    def test_roundtrip_load_into_wrapped_module(self):
        # a wrapped module must load the remapped old checkpoint with only lora_* missing
        base = torch.nn.Sequential()
        base.add_module('q_proj', torch.nn.Linear(8, 8))
        ckpt = {f'q_proj.{k}': v.clone() for k, v in base.q_proj.state_dict().items()}
        wrapped = inject_lora(base, r=2)
        remapped = remap_lora_checkpoint(ckpt, wrapped)
        missing, unexpected = base.load_state_dict(remapped, strict=False)
        assert not unexpected
        assert all('lora_' in k for k in missing)
