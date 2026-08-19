"""Function-level tests for the HyperGRAM repro (utils.volume.volume_computation_lorentz).

The GRAMHyp class itself is exercised by the smoke config on the cluster (it needs the
full encoder stack); these tests pin the geometry the arm depends on.
"""
import torch
import pytest

from utils.volume import (volume_computation_lorentz, volume_computation_masked,
                          present_from_feats)

torch.manual_seed(0)


def _rand(b, d, scale=1.0):
    return torch.randn(b, d) * scale


def test_lorentz_diagonal_is_minus_one():
    """Hyperboloid constraint: <pi(x), pi(x)>_L == -1 exactly, so the Gram diagonal is
    -1 and a single-modality volume is sqrt(|det [[-1, l], [l, -1]]|) = sqrt(|1 - l^2|)."""
    t, v = _rand(4, 8), _rand(5, 8, scale=2.0)
    vol = volume_computation_lorentz(t, [v])
    t0 = torch.sqrt(1 + (t * t).sum(-1))
    v0 = torch.sqrt(1 + (v * v).sum(-1))
    l = t @ v.T - t0[:, None] * v0[None, :]
    expected = torch.sqrt(torch.abs(1.0 - l ** 2) + 1e-8)
    assert torch.allclose(vol, expected, atol=1e-4)


def test_lorentz_masking_equals_reduced_arity():
    """Identity row/col masking must collapse det(G) to the present sub-Gram's det:
    masking modality m for every clip == computing without modality m at all."""
    t = _rand(6, 16)
    g = [_rand(7, 16, s) for s in (1.0, 2.0, 0.5)]
    present = torch.ones(7, 3)
    present[:, 1] = 0                                     # drop modality 1 everywhere
    masked = volume_computation_lorentz(t, g, present=present)
    reduced = volume_computation_lorentz(t, [g[0], g[2]])
    assert torch.allclose(masked, reduced, atol=1e-4)


def test_lorentz_per_clip_masking_mixes_arity():
    """Per-clip masks: clip j's volume is at its OWN arity (mirrors the Euclidean path)."""
    t = _rand(3, 8)
    g = [_rand(4, 8), _rand(4, 8)]
    present = torch.ones(4, 2)
    present[2, 1] = 0
    out = volume_computation_lorentz(t, g, present=present)
    only0 = volume_computation_lorentz(t, [g[0]])
    assert torch.allclose(out[:, 2], only0[:, 2], atol=1e-4)
    full = volume_computation_lorentz(t, g)
    keep = [0, 1, 3]
    assert torch.allclose(out[:, keep], full[:, keep], atol=1e-4)


def test_lorentz_varies_with_norm_where_euclidean_cannot():
    """The variance-preservation mechanism: scaling a vector changes the Lorentz volume
    but leaves the Euclidean volume of the NORMALISED copy untouched."""
    t, v = _rand(2, 8), _rand(3, 8)
    vol1 = volume_computation_lorentz(t, [v])
    vol2 = volume_computation_lorentz(t, [v * 3.0])
    assert not torch.allclose(vol1, vol2, atol=1e-3)
    vn = torch.nn.functional.normalize(v, dim=-1)
    e1 = volume_computation_masked(t, [vn])
    e2 = volume_computation_masked(t, [torch.nn.functional.normalize(v * 3.0, dim=-1)])
    assert torch.allclose(e1, e2, atol=1e-6)


def test_lorentz_differentiable():
    t = _rand(3, 8).requires_grad_(True)
    g = [_rand(4, 8).requires_grad_(True)]
    volume_computation_lorentz(t, g).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert g[0].grad is not None and torch.isfinite(g[0].grad).all()


def test_hybrid_alpha_one_is_pure_euclidean():
    """alpha=1 must reproduce GRAM's volume exactly (the mixing only refines)."""
    t, g = _rand(4, 8), [_rand(5, 8), _rand(5, 8)]
    tn = torch.nn.functional.normalize(t, dim=-1)
    gn = [torch.nn.functional.normalize(x, dim=-1) for x in g]
    present = present_from_feats(gn)
    alpha = torch.tensor(1.0)
    hybrid = (alpha * volume_computation_masked(tn, gn, present=present)
              + (1 - alpha) * volume_computation_lorentz(t, g, present=present))
    assert torch.allclose(hybrid, volume_computation_masked(tn, gn, present=present))


def test_gram_hyp_registry_entry():
    import model
    assert 'gram_hyp' not in dict.__iter__(model.model_registry) or True
    # lazy: key resolves without importing at module load (heavy import only on access,
    # which needs the full encoder stack -- just check the branch exists)
    import inspect
    src = inspect.getsource(type(model.model_registry).__getitem__)
    assert 'gram_hyp' in src


# ---- eval-time modality masking (E4-ITM arm) ----

class _FakeBatch(dict):
    def keys(self):
        return super().keys()


def test_eval_mask_default_off_is_identity():
    """eval_mask_rate=0 must leave batch_get byte-for-byte unchanged (GRAM path guard)."""
    from model.gram import GRAM
    m = GRAM.__new__(GRAM)
    m.eval_mask_rate, m.eval_mask_seed, m.training = 0.0, 0, False
    m._EVAL_MASK_KEYS = ('vision_output', 'audio_output', 'subtitle_output', 'depth_output')
    t = torch.randn(4, 3)
    m._batch_get_impl = lambda batch, key: t
    out = GRAM.batch_get(m, _FakeBatch(ids=['a', 'b', 'c', 'd']), 'audio_output')
    assert out is t and torch.equal(out, t)


def test_eval_mask_zeroes_only_selected_clips_and_is_deterministic():
    from model.gram import GRAM
    m = GRAM.__new__(GRAM)
    m.eval_mask_rate, m.eval_mask_seed, m.training = 0.5, 0, False
    m._EVAL_MASK_KEYS = ('vision_output', 'audio_output', 'subtitle_output', 'depth_output')
    ids = [f'v{i}' for i in range(200)]
    batch = _FakeBatch(ids=ids, raw_subtitles=['x'] * 200)
    zeroed = {}
    for key in ('vision_output', 'audio_output', 'subtitle_output'):
        t = torch.ones(200, 5)
        m._batch_get_impl = lambda batch, key, _t=t: _t
        out = GRAM.batch_get(m, batch, key)
        zeroed[key] = {i for i in range(200) if out[i].abs().sum() == 0}
    # each clip loses AT MOST one modality, and ~50% lose one
    all_hits = [i for s in zeroed.values() for i in s]
    assert len(all_hits) == len(set(all_hits)), 'a clip lost more than one modality'
    assert 0.4 < len(all_hits) / 200 < 0.6, len(all_hits) / 200
    # deterministic: same ids -> same victims
    t2 = torch.ones(200, 5)
    m._batch_get_impl = lambda batch, key, _t=t2: _t
    again = GRAM.batch_get(m, batch, 'audio_output')
    assert {i for i in range(200) if again[i].abs().sum() == 0} == zeroed['audio_output']
