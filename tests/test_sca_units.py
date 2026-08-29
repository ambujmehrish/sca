"""SCA unit tests: the §1.4 guards from the k=2 analysis + the GRAM regression contract
(present=None == plain GRAM byte-for-byte). CPU-only, no encoder stacks -- runnable in CI.

    python -m pytest tests/ -q
"""
import os
import sys
import math
import json

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.volume import (volume_computation, volume_computation2, volume_computation3,
                          volume_computation4, volume_computation5,
                          volume_computation_masked, volume_computation_mean_imputed,
                          present_from_feats)
from model.centroid import masked_spherical_mean, concept_resultant, centroid_scores
from model.prototypes import PrototypeMemory
from model.losses_sca import (l_align, l_sem, l_mask, l_concept, l_unif,
                              sharpen_targets, check_calibration_config)
from model.pmrl_loss import pmrl_lambda1, pmrl_loss
from model.lora import LoRALinear, LoRAQKVLinear, inject_lora, merge_all, unmerge_all
from model.hypergraph import concept_incidence, doc_incidence
from data.mask_sampler import MaskSampler
from evaluation.eval_missing import (drop_mask, score, recall_at_k, missing_grid,
                                     fit_affine_calibration, apply_affine_calibration)
from evaluation.eval_calibration import calibration_regression, graded_ndcg

torch.manual_seed(0)


def unit(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape), dim=-1)


# --------------------------------------------------------------------------- volume / GRAM

class TestGramRegression:
    """The GRAM path must remain byte-for-byte intact (plan §0 / DoD 6)."""

    def test_masked_none_equals_plain(self):
        t, g = unit(6, 32), [unit(5, 32) for _ in range(3)]
        assert torch.allclose(volume_computation_masked(t, g, present=None),
                              torch.sqrt(torch.abs(torch.det(
                                  _gram(t, g))) + 1e-8), atol=1e-6)
        # and against GRAM's own generic form (up to its missing eps guard)
        ref = volume_computation(t, *g)
        out = volume_computation_masked(t, g, present=None)
        assert torch.allclose(out, torch.sqrt(ref ** 2 + 1e-8), atol=1e-5)

    def test_masked_all_ones_equals_none(self):
        t, g = unit(4, 16), [unit(7, 16) for _ in range(2)]
        ones = torch.ones(7, 2)
        assert torch.equal(volume_computation_masked(t, g, present=ones),
                           volume_computation_masked(t, g, present=None))

    def test_masked_reduces_arity(self):
        # dropping a modality via present == computing the lower-arity volume directly
        t = unit(4, 16)
        v, a, s = unit(5, 16), unit(5, 16), unit(5, 16)
        present = torch.tensor([[1., 1., 0.]]).expand(5, -1)
        got = volume_computation_masked(t, [v, a, s], present=present)
        want = volume_computation_masked(t, [v, a], present=None)
        assert torch.allclose(got, want, atol=1e-5)

    def test_masked_is_not_the_degenerate_zero_fill(self):
        """The baseline in the missing-modality table is NOT scored by zero-filling.

        This is the misreading the table invites, and it would invalidate the whole
        experiment: feeding a zero vector to the Gramian puts a zero row AND a zero column
        into G, so det(G) = 0 for every masked clip, every clip scores the same constant,
        and the ranking among them is arbitrary. What the baseline is actually given is the
        phantom-identity construction of volume_computation_masked (utils/volume.py): the
        missing axis is made orthonormal, contributes a factor of exactly 1 to the
        determinant, and the score reduces to that clip's volume over the modalities it
        still has -- strictly positive and still discriminative. test_masked_reduces_arity
        above proves the identity; this proves the contrast with the degenerate case."""
        t = unit(6, 16)
        v, a, s = unit(9, 16), unit(9, 16), unit(9, 16)
        present = torch.ones(9, 3)
        present[:, 2] = 0.0                               # every gallery clip loses S
        got = volume_computation_masked(t, [v, a, s], present=present)
        assert (got > 1e-2).all(), 'masked volume collapsed toward zero'
        assert got.std(dim=1).min() > 1e-2, 'masked volume is constant across the gallery'
        # the degenerate alternative, for contrast: zero-filled S, no presence mask
        degenerate = volume_computation_masked(t, [v, a, torch.zeros_like(s)], present=None)
        assert degenerate.max() < 1e-3, 'fixture does not exhibit the degenerate case'

    def test_fixed_arity_functions_match_generic(self):
        t = unit(3, 24)
        mods = [unit(4, 24) for _ in range(4)]
        for fn, k in ((volume_computation3, 2), (volume_computation4, 3), (volume_computation5, 4)):
            ref = fn(t, *mods[:k])
            gen = volume_computation(t, *mods[:k])
            assert torch.allclose(ref, gen, atol=1e-5)

    def test_present_from_feats(self):
        f1, f2 = unit(4, 8), unit(4, 8)
        f2[1] = 0.0                                       # loader zero-fill
        p = present_from_feats([f1, f2])
        assert p.tolist() == [[1, 1], [1, 0], [1, 1], [1, 1]]

    def test_mean_imputed_variant(self):
        t, g = unit(4, 16), [unit(5, 16) for _ in range(3)]
        ones = torch.ones(5, 3)
        # all-present == plain volume byte-for-byte
        assert torch.equal(volume_computation_mean_imputed(t, g, present=ones),
                           volume_computation_masked(t, g, present=None))
        # imputing a missing modality with the mean of present ones == manual construction
        present = torch.ones(5, 3); present[2, 1] = 0.0
        got = volume_computation_mean_imputed(t, g, present=present)
        g_manual = [x.clone() for x in g]
        g_manual[1][2] = (g[0][2] + g[2][2]) / 2
        want = volume_computation_masked(t, g_manual, present=None)
        assert torch.allclose(got, want, atol=1e-6)


def _gram(t, gallery):
    B1, B2, L = t.shape[0], gallery[0].shape[0], len(gallery)
    ll = torch.einsum('bi,bi->b', t, t).unsqueeze(1).expand(-1, B2)
    li = [t @ x.T for x in gallery]
    rows = [torch.stack([ll] + li, dim=-1)]
    for i in range(L):
        row = [li[i]]
        for j in range(L):
            row.append(torch.einsum('bi,bi->b', gallery[i], gallery[j]).unsqueeze(0).expand(B1, -1))
        rows.append(torch.stack(row, dim=-1))
    return torch.stack(rows, dim=-2).float()


# --------------------------------------------------------------------------- centroid

class TestCentroid:
    def test_all_ones_equals_plain_mean(self):
        z = unit(6, 3, 32)
        mu_m, A_m, _ = masked_spherical_mean(z, torch.ones(6, 3))
        mu_p, A_p, _ = masked_spherical_mean(z, None)
        assert torch.allclose(mu_m, mu_p) and torch.allclose(A_m, A_p)
        assert torch.allclose(mu_p.norm(dim=-1), torch.ones(6), atol=1e-5)

    def test_single_modality_branch(self):
        # |M| = 1: mu is exactly the surviving embedding, A == 1 with NO gradient
        z = unit(4, 3, 16).requires_grad_(True)
        present = torch.zeros(4, 3); present[:, 1] = 1.0
        mu, A, n = masked_spherical_mean(z, present)
        assert torch.allclose(mu, torch.nn.functional.normalize(z[:, 1], dim=-1), atol=1e-6)
        assert torch.allclose(A, torch.ones(4), atol=1e-5)
        # the degenerate branch is detached: backprop through A must leave z ungradiented
        A.sum().backward()
        assert z.grad is None or torch.all(z.grad == 0)
        assert n.tolist() == [1.0] * 4

    def test_near_antipodal_counter(self):
        # two antipodal vectors: resultant ~ 0 -- must stay finite (eps guard)
        v = unit(1, 16)
        z = torch.stack([v, -v], dim=1)                    # (1, 2, d)
        mu, A, _ = masked_spherical_mean(z, torch.ones(1, 2))
        assert torch.isfinite(mu).all() and torch.isfinite(A).all()
        assert A.item() < 1e-4

    def test_gradient_flows(self):
        z = unit(5, 3, 16).clone().requires_grad_(True)
        mu, A, _ = masked_spherical_mean(z, torch.ones(5, 3))
        (mu.sum() + A.sum()).backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()

    def test_concept_resultant(self):
        mu = unit(6, 8)
        labels = torch.tensor([0, 0, 1, 1, 1, 0])
        A_c, cnt = concept_resultant(mu, labels, num_classes=2)
        assert cnt.tolist() == [3.0, 3.0]
        assert ((A_c >= 0) & (A_c <= 1 + 1e-5)).all()
        # identical members -> A(c) == 1
        same = unit(1, 8).expand(4, -1)
        A_c1, _ = concept_resultant(same, torch.zeros(4, dtype=torch.long), num_classes=1)
        assert abs(A_c1.item() - 1.0) < 1e-5


# --------------------------------------------------------------------------- P5 ablation knobs

class TestGatedCentroid:
    def test_zero_init_equals_uniform(self):
        z = unit(6, 3, 32)
        present = torch.ones(6, 3); present[2, 1] = 0.0
        mu_u, A_u, _ = masked_spherical_mean(z, present)
        mu_g, A_g, _ = masked_spherical_mean(z, present, gates=torch.zeros(3))
        assert torch.allclose(mu_g, mu_u, atol=1e-5)          # zero-init == uniform centroid
        assert torch.allclose(A_g, A_u, atol=1e-5)

    def test_gates_never_weight_missing_modalities(self):
        z = unit(4, 3, 16)
        z[:, 2] = 0.0
        present = torch.tensor([[1., 1., 0.]]).expand(4, -1)
        gates = torch.tensor([0.0, 0.0, 10.0])                # huge gate on the MISSING one
        mu, _, _ = masked_spherical_mean(z, present, gates=gates)
        want, _, _ = masked_spherical_mean(z, present)        # must reduce to present-only
        assert torch.allclose(mu, want, atol=1e-5)

    def test_gates_receive_gradient_and_A_is_gate_independent(self):
        z = unit(5, 3, 16)
        gates = torch.zeros(3, requires_grad=True)
        mu, A, _ = masked_spherical_mean(z, None, gates=gates)
        mu.sum().backward()
        assert gates.grad is not None and torch.isfinite(gates.grad).all()
        _, A_u, _ = masked_spherical_mean(z, None)
        assert torch.allclose(A, A_u, atol=1e-6)              # A(M) is a set property


class TestBatchPrototypes:
    def test_means_and_absent_classes(self):
        from model.prototypes import batch_prototypes
        mu = unit(4, 8)
        labels = torch.tensor([0, 0, 2, 2])
        protos, has = batch_prototypes(mu, labels, num_concepts=3)
        assert has.tolist() == [True, False, True]
        want = torch.nn.functional.normalize(mu[:2].mean(0), dim=-1)
        assert torch.allclose(protos[0], want, atol=1e-5)
        assert protos[1].abs().sum() == 0                     # absent class stays zero


class TestSetMeasures:
    def test_bounds_and_collinear_extremes(self):
        from evaluation.measure_comparison import set_measures
        v = unit(1, 16)
        z_same = torch.cat([v, v, v]).unsqueeze(0)            # (1, 3, d) collinear
        m = set_measures(z_same)
        assert abs(m['A'].item() - 1.0) < 1e-4                # perfectly aligned set
        assert abs(m['lambda1_norm'].item() - 1.0) < 1e-4     # lambda1 = |M|
        assert m['logdetG'].item() < -5.0                     # det -> 0 => logdet -> -inf
        z_rand = unit(8, 3, 64)
        r = set_measures(z_rand)
        assert ((r['A'] >= 0) & (r['A'] <= 1 + 1e-5)).all()
        assert (r['lambda1_norm'] <= 1 + 1e-4).all()

    def test_masked_reduces_to_subset(self):
        from evaluation.measure_comparison import set_measures
        z = unit(6, 3, 32)
        present = torch.tensor([[1., 1., 0.]]).expand(6, -1)
        got = set_measures(z, present)
        want = set_measures(z[:, :2])
        assert torch.allclose(got['A'], want['A'], atol=1e-5)
        assert torch.allclose(got['lambda1_norm'], want['lambda1_norm'], atol=1e-4)


class TestDiagnostics:
    def test_rankme_bounds(self):
        from evaluation.diagnostics import rankme
        iso = torch.randn(500, 32)
        r = rankme(iso)
        assert 25.0 < r['rankme'] <= 32.0                     # isotropic: near full rank
        collapsed = torch.randn(500, 1) @ torch.randn(1, 32)
        assert rankme(collapsed)['rankme'] < 1.5              # rank-1 collapse

    def test_modality_gap_zero_for_identical(self):
        from evaluation.diagnostics import modality_gap
        x = unit(100, 16)
        g = modality_gap({'a': x, 'b': x.clone(), 'c': unit(100, 16)})
        assert g['a-b']['gap'] < 1e-6 and g['a-c']['gap'] > 0.0

    def test_align_unif_directions(self):
        from evaluation.diagnostics import align_unif
        x = unit(200, 16)
        perfect = align_unif(x, x.clone())
        noisy = align_unif(x, unit(200, 16))
        assert perfect['align'] < 1e-6 < noisy['align']

    def test_tsne_without_sklearn_is_hard_error_or_runs(self):
        from evaluation.diagnostics import project_2d
        feats = {'t': unit(60, 8), 'v': unit(60, 8)}
        try:
            import sklearn  # noqa: F401
            coords, names, slices = project_2d(feats, method='tsne')
            assert coords.shape == (120, 2)
        except ImportError:
            with pytest.raises(ImportError, match='refusing to substitute'):
                project_2d(feats, method='tsne')
        coords, _, _ = project_2d(feats, method='pca')        # pca always available
        assert coords.shape == (120, 2)


class TestLatexTables:
    def test_null_renders_dash_and_gram_rows_present(self, tmp_path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark_eval'))
        from make_latex_tables import section_table, load_measured
        from import_published_rows import load_published_rows
        pub = load_published_rows()
        measured = {'zeroshot_t2v': {'SCA (ours)': {'MSR-VTT|T-V': [55.0, 84.0]}}}
        tex = section_table('zeroshot_t2v', 'test', pub, measured)
        assert '55.0' in tex and '52.8' in tex                # measured + GRAM published
        assert '--' in tex                                    # SCA's unfilled cells dash out
        assert ' 0.0 ' not in tex                             # nulls NEVER render as zero
        # measured-row schema is enforced
        bad = tmp_path / 'bad.json'
        bad.write_text(json.dumps({'method': 'X', 'section': 'zeroshot_t2v',
                                   'rows': {'MSR-VTT|T-V': [1.0]}}))
        with pytest.raises(ValueError):
            load_measured(str(tmp_path))


# --------------------------------------------------------------------------- prototypes

class TestPrototypes:
    def test_init_from_first_means_then_ema(self):
        mem = PrototypeMemory(3, 8, eta=0.9)
        mu1 = unit(6, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        mem.update(mu1, labels)
        assert mem.initialized.all()
        # first update == class mean (renormed), no EMA blending yet
        want = torch.nn.functional.normalize(mu1[:2].mean(0, keepdim=True), dim=-1)
        assert torch.allclose(mem.protos[0], want[0], atol=1e-5)
        # second update blends with eta and renorms
        p_before = mem.protos.clone()
        mu2 = unit(6, 8)
        mem.update(mu2, labels)
        assert not torch.allclose(mem.protos, p_before)
        assert torch.allclose(mem.protos.norm(dim=-1), torch.ones(3), atol=1e-5)

    def test_staleness_reset(self):
        mem = PrototypeMemory(2, 8, eta=0.99)
        labels = torch.tensor([0, 1])
        for _ in range(5):
            mem.update(unit(2, 8), labels)
        assert (mem.run_count > 0).all()
        mem.reset_from_running()
        assert (mem.run_count == 0).all() and mem.initialized.all()
        assert torch.allclose(mem.protos.norm(dim=-1), torch.ones(2), atol=1e-5)

    def test_no_grad(self):
        mem = PrototypeMemory(2, 8)
        mu = unit(4, 8).requires_grad_(True)
        mem.update(mu, torch.tensor([0, 0, 1, 1]))
        assert mem.protos.grad_fn is None


# --------------------------------------------------------------------------- losses

class TestLosses:
    def test_l_align_perfect_alignment_is_minimal(self):
        t = unit(8, 16)
        targets = torch.arange(8)
        aligned = l_align(t, t, t, t, 0.07, targets, label_smoothing=0.0)
        shuffled = l_align(t, t[torch.randperm(8)], t, t, 0.07, targets, label_smoothing=0.0)
        assert aligned < shuffled

    def test_l_sem_zero_at_target(self):
        # when the score matrix already row-softmaxes to P*, the KL vanishes
        s_star = torch.eye(4)
        tau, tau_star = 0.5, 0.5
        sim = s_star.clone()                              # sim/tau == s*/tau* row-wise
        loss = l_sem(sim, s_star, tau, tau_star, calibration='none')
        assert loss.item() < 1e-6

    def test_l_sem_regression_pins_scale(self):
        s_star = torch.eye(3)
        sim_cal = 2.0 * s_star - 1.0                      # exactly the calibration target
        sim_off = sim_cal + 0.5
        l_cal = l_sem(sim_cal, s_star, 0.07, calibration='regression', cal_w=1.0)
        l_off = l_sem(sim_off, s_star, 0.07, calibration='regression', cal_w=1.0)
        assert l_cal < l_off                              # KL alone cannot see the shift

    def test_l_sem_regression_ignores_sparsified_zeros(self):
        # a sparsified S* stores 0 for "unknown"; the regression must not read that as
        # "push this pair to cosine -1"
        s_star = torch.eye(4)                              # off-diag = unknown (sparsified out)
        sim = torch.full((4, 4), 0.3)
        sim.fill_diagonal_(1.0)                            # positives perfectly calibrated
        known = l_sem(sim, s_star, 0.07, calibration='regression', cal_w=1.0,
                      cal_known_only=True)
        dense = l_sem(sim, s_star, 0.07, calibration='regression', cal_w=1.0,
                      cal_known_only=False)
        assert known < dense                               # dense penalizes the unknown pairs
        # with known-only, the regression term is exactly 0 here (diagonal fits perfectly)
        kl_only = l_sem(sim, s_star, 0.07, calibration='none')
        assert torch.allclose(known, kl_only, atol=1e-6)

    def test_calibration_config_guard(self):
        with pytest.raises(ValueError):
            check_calibration_config('regression', tau_learnable=True)
        with pytest.raises(ValueError):
            check_calibration_config('fixed_tau', tau_learnable=True)
        check_calibration_config('none', tau_learnable=True)      # allowed

    def test_l_mask_zero_when_views_agree(self):
        mu = unit(6, 16)
        s = torch.rand(6)
        assert l_mask(mu, mu, s, s).item() < 1e-6
        assert l_mask(mu, -mu, term1=True, term2=False).item() > 1.0

    def test_l_mask_term2_detaches_full_view(self):
        mu_M, mu_K = unit(4, 8).requires_grad_(True), unit(4, 8).requires_grad_(True)
        s_M = torch.rand(4, requires_grad=True)
        s_K = torch.rand(4, requires_grad=True)
        l_mask(mu_M.detach(), mu_K.detach(), s_M, s_K, term1=False).backward()
        assert s_M.grad is not None and s_K.grad is None   # full view is the reference

    def test_l_concept_eps_floor(self):
        protos = unit(2, 16)
        labels = torch.tensor([0, 1])
        mu_close = torch.nn.functional.normalize(protos + 0.001 * torch.randn(2, 16), dim=-1)
        assert l_concept(mu_close, labels, protos, eps_floor=0.05).item() == 0.0
        mu_far = unit(2, 16)
        assert l_concept(mu_far, labels, protos, eps_floor=0.0).item() > 0.0

    def test_l_unif_prefers_spread(self):
        spread = unit(16, 8)
        clumped = torch.nn.functional.normalize(
            unit(1, 8) + 0.01 * torch.randn(16, 8), dim=-1)
        assert l_unif(spread) < l_unif(clumped)

    def test_l_unif_weighting_exempts_similar(self):
        mu = unit(8, 16)
        s_star = torch.eye(8)
        w_loss = l_unif(mu, s_star=s_star)
        assert torch.isfinite(w_loss)

    def test_sharpen_targets_rows_sum_to_one(self):
        p = sharpen_targets(torch.rand(5, 5), tau_star=0.3)
        assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)


# --------------------------------------------------------------------------- mask sampler

class TestMaskSampler:
    def test_schedule_endpoints(self):
        ms = MaskSampler(3, p_full_start=1.0, p_full_end=0.5, schedule_steps=100)
        assert ms.p_full(0) == 1.0
        assert abs(ms.p_full(100) - 0.5) < 1e-9
        assert abs(ms.p_full(10 ** 6) - 0.5) < 1e-9        # clamps after schedule end

    def test_p_full_1_never_masks(self):
        ms = MaskSampler(3, p_full_start=1.0, p_full_end=0.5, schedule_steps=100)
        m = ms.sample(64, 0, torch.device('cpu'))
        assert m.min() == 1.0

    def test_never_drops_below_one(self):
        ms = MaskSampler(3, p_full_start=0.0, p_full_end=0.0, n_drop=3)
        present = torch.ones(128, 3)
        for step in range(5):
            m = ms.sample(128, step, torch.device('cpu'), present=present)
            assert ((present * m).sum(1) >= 1).all()

    def test_respects_upstream_absence(self):
        ms = MaskSampler(3, p_full_start=0.0, p_full_end=0.0)
        present = torch.ones(64, 3); present[:, 2] = 0.0   # modality 2 really missing
        for step in range(5):
            m = ms.sample(64, step, torch.device('cpu'), present=present)
            assert ((present * m).sum(1) >= 1).all()

    def test_from_config_reads_train_mask_knobs(self):
        from easydict import EasyDict as edict
        cfg = edict(train_mask_p_full_start=0.9, train_mask_p_full_end=0.4,
                    train_mask_schedule_steps=100, train_mask_mode='uniform',
                    train_mask_n_drop=2)
        ms = MaskSampler.from_config(cfg)
        assert ms.p_full(0) == 0.9 and abs(ms.p_full(100) - 0.4) < 1e-9 and ms.n_drop == 2
        # defaults when knobs are absent (the gram.py hook with bare train_mask=true)
        ms2 = MaskSampler.from_config(edict())
        assert ms2.p_full(0) == 1.0 and abs(ms2.p_full(10 ** 6) - 0.5) < 1e-9

    def test_sample_and_apply_zero_fills_and_respects_absence(self):
        ms = MaskSampler(3, p_full_start=0.0, p_full_end=0.0)
        feats = [unit(64, 16), unit(64, 16), unit(64, 16)]
        feats[2][:32] = 0.0                                     # modality 2 really absent
        g = torch.Generator().manual_seed(0)
        masked, mask = ms.sample_and_apply(feats, step=0, generator=g)
        pres = present_from_feats(masked)
        assert (pres <= present_from_feats(feats)).all()        # never resurrects absence
        assert (pres.sum(1) >= 1).all()                         # |M|=1 floor holds
        assert (pres == present_from_feats(feats) * mask).all() # zero-fill == the draw

    def test_apply_mask_zero_fills_for_present_from_feats(self):
        feats = [unit(8, 16), unit(8, 16)]
        mask = torch.ones(8, 2); mask[3, 1] = 0.0
        out = MaskSampler.apply_mask(feats, mask)
        assert present_from_feats(out).tolist() == mask.tolist()

    def test_freq_weighted_draw(self):
        ms = MaskSampler(2, p_full_start=0.0, p_full_end=0.0, mode='freq', freq=[0.0, 1.0])
        m = ms.sample(256, 0, torch.device('cpu'))
        assert (m[:, 0] == 1.0).all() and (m[:, 1] == 0.0).all()

    def test_freq_zero_means_never_drop_no_uniform_fallback(self):
        # after the freq-1.0 modality is dropped, the only remaining candidate has freq 0:
        # the second draw must NOT fall back to uniform -- the clip keeps it
        ms = MaskSampler(3, p_full_start=0.0, p_full_end=0.0, mode='freq',
                         freq=[0.0, 0.0, 1.0], n_drop=2)
        m = ms.sample(128, 0, torch.device('cpu'))
        assert (m[:, 0] == 1.0).all() and (m[:, 1] == 1.0).all() and (m[:, 2] == 0.0).all()

    def test_freq_all_zero_rejected(self):
        with pytest.raises(AssertionError):
            MaskSampler(2, mode='freq', freq=[0.0, 0.0])


# --------------------------------------------------------------------------- LoRA

class TestLoRA:
    def test_zero_init_is_identity(self):
        base = torch.nn.Linear(16, 16)
        lora = LoRALinear(base, r=4)
        x = torch.randn(3, 16)
        assert torch.allclose(lora(x), base(x), atol=1e-6)

    def test_merge_unmerge_roundtrip(self):
        base = torch.nn.Linear(16, 16)
        lora = LoRALinear(base, r=4)
        with torch.no_grad():
            lora.lora_B.normal_()
        x = torch.randn(3, 16)
        y = lora(x)
        lora.merge()
        assert torch.allclose(lora(x), y, atol=1e-5)       # merged == unmerged forward
        lora.unmerge()
        assert torch.allclose(lora(x), y, atol=1e-5)

    def test_qkv_only_updates_q_and_v(self):
        base = torch.nn.Linear(16, 48, bias=False)
        lora = LoRAQKVLinear(base, r=4)
        with torch.no_grad():
            lora.lora_B_q.normal_(); lora.lora_B_v.normal_()
        x = torch.randn(3, 16)
        y0, y1 = base(x), lora(x)
        assert not torch.allclose(y0[:, :16], y1[:, :16])       # q changed
        assert torch.allclose(y0[:, 16:32], y1[:, 16:32])       # k untouched
        assert not torch.allclose(y0[:, 32:], y1[:, 32:])       # v changed
        lora.merge()
        assert torch.allclose(lora(x), y1, atol=1e-5)
        lora.unmerge()

    def test_inject_targets_and_freeze(self):
        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query = torch.nn.Linear(8, 8)
                self.key = torch.nn.Linear(8, 8)
                self.value = torch.nn.Linear(8, 8)
        m = torch.nn.Sequential(Attn())
        wrapped = inject_lora(m, r=2)
        assert len(wrapped) == 2                            # query + value, never key
        assert isinstance(m[0].query, LoRALinear) and isinstance(m[0].value, LoRALinear)
        assert not m[0].query.base.weight.requires_grad
        assert m[0].query.lora_A.requires_grad
        assert not isinstance(m[0].key, LoRALinear)
        # idempotent
        assert inject_lora(m, r=2) == []
        merge_all(m); unmerge_all(m)


# --------------------------------------------------------------------------- PMRL

class TestPMRL:
    def test_lambda1_bounds(self):
        t = unit(1, 16)
        z = [t.clone(), t.clone()]                          # all collinear
        lam = pmrl_lambda1(t, z)
        assert abs(lam.item() - 3.0) < 1e-4                 # M identical unit vectors -> M
        lam_n = pmrl_lambda1(t, z, variant='norm')
        assert abs(lam_n.item() - 1.0) < 1e-4

    def test_masked_matches_reduced_arity(self):
        t = unit(3, 16)
        v, a, s = unit(4, 16), unit(4, 16), unit(4, 16)
        present = torch.tensor([[1., 1., 0.]]).expand(4, -1)
        got = pmrl_lambda1(t, [v, a, s], present=present)
        want = pmrl_lambda1(t, [v, a])
        assert torch.allclose(got, want, atol=1e-4)

    def test_loss_runs_and_backprops(self):
        t = unit(4, 16).requires_grad_(True)
        z = [unit(4, 16).requires_grad_(True) for _ in range(2)]
        loss = pmrl_loss(t, z, torch.arange(4))
        loss.backward()
        assert torch.isfinite(loss) and t.grad is not None


# --------------------------------------------------------------------------- hypergraph

class TestConceptIncidence:
    def test_shapes_and_membership(self):
        labels = torch.tensor([3, 1, 3, 7])
        mask = ('V', 'A', 'S')
        H, uniq = concept_incidence(labels, mask, torch.device('cpu'))
        assert H.shape == (4 * 3, 3) and uniq.tolist() == [1, 3, 7]
        # doc 0 and doc 2 share concept 3 -> same column
        col3 = uniq.tolist().index(3)
        assert H[0 * 3 + 0, col3] == 1.0 and H[2 * 3 + 1, col3] == 1.0
        assert H.sum() == 4 * 3                             # every vertex in exactly one edge

    def test_present_disconnects(self):
        labels = torch.tensor([0, 0])
        present = torch.tensor([[1., 0.], [1., 1.]])
        H, _ = concept_incidence(labels, ('V', 'A'), torch.device('cpu'), present=present)
        assert H[1, 0] == 0.0                               # doc 0's absent audio disconnected
        # same convention as doc_incidence
        Hd = doc_incidence(2, ('V', 'A'), torch.device('cpu'), present=present)
        assert Hd[1, 0] == 0.0


# --------------------------------------------------------------------------- eval harnesses

class TestEvalMissing:
    def test_drop_mask_rates(self):
        g = torch.Generator().manual_seed(0)
        p = drop_mask(100, 3, 0.25, generator=g)
        assert int((p.sum(1) < 3).sum()) == 25
        assert drop_mask(100, 3, 0.0).min() == 1.0

    def test_scorers_agree_on_ranking_when_full(self):
        t, g = unit(10, 32), [unit(10, 32) for _ in range(2)]
        gt = torch.arange(10)
        full = torch.ones(10, 2)
        for method in ('centroid', 'volume_masked', 'volume_imputed', 'pmrl_raw', 'pmrl_norm'):
            d = score(t, g, full, method)
            assert d.shape == (10, 10) and torch.isfinite(d).all()

    def test_recall_perfect_when_gallery_is_query(self):
        t = unit(20, 16)
        d = 1.0 - t @ t.T
        log, _ = recall_at_k(d, torch.arange(20))
        assert log['R@1'] == 100.0

    def test_missing_grid_runs(self):
        t, g = unit(12, 16), [unit(12, 16) for _ in range(3)]
        res = missing_grid(t, g, torch.arange(12), methods=('centroid',),
                           rates=(0.0, 0.5), calibrate=True)
        assert '0%|rand' in res['centroid'] and '50%|rand' in res['centroid']
        assert 'calibrated' in res['centroid']['50%|rand']

    def test_affine_calibration_matches_moments(self):
        t, g = unit(30, 16), [unit(30, 16) for _ in range(2)]
        p = drop_mask(30, 2, 0.5, generator=torch.Generator().manual_seed(1))
        d = score(t, g, p, 'volume_masked')
        cal = fit_affine_calibration(d, p)
        d2 = apply_affine_calibration(d, p, cal)
        card = p.sum(1)
        m_low = d2[:, card == card.min()].mean()
        m_full = d2[:, card == card.max()].mean()
        assert abs(m_low - m_full) < 0.05                   # score shift removed


class TestEvalCalibration:
    def test_r2_perfect_when_scores_equal_target(self):
        s_star = torch.rand(8, 8)
        sim = 2.0 * s_star - 1.0
        out = calibration_regression(sim, s_star)
        assert out['overall']['r2'] > 0.999 and abs(out['overall']['slope'] - 1.0) < 1e-4

    def test_graded_ndcg_bounds(self):
        s_star = torch.rand(6, 20)
        perfect = graded_ndcg(2 * s_star - 1, s_star, k=5)
        assert abs(perfect - 1.0) < 1e-5
        rand = graded_ndcg(torch.randn(6, 20), s_star, k=5)
        assert 0.0 <= rand <= 1.0 + 1e-6


# --------------------------------------------------------------------------- P4 grid driver

class TestRunEvalGrids:
    def test_run_grids_structure_synthetic_free(self):
        # real CLIP features from the A10 workdir when available; otherwise loudly skipped
        feats_path = os.path.join(os.path.dirname(__file__), '..',
                                  'experiments/a10_workdir/features_test.pt')
        if not os.path.exists(feats_path):
            pytest.skip('A10 Flickr8k features not built (run scripts/smoke_test.sh stage 2)')
        from evaluation.run_eval_grids import run_grids
        d = torch.load(feats_path, map_location='cpu')
        # k=2: one gallery modality -- rates cannot drop it (|M|=1 floor), grid still runs
        res = run_grids(d['txt'][:200, 0], [d['img'][:200]],
                        methods=('centroid', 'volume_masked'), rates=(0.0, 0.5))
        assert res['setup']['k_gallery_modalities'] == 1
        for m in ('centroid', 'volume_masked'):
            assert res['e4'][m]['0%|rand']['R@1'] > 20.0        # real CLIP feats retrieve
        # k=1 gallery: identical scores at every rate (nothing was droppable)
        assert res['e4']['centroid']['0%|rand']['R@1'] == res['e4']['centroid']['50%|rand']['R@1']

    def test_feature_file_loader_formats(self, tmp_path):
        from evaluation.run_eval_grids import _load_features
        t, g = unit(4, 8), unit(4, 8)
        p1 = tmp_path / 'a.pt'
        torch.save({'feat_t': t, 'gallery': {'v': g, 'a': g}}, p1)
        ft, gal, gt, ids = _load_features(str(p1))
        assert len(gal) == 2 and gt is None
        p2 = tmp_path / 'bad.pt'
        torch.save({'x': t}, p2)
        with pytest.raises(KeyError):
            _load_features(str(p2))


# --------------------------------------------------------------------------- config expansion

class TestEnvExpansion:
    def test_expands_set_vars(self, monkeypatch):
        from utils.args import expand_env_vars
        monkeypatch.setenv('DATA_ROOT', '/data')
        assert expand_env_vars({'p': '${DATA_ROOT}/x'})['p'] == '/data/x'

    def test_unset_var_is_hard_error(self, monkeypatch):
        from utils.args import expand_env_vars
        monkeypatch.delenv('SCA_UNSET_VAR_XYZ', raising=False)
        with pytest.raises(EnvironmentError):
            expand_env_vars({'p': '${SCA_UNSET_VAR_XYZ}/x'})

    def test_plain_paths_pass_through(self):
        from utils.args import expand_env_vars
        cfg = {'p': '/leonardo_scratch/abs/path', 'n': 3, 'l': ['a', 1]}
        assert expand_env_vars(cfg) == cfg


# --------------------------------------------------------------------------- S* gather

class TestAnnotationReader:
    """The S* builder's input contract: no skipped items, no index-substituted ids,
    no guessed dict layouts."""

    def _write(self, tmp_path, obj):
        p = tmp_path / 'anno.json'
        p.write_text(json.dumps(obj))
        return str(p)

    def test_valid_list_and_key_resolution(self, tmp_path):
        from data.semantic_targets import _read_annotations
        ids, caps, id_key, cap_key = _read_annotations(self._write(tmp_path, [
            {'video_id': 'a', 'caption': 'x'}, {'video_id': 'b', 'caption': 'y'}]))
        assert ids == ['a', 'b'] and caps == ['x', 'y']
        assert id_key == 'video_id' and cap_key == 'caption'

    def test_vast27m_schema(self, tmp_path):
        # the real vast27m_150k item shape: clip_id + vast_cap (list) among other caps
        from data.semantic_targets import _read_annotations
        item = {'clip_id': 'G1DRYgjsZTw.63', 'clip_span': ['a', 'b'], 'url': 'u',
                'vision_cap': ['v'], 'audio_cap': ['a'], 'subtitle': 's',
                'vast_cap': ['the omni caption']}
        ids, caps, id_key, cap_key = _read_annotations(self._write(tmp_path, [item]))
        assert ids == ['G1DRYgjsZTw.63'] and caps == ['the omni caption']
        assert id_key == 'clip_id' and cap_key == 'vast_cap'
        # explicit override picks a different field
        _, caps2, _, ck2 = _read_annotations(self._write(tmp_path, [item]),
                                             caption_key='vision_cap')
        assert caps2 == ['v'] and ck2 == 'vision_cap'

    def test_inconsistent_schema_is_hard_error(self, tmp_path):
        # caption key resolved from item 0 must exist on EVERY item
        from data.semantic_targets import _read_annotations
        with pytest.raises(ValueError, match='inconsistent schema'):
            _read_annotations(self._write(tmp_path, [
                {'video_id': 'a', 'caption': 'x'}, {'video_id': 'b', 'desc': 'y'}]))

    def test_missing_caption_on_first_item_is_hard_error(self, tmp_path):
        from data.semantic_targets import _read_annotations
        with pytest.raises(ValueError, match='caption'):
            _read_annotations(self._write(tmp_path, [{'video_id': 'b'}]))

    def test_missing_id_is_hard_error(self, tmp_path):
        from data.semantic_targets import _read_annotations
        with pytest.raises(ValueError, match='an id'):
            _read_annotations(self._write(tmp_path, [{'caption': 'x'}]))

    def test_dict_without_data_key_is_hard_error(self, tmp_path):
        from data.semantic_targets import _read_annotations
        with pytest.raises(ValueError, match='refusing to guess'):
            _read_annotations(self._write(tmp_path, {'items': [{'video_id': 'a',
                                                                'caption': 'x'}]}))
        ids, _, _, _ = _read_annotations(self._write(tmp_path, {'data': [{'video_id': 'a',
                                                                          'caption': 'x'}]}))
        assert ids == ['a']

    def test_embedding_impl_is_explicit(self):
        from data.semantic_targets import _embed_captions
        with pytest.raises(ValueError, match='unknown embedding impl'):
            _embed_captions(['x'], 'any-model', 'cpu', impl='auto')


class TestSemanticTargets:
    def test_gather_roundtrip(self, tmp_path):
        import torch as T
        from data.semantic_targets import SemanticTargets
        ids = ['a', 'b', 'c']
        cache = {'ids': ids,
                 'topk_idx': T.tensor([[0, 1], [1, 0], [2, 1]]),
                 'topk_val': T.tensor([[1.0, 0.5], [1.0, 0.5], [1.0, 0.25]]).half(),
                 'meta': {}}
        p = tmp_path / 's.pt'
        T.save(cache, p)
        st = SemanticTargets(str(p))
        s = st.gather(['a', 'b'])
        assert s.shape == (2, 2)
        assert s[0, 0] == 1.0 and abs(s[0, 1].item() - 0.5) < 1e-3
        # unknown id is a HARD ERROR (cache/annotation mismatch), never a one-hot fallback
        with pytest.raises(KeyError):
            st.gather(['a', 'zzz'])
