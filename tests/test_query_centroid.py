"""Query-conditioned centroid weights: w_m proportional to exp(<t, z_m>/tau).

The property that matters for the paper is that this is still a centroid -- a convex
combination of unit vectors -- with tau interpolating between the uniform mean and
max-over-modalities, and that it is defined at every arity including k=2, where the earlier
reliability weighting was degenerate by symmetry."""
import torch

from model.centroid import (masked_spherical_mean, query_centroid_scores, query_weights)


def _fixture(seed=0, Ng=13, L=3, d=16, Nt=9, drop=0.3):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(Nt, d, generator=g)
    t = t / t.norm(dim=-1, keepdim=True)
    z = torch.randn(Ng, L, d, generator=g)
    z = z / z.norm(dim=-1, keepdim=True)
    present = (torch.rand(Ng, L, generator=g) > drop).float()
    present[present.sum(1) == 0, 0] = 1.0
    return t, z * present.unsqueeze(-1), present


def test_closed_form_matches_the_per_pair_centroid():
    t, z, present = _fixture()
    naive = torch.stack([
        (t[i].unsqueeze(0) * masked_spherical_mean(
            z, present, weighting='query', tau_w=0.1,
            query=t[i].expand(z.shape[0], z.shape[-1]))[0]).sum(-1)
        for i in range(t.shape[0])])
    assert torch.allclose(naive, query_centroid_scores(t, z, present, tau=0.1), atol=1e-4)


def test_large_tau_recovers_the_uniform_centroid():
    t, z, present = _fixture()
    mu, _, _ = masked_spherical_mean(z, present)
    assert torch.allclose(t @ mu.T, query_centroid_scores(t, z, present, tau=1e6), atol=1e-4)


def test_small_tau_recovers_max_over_modalities():
    """The limit is asymptotic in tau relative to the top-1/top-2 cosine GAP, not in tau
    alone: the softmax saturates once gap/tau is large. On this fixture the smallest gap is
    ~1e-3, so tau=1e-3 is still a soft mix (off by 6e-2) while tau=1e-5 is exact to 6e-8."""
    t, z, present = _fixture()
    sims = torch.einsum('td,gld->tgl', t, z).masked_fill(present.unsqueeze(0) <= 0, float('-inf'))
    assert torch.allclose(sims.max(-1).values,
                          query_centroid_scores(t, z, present, tau=1e-5), atol=1e-5)


def test_absent_modalities_never_receive_weight():
    t, z, present = _fixture(Ng=9, Nt=9)
    w = query_weights(z, t, present, tau=0.1)
    assert (w[present <= 0] == 0).all()
    assert torch.allclose(w.sum(1), torch.ones(w.shape[0]), atol=1e-5)


def test_defined_at_k2_unlike_reliability_weighting():
    """Reliability weighting is forced to 0.5/0.5 at k=2 by symmetry; the query breaks it.
    DiDeMo, ActivityNet and AudioCaps are all k=2 galleries, so this is the case that
    decides whether the mechanism is usable at all."""
    t, z, present = _fixture(L=2, drop=0.0, Ng=9, Nt=9)
    w = query_weights(z, t, present, tau=0.1)
    assert (w - 0.5).abs().max() > 0.05, 'weights collapsed to uniform at k=2'


def test_scoring_is_chunk_invariant():
    t, z, present = _fixture(Nt=17)
    a = query_centroid_scores(t, z, present, tau=0.1, chunk=4)
    b = query_centroid_scores(t, z, present, tau=0.1, chunk=1000)
    assert torch.allclose(a, b, atol=1e-6)


def test_query_weighting_without_a_query_raises():
    _t, z, present = _fixture()
    try:
        masked_spherical_mean(z, present, weighting='query')
    except ValueError as e:
        assert 'query' in str(e)
    else:
        raise AssertionError('silently fell back to the uniform centroid')


def test_score_matrix_is_pairwise_not_self_conditioned():
    """The bug this guards: a query-weighted centroid is a function of the text it is
    scored against, so building mu_j from t_j and then doing feat_t @ mu.T conditions the
    positive on its own text and every negative on the wrong one. The model can then win by
    sharpening the weights instead of learning features, and it does not match inference.

    The correct matrix is S[i, j] = <t_i, mu(z_j | t_i)>. Its DIAGONAL coincides with the
    self-conditioned form; its off-diagonal must not."""
    t, z, present = _fixture(Ng=6, Nt=6)
    S = query_centroid_scores(t, z, present, tau=0.1)

    self_cond = torch.stack([
        masked_spherical_mean(z, present, weighting='query', tau_w=0.1,
                              query=t[j].expand(z.shape[0], z.shape[-1]))[0][j]
        for j in range(z.shape[0])])                      # mu_j built from t_j
    naive = t @ self_cond.T                               # what l_align used to compute

    assert torch.allclose(S.diagonal(), naive.diagonal(), atol=1e-5), \
        'diagonal must agree: both condition clip j on text j'
    off = ~torch.eye(6, dtype=torch.bool)
    assert (S[off] - naive[off]).abs().max() > 1e-3, \
        'off-diagonal identical -- the weighting is not query-dependent, so this test is vacuous'
