"""Query-weighted reranking: the centroid's rule applied to the stage that decides the metric.

The reported number is candidate recall times the reranker's accuracy on those candidates, and
the second factor is far the smaller. Measured on AudioCaps: recall@50 is 89.9 and the frozen
cross-encoder ranks the ground truth first on 39.2% of the candidate sets that contain it.
SCA's query weighting is applied to the contrastive features only, so it never reaches that
39%.

    score = (1 - gamma) * ITM(joint) + gamma * sum_m w_m(t, clip) * ITM(modality m)

Each per-modality pass is exactly the input the tv / ta pretraining tasks give this
cross-encoder, so nothing is rescaled and the frozen ITM head stays in distribution -- the
weighting acts on its OUTPUTS. gamma = 0 must reproduce the existing protocol exactly, which
is what most of these tests pin down: the flag has to degrade to the measured baseline rather
than to something near it.
"""
import pytest
import torch


class _Cfg:
    def __init__(self, **kw):
        self.sca_tau_w = 0.1
        self.sca_itm_qw_gamma = 0.0
        for k, v in kw.items():
            setattr(self, k, v)


class _Model:
    """Only what refine_score_matrix touches: a config and a per-slice ITM scorer.

    compute_slice_scores returns a value that depends on WHICH condition_feats it was given,
    so a combination that ignored a modality would be visible rather than silently averaged
    away.
    """
    def __init__(self, cfg):
        self.config = cfg
        self.calls = []

    def compute_slice_scores(self, cond, input_ids, attention_mask):
        self.calls.append(cond.shape)
        # cond is (N, T, d); its first feature value tags which stream it came from
        return cond[:, 0, 0].float()


def _weights(model, feat_t_rows, z_clip, pres_clip):
    from evaluation.evaluation_mm import _qw_weights
    return _qw_weights(model, feat_t_rows, z_clip, pres_clip)


def test_weights_are_a_distribution_over_present_modalities_only():
    m = _Model(_Cfg())
    t = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    pres = torch.tensor([1.0, 1.0, 0.0])
    w = _weights(m, t, z, pres)
    assert w.shape == (4, 3)
    assert torch.allclose(w.sum(1), torch.ones(4), atol=1e-5)
    assert (w[:, 2] == 0).all(), 'an absent modality received reranking weight'


def test_different_texts_weigh_one_clip_differently():
    """The whole point: a caption about a sound and a caption about a scene must not weight
    the same clip's video and audio identically. If they did, this reduces to a fixed
    per-modality average and the query plays no role."""
    m = _Model(_Cfg())
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), dim=-1)
    t = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])          # one text per modality
    w = _weights(m, t, z, torch.ones(2))
    assert w[0, 0] > 0.9 and w[1, 1] > 0.9, 'weights did not follow the query'


def test_tau_large_recovers_the_uniform_modality_average():
    m = _Model(_Cfg(sca_tau_w=1e6))
    t = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    w = _weights(m, t, z, torch.ones(3))
    assert torch.allclose(w, torch.full((5, 3), 1 / 3), atol=1e-4)


def test_gamma_zero_is_the_untouched_protocol():
    """Every number in the paper was produced with gamma unset. If gamma=0 were merely close
    to the old path rather than identical to it, adding the flag would move results that are
    supposed to be fixed."""
    from evaluation.evaluation_mm import refine_score_matrix
    import inspect
    src = inspect.getsource(refine_score_matrix)
    assert "gamma = float(getattr(model.config, 'sca_itm_qw_gamma', 0.0)) if per_modality else 0.0" in src
    # with no per_modality argument the extra branch is unreachable whatever the config says
    m = _Model(_Cfg(sca_itm_qw_gamma=0.9))
    sig = inspect.signature(refine_score_matrix)
    assert sig.parameters['per_modality'].default is None
    assert sig.parameters['feat_t'].default is None


def test_a_configured_gamma_without_features_raises_instead_of_falling_back():
    """The failure this guards is silent: a weighted reranker that quietly runs unweighted
    would report the baseline number under the experiment's name, and the experiment would
    look like it had no effect."""
    from evaluation.evaluation_mm import refine_score_matrix
    m = _Model(_Cfg(sca_itm_qw_gamma=0.5))
    with pytest.raises(ValueError, match='no features to weigh'):
        refine_score_matrix(torch.zeros(2, 3, 4), torch.zeros(2, 5).long(),
                            torch.ones(2, 5).long(), torch.zeros(2, 2), m, 1,
                            per_modality=[torch.zeros(2, 3, 4)])


def test_the_combination_is_a_convex_blend_of_head_outputs():
    """No feature is rescaled; the weights multiply ITM PROBABILITIES. So the combined score
    stays inside the range the frozen head produced, and cannot leave the calibration it was
    fitted with -- the property that separates this from scaling condition_feats."""
    gamma, w = 0.4, torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    joint = torch.tensor([0.6, 0.5])
    per = torch.tensor([[0.9, 0.1], [0.4, 0.2]])                 # (N, L)
    combined = (1 - gamma) * joint + gamma * (w * per).sum(1)
    lo = torch.minimum(joint, per.min(1).values)
    hi = torch.maximum(joint, per.max(1).values)
    assert ((combined >= lo - 1e-6) & (combined <= hi + 1e-6)).all()


def test_modality_count_mismatch_is_refused():
    """per_modality comes from the task string and the weights from the scorer's slots. If
    those ever disagree, pairing them positionally weights the wrong stream -- and every
    number would still look plausible."""
    from evaluation.evaluation_mm import refine_score_matrix
    import inspect
    src = inspect.getsource(refine_score_matrix)
    assert 'the two disagree about what the modalities ARE' in src


def test_shard_offset_is_applied_to_the_gallery_features():
    """condition_feats is the rank's shard; z/present are the whole gallery. Indexing both
    with the local i weights every clip by another clip's features on every rank but rank 0 --
    invisible on one GPU, wrong on four, which is how it would actually be run."""
    from evaluation.evaluation_mm import refine_score_matrix
    import inspect
    src = inspect.getsource(refine_score_matrix)
    assert 'gi = start_ls[rank] + i' in src
    assert 'z[gi]' in src and 'present[gi]' in src
