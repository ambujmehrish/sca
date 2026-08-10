import torch
import torch.nn.functional as F

# SCA loss zoo (plan item 1.3.4). Every term operates on L2-normalised text features and
# centroids, so scores are cosines in [-1, 1] at every cardinality. Weights: alpha -> L_sem,
# beta -> L_mask, delta -> L_concept, lambda -> L_unif; L_align is the base retrieval term.


def info_nce(sim, targets, label_smoothing=0.1):
    """Cross-entropy over a (B, N) similarity-logit matrix; identical recipe to GRAM's volume
    softmax (which uses -volume as the logit) so ablations differ only in the geometry."""
    return F.cross_entropy(sim, targets, label_smoothing=label_smoothing)


def l_align(feat_t, mu_all, mu, feat_t_all, tau, targets, label_smoothing=0.1):
    """Symmetric InfoNCE between text and gallery centroids (both directions, gathered
    negatives), the SCA counterpart of GRAM's loss_area."""
    sim_t2m = feat_t @ mu_all.T / tau
    sim_m2t = mu @ feat_t_all.T / tau
    return (info_nce(sim_t2m, targets, label_smoothing)
            + info_nce(sim_m2t, targets, label_smoothing)) / 2


def sharpen_targets(s_star, tau_star=1.0):
    """Row-stochastic target distribution P* from the S* affinity block: temperature-sharpened
    softmax over each row. tau_star < 1 sharpens toward the diagonal; S* = I reproduces plain
    one-hot InfoNCE targets (ablation A3)."""
    return F.softmax(s_star / tau_star, dim=-1)


def l_sem(sim, s_star, tau, tau_star=1.0, calibration='regression', cal_w=1.0,
          cal_known_only=True):
    """Semantic calibration loss (headline E6), with the A10 calibration mechanism.

    sim    : (B, B) raw cosine scores between the batch texts and the batch centroids.
    s_star : (B, B) semantic target affinities in [0, 1] (diagonal 1).
    KL term: KL( P* || row-softmax(sim / tau) ) -- shift-invariant, shapes the ranking.
    Calibration term (default ON): cal_w * || sim - (2 S* - 1) ||^2 pins the absolute cosine
    scale to the target affinity (2S*-1 maps [0,1] -> [-1,1]), which softmax alone cannot do
    (shift invariance). calibration='fixed_tau' relies on tau being frozen at tau* instead
    (the config layer must forbid a learnable tau in that mode); 'none' = KL only.

    cal_known_only (default True): the S* cache is top-k SPARSIFIED, so an absent entry means
    "affinity unknown (below the k-th value)", NOT "affinity exactly 0". Regressing those
    zeros would push every non-top-k pair toward cosine -1 -- a storage artifact, not a
    semantic target -- so the regression is fitted only where S* > 0 (the diagonal is always
    1, hence always included). Set False only with a genuinely dense S*.
    """
    p_star = sharpen_targets(s_star, tau_star)
    log_p = F.log_softmax(sim / tau, dim=-1)
    kl = F.kl_div(log_p, p_star, reduction='batchmean')
    if calibration == 'regression':
        sq = (sim - (2.0 * s_star - 1.0)) ** 2
        if cal_known_only:
            known = (s_star > 0).to(sq.dtype)
            kl = kl + cal_w * (sq * known).sum() / known.sum().clamp(min=1.0)
        else:
            kl = kl + cal_w * sq.mean()
    return kl


def l_mask(mu_M, mu_K, s_M=None, s_K=None, term1=True, term2=True):
    """Masking-consistency loss: mu_M is the centroid of the virtually-masked modality set,
    mu_K the full-set centroid, from the SAME forward pass (mask_sampler bookkeeping).

    term1: centroid consistency 1 - cos(mu_M, mu_K) -- the reduced-arity view must land where
           the full view lands.
    term2: cross-cardinality score calibration (E5): the positive-pair score must not shift
           when modalities drop, so (s_M - stopgrad(s_K))^2 with the full view as reference.
           Toggleable independently (ablation A1 switches term2 off).
    """
    loss = mu_M.new_zeros(())
    if term1:
        loss = loss + (1.0 - (mu_M * mu_K).sum(-1)).mean()
    if term2 and s_M is not None and s_K is not None:
        loss = loss + ((s_M - s_K.detach()) ** 2).mean()
    return loss


def l_concept(mu, labels, protos, eps_floor=0.05):
    """Level-2 concept loss: pull each clip centroid toward its concept prototype nu_c, with an
    eps-floor on the residual 1 - A(c) so concepts are never squeezed to a point (collapse
    guard): only the dispersion ABOVE the floor is penalised."""
    nu = protos[labels]                                   # (B, d), memory is no-grad
    resid = 1.0 - (mu * nu.detach()).sum(-1)              # 1 - A(c) per member
    return F.relu(resid - eps_floor).mean()


def l_unif(mu, t=2.0, s_star=None):
    """Uniformity on the hypersphere (Wang & Isola): log mean exp(-t ||mu_i - mu_j||^2) over
    distinct pairs. Optional (1 - S*) weighting (ablation A8): semantically-close pairs are
    exempted from repulsion instead of being pushed apart."""
    B = mu.shape[0]
    if B < 2:
        return mu.new_zeros(())
    sq_dist = torch.cdist(mu, mu, p=2) ** 2               # (B, B)
    off = ~torch.eye(B, dtype=torch.bool, device=mu.device)
    e = torch.exp(-t * sq_dist)
    if s_star is not None:
        w = (1.0 - s_star).clamp(min=0.0)
        num = (e * w)[off].sum()
        den = w[off].sum().clamp(min=1e-8)
        return torch.log(num / den + 1e-12)
    return torch.log(e[off].mean() + 1e-12)


def check_calibration_config(calibration, tau_learnable):
    """Guard from the k=2 analysis: any calibration claim is meaningless if tau can drift
    (softmax shift/scale-invariance would silently re-absorb the calibration). The config
    layer calls this once at model construction."""
    if calibration in ('regression', 'fixed_tau') and tau_learnable:
        raise ValueError(
            f"sca_calibration='{calibration}' requires a FIXED temperature "
            "(sca_tau_learnable=false): a learnable tau re-absorbs the calibration term "
            "and voids the E6 claim.")
