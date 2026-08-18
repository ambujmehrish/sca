import torch

# E6 harness (plan item 1.3.8): semantic-calibration quality of the raw scores.
# For SCA the claim is that the cosine S(t_i, mu_j) REGRESSES the semantic target S*_ij
# (via 2S*-1), per cardinality |M|; GRAM/PMRL volumes are uncalibrated by construction and
# are reported through the same functions for contrast. Plus graded nDCG: retrieval quality
# when relevance is the graded S* rather than the binary ground truth.

from evaluation.eval_missing import drop_mask, score


def r2_score(pred, target):
    """Plain coefficient of determination."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum().clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).item()


@torch.no_grad()
def calibration_regression(sim, s_star, present=None, known_only=False):
    """S vs S* regression, overall and per gallery cardinality.

    sim    : (T, G) SIMILARITY matrix (higher = closer; pass 1 - dist for distance scorers).
    s_star : (T, G) semantic targets in [0, 1].
    The calibration target is 2S*-1 (the scale L_sem's regression term pins the cosines to).
    known_only: S* caches are top-k sparsified (absent = unknown, not 0); True restricts the
    fit to pairs with a stored affinity (S* > 0), matching l_sem's cal_known_only definition.
    Returns {'overall': {'r2','pearson','slope','intercept'}, 'per_cardinality': {|M|: ...}}.
    """
    target = 2.0 * s_star.float() - 1.0
    known = (s_star > 0) if known_only else None

    def _fit(x, y, mask=None):
        if mask is not None:
            x, y = x[mask], y[mask]
        x, y = x.flatten().float(), y.flatten().float()
        vx, vy = x - x.mean(), y - y.mean()
        denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt()).clamp(min=1e-12)
        pearson = (vx * vy).sum() / denom
        slope = (vx * vy).sum() / vx.pow(2).sum().clamp(min=1e-12)
        intercept = y.mean() - slope * x.mean()
        return {'r2': r2_score(x, y), 'pearson': pearson.item(),
                'slope': slope.item(), 'intercept': intercept.item(), 'n': x.numel()}

    out = {'overall': _fit(sim, target, known)}
    if present is not None:
        card = present.sum(dim=1)
        out['per_cardinality'] = {}
        for m in card.unique().tolist():
            sel = card == m
            out['per_cardinality'][int(m)] = _fit(
                sim[:, sel], target[:, sel], known[:, sel] if known is not None else None)
    return out


@torch.no_grad()
def graded_ndcg(sim, s_star, k=10):
    """nDCG@k with the GRADED relevance rel_ij = S*_ij (not binary ground truth): rewards a
    scorer that ranks semantically-close non-matches high -- exactly what a calibrated S
    should do and a purely discriminative one need not."""
    T = sim.shape[0]
    order = sim.argsort(dim=1, descending=True)[:, :k]
    rel = torch.gather(s_star.float(), 1, order)
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=sim.device))
    dcg = (rel * discount).sum(dim=1)
    ideal = (s_star.float().sort(dim=1, descending=True).values[:, :k] * discount).sum(dim=1)
    return (dcg / ideal.clamp(min=1e-12)).mean().item()


@torch.no_grad()
def calibration_grid(feat_t, gallery_feats, s_star, methods=('centroid', 'volume_masked'),
                     rates=(0.0, 0.25, 0.5), seed=0, ndcg_k=10):
    """E6 sweep: per method x missing-rate, the S-vs-S* regression (per cardinality) and the
    graded nDCG. Distance scorers are negated into similarities so every method is judged on
    the same footing (their affine scale is exactly what the regression measures)."""
    L = len(gallery_feats)
    G = gallery_feats[0].shape[0]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    results = {m: {} for m in methods}
    for rate in rates:
        present = drop_mask(G, L, rate, generator=gen).to(gallery_feats[0].device)
        for m in methods:
            d = score(feat_t, gallery_feats, present, m)
            sim = 1.0 - d if m == 'centroid' else -d
            # known-pairs-only is THE calibration metric (A10): sparsified S* zeros mean
            # "unknown", and fitting against them is a storage artifact, not miscalibration.
            # The all-pairs fit is kept as a secondary diagnostic under 'overall_allpairs'.
            entry = calibration_regression(sim, s_star, present=present, known_only=True)
            entry['overall_allpairs'] = calibration_regression(sim, s_star)['overall']
            entry['graded_ndcg'] = graded_ndcg(sim, s_star, k=ndcg_k)
            results[m][f'{int(rate * 100)}%'] = entry
    return results
