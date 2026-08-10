import torch

# E4 / E5 harness (plan item 1.3.7): controlled missing-modality grids over CACHED features.
# Works on the feature dict any eval pass produces (feat_t + per-modality gallery feats), so
# one encoder pass serves the whole {0, 25, 50, 75}% x which-modality x scorer grid for every
# method (SCA centroid, GRAM masked-(i)/(ii), PMRL raw / /|M|) -- the honest-baseline design.

from utils.volume import volume_computation_masked, volume_computation_mean_imputed
from model.centroid import masked_spherical_mean
from model.pmrl_loss import pmrl_lambda1

MISS_RATES = (0.0, 0.25, 0.5, 0.75)


def drop_mask(B, L, rate, which=None, generator=None, device=None):
    """Presence mask (B, L) with `rate` of the gallery clips losing a modality.
    which=None: the dropped modality is drawn uniformly per clip; which=m: always modality m
    (the which-modality axis of the grid). rate=0 -> all-ones."""
    present = torch.ones(B, L, device=device)
    if rate <= 0 or L < 2:
        return present
    n_hit = int(round(B * rate))
    if n_hit == 0:
        return present
    hit = torch.randperm(B, generator=generator, device=device)[:n_hit]
    if which is None:
        m = torch.randint(L, (n_hit,), generator=generator, device=device)
    else:
        m = torch.full((n_hit,), int(which), device=device)
    present[hit, m] = 0.0
    return present


def score(feat_t, gallery_feats, present, method='centroid'):
    """Distance matrix (T, G), smaller = better, under a scorer:
    'centroid'      : 1 - cos(t, masked spherical mean)          (SCA)
    'volume_masked' : reduced-arity Gramian volume, variant (i)  (GRAM-masked, exists)
    'volume_imputed': mean-imputed full-arity volume, variant (ii)
    'pmrl_raw'      : -lambda_1 of the zero-masked Gram          (PMRL)
    'pmrl_norm'     : -lambda_1 / |M|
    """
    feats = [f.float() for f in gallery_feats]
    t = feat_t.float()
    if method == 'centroid':
        mu, _, _ = masked_spherical_mean(torch.stack(feats, dim=1), present)
        return 1.0 - t @ mu.T
    if method == 'volume_masked':
        return volume_computation_masked(t, feats, present=present)
    if method == 'volume_imputed':
        return volume_computation_mean_imputed(t, feats, present=present)
    if method == 'pmrl_raw':
        return -pmrl_lambda1(t, feats, present=present, variant='raw')
    if method == 'pmrl_norm':
        return -pmrl_lambda1(t, feats, present=present, variant='norm')
    raise ValueError(method)


def recall_at_k(dist, gt_cols, ks=(1, 5, 10)):
    """dist (T, G) distances; gt_cols (T,) index of each text's true gallery item."""
    order = dist.argsort(dim=1)
    rank = (order == gt_cols.unsqueeze(1)).float().argmax(dim=1)
    out = {f'R@{k}': (rank < k).float().mean().item() * 100 for k in ks}
    out['MedR'] = (rank.float().median() + 1).item()
    return out, rank


def per_cardinality_stats(dist, present, gt_cols):
    """Positive-pair score statistics split by gallery cardinality |M| -- the E5 evidence.
    Returns {|M|: {'mean','std','n'}} of the positive scores."""
    card = present.sum(dim=1)
    pos = dist[torch.arange(dist.shape[0]), gt_cols]
    stats = {}
    for m in card.unique().tolist():
        sel = card[gt_cols] == m
        if sel.any():
            s = pos[sel]
            stats[int(m)] = {'mean': s.mean().item(),
                             'std': s.std().item() if s.numel() > 1 else 0.0,
                             'n': int(sel.sum())}
    return stats


def rank_displacement_bias(dist_full, dist_masked, present_masked, gt_cols):
    """How much a clip's rank moves purely because it lost a modality: mean signed rank change
    of the positives, split into hit (lost a modality) vs intact clips. A calibrated scorer
    shows ~0 gap between the two groups."""
    def _ranks(d):
        order = d.argsort(dim=1)
        return (order == gt_cols.unsqueeze(1)).float().argmax(dim=1).float()
    r0, r1 = _ranks(dist_full), _ranks(dist_masked)
    disp = r1 - r0
    hit = present_masked.sum(dim=1) < present_masked.shape[1]
    hit_t = hit[gt_cols]
    return {'disp_hit': disp[hit_t].mean().item() if hit_t.any() else 0.0,
            'disp_intact': disp[~hit_t].mean().item() if (~hit_t).any() else 0.0}


def fit_affine_calibration(dist, present):
    """Per-cardinality affine fit: for each |M|, (a_m, b_m) mapping that cardinality's score
    distribution onto the full-cardinality one (match mean/std). Returns {m: (a, b)}."""
    card = present.sum(dim=1)
    full = int(card.max().item())
    ref = dist[:, card == full]
    mu_r, sd_r = ref.mean().item(), ref.std().item()
    cal = {}
    for m in card.unique().tolist():
        col = dist[:, card == m]
        mu_m, sd_m = col.mean().item(), col.std().item()
        a = sd_r / max(sd_m, 1e-8)
        cal[int(m)] = (a, mu_r - a * mu_m)
    return cal


def apply_affine_calibration(dist, present, cal):
    card = present.sum(dim=1)
    out = dist.clone()
    for m, (a, b) in cal.items():
        sel = card == m
        out[:, sel] = a * out[:, sel] + b
    return out


@torch.no_grad()
def missing_grid(feat_t, gallery_feats, gt_cols, methods=('centroid', 'volume_masked'),
                 rates=MISS_RATES, seed=0, calibrate=False):
    """The full E4 grid: {rate} x {which-modality + random} x {method}.
    gallery_feats: list of L (G, d) tensors; gt_cols: (T,) ground-truth column per text.
    Returns nested dict results[method][f'{rate}|{which}'] = {'R@1', ..., 'card_stats',
    'rank_disp', and (if calibrate) calibrated R@k}."""
    L = len(gallery_feats)
    G = gallery_feats[0].shape[0]
    device = gallery_feats[0].device
    gen = torch.Generator(device='cpu').manual_seed(seed)
    results = {m: {} for m in methods}
    full_present = torch.ones(G, L, device=device)
    dist_full = {m: score(feat_t, gallery_feats, full_present, m) for m in methods}

    for rate in rates:
        which_axis = [None] + (list(range(L)) if rate > 0 else [])
        for which in which_axis:
            present = drop_mask(G, L, rate, which=which, generator=gen).to(device)
            key = f"{int(rate * 100)}%|{'rand' if which is None else f'mod{which}'}"
            for m in methods:
                d = score(feat_t, gallery_feats, present, m)
                log, _ = recall_at_k(d, gt_cols)
                log['card_stats'] = per_cardinality_stats(d, present, gt_cols)
                if rate > 0:
                    log['rank_disp'] = rank_displacement_bias(dist_full[m], d, present, gt_cols)
                if calibrate and rate > 0:
                    cal = fit_affine_calibration(d, present)
                    log_cal, _ = recall_at_k(apply_affine_calibration(d, present, cal), gt_cols)
                    log['calibrated'] = log_cal
                results[m][key] = log
    return results
