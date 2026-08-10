import torch
import torch.nn.functional as F

# PMRL baseline head (plan item 1.3.10, reimplemented -- the released code is not importable
# here). PMRL scores a (text, gallery) pair by the LARGEST eigenvalue lambda_1 of the Gram
# matrix of the stacked vectors [t, z_1..z_L]: perfectly aligned unit vectors give
# lambda_1 = M (all energy on one principal direction), orthogonal ones give 1. Trained with a
# softmax over the gallery on lambda_1 logits, plus an eigenvalue-concentration regulariser
# standing in for the paper's eigenvector-orthogonality term (energy outside the top
# eigenvalue is penalised on positive pairs).
#
# Masked variants for E4's honest baselines (A2 uses lambda_1/|M| as the alignment measure):
#   'raw' : lambda_1 of the zero-masked Gram (zero rows/cols only add zero eigenvalues, so
#           this IS the reduced-arity lambda_1 -- but its scale grows with |M|).
#   'norm': lambda_1 / |M| -- cardinality-normalised, comparable across missing patterns.


def _pairwise_gram(language, inputs, present=None):
    """(B1, B2, M, M) Gram of [t_i, z_j1..z_jL] for every (i, j) pair, missing modalities
    zero-masked. Mirrors utils.volume.volume_computation_masked's construction (no phantom
    axis: lambda_1 needs plain zero rows/cols, det needs the identity trick)."""
    B1, B2, L = language.shape[0], inputs[0].shape[0], len(inputs)
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, B2)
    l_inputs = [language @ x.T for x in inputs]
    rows = [torch.stack([ll] + l_inputs, dim=-1)]
    for i in range(L):
        row = [l_inputs[i]]
        for j in range(L):
            row.append(torch.einsum('bi,bi->b', inputs[i], inputs[j]).unsqueeze(0).expand(B1, -1))
        rows.append(torch.stack(row, dim=-1))
    G = torch.stack(rows, dim=-2).float()                        # (B1, B2, M, M)
    if present is not None:
        ones = torch.ones(B2, 1, device=G.device, dtype=G.dtype)
        pres_full = torch.cat([ones, present.to(G.dtype)], dim=1)          # text always present
        keep = (pres_full.unsqueeze(2) * pres_full.unsqueeze(1)).unsqueeze(0)
        G = G * keep
    return G


def pmrl_lambda1(language, inputs, present=None, variant='raw'):
    """(B1, B2) top-eigenvalue score for every (text, gallery) pair.
    variant 'raw' or 'norm' (divide by the pair's effective arity |M|+1)."""
    G = _pairwise_gram(language, inputs, present=present)
    lam = torch.linalg.eigvalsh(G)                               # ascending
    lam1 = lam[..., -1]
    if variant == 'norm':
        if present is None:
            m = torch.full((G.shape[1],), float(len(inputs) + 1), device=G.device)
        else:
            m = present.float().sum(dim=1) + 1.0
        lam1 = lam1 / m.unsqueeze(0)
    return lam1


def pmrl_loss(language, inputs, targets, temp=0.07, present=None, variant='raw',
              ortho_w=0.1, label_smoothing=0.1):
    """Softmax retrieval loss on lambda_1 logits (one direction; the caller mirrors it with
    gathered features for the transpose) + eigenvalue-concentration penalty on positives.

    ortho term: for each positive pair, the fraction of Gram energy NOT captured by lambda_1
    (sum_{k>=2} lambda_k / trace). Zero iff the stacked vectors are rank-1 (all modalities and
    the text collinear), the same optimum the eigvec-orthogonality of the original enforces.
    """
    G = _pairwise_gram(language, inputs, present=present)
    lam = torch.linalg.eigvalsh(G)
    lam1 = lam[..., -1]
    logits = lam1
    if variant == 'norm':
        if present is None:
            m = torch.full((G.shape[1],), float(len(inputs) + 1), device=G.device)
        else:
            m = present.float().sum(dim=1) + 1.0
        logits = logits / m.unsqueeze(0)
    loss = F.cross_entropy(logits / temp, targets, label_smoothing=label_smoothing)

    if ortho_w > 0:
        pos = lam[torch.arange(targets.shape[0], device=lam.device), targets]   # (B, M)
        trace = pos.sum(-1).clamp(min=1e-6)
        residual = (trace - pos[..., -1]) / trace
        loss = loss + ortho_w * residual.mean()
    return loss
