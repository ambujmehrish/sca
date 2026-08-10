import torch
import torch.nn.functional as F

# SCA Level-1 aggregation: the gallery representation of a clip is the masked spherical mean of
# its PRESENT modality embeddings {V, A, S, (D)} on the unit sphere. Text is never averaged in
# (leak-free, same principle as the hypergraph: text = query side only). Arity-invariant by
# construction: the same code serves k=2..5 with zero changes (E10).


def masked_spherical_mean(z, present=None, eps=1e-6):
    """Masked spherical mean mu(Z, present) and concentration A(M).

    z       : (B, L, d) stacked modality embeddings, each L2-normalised (a missing modality may
              be the zero vector -- it is excluded via `present`, so its value never matters).
    present : (B, L) 0/1 (or bool); present[j, m] = 1 iff clip j really has modality m.
              None => all ones == plain spherical mean (no-mask regression path).
    eps     : guard for the renormalisation; near-antipodal sets (||sum z|| ~ 0) stay finite.

    Returns:
    - mu : (B, d) L2-normalised centroid over the present modalities only.
    - A  : (B,) mean resultant length A(M) = ||(1/|M|) sum_{m in M} z_m|| in [0, 1].
           |M| = 1 is degenerate: A == 1 identically, so its (meaningless) gradient is cut and
           mu is exactly the single present embedding.
    - n  : (B,) number of present modalities |M| (clamped to >= 1 for the divisions; an
           all-absent row yields mu = 0, A = 0 rather than NaN).
    """
    B, L, _ = z.shape
    if present is None:
        present = z.new_ones(B, L)
    present = present.to(z.dtype)

    s = (z * present.unsqueeze(-1)).sum(dim=1)                    # (B, d) masked resultant
    n = present.sum(dim=1)                                        # (B,)  |M|
    n_safe = n.clamp(min=1.0)

    norm_s = s.norm(dim=-1)
    A = norm_s / n_safe
    # |M|=1: A is identically 1 -- skip its gradient (guard from the k=2 analysis)
    A = torch.where(n <= 1.0, A.detach(), A)

    mu = s / norm_s.clamp(min=eps).unsqueeze(-1)                  # eps: centroid-norm blowup guard
    return mu, A, n


def concept_resultant(mu, labels, num_classes=None, eps=1e-6):
    """Per-concept mean resultant length A(c) over the member centroids in the batch.

    mu     : (B, d) L2-normalised clip centroids.
    labels : (B,) long concept ids.
    Returns (A_c, count_c): (C,) resultant length per concept (0 where no members) and member
    counts. Used by L_concept's eps-floor and by the wandb A(M)/A(c) diagnostics.
    """
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1 if labels.numel() else 0
    sums = mu.new_zeros(num_classes, mu.shape[-1])
    sums.index_add_(0, labels, mu)
    counts = mu.new_zeros(num_classes)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=mu.dtype))
    A_c = sums.norm(dim=-1) / counts.clamp(min=1.0)
    return A_c, counts


def centroid_scores(feat_t, mu):
    """Cosine score matrix S = feat_t @ mu.T, (B1, B2). Both inputs L2-normalised, so S is in
    [-1, 1] regardless of how many modalities produced each mu -- the cardinality-invariance
    that GRAM's volume does not have (E5)."""
    return feat_t @ mu.T
