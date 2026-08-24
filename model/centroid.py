import torch
import torch.nn.functional as F

# SCA Level-1 aggregation: the gallery representation of a clip is the masked spherical mean of
# its PRESENT modality embeddings {V, A, S, (D)} on the unit sphere. Text is never averaged in
# (leak-free, same principle as the hypergraph: text = query side only). Arity-invariant by
# construction: the same code serves k=2..5 with zero changes (E10).


def masked_spherical_mean(z, present=None, eps=1e-6, gates=None,
                          weighting='uniform', tau_w=0.1, query=None):
    """Masked spherical mean mu(Z, present) and concentration A(M).

    z       : (B, L, d) stacked modality embeddings, each L2-normalised (a missing modality may
              be the zero vector -- it is excluded via `present`, so its value never matters).
    present : (B, L) 0/1 (or bool); present[j, m] = 1 iff clip j really has modality m.
              None => all ones == plain spherical mean (no-mask regression path).
    eps     : guard for the renormalisation; near-antipodal sets (||sum z|| ~ 0) stay finite.
    gates   : optional (L,) learnable logits (ablation A7 "learned gates" arm): per-modality
              weights w = softmax(gates) restricted to the PRESENT set and renormalised, so a
              missing modality never receives weight. gates=None (or all-equal, e.g. the
              zero-init) reproduces the uniform centroid exactly -- the arm only refines.
              A(M) stays the UNWEIGHTED resultant (it is a property of the set, not the mix).

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

    if weighting == 'query':
        if query is None:
            raise ValueError("weighting='query' needs the text embedding -- pass query=(B, d). "
                             "Falling back to uniform here would silently report the old "
                             "centroid under the new name.")
        w = query_weights(z, query, present, tau=tau_w, eps=eps)
        s = (z * w.unsqueeze(-1)).sum(dim=1)
    elif weighting == 'reliability':
        # per-sample, content-dependent weights (see reliability_weights)
        w = reliability_weights(z, present, tau=tau_w, eps=eps)
        s = (z * w.unsqueeze(-1)).sum(dim=1)
    elif gates is not None:
        w = torch.softmax(gates.to(z.dtype)[:L], dim=0).unsqueeze(0) * present   # (B, L)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=eps)
        s = (z * w.unsqueeze(-1)).sum(dim=1)                      # gated resultant (renormed)
    else:
        s = (z * present.unsqueeze(-1)).sum(dim=1)                # (B, d) masked resultant
    n = present.sum(dim=1)                                        # (B,)  |M|
    n_safe = n.clamp(min=1.0)

    # A(M) is always the UNWEIGHTED mean resultant length (set property, gate-independent)
    s_plain = ((z * present.unsqueeze(-1)).sum(dim=1)
               if (gates is not None or weighting in ('reliability', 'query')) else s)
    A = s_plain.norm(dim=-1) / n_safe
    # |M|=1: A is identically 1 -- skip its gradient (guard from the k=2 analysis)
    A = torch.where(n <= 1.0, A.detach(), A)

    mu = s / s.norm(dim=-1).clamp(min=eps).unsqueeze(-1)          # eps: centroid-norm blowup guard
    return mu, A, n


def _masked_softmax(logits, mask, dim=-1, eps=1e-6):
    """softmax over the entries mask marks present, without an -inf sentinel.

    The original filled absent entries with torch.finfo(dtype).min. That is ~-3.4e38, a
    FINITE fp32 value that overflows when autocast casts it to fp16 -- "RuntimeError: value
    cannot be converted to type at::Half without overflow", which is what killed T6 on its
    first training step. Negative infinity has no such problem: it is representable in every
    float dtype, so it converts rather than overflowing.

    The max must be taken over the PRESENT entries only. Shifting by a max that an absent
    entry happened to set drives every present logit far negative, exp underflows them all to
    zero, and the row normalises to zeros -- silently returning no weights at all rather than
    a distribution over what is there.
    """
    mask = mask.to(logits.dtype)
    masked = torch.where(mask > 0, logits,
                         torch.tensor(float('-inf'), dtype=logits.dtype, device=logits.device))
    m = masked.max(dim=dim, keepdim=True).values
    # an all-absent row has max -inf; shifting by it would give -inf - -inf = nan
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.exp(masked - m)                       # exp(-inf) == 0, so absent entries vanish
    return w / w.sum(dim=dim, keepdim=True).clamp(min=eps)


def query_weights(z, query, present, tau=0.1, eps=1e-6):
    """Per-modality weights from the QUERY: w_m ∝ exp(<t, z_m> / tau) over the present set.

    Why this and not reliability weighting. Every arm shows NEGATIVE aggregation gain -- the
    fused score is worse than the best single modality it fused. Measured against its own best
    modality, SCA loses 9.5 R@1 on VATEX where the released GRAM checkpoint loses 1.9, and
    4-5 points on DiDeMo and ActivityNet. The cause is structural: a uniform mean gives a
    subtitle stream retrieving at 15.1 the same share as video retrieving at 81.4. A Gramian
    determinant discounts a near-uninformative axis by itself; a uniform mean cannot. That
    implicit discounting is the only thing GRAM's extra machinery buys.

    The text is the natural source of that weighting -- which modality matters is a property
    of the QUERY, not of the clip. "a dog barking" should lean on audio; "a red car turns
    left" on video. Reliability weighting tried to answer the same question from the clip
    alone and was degenerate at k=2 by symmetry (cos(z_0,z_1) == cos(z_1,z_0) forces 0.5/0.5).
    The query breaks that symmetry, so this is defined at EVERY arity, k=2 included -- and
    k=2 is where DiDeMo, ActivityNet and AudioCaps live.

    Still a centroid: a convex combination of unit vectors, renormalised. No determinant, so
    Proposition 1 is untouched; arity-invariant; O(k). And no new parameters -- tau alone
    interpolates between the two regimes, tau -> inf giving back the uniform centroid exactly
    and tau -> 0 giving max_m cos(t, z_m). Being parameter-free is what makes it testable on
    an already-trained checkpoint, before any GPU time is spent.

    The cost is that the gallery representation becomes query-dependent, so a clip no longer
    has one precomputable embedding. Scoring stays cheap -- see query_centroid_scores, which
    evaluates it in closed form without ever forming a per-pair centroid.

    z       : (B, L, d) L2-normalised modality embeddings (absent ones may be zero).
    query   : (B, d) L2-normalised text embedding, aligned row-wise with z.
    present : (B, L) 0/1 mask.
    Returns : (B, L) weights, zero on absent modalities, rows summing to 1.
    """
    present = present.to(z.dtype)
    agree = torch.einsum('bd,bld->bl', query.to(z.dtype), z)        # (B, L) cos(t, z_m)
    w = _masked_softmax(agree / max(tau, eps), present, dim=1, eps=eps)
    # an all-absent row would be 0/0; leave it at zero rather than NaN, matching the uniform
    # path where such a row yields mu = 0
    return torch.where(present.sum(dim=1, keepdim=True) > 0, w, torch.zeros_like(w))


def query_centroid_scores(feat_t, z, present, tau=0.1, eps=1e-6, chunk=256):
    """Full (Nt, Ng) score matrix for the query-weighted centroid, in closed form.

    The naive route materialises a centroid per (text, clip) pair -- (Nt, Ng, d), which is
    ~100 GB on ActivityNet. It is unnecessary. With w = softmax_m(<t, z_m>/tau) and
    S = sum_m w_m z_m, the score is <t, S/||S||>, and both halves are computable from
    quantities that never carry the embedding dimension:

        <t, S>   = sum_m w_m <t, z_m>                        -- from the (Nt, Ng, L) sims
        ||S||^2  = sum_{m,n} w_m w_n <z_m, z_n>              -- from the per-clip Gram (Ng, L, L)

    so the peak allocation is (chunk, Ng, L), and the result is exact rather than an
    approximation of the pairwise form.

    feat_t  : (Nt, d) L2-normalised text embeddings.
    z       : (Ng, L, d) L2-normalised gallery modality embeddings.
    present : (Ng, L) 0/1 mask.
    Returns : (Nt, Ng) similarity, higher = better (a cosine, so in [-1, 1]).
    """
    z = z.float()
    present = present.to(z.dtype)
    gram = torch.einsum('gld,gmd->glm', z, z)                       # (Ng, L, L) per-clip Gram
    mask = present.unsqueeze(1) * present.unsqueeze(2)              # zero out absent pairs
    gram = gram * mask
    out = []
    for i in range(0, feat_t.shape[0], chunk):
        t = feat_t[i:i + chunk].float()                             # (c, d)
        sims = torch.einsum('cd,gld->cgl', t, z)                    # (c, Ng, L) = <t, z_m>
        w = _masked_softmax(sims / max(tau, eps), present.unsqueeze(0), dim=-1, eps=eps)
        num = (w * sims).sum(dim=-1)                                # (c, Ng) = <t, S>
        den = torch.einsum('cgl,glm,cgm->cg', w, gram, w).clamp(min=eps).sqrt()
        out.append(num / den)
    return torch.cat(out, dim=0)


def reliability_weights(z, present, tau=0.1, eps=1e-6):
    """Per-sample modality weights from leave-one-out consensus.

    The uniform centroid gives every present modality an equal share, which is the wrong
    prior when a modality carries no signal for a given corpus. Measured on the zero-shot
    grids: SCA's margin over the released GRAM checkpoint tracks how informative the
    non-video modalities are almost monotonically --

        AudioCaps  T->A 25.1 R@1 (audio share .65)   SCA -3.0 better
        VATEX      T->A 15.1     (.16)               SCA +0.3
        DiDeMo     T->A  4.1     (.10)               SCA -0.7
        ActivityNet T->A 3.0     (.07)               SCA -3.7

    On ActivityNet the audio stream retrieves at 3.0 R@1 -- it is close to noise -- and yet
    it contributes a third of the centroid. A Gramian volume is far less sensitive to a
    noise axis than a mean is, which is the shape of the deficit.

    The fix keeps the centroid but stops assuming equal reliability. Each modality is scored
    by how well it agrees with the consensus of the OTHERS,

        a_m = cos(z_m, mu_{-m}),    mu_{-m} = normalise(sum_{k in M, k != m} z_k)

    and weights are softmax(a / tau) over the present set. A modality that disagrees with
    everything else is down-weighted for that clip; one that corroborates is up-weighted.

    Properties that matter here: no new parameters (one temperature), nothing learned, so it
    applies to an ALREADY-TRAINED checkpoint and can be measured on cached feature dumps
    without retraining. It is arity-invariant like the plain mean, and it involves no
    determinant, so Proposition 1 is untouched. With |M| <= 1 there is no consensus to
    compare against and it falls back to the uniform weights exactly.

    LIMITATION -- degenerate at L=2, which is where we need it. With two present
    modalities mu_{-0} = z_1 and mu_{-1} = z_0, so a_0 = cos(z_0, z_1) = cos(z_1, z_0) = a_1
    identically and the weights are forced to 0.5/0.5 whatever tau is. DiDeMo, ActivityNet
    and AudioCaps are all T-VA, i.e. a two-modality gallery, so this mechanism is a no-op on
    both benchmarks where SCA trails GRAM and acts only on MSR-VTT and VATEX where it
    already leads. It is also undefined at k=1. A weighting that only engages at k>=3 is not
    arity-invariant and contradicts the property that motivates the centroid, so this is
    kept as a measured negative result, NOT as a proposed fix. A general scheme would have
    to score a modality from itself alone -- pre-normalisation norm, or typicality against
    that modality's own distribution -- so that it is defined at every arity.

    z       : (B, L, d) L2-normalised modality embeddings (absent ones may be zero).
    present : (B, L) 0/1 mask.
    tau     : softmax temperature. tau -> inf recovers the uniform centroid.
    Returns : (B, L) weights, zero on absent modalities, rows summing to 1.
    """
    present = present.to(z.dtype)
    s = (z * present.unsqueeze(-1)).sum(dim=1, keepdim=True)        # (B, 1, d) resultant
    loo = s - z * present.unsqueeze(-1)                             # (B, L, d) leave-one-out
    loo = loo / loo.norm(dim=-1, keepdim=True).clamp(min=eps)
    agree = (z * loo).sum(dim=-1)                                   # (B, L) cos(z_m, mu_-m)

    # absent modalities must never win weight; masked softmax, no -inf sentinel (fp16-safe)
    w = _masked_softmax(agree / max(tau, eps), present, dim=1, eps=eps)

    # |M| <= 1: mu_{-m} is the zero vector and the agreement is meaningless -- use uniform.
    n = present.sum(dim=1, keepdim=True)
    uniform = present / n.clamp(min=1.0)
    return torch.where(n > 1.0, w, uniform)


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
