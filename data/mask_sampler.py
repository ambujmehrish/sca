import torch

# Training-time modality masking for SCA (plan item 1.3.6). The sampler decides, per clip, which
# modalities to VIRTUALLY drop this step; sca.py computes both mu_M (masked centroid) and mu_K
# (full centroid) from the one forward pass, so masking costs no extra encoder time. apply_mask
# zero-fills features so that anything downstream that infers presence from norms
# (present_from_feats, doc_incidence) sees exactly the same signal a loader-side missing
# modality produces.


class MaskSampler:
    """p_full schedule (1.0 -> 0.5 by default) + m-dagger draw.

    num_modalities : L, order matches the gallery stack (e.g. V, A, S[, D]).
    p_full_start/end, schedule_steps : linear schedule of the probability that a clip keeps its
        FULL modality set; after schedule_steps the probability stays at p_full_end.
    mode : 'uniform' -- m-dagger uniform over the clip's present modalities;
           'freq'    -- weighted by `freq` (higher weight = dropped more often), e.g. the
                        corpus missing-frequency so training matches the test-time pattern.
    n_drop : how many modalities a masked clip loses (1 default; 2 for the A5 2-drop arm).
        Never drops a clip below one present modality, and never "drops" a modality the clip
        does not have (the upstream `present` is respected and the mask composed into it).
    """

    def __init__(self, num_modalities, p_full_start=1.0, p_full_end=0.5, schedule_steps=1000,
                 mode='uniform', freq=None, n_drop=1):
        assert num_modalities >= 1
        assert mode in ('uniform', 'freq')
        self.L = num_modalities
        self.p_full_start = float(p_full_start)
        self.p_full_end = float(p_full_end)
        self.schedule_steps = max(1, int(schedule_steps))
        self.mode = mode
        self.n_drop = int(n_drop)
        if mode == 'freq':
            assert freq is not None and len(freq) == num_modalities
            self.freq = torch.as_tensor(freq, dtype=torch.float)
            assert (self.freq >= 0).all() and self.freq.sum() > 0, \
                'mask freq weights must be non-negative with at least one positive entry'
        else:
            self.freq = torch.ones(num_modalities)

    def p_full(self, step):
        t = min(max(step, 0), self.schedule_steps) / self.schedule_steps
        return self.p_full_start + t * (self.p_full_end - self.p_full_start)

    def sample(self, batch_size, step, device, present=None, generator=None):
        """Returns mask (B, L) 0/1: 1 = keep. The effective presence for mu_M is
        present * mask; a clip chosen to stay full gets an all-ones mask.
        L follows `present` when given (a batch may carry fewer modalities than the
        sampler's capacity, e.g. V,A only during pretrain); freq is sliced to match."""
        L = present.shape[1] if present is not None else self.L
        if present is None:
            present = torch.ones(batch_size, L, device=device)
        present = present.float()
        mask = torch.ones(batch_size, L, device=device)

        p_full = self.p_full(step)
        masked_rows = torch.rand(batch_size, device=device, generator=generator) >= p_full
        if not masked_rows.any():
            return mask

        freq = self.freq[:L] if self.freq.shape[0] >= L else torch.ones(L)
        w = freq.to(device).unsqueeze(0) * present               # only present mods can drop
        for _ in range(self.n_drop):
            # a drop is only legal while the clip keeps >= 2 present modalities
            droppable = masked_rows & ((present * mask).sum(dim=1) >= 2.0)
            if not droppable.any():
                break
            w_eff = w * mask                                     # don't re-drop the same one
            # freq = 0 means "never drop this modality" -- a clip whose remaining droppable
            # modalities all have zero weight simply keeps them (no silent uniform redraw,
            # which would contradict the configured distribution)
            droppable = droppable & (w_eff.sum(dim=1) > 0)
            if not droppable.any():
                break
            w_rows = w_eff[droppable]
            m_dagger = torch.multinomial(w_rows, 1, generator=generator).squeeze(1)
            rows = droppable.nonzero(as_tuple=True)[0]
            mask[rows, m_dagger] = 0.0
        return mask

    @classmethod
    def from_config(cls, cfg, num_modalities=4, prefix='train_mask_'):
        """Build a sampler from config knobs named <prefix>{p_full_start,p_full_end,
        schedule_steps,mode,freq,n_drop} -- shared by the E4 2x2 train-masking arm of the
        GRAM/PMRL baselines (gram.py) and any future consumer."""
        get = lambda k, d: getattr(cfg, prefix + k, d)
        mode = get('mode', 'uniform')
        return cls(num_modalities,
                   p_full_start=float(get('p_full_start', 1.0)),
                   p_full_end=float(get('p_full_end', 0.5)),
                   schedule_steps=int(get('schedule_steps', 2000)),
                   mode=mode,
                   freq=get('freq', None) if mode == 'freq' else None,
                   n_drop=int(get('n_drop', 1)))

    def sample_and_apply(self, feats, step, generator=None):
        """Draw a mask for a list of (B, d) modality features and zero-fill the drops so
        every downstream present_from_feats consumer sees them. Respects real absence
        (zero-filled inputs stay absent). Returns (masked_feats, mask)."""
        present = torch.stack([(f.float().norm(dim=-1) > 0.5).float() for f in feats], dim=1)
        mask = self.sample(present.shape[0], step, feats[0].device, present=present,
                           generator=generator)
        return self.apply_mask(feats, mask), mask

    @staticmethod
    def apply_mask(feats, mask):
        """Zero-fill dropped modalities so present_from_feats / doc_incidence see the drop.
        feats: list of L tensors (B, d) (or a (B, L, d) stack); mask: (B, L)."""
        if isinstance(feats, torch.Tensor):
            return feats * mask.unsqueeze(-1).to(feats.dtype)
        return [f * mask[:, i:i + 1].to(f.dtype) for i, f in enumerate(feats)]
