import torch
import torch.nn as nn
import torch.distributed as dist

# SCA Level-2 grouping: a persistent EMA memory of concept prototypes {nu_c} on the unit sphere.
# Paper narrative vs the GatedHGNN arm (A7): persistent EMA prototypes replace train-time-only
# transductive kNN hyperedges. All state lives in buffers (saved/restored with the checkpoint),
# updates are strictly no-grad.


class PrototypeMemory(nn.Module):
    """EMA memory of L2-normalised concept prototypes.

    - init: a prototype is created from the running class mean accumulated during the first
      epoch / warmup (`initialized` flips per class on its first flush).
    - update: no-grad EMA nu_c <- eta * nu_c + (1 - eta) * batch_mean_c, then renormalise.
      Batch class sums/counts are all-reduced across DDP ranks first, so every rank holds the
      same memory.
    - staleness guard (from the k=2 analysis): prototypes drift while the encoder warms up, so
      L_concept is delayed to warmup end and `reset_from_running()` re-initialises every
      prototype from the accumulator collected since the last reset.
    """

    def __init__(self, num_concepts, dim, eta=0.99):
        super().__init__()
        self.num_concepts = num_concepts
        self.eta = float(eta)
        self.register_buffer('protos', torch.zeros(num_concepts, dim))
        self.register_buffer('initialized', torch.zeros(num_concepts, dtype=torch.bool))
        # running accumulator (since last reset) -- feeds first-init and staleness resets
        self.register_buffer('run_sum', torch.zeros(num_concepts, dim))
        self.register_buffer('run_count', torch.zeros(num_concepts))

    @torch.no_grad()
    def _allreduce(self, sums, counts):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        return sums, counts

    @torch.no_grad()
    def update(self, mu, labels):
        """mu: (B, d) L2-normalised centroids; labels: (B,) long concept ids in [0, C)."""
        mu = mu.detach().float()
        sums = torch.zeros_like(self.run_sum)
        sums.index_add_(0, labels, mu)
        counts = torch.zeros_like(self.run_count)
        counts.index_add_(0, labels, torch.ones(labels.shape[0], device=mu.device,
                                                dtype=self.run_count.dtype))
        sums, counts = self._allreduce(sums, counts)

        self.run_sum += sums
        self.run_count += counts

        seen = counts > 0
        batch_mean = sums[seen] / counts[seen].unsqueeze(-1)
        batch_mean = torch.nn.functional.normalize(batch_mean, dim=-1)

        init_mask = seen & ~self.initialized
        if init_mask.any():
            # first sighting of a class: prototype = its (running) class mean, no EMA yet
            m = self.run_sum[init_mask] / self.run_count[init_mask].clamp(min=1.0).unsqueeze(-1)
            self.protos[init_mask] = torch.nn.functional.normalize(m, dim=-1)
            self.initialized |= init_mask

        ema_mask = seen & self.initialized & ~init_mask
        if ema_mask.any():
            sub = torch.zeros_like(self.protos[ema_mask])
            idx_map = ema_mask.nonzero(as_tuple=True)[0]
            means = sums[idx_map] / counts[idx_map].unsqueeze(-1)
            means = torch.nn.functional.normalize(means, dim=-1)
            sub = self.eta * self.protos[ema_mask] + (1.0 - self.eta) * means
            self.protos[ema_mask] = torch.nn.functional.normalize(sub, dim=-1)

    @torch.no_grad()
    def reset_from_running(self):
        """Staleness reset at warmup end: re-init every seen prototype from the accumulator
        gathered since the last reset, then clear the accumulator."""
        seen = self.run_count > 0
        if seen.any():
            m = self.run_sum[seen] / self.run_count[seen].unsqueeze(-1)
            self.protos[seen] = torch.nn.functional.normalize(m, dim=-1)
            self.initialized |= seen
        self.run_sum.zero_()
        self.run_count.zero_()

    def get(self, labels):
        """(B, d) prototype for each label (no grad through the memory)."""
        return self.protos[labels]


@torch.no_grad()
def batch_prototypes(mu, labels, num_concepts):
    """A4 'batch-only nu_c' arm: prototypes are THIS batch's normalised class means -- no
    persistent memory, no EMA. Returns (protos (C, d), has (C,) bool: classes present in
    the batch; members of absent classes must be excluded from the loss by the caller)."""
    protos = mu.new_zeros(num_concepts, mu.shape[-1])
    protos.index_add_(0, labels, mu.detach().float().to(mu.dtype))
    counts = mu.new_zeros(num_concepts)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=mu.dtype))
    has = counts > 0
    protos[has] = torch.nn.functional.normalize(
        protos[has] / counts[has].unsqueeze(-1), dim=-1)
    return protos, has
