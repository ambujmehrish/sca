import math
import torch
import torch.nn as nn

# LoRA for the three frozen backbones (plan item 1.3.1) -- genuinely new here: the trunk has no
# adapter code anywhere (GRAM full-finetunes). Injected into the attention W_q and W_v only,
# per-modality rank r in {4, 8, 16} (A6 sweeps ranks incl. asymmetric). merge()/unmerge() folds
# the update into the frozen weight for zero-overhead inference and exact round-trip.
#
# Attention naming across the backbones:
#   text  (BERT)          : .query / .value          (separate nn.Linear)
#   audio (BEATs)         : .q_proj / .v_proj        (separate nn.Linear)
#   vision (EVA-CLIP)     : .q_proj / .v_proj  OR a fused .qkv nn.Linear (subln off)
# Separate layers get LoRALinear; a fused qkv gets LoRAQKVLinear updating only the q and v
# slices of the output.

_Q_NAMES = ('q_proj', 'query')
_V_NAMES = ('v_proj', 'value')
_QKV_NAMES = ('qkv',)


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a rank-r update: y = W x + (alpha/r) * B A x."""

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))     # B zero-init => identity at step 0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def _delta(self):
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            h = self.lora_dropout(x) @ self.lora_A.T.to(x.dtype)
            out = out + (h @ self.lora_B.T.to(x.dtype)) * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self._delta().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self._delta().to(self.base.weight.dtype)
            self.merged = False


class LoRAQKVLinear(nn.Module):
    """LoRA on the q and v slices of a fused qkv nn.Linear (out_features = 3 * d).
    k stays frozen with no update, matching the W_q/W_v-only recipe."""

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base, nn.Linear) and base.out_features % 3 == 0
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.d = base.out_features // 3
        self.r = r
        self.scaling = alpha / r
        self.lora_A_q = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(self.d, r))
        self.lora_A_v = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B_v = nn.Parameter(torch.zeros(self.d, r))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def _delta(self):
        delta = torch.zeros_like(self.base.weight, dtype=self.lora_A_q.dtype)
        delta[:self.d] = (self.lora_B_q @ self.lora_A_q) * self.scaling
        delta[2 * self.d:] = (self.lora_B_v @ self.lora_A_v) * self.scaling
        return delta

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            xd = self.lora_dropout(x)
            dq = (xd @ self.lora_A_q.T.to(x.dtype)) @ self.lora_B_q.T.to(x.dtype) * self.scaling
            dv = (xd @ self.lora_A_v.T.to(x.dtype)) @ self.lora_B_v.T.to(x.dtype) * self.scaling
            out = torch.cat((out[..., :self.d] + dq,
                             out[..., self.d:2 * self.d],
                             out[..., 2 * self.d:] + dv), dim=-1)
        return out

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self._delta().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self._delta().to(self.base.weight.dtype)
            self.merged = False


def inject_lora(module, r=8, alpha=16, dropout=0.0, prefix=''):
    """Replace every attention W_q / W_v (or fused qkv) nn.Linear under `module` with its LoRA
    wrapper. Returns the list of wrapped module paths. Idempotent: already-wrapped layers are
    skipped."""
    wrapped = []
    for name, child in list(module.named_children()):
        path = f'{prefix}.{name}' if prefix else name
        if isinstance(child, (LoRALinear, LoRAQKVLinear)):
            continue
        if isinstance(child, nn.Linear) and name in _QKV_NAMES:
            setattr(module, name, LoRAQKVLinear(child, r=r, alpha=alpha, dropout=dropout))
            wrapped.append(path)
        elif isinstance(child, nn.Linear) and name in _Q_NAMES + _V_NAMES:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            wrapped.append(path)
        else:
            wrapped += inject_lora(child, r=r, alpha=alpha, dropout=dropout, prefix=path)
    return wrapped


def merge_all(module):
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAQKVLinear)):
            m.merge()


def unmerge_all(module):
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAQKVLinear)):
            m.unmerge()


def remap_lora_checkpoint(checkpoint, wrapped_paths):
    """A LoRA-wrapped linear renames <path>.weight -> <path>.base.weight. Remap a pretrained
    state dict (VAST ckpt, GRAM ckpts) so the frozen base weights still load after injection.
    Non-wrapped keys pass through untouched; LoRA A/B keys are absent from old checkpoints by
    construction (load_state_dict(strict=False) reports them as missing, which is correct)."""
    wrapped = set(wrapped_paths)
    remapped = {}
    for k, v in checkpoint.items():
        stem, _, leaf = k.rpartition('.')
        if stem in wrapped and leaf in ('weight', 'bias'):
            remapped[f'{stem}.base.{leaf}'] = v
        else:
            remapped[k] = v
    return remapped


def lora_parameters(module):
    """Iterator over (name, param) of all LoRA A/B matrices -- the optimizer's LoRA group."""
    for name, p in module.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            yield name, p
