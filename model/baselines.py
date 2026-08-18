import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.logger import LOGGER
from utils.distributed import concat_all_gather
from .gram import GRAM
from .pmrl_loss import pmrl_lambda1
from utils.volume import present_from_feats

# P3 baseline heads on the SAME trunk as GRAM/SCA (plan §4): every baseline shares encoders,
# projections, gathering and ITM, so E4/E5/E6 comparisons isolate the geometry, not the
# backbone. GRAM itself is model_type 'gram' (untouched); this file adds:
#   'gram_lora' -- GRAM's volume objective with LoRA backbones (LoRA-parity arm: separates
#                  "SCA's objective" from "SCA's adapter budget").
#   'pmrl'      -- the PMRL head: lambda_1-of-Gram softmax retrieval loss (+ eigenvalue-
#                  concentration term), masked raw / /|M| variants; optional LoRA for the
#                  parity arm. Eval scoring via score_mode 'pmrl_raw' / 'pmrl_norm'.


def _gather(t):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return concat_all_gather(t)
    return t


class _LoRACkptMixin:
    _lora_wrapped = ()

    def modify_checkpoint(self, checkpoint):
        checkpoint = super().modify_checkpoint(checkpoint)
        if self._lora_wrapped:
            from .lora import remap_lora_checkpoint
            checkpoint = remap_lora_checkpoint(checkpoint, self._lora_wrapped)
        return checkpoint


class GRAMLoRA(_LoRACkptMixin, GRAM):
    """GRAM byte-identical objective, LoRA-adapted backbones (parity baseline)."""

    def __init__(self, config):
        super().__init__(config)
        if not bool(getattr(self.config, 'use_lora', False)):
            raise ValueError("model_type 'gram_lora' requires use_lora=true -- for plain "
                             "GRAM use model_type 'gram' (kept byte-for-byte intact).")
        from .lora import setup_lora_backbones
        self._lora_wrapped = setup_lora_backbones(self, self.config, logger=LOGGER)


class _ScoreITM:
    """GRAM's hard-negative ITM, with negatives drawn from an arbitrary similarity matrix
    (shared by the PMRL and GRAMHyp heads; sim/simT are SIMILARITIES, higher = better)."""

    def _itm_from_scores(self, batch, sim, simT, rank, bs, temp=1.0):
        caption_tokens = self.batch_get(batch, 'caption_tokens')
        input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
        input_ids_collate = _gather(input_ids)
        attention_mask_collate = _gather(attention_mask)
        from utils.distributed import all_gather_with_grad
        condition_feats = self.batch_get(batch, 'condition_feats_va')
        condition_feats_collate = (all_gather_with_grad(condition_feats)
                                   if dist.is_initialized() else condition_feats)
        with torch.no_grad():
            weights_t2cond = F.softmax(sim / temp, dim=1) + 1e-4
            weights_t2cond[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
            weights_cond2t = F.softmax(simT / temp, dim=1) + 1e-4
            weights_cond2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
        condition_feats_neg = torch.stack(
            [condition_feats_collate[torch.multinomial(weights_t2cond[b], 1).item()]
             for b in range(bs)], dim=0)
        text_ids_neg = torch.stack(
            [input_ids_collate[torch.multinomial(weights_cond2t[b], 1).item()]
             for b in range(bs)], dim=0)
        text_atts_neg = torch.stack(
            [attention_mask_collate[torch.multinomial(weights_cond2t[b], 1).item()]
             for b in range(bs)], dim=0)
        input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
        attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
        condition_feats = torch.cat((condition_feats, condition_feats_neg, condition_feats), dim=0)
        output = self.multimodal_encoder.bert(input_ids=input_ids_1,
                                              attention_mask=attention_mask_1,
                                              encoder_hidden_states=condition_feats).last_hidden_state
        logits = self.itm_head(output[:, 0].half())
        ground_truth = torch.zeros(bs * 3).long().to(logits.device)
        ground_truth[:bs] = 1
        return self.itm_ratio * F.cross_entropy(logits, ground_truth)


class PMRL(_LoRACkptMixin, _ScoreITM, GRAM):
    """PMRL retrieval head: the pair score is lambda_1 of the Gram matrix of
    [t, z_V, z_A, (z_S, z_D)]; trained with a softmax over the gallery in both directions
    plus an eigenvalue-concentration regulariser on positives (see model/pmrl_loss.py).

    config: pmrl_variant 'raw' | 'norm' (lambda_1 vs lambda_1/|M| -- the A2 masked
    variants), pmrl_ortho_w, pmrl_temp; use_lora for the parity arm. Missing modalities are
    zero-masked per clip via present_from_feats, mirroring the masked GRAM/SCA paths."""

    def __init__(self, config):
        super().__init__(config)
        self.pmrl_variant = getattr(self.config, 'pmrl_variant', 'raw')
        assert self.pmrl_variant in ('raw', 'norm'), self.pmrl_variant
        self.pmrl_ortho_w = float(getattr(self.config, 'pmrl_ortho_w', 0.1))
        self.pmrl_temp = float(getattr(self.config, 'pmrl_temp', 0.07))
        # declare the eval geometry (matching the trained variant) so a bare eval config
        # can never silently score a PMRL checkpoint with GRAM's volume
        if not getattr(self.config, 'score_mode', None):
            self.config.score_mode = f'pmrl_{self.pmrl_variant}'
        self._lora_wrapped = []
        if bool(getattr(self.config, 'use_lora', False)):
            from .lora import setup_lora_backbones
            self._lora_wrapped = setup_lora_backbones(self, self.config, logger=LOGGER)

    def forward_ret(self, batch, task, compute_loss=True):
        if not compute_loss:
            return super().forward_ret(batch, task, compute_loss=False)

        if isinstance(batch.raw_captions[0], list):
            batch.raw_captions = [i for j in batch.raw_captions for i in j]

        loss_dict = {}
        feat_t = self.batch_get(batch, 'feat_t').float()
        gallery = [self.batch_get(batch, 'feat_v').float(),
                   self.batch_get(batch, 'feat_a').float()]
        if 'raw_subtitles' in batch.keys():
            gallery.append(self.batch_get(batch, 'feat_s').float())
        if 'depth_pixels' in batch.keys():
            gallery.append(self.batch_get(batch, 'feat_d').float())
        gallery = self._apply_train_mask(gallery)          # E4 train-masking arm (no-op off)
        present = present_from_feats(gallery)

        feat_t_all = _gather(feat_t)
        gallery_all = [_gather(g) for g in gallery]
        present_all = present_from_feats(gallery_all)

        rank = dist.get_rank() if dist.is_initialized() else 0
        bs = feat_t.shape[0]
        targets = torch.arange(rank * bs, rank * bs + bs, dtype=torch.long,
                               device=feat_t.device)

        # lambda_1 is a SIMILARITY (max = arity when collinear): softmax over +lambda_1
        lam_t2g = pmrl_lambda1(feat_t, gallery_all, present=present_all,
                               variant=self.pmrl_variant)
        lam_g2t = pmrl_lambda1(feat_t_all, gallery, present=present,
                               variant=self.pmrl_variant).T
        loss_ret = (F.cross_entropy(lam_t2g / self.pmrl_temp, targets, label_smoothing=0.1)
                    + F.cross_entropy(lam_g2t / self.pmrl_temp, targets,
                                      label_smoothing=0.1)) / 2
        loss_dict['loss_pmrl'] = loss_ret

        if self.pmrl_ortho_w > 0:
            # eigenvalue concentration on the POSITIVE pairs (local block)
            from .pmrl_loss import _pairwise_gram
            G = _pairwise_gram(feat_t, gallery, present=present)
            lam = torch.linalg.eigvalsh(G[torch.arange(bs), torch.arange(bs)])
            trace = lam.sum(-1).clamp(min=1e-6)
            loss_dict['loss_ortho'] = self.pmrl_ortho_w * ((trace - lam[..., -1]) / trace).mean()

        if self.itm_ratio > 0:
            loss_dict['loss_itm'] = self._itm_from_scores(batch, lam_t2g, lam_g2t, rank, bs,
                                                          temp=self.pmrl_temp)
        return loss_dict


class GRAMHyp(_ScoreITM, GRAM):
    """HyperGRAM repro arm (Na et al., CVPR 2026) on this trunk, model_type 'gram_hyp'.

    Training objective: hybrid Gramian volume V = alpha * V_euc + (1 - alpha) * V_hyp
    with learnable alpha (init 0.5, clamped to [0, 1] -- their Eq. 12's projected
    update), where V_euc is GRAM's masked Euclidean volume on the L2-NORMALISED
    features and V_hyp is the Lorentzian pseudo-volume on the PRE-normalisation
    contra-head outputs (varying spatial norms are the paper's variance-preservation
    mechanism; the pooled encoder outputs are cached by batch_get, so re-applying the
    bias-free linear heads costs one matmul per modality). Same contrastive CE(-V) as
    GRAM, same hard-negative ITM (negatives drawn from the hybrid similarity).

    Evaluation is deliberately IDENTICAL to every other arm (standard eval branch:
    Euclidean-volume raw diagnostic + ITM-reranked table metric), so its rows are
    directly comparable in our single anchored environment."""

    def __init__(self, config):
        super().__init__(config)
        import torch.nn as nn
        if getattr(self, 'train_mask', False):
            raise ValueError("gram_hyp is the faithful HyperGRAM repro -- no masked-"
                             "training variant is defined for it (train_mask must be off).")
        self.hyp_alpha = nn.Parameter(
            torch.tensor(float(getattr(self.config, 'hyp_alpha_init', 0.5))))
        self._alpha_log_every = 200
        self._alpha_step = 0

    def _prenorm_feats(self, batch):
        """Pre-normalisation contra-head projections, SAME modality order as the
        normalised gallery ([v, a, (s), (d)]); text last. Encoder outputs come from the
        batch_get cache -- no encoder re-run, heads are deterministic bias-free linears."""
        pv = self.contra_head_v(self.pool_vision_for_contra(self.batch_get(batch, 'vision_output')))
        pa = self.contra_head_a(self.pool_audio_for_contra(self.batch_get(batch, 'audio_output')))
        gallery = [pv, pa]
        if 'raw_subtitles' in batch.keys():
            gallery.append(self.contra_head_s(
                self.pool_text_for_contra(self.batch_get(batch, 'subtitle_output'))))
        if 'depth_pixels' in batch.keys():
            gallery.append(self.contra_head_d(
                self.pool_vision_for_contra(self.batch_get(batch, 'depth_output'))))
        pt = self.contra_head_t(self.pool_text_for_contra(self.batch_get(batch, 'caption_output')))
        return pt, gallery

    def forward_ret(self, batch, task, compute_loss=True):
        if not compute_loss:
            return super().forward_ret(batch, task, compute_loss=False)
        from utils.volume import volume_computation_masked, volume_computation_lorentz

        if isinstance(batch.raw_captions[0], list):
            batch.raw_captions = [i for j in batch.raw_captions for i in j]

        feat_t = self.batch_get(batch, 'feat_t').float()
        gallery = [self.batch_get(batch, 'feat_v').float(),
                   self.batch_get(batch, 'feat_a').float()]
        if 'raw_subtitles' in batch.keys():
            gallery.append(self.batch_get(batch, 'feat_s').float())
        if 'depth_pixels' in batch.keys():
            gallery.append(self.batch_get(batch, 'feat_d').float())
        pren_t, pren_g = self._prenorm_feats(batch)
        assert len(pren_g) == len(gallery), (len(pren_g), len(gallery))
        present = present_from_feats(gallery)

        feat_t_all = _gather(feat_t)
        gallery_all = [_gather(g) for g in gallery]
        present_all = present_from_feats(gallery_all)
        pren_t_all = _gather(pren_t)
        pren_g_all = [_gather(g) for g in pren_g]

        rank = dist.get_rank() if dist.is_initialized() else 0
        bs = feat_t.shape[0]
        targets = torch.arange(rank * bs, rank * bs + bs, dtype=torch.long,
                               device=feat_t.device)

        alpha = self.hyp_alpha.clamp(0.0, 1.0)
        vol = (alpha * volume_computation_masked(feat_t, gallery_all, present=present_all)
               + (1.0 - alpha) * volume_computation_lorentz(pren_t, pren_g_all,
                                                            present=present_all))
        vol = vol / self.contra_temp
        volT = (alpha * volume_computation_masked(feat_t_all, gallery, present=present)
                + (1.0 - alpha) * volume_computation_lorentz(pren_t_all, pren_g,
                                                             present=present)).T
        volT = volT / self.contra_temp

        loss_dict = {'loss_hyp_volume':
                     (F.cross_entropy(-vol, targets, label_smoothing=0.1)
                      + F.cross_entropy(-volT, targets, label_smoothing=0.1)) / 2}
        if self.itm_ratio > 0:
            loss_dict['loss_itm'] = self._itm_from_scores(batch, -vol, -volT, rank, bs)

        self._alpha_step += 1
        if self._alpha_step % self._alpha_log_every == 1 and rank == 0:
            LOGGER.info(f'[gram_hyp] step~{self._alpha_step} alpha={alpha.item():.4f}')
        return loss_dict
