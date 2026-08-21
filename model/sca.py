import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from easydict import EasyDict as edict

from utils.logger import LOGGER
from utils.distributed import all_gather_with_grad, concat_all_gather
from .gram import GRAM
from .centroid import (masked_spherical_mean, concept_resultant,
                       query_centroid_scores)
from .prototypes import PrototypeMemory
from .losses_sca import (l_align, l_sem, l_mask, l_concept, l_unif,
                         check_calibration_config, info_nce)
from data.mask_sampler import MaskSampler


def _gather(t):
    """concat_all_gather that degrades to identity outside DDP (single-GPU smoke runs)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return concat_all_gather(t)
    return t

# SCA: Spherical Centroid Alignment, a sibling of GRAM on the SAME trunk (config switch
# model_type = gram | sca). Everything up to the contrastive features -- encoders, projections,
# gathering, ITM machinery, cap/qa heads, the eval branch -- is inherited from GRAM unchanged;
# only the forward_ret LOSS BLOCK is swapped: the gallery clip is represented by the masked
# spherical mean of its present modality embeddings (arity-invariant, k=2..5 with zero code
# change -- E10) and trained with L_align + alpha*L_sem + beta*L_mask + delta*L_concept +
# lambda*L_unif. Text stays the query side only (leak-free, as in the hypergraph arm).


class SCA(GRAM):

    def __init__(self, config):
        super().__init__(config)

        cfg = self.config
        # the model declares its own eval geometry: without this, an eval config lacking
        # score_mode would silently score an SCA checkpoint with GRAM's volume
        if not getattr(cfg, 'score_mode', None):
            cfg.score_mode = 'centroid'
        if bool(getattr(cfg, 'train_mask', False)):
            raise ValueError(
                "train_mask=true is the GRAM/PMRL 2x2 arm; SCA masks via its own "
                "mask_p_full schedule (mu_M/mu_K virtual masking). Combining both would "
                "double-mask -- use mask_p_full_* on SCA instead.")
        # ---- loss weights ----
        self.sca_alpha = float(getattr(cfg, 'sca_alpha', 1.0))     # L_sem
        self.sca_beta = float(getattr(cfg, 'sca_beta', 1.0))       # L_mask
        self.sca_delta = float(getattr(cfg, 'sca_delta', 0.5))     # L_concept
        self.sca_lambda = float(getattr(cfg, 'sca_lambda', 0.1))   # L_unif

        # ---- temperature + calibration mechanism (A10) ----
        self.sca_calibration = getattr(cfg, 'sca_calibration', 'regression')
        self.sca_cal_w = float(getattr(cfg, 'sca_cal_w', 1.0))
        # S* caches are top-k sparsified: absent = "unknown", not "0" -- regress known pairs only
        self.sca_cal_known_only = bool(getattr(cfg, 'sca_cal_known_only', True))
        self.sca_tau_star = float(getattr(cfg, 'sca_tau_star', 0.5))
        tau_learnable = bool(getattr(cfg, 'sca_tau_learnable', False))
        check_calibration_config(self.sca_calibration, tau_learnable)
        tau0 = float(getattr(cfg, 'sca_tau', 0.07))
        if self.sca_calibration == 'fixed_tau':
            tau0 = self.sca_tau_star                               # tau frozen AT tau*
        if tau_learnable:
            self.sca_tau = nn.Parameter(torch.tensor(tau0))
        else:
            self.register_buffer('sca_tau', torch.tensor(tau0))

        # ---- warmup (guards from the k=2 analysis): L_align-only before L_sem; L_concept
        # delayed to warmup end, where the prototype memory gets its staleness reset ----
        self.sca_warmup_steps = int(getattr(cfg, 'sca_warmup_steps', 500))
        self.register_buffer('sca_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('sca_warmup_reset_done', torch.zeros((), dtype=torch.bool))

        # ---- mask sampler (virtual masking; mu_M and mu_K from ONE forward pass) ----
        self.mask_sampler = MaskSampler(
            num_modalities=4,                                      # capacity V,A,S,D; per-batch L <= 4
            p_full_start=float(getattr(cfg, 'mask_p_full_start', 1.0)),
            p_full_end=float(getattr(cfg, 'mask_p_full_end', 0.5)),
            schedule_steps=int(getattr(cfg, 'mask_schedule_steps', 2000)),
            mode=getattr(cfg, 'mask_mode', 'uniform'),
            freq=getattr(cfg, 'mask_freq', None) if getattr(cfg, 'mask_mode', 'uniform') == 'freq' else None,
            n_drop=int(getattr(cfg, 'mask_n_drop', 1)))
        self.l_mask_term1 = bool(getattr(cfg, 'l_mask_term1', True))
        self.l_mask_term2 = bool(getattr(cfg, 'l_mask_term2', True))
        self.l_unif_weighted = bool(getattr(cfg, 'l_unif_weighted', False))
        self.l_unif_t = float(getattr(cfg, 'l_unif_t', 2.0))

        # ---- Level-2 concept prototypes (EMA memory; A4 sweeps mode/eta/eps-floor) ----
        self.sca_num_concepts = int(getattr(cfg, 'sca_num_concepts', 0))
        self.sca_eps_floor = float(getattr(cfg, 'sca_eps_floor', 0.05))
        self.sca_proto_mode = getattr(cfg, 'sca_proto_mode', 'ema')      # 'ema' | 'batch'
        assert self.sca_proto_mode in ('ema', 'batch'), self.sca_proto_mode
        if self.sca_num_concepts > 0:
            self.prototypes = PrototypeMemory(self.sca_num_concepts, cfg.contra_dim,
                                              eta=float(getattr(cfg, 'sca_eta', 0.99)))
        else:
            self.prototypes = None

        # ---- A7 "learned gates" arm: per-modality centroid weights, zero-init == uniform ----
        if bool(getattr(cfg, 'sca_centroid_gates', False)):
            self.centroid_gates = nn.Parameter(torch.zeros(4))           # capacity V,A,S,D
        else:
            self.centroid_gates = None

        # ---- frame slots: the video contributes one slot per frame, not one pooled vector.
        # pool_vision_for_contra averages the per-frame CLS tokens before the contrastive
        # head, so length stops being representable there -- upstream of every aggregator.
        # ---- query weighting: w_m proportional to exp(<t, z_m>/tau_w) over the present
        # slots, which is what lets a slot the query does not care about be discounted. On a
        # trained checkpoint, scoring against the frame set at tau_w=0.1 beats mean-pooling
        # by +1.9 to +3.1 R@1 on the video pathway on all four benchmarks, and beats
        # max-over-frames too, so it is soft fusion rather than selection.
        self.frame_slots = bool(getattr(cfg, 'sca_frame_slots', False))
        self.query_weighting = bool(getattr(cfg, 'sca_query_weighting', False))
        self.sca_tau_w = float(getattr(cfg, 'sca_tau_w', 0.1))
        if self.frame_slots and not self.query_weighting:
            # Frames are slots, and a UNIFORM mean weights slots equally -- so 8 video frames
            # beside one audio and one subtitle slot hand video 8/10 of the centroid where it
            # previously had 1/3. That is a silent change to the modality balance masquerading
            # as a change to the video representation, and the two effects could never be
            # separated afterwards. Query weighting sets the shares from the text instead of
            # from how many slots a modality happens to own, which is what makes the frame
            # expansion well-posed.
            raise RuntimeError(
                '[SCA] sca_frame_slots without sca_query_weighting: a uniform mean would give '
                'video one share PER FRAME, silently re-weighting the modalities. Enable '
                'sca_query_weighting (weights come from the query, not the slot count), or '
                'turn off sca_frame_slots.')

        # ---- S* semantic targets (built offline by data/semantic_targets.py) ----
        self.s_star_path = os.path.expandvars(getattr(cfg, 's_star_path', '') or '')
        # A3 ablation arm (S* = I) must be requested EXPLICITLY -- it is never a fallback
        self.s_star_identity = bool(getattr(cfg, 'sca_s_star_identity', False))
        if self.sca_alpha > 0 and not self.s_star_identity and not self.s_star_path:
            raise ValueError(
                'sca_alpha > 0 (L_sem on) but no S* source: set s_star_path to a cache built '
                'by data/semantic_targets.py, or set sca_s_star_identity=true for the explicit '
                'S*=I (A3) arm, or set sca_alpha=0.')
        self.semantic_targets = None

        # ---- LoRA into the attention W_q/W_v of the three backbones ----
        self.use_lora = bool(getattr(cfg, 'use_lora', False))
        self._lora_wrapped = []
        if self.use_lora:
            from .lora import setup_lora_backbones
            self._lora_wrapped = setup_lora_backbones(self, cfg, logger=LOGGER)

    def modify_checkpoint(self, checkpoint):
        checkpoint = super().modify_checkpoint(checkpoint)
        if self._lora_wrapped:
            from .lora import remap_lora_checkpoint
            checkpoint = remap_lora_checkpoint(checkpoint, self._lora_wrapped)
        return checkpoint

    # ------------------------------------------------------------------ helpers

    def _load_semantic_targets(self):
        if self.semantic_targets is None and self.s_star_path:
            if not os.path.exists(self.s_star_path):
                # hard error: silently training without the configured S* would make every
                # E6/A10 number a lie (it would really be the A3 S*=I arm)
                raise FileNotFoundError(
                    f'[SCA] s_star_path {self.s_star_path} does not exist. Build it with '
                    'data/semantic_targets.py, or explicitly run the S*=I arm via '
                    'sca_s_star_identity=true.')
            from data.semantic_targets import SemanticTargets
            self.semantic_targets = SemanticTargets(self.s_star_path)
            LOGGER.info(f'[SCA] loaded S* cache {self.s_star_path} '
                        f'({len(self.semantic_targets.row_of)} rows)')
        return self.semantic_targets

    def _gallery_feats(self, batch):
        """Ordered dict of the batch's gallery modalities (text excluded)."""
        feats = {'v': self.batch_get(batch, 'feat_v'),
                 'a': self.batch_get(batch, 'feat_a')}
        if 'raw_subtitles' in batch.keys():
            feats['s'] = self.batch_get(batch, 'feat_s')
        if 'depth_pixels' in batch.keys():
            feats['d'] = self.batch_get(batch, 'feat_d')
        return feats

    def _gallery_slots(self, batch, gallery, mods):
        """(z, owner): the slot set the centroid is taken over, and each slot's modality.

        Default -- one slot per modality, owner = [0, 1, ...], i.e. exactly the previous
        behaviour and byte-identical to it.

        With sca_frame_slots, video contributes ONE SLOT PER FRAME instead of one pooled
        vector. pool_vision_for_contra (general_module.py:426) averages the per-frame CLS
        tokens before the contrastive head, so every clip reaches the centroid as a single
        point regardless of length. Measured on a trained checkpoint, scoring against the
        frame set instead of the pooled vector is worth +1.9 to +3.1 R@1 on the video
        pathway across all four benchmarks -- and beats BOTH limits (mean-pooling and
        max-over-frames) everywhere, so it is soft fusion over frames rather than frame
        selection.

        Nothing downstream needs to change: the centroid is arity-invariant, so frames are
        simply more slots. `owner` is what keeps the semantics right -- the mask sampler
        draws at MODALITY level and the draw is expanded over that modality's slots, so
        dropping video still drops the whole video rather than one frame of it.
        """
        z = torch.stack([gallery[m].float() for m in mods], dim=1)          # (B, L, d)
        owner = list(range(len(mods)))
        if not self.frame_slots or 'v' not in mods:
            return z, owner
        zf = self.batch_get(batch, 'feat_v_frames').float()                 # (B, F, d)
        vi = mods.index('v')
        slots, owner = [], []
        for i, m in enumerate(mods):
            if i == vi:
                slots += [zf[:, f] for f in range(zf.shape[1])]
                owner += [vi] * zf.shape[1]
            else:
                slots.append(gallery[m].float())
                owner.append(i)
        return torch.stack(slots, dim=1), owner

    # ------------------------------------------------------------------ forward

    def forward_ret(self, batch, task, compute_loss=True):
        if not compute_loss:
            # eval branch identical to GRAM (raw per-modality feats out; scoring happens in
            # evaluation_mm, which builds the centroid when score_mode == 'centroid')
            return super().forward_ret(batch, task, compute_loss=False)

        if isinstance(batch.raw_captions[0], list):
            batch.raw_captions = [i for j in batch.raw_captions for i in j]

        loss_dict = {}
        step = int(self.sca_step.item())
        warm = step < self.sca_warmup_steps

        feat_t = self.batch_get(batch, 'feat_t')
        gallery = self._gallery_feats(batch)
        mods = list(gallery.keys())
        L = len(mods)
        z, owner = self._gallery_slots(batch, gallery, mods)                 # (B, S, d)
        B = z.shape[0]
        owner_idx = torch.tensor(owner, device=z.device, dtype=torch.long)

        # real per-clip presence, at MODALITY level then expanded over that modality's slots
        # (loader zero-fills a modality it could not load). With one slot per modality this
        # is the previous expression unchanged.
        present_mod = torch.stack([(gallery[m].float().norm(dim=-1) > 0.5).float()
                                   for m in mods], dim=1)                   # (B, L)
        present = present_mod[:, owner_idx]                                 # (B, S)

        # virtual mask: train-time m-dagger draws on top of the real presence. Drawn at
        # modality level so a drop removes a whole modality, not one frame of the video.
        if self.training:
            vmask = self.mask_sampler.sample(B, step, z.device, present=present_mod)[:, owner_idx]
        else:
            vmask = torch.ones_like(present)
        present_M = present * vmask

        # both centroids from the ONE forward pass (virtual-mask bookkeeping)
        g = self.centroid_gates
        # query weighting needs the text, and the gates are per-slot; they are mutually
        # exclusive arms, so a config asking for both is a mistake worth refusing
        if self.query_weighting and g is not None:
            raise RuntimeError('[SCA] sca_query_weighting and centroid gates are two different '
                               'weightings of the same sum; enable one, not both.')
        wkw = ({'weighting': 'query', 'tau_w': self.sca_tau_w, 'query': feat_t.float()}
               if self.query_weighting else {'gates': g})
        mu_K, A_K, n_K = masked_spherical_mean(z, present, **wkw)           # full view
        mu_M, A_M, n_M = masked_spherical_mean(z, present_M, **wkw)         # masked view

        feat_t32 = feat_t.float()
        tau = self.sca_tau if torch.is_tensor(self.sca_tau) else torch.tensor(self.sca_tau)
        tau = tau.clamp(min=1e-3)

        rank = dist.get_rank() if dist.is_initialized() else 0
        feat_t_all = _gather(feat_t32)
        mu_M_all = _gather(mu_M)
        targets = torch.arange(rank * B, rank * B + B, dtype=torch.long, device=z.device)

        # ---- L_align: symmetric InfoNCE text <-> masked centroid (the training view) ----
        if self.query_weighting:
            # A query-weighted centroid is a function of the TEXT it is scored against, so
            # feat_t @ mu_all.T is the wrong objective: mu_j was built from t_j, giving the
            # positive pair a centroid weighted for its own text while every negative is
            # weighted by the wrong one. The model could then win by sharpening the weights
            # rather than by learning features, and it would not match inference, where every
            # candidate is conditioned on the query. Both directions therefore need the
            # PAIRWISE score S[i, j] = <t_i, mu(z_j | t_i)>.
            z_all = _gather(z)                                          # (B_all, S, d)
            presM_all = _gather(present_M)                              # (B_all, S)
            sim_t2m = query_centroid_scores(feat_t32, z_all, presM_all,
                                            tau=self.sca_tau_w) / tau
            # clip-to-text: for each LOCAL clip, a distribution over ALL texts. Same pairwise
            # definition, computed with the roles of the gathered side swapped.
            sim_m2t = query_centroid_scores(feat_t_all, z, present_M,
                                            tau=self.sca_tau_w).T / tau
            loss_dict['loss_align'] = (info_nce(sim_t2m, targets, 0.1)
                                       + info_nce(sim_m2t, targets, 0.1)) / 2
        else:
            loss_dict['loss_align'] = l_align(feat_t32, mu_M_all, mu_M, feat_t_all, tau, targets)

        # ---- L_sem + calibration (E6; gated to post-warmup) ----
        if not warm and self.sca_alpha > 0:
            sim_local = feat_t32 @ mu_M.T                                   # (B, B) raw cosines
            if self.s_star_identity:
                s_star = torch.eye(B, device=z.device)                      # explicit A3 arm only
            else:
                st = self._load_semantic_targets()
                if 'ids' not in batch.keys():
                    raise RuntimeError(
                        '[SCA] L_sem needs batch.ids to gather S* rows, but the batch has no '
                        'ids field -- the dataset/collate must pass clip ids through.')
                s_star = st.gather(batch.ids, device=z.device)
            loss_dict['loss_sem'] = self.sca_alpha * l_sem(
                sim_local, s_star, tau, tau_star=self.sca_tau_star,
                calibration=self.sca_calibration, cal_w=self.sca_cal_w,
                cal_known_only=self.sca_cal_known_only)
        else:
            s_star = None

        # ---- L_mask: masked view must agree with the full view (terms toggleable, A1) ----
        if self.sca_beta > 0 and self.training:
            s_M = (feat_t32 * mu_M).sum(-1)                                 # positive-pair scores
            s_K = (feat_t32 * mu_K).sum(-1)
            loss_dict['loss_mask'] = self.sca_beta * l_mask(
                mu_M, mu_K, s_M=s_M, s_K=s_K,
                term1=self.l_mask_term1, term2=self.l_mask_term2)

        # ---- L_concept: Level-2 EMA prototypes (delayed to warmup end + staleness reset) ----
        labels = batch.get('concept_labels', batch.get('labels', None))
        if self.prototypes is not None and labels is None and self.training:
            raise RuntimeError(
                '[SCA] sca_num_concepts > 0 (L_concept on) but the batch carries no '
                'concept_labels/labels -- L_concept would silently never train. Provide labels '
                'in the dataset or set sca_num_concepts=0.')
        if self.prototypes is not None and labels is not None:
            labels = torch.as_tensor(labels, device=z.device).long()
            if not warm:
                if not bool(self.sca_warmup_reset_done):
                    self.prototypes.reset_from_running()                    # staleness guard
                    self.sca_warmup_reset_done.fill_(True)
                if self.sca_delta > 0:
                    if self.sca_proto_mode == 'batch':
                        # A4 batch-only arm: nu_c from THIS batch's class means, no memory
                        from .prototypes import batch_prototypes
                        protos, has = batch_prototypes(mu_K, labels, self.sca_num_concepts)
                        keep = has[labels]
                        loss_dict['loss_concept'] = self.sca_delta * l_concept(
                            mu_K[keep], labels[keep], protos, eps_floor=self.sca_eps_floor)
                    else:
                        loss_dict['loss_concept'] = self.sca_delta * l_concept(
                            mu_K, labels, self.prototypes.protos, eps_floor=self.sca_eps_floor)
            self.prototypes.update(mu_K, labels)                            # no-grad EMA + DDP reduce

        # ---- L_unif: hypersphere uniformity, optional (1 - S*) weighting (A8) ----
        if not warm and self.sca_lambda > 0:
            w = s_star if (self.l_unif_weighted and s_star is not None) else None
            loss_dict['loss_unif'] = self.sca_lambda * l_unif(mu_M, t=self.l_unif_t, s_star=w)

        # ---- ITM on the shared trunk (GRAM recipe, negatives drawn from centroid sims) ----
        if self.itm_ratio > 0:
            loss_dict['loss_itm'] = self._itm_loss(batch, feat_t32, mu_M_all, feat_t_all,
                                                   mu_M, tau, rank, B)

        # diagnostics for wandb (NOT in loss_dict -- the pipeline sums every loss_dict value);
        # picked up by the training loop / hooks via model.sca_stats
        self.sca_stats = {'A_M_mean': A_M.mean().item(),
                          'A_K_mean': A_K.mean().item(),
                          'min_resultant': (A_M * n_M.clamp(min=1.0)).min().item(),
                          'p_full': self.mask_sampler.p_full(step)}
        if self.training:
            self.sca_step += 1
        return loss_dict

    def _itm_loss(self, batch, feat_t32, mu_M_all, feat_t_all, mu_M, tau, rank, bs):
        """GRAM's hard-negative ITM, with negatives sampled from the centroid similarities
        instead of the volume (same trunk, same heads)."""
        caption_tokens = self.batch_get(batch, 'caption_tokens')
        input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
        input_ids_collate = _gather(input_ids)
        attention_mask_collate = _gather(attention_mask)

        sim = feat_t32 @ mu_M_all.T / tau
        simT = mu_M @ feat_t_all.T / tau

        condition_feats = self.batch_get(batch, 'condition_feats_va')
        condition_feats_collate = (all_gather_with_grad(condition_feats)
                                   if dist.is_initialized() else condition_feats)
        with torch.no_grad():
            weights_t2cond = F.softmax(sim, dim=1) + 1e-4
            weights_t2cond[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
            weights_cond2t = F.softmax(simT, dim=1) + 1e-4
            weights_cond2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0)

        condition_feats_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
            condition_feats_neg.append(condition_feats_collate[neg_idx])
        condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

        text_ids_neg, text_atts_neg = [], []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

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
