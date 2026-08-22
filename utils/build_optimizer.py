import math
import re
import torch
from torch.optim import Adam, Adamax, Optimizer
from .logger import LOGGER






def build_optimizer(model, args, checkpoint_optim):

    vision_clip = 'vision_encoder_type' in args.model_cfg and 'clip' in args.model_cfg.vision_encoder_type

    # hgnn.gates is a zero-init scalar residual gate, not a weight -> must not be weight-decayed
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'gates']

    # ---- SCA LoRA regime: freeze the three backbones, train LoRA A/B (own lr, default
    # 0.1x the base lr) + the projection heads at the base lr. GRAM path (use_lora unset)
    # is byte-for-byte unchanged.
    use_lora = bool(getattr(args.model_cfg, 'use_lora', False))
    # multimodal_encoder is the ITM CROSS-ENCODER (model/sca.py: self.multimodal_encoder.bert
    # feeds self.itm_head). Freezing it with a rank-8 adapter while GRAM full-finetunes it is
    # where SCA's dual-encoder advantage is lost: measured against the released GRAM
    # checkpoint, SCA leads by +4.5 (DiDeMo) and +2.3 (ActivityNet) on the aggregator's own
    # score and then trails by 0.7 and 3.7 after reranking -- a 5-6 point swing at the one
    # stage it under-trains. lora_freeze_multimodal=false keeps vision/audio adapted but lets
    # the reranker train, which is the targeted version of that fix.
    freeze_mm = bool(getattr(args.model_cfg, 'lora_freeze_multimodal', True))
    # Unfreezing the cross-encoder while an adapter is still injected into it trains the SAME
    # matrices twice per step: W_q and W_v get the base update at learning_rate AND a rank-r
    # update at lora_lr, scaled by alpha/r, while every other projection in the layer (key,
    # output.dense, the FFN) gets only the base update. The layer's attention then moves at a
    # different effective rate from the rest of it, which is not what full fine-tuning is and
    # not what GRAM does.
    #
    # Three arms were run this way before it was noticed. x1_xenc_full_lr2e5 validated
    # 52.6 / 45.8 / 50.3 / 48.7 / 40.7 / 40.3 / 43.7 / 49.5 / 49.8 / 49.2 -- oscillation, which
    # was misread as overtraining decay and used to conclude that training the cross-encoder
    # does not work. That conclusion was drawn from a defect, not from the method.
    #
    # To train the cross-encoder, set lora_r_text = 0 alongside this flag: vision and audio
    # keep their adapters, the cross-encoder is fine-tuned cleanly, and the parameterization
    # matches GRAM's.
    if use_lora and not freeze_mm and int(getattr(args.model_cfg, 'lora_r_text', 8)) > 0:
        raise ValueError(
            'lora_freeze_multimodal=false with lora_r_text=%d: the cross-encoder would be '
            'trained by its base weights AND by a LoRA adapter on the same W_q/W_v at once. '
            'Set lora_r_text=0 to fine-tune it cleanly, or keep it frozen.'
            % int(getattr(args.model_cfg, 'lora_r_text', 8)))
    backbone_prefixes = (('vision_encoder', 'audio_encoder', 'multimodal_encoder')
                         if freeze_mm else ('vision_encoder', 'audio_encoder'))

    # ---- How MUCH to move the cross-encoder, not whether to.
    #
    # Fine-tuning it at the base rate for the full schedule destroys it: X3/X4/X5 read 51.4,
    # 51.1 and 50.9 against 54.8 with it frozen, and every one of them is HIGHEST at its first
    # validation and falling. That is forgetting, and two things drive it.
    #
    # Scale. GRAM's released weights are model_step_459 on the same VAST foundation we start
    # from -- 459 steps of adaptation. We run 5330, on 150k clips, against the 27M this
    # component was pretrained on.
    #
    # Modality mix. The ITM loss trains on condition_feats_va (gram.py:732, hardcoded), and
    # our training set carries no subtitles at all. MSR-VTT and VATEX are then evaluated with
    # tvas. Fine-tuning here therefore erases a subtitle pathway that no gradient in this
    # recipe can restore, and MSR-VTT is exactly where the gap to HyperGRAM sits.
    #
    # xenc_lr gives the cross-encoder its own, much smaller step while the rest of the model
    # trains normally. xenc_train_layers keeps the lower BERT layers frozen and moves only the
    # top K, so the general representation survives and only the task head adapts. Both are
    # off by default, and both require the cross-encoder to be unfrozen to mean anything.
    xenc_lr = getattr(args.run_cfg, 'xenc_lr', None)
    xenc_layers = int(getattr(args.model_cfg, 'xenc_train_layers', 0) or 0)
    if (xenc_lr or xenc_layers) and freeze_mm:
        raise ValueError(
            'xenc_lr/xenc_train_layers set while lora_freeze_multimodal is true: the '
            'cross-encoder is frozen, so neither does anything. Set '
            'lora_freeze_multimodal=false (with lora_r_text=0), or drop these keys.')
    xenc_frozen_by_depth = []
    if xenc_layers and not freeze_mm:
        idx = []
        for k, _ in model.named_parameters():
            m = re.search(r'multimodal_encoder\..*?\.layer\.(\d+)\.', k)
            if m:
                idx.append(int(m.group(1)))
        if not idx:
            raise RuntimeError(
                'xenc_train_layers=%d but no multimodal_encoder layer index matched -- the '
                'BERT naming has drifted, and silently training ALL layers under a flag that '
                'says otherwise is the failure this refuses to have.' % xenc_layers)
        cutoff = max(idx) + 1 - xenc_layers
        for k, v in model.named_parameters():
            m = re.search(r'multimodal_encoder\..*?\.layer\.(\d+)\.', k)
            if m and int(m.group(1)) < cutoff:
                v.requires_grad = False
                xenc_frozen_by_depth.append(k)
        LOGGER.info('[XENC] training the top %d of %d layers (froze %d tensors below layer %d)'
                    % (xenc_layers, max(idx) + 1, len(xenc_frozen_by_depth), cutoff))
    lora_params = []
    lora_params_name = []
    if use_lora:
        for k, v in model.named_parameters():
            if any(bp in k for bp in backbone_prefixes) and 'lora_' not in k:
                v.requires_grad = False
    lora_lr = getattr(args.run_cfg, 'lora_lr', None) or args.run_cfg.learning_rate * 0.1

    basic_params = []
    basic_params_name = []
    basic_params_no_decay = []
    clip_params_visual = []
    clip_params_name_visual = []
    clip_params_no_decay_visual = []
    clip_params_text = []
    clip_params_name_text = []
    clip_params_no_decay_text = []
    new_params = []
    new_params_name = []
    new_params_no_decay = []


    xenc_params, xenc_params_no_decay, xenc_params_name = [], [], []
    for k, v in model.named_parameters():
        if use_lora and not v.requires_grad:
            continue
        if xenc_lr and k.startswith('multimodal_encoder'):
            # its own group, so the cross-encoder can move at 1e-6 while the heads move at
            # 2e-5. Checked before the lora branch because with lora_r_text=0 there are no
            # adapters here anyway, and after it the params would land in basic_params at the
            # base rate -- which is the setting already measured to destroy this component.
            (xenc_params_no_decay if any(nd in k for nd in no_decay) else xenc_params).append(v)
            xenc_params_name.append(k)
        elif use_lora and ('lora_A' in k or 'lora_B' in k):
            lora_params.append(v)
            lora_params_name.append(k)
        elif any(nd in k for nd in args.run_cfg.new_params_name) and not any(nd in k for nd in no_decay):
            new_params.append(v)
            new_params_name.append(k)
        elif any(nd in k for nd in args.run_cfg.new_params_name) and any(nd in k for nd in no_decay):
            new_params_no_decay.append(v) 
            new_params_name.append(k)
        elif  vision_clip  and 'visual' in k and not any(nd in k for nd in no_decay):
            clip_params_visual.append(v)
            clip_params_name_visual.append(k)
        elif vision_clip  and  'visual' in k and  any(nd in k for nd in no_decay):
            clip_params_no_decay_visual.append(v)
            clip_params_name_visual.append(k)
     
        
        elif not any(nd in k for nd in no_decay):
            basic_params.append(v)
            basic_params_name.append(k)
        elif any(nd in k for nd in no_decay):
            basic_params_no_decay.append(v)
            basic_params_name.append(k)

    # print(new_params)
    optimizer_grouped_parameters = [
        {'params': basic_params, 'weight_decay': args.run_cfg.weight_decay, 'lr': args.run_cfg.learning_rate},
        {'params': basic_params_no_decay, 'weight_decay': 0.0, 'lr': args.run_cfg.learning_rate},
        {'params': new_params, 'weight_decay': args.run_cfg.weight_decay, 'lr': args.run_cfg.new_lr},
        {'params': new_params_no_decay, 'weight_decay': 0.0, 'lr': args.run_cfg.new_lr},
        {'params': clip_params_visual, 'weight_decay': args.run_cfg.weight_decay, 'lr': args.run_cfg.clip_lr},
        {'params': clip_params_no_decay_visual, 'weight_decay': 0.0, 'lr': args.run_cfg.clip_lr},
    ]
    if xenc_lr:
        optimizer_grouped_parameters += [
            {'params': xenc_params, 'weight_decay': args.run_cfg.weight_decay, 'lr': xenc_lr},
            {'params': xenc_params_no_decay, 'weight_decay': 0.0, 'lr': xenc_lr},
        ]
        if not xenc_params and not xenc_params_no_decay:
            raise RuntimeError(
                'xenc_lr=%s but no trainable multimodal_encoder parameter reached the '
                'optimizer. The cross-encoder would train at no rate at all while the config '
                'claims a discriminative one.' % xenc_lr)
        LOGGER.info('[XENC] %d tensors at lr %s (base lr %s)'
                    % (len(xenc_params) + len(xenc_params_no_decay), xenc_lr,
                       args.run_cfg.learning_rate))
    if use_lora:
        optimizer_grouped_parameters.append(
            {'params': lora_params, 'weight_decay': args.run_cfg.weight_decay, 'lr': lora_lr})
        LOGGER.info(f'[LoRA] {len(lora_params)} adapter tensors at lr {lora_lr}; '
                    f'backbones {backbone_prefixes} frozen'
                    + ('' if freeze_mm else '; multimodal_encoder (ITM cross-encoder) TRAINABLE'))

    # print(basic_params_name)
    # print(clip_params_visual)
    # currently Adam only
    if args.run_cfg.optim == 'adam':
        OptimCls = Adam
    elif args.run_cfg.optim == 'adamax':
        OptimCls = Adamax
    elif args.run_cfg.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')

    for i in optimizer_grouped_parameters:
        i['init_lr'] = i['lr']
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=args.run_cfg.learning_rate, betas=args.run_cfg.betas)

    optimizer.new_params_name = new_params_name
    optimizer.new_lr = args.run_cfg.new_lr
    optimizer.basic_lr = args.run_cfg.learning_rate
    optimizer.clip_lr_visual = args.run_cfg.clip_lr
    optimizer.clip_lr_visual_len = len(clip_params_visual)

    optimizer.zero_grad()

    if checkpoint_optim:
        optimizer.load_state_dict(checkpoint_optim)
        del(checkpoint_optim)

    LOGGER.info('==='*6+'learning_rate_settings'+'==='*6+'\n')
    LOGGER.info(f"  basic_lr : {optimizer.basic_lr}")
    LOGGER.info(f"  clip_lr_visual : {optimizer.clip_lr_visual}")
    LOGGER.info(f"  clip_lr_visual_len : {optimizer.clip_lr_visual_len}")
    LOGGER.info(f"  new_lr : {optimizer.new_lr}")
    LOGGER.info(f"  new_params_name: {optimizer.new_params_name}")

    return  optimizer





class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2).
            Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias
            in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - "
                             "should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - "
                             "should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - "
                             "should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse '
                        'gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = (step_size * math.sqrt(bias_correction2)
                                 / bias_correction1)

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is
                # *not* the correct way of using L2 regularization/weight decay
                # with Adam, since that will interact with the m and v
                # parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't
                # interact with the m/v parameters. This is equivalent to
                # adding the square of the weights to the loss with plain
                # (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss