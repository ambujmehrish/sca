"""Make an eval config's LoRA geometry match the checkpoint it is about to load.

The eval configs in benchmark_eval/ hardcode rank 8 (the default arm's value), so evaluating
any arm trained at a different rank builds a model whose adapters do not match the
checkpoint. Two ways that goes wrong, and the quiet one is the dangerous one:

  LOUD  -- a rank mismatch raises on load_state_dict, e.g. the x2_xenc_r64 arms
           (lora_r_text 64) against a rank-8 eval config:
             size mismatch for multimodal_encoder...query.lora_A: copying a param with
             shape torch.Size([64, 768]) ... current model is torch.Size([8, 768])

  QUIET -- an ALPHA mismatch does not. LoRALinear scales its update by alpha/r
           (model/lora.py:52), and alpha is not stored in the checkpoint. An arm trained at
           r=32, alpha=64 loads cleanly into a config saying r=32, alpha=16 and then runs
           with a quarter of the adapter strength it was trained with. Every number that
           comes out is wrong, and nothing says so.

The B-grid arms (rank 32, alpha 64) are exactly the second case, so this is not hypothetical.

The rank is read from the checkpoint's own tensor shapes, which cannot disagree with the
weights being loaded. Alpha has to come from the run's recorded hps.json; where that is
missing and the rank does not already match, this raises rather than guessing, because a
wrong alpha produces plausible numbers instead of an error.
"""
import glob
import json
import os

import torch

# checkpoint prefix -> the model_cfg field that sets that encoder's rank, per
# model/lora.py:setup_lora_backbones
ENCODER_RANK_KEY = (('vision_encoder', 'lora_r_vision'),
                    ('audio_encoder', 'lora_r_audio'),
                    ('multimodal_encoder', 'lora_r_text'))


def ranks_from_checkpoint(path):
    """-> {model_cfg field: rank} inferred from the lora_A tensors actually in the file."""
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    if not isinstance(state, dict):
        return {}
    found = {}
    for key, tensor in state.items():
        if not key.endswith('.lora_A') or not torch.is_tensor(tensor):
            continue
        name = key.replace('module.', '')
        for prefix, cfg_key in ENCODER_RANK_KEY:
            if name.startswith(prefix + '.'):
                # lora_A is (r, in_features)
                found.setdefault(cfg_key, tensor.shape[0])
    return found


def recorded_model_cfg(ckpt_path):
    """The model_cfg the run that produced this checkpoint recorded, if it is on disk.

    Checkpoints live at <workdir>/ckpt/model_step_N.pt and the config at <workdir>/log/hps.json.
    """
    workdir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    for pat in ('log/hps.json', 'hps.json'):
        hits = sorted(glob.glob(os.path.join(workdir, pat)))
        if hits:
            try:
                return json.load(open(hits[0])).get('model_cfg', {}), hits[0]
            except (ValueError, IOError):
                return None, hits[0]
    return None, None


def sync_lora_geometry(args, logger=None):
    """Align args.model_cfg's LoRA fields with the checkpoint. Raises rather than guess.

    No-op when there is no checkpoint or the checkpoint carries no LoRA tensors, so a
    non-LoRA arm (full finetuning, the released GRAM weights) is untouched.
    """
    ckpt = getattr(args.run_cfg, 'checkpoint', None)
    if not ckpt or not os.path.isfile(ckpt):
        return
    ranks = ranks_from_checkpoint(ckpt)
    if not ranks:
        return                                   # not a LoRA checkpoint

    mcfg = args.model_cfg
    changed = []
    for cfg_key, r in sorted(ranks.items()):
        if int(getattr(mcfg, cfg_key, 8)) != int(r):
            changed.append('%s %s->%d' % (cfg_key, getattr(mcfg, cfg_key, None), r))
            setattr(mcfg, cfg_key, int(r))

    recorded, hps_path = recorded_model_cfg(ckpt)
    if recorded is not None and 'lora_alpha' in recorded:
        alpha = int(recorded['lora_alpha'])
        if int(getattr(mcfg, 'lora_alpha', 16)) != alpha:
            changed.append('lora_alpha %s->%d' % (getattr(mcfg, 'lora_alpha', None), alpha))
            setattr(mcfg, 'lora_alpha', alpha)
        for extra in ('lora_dropout', 'lora_freeze_multimodal'):
            if extra in recorded:
                setattr(mcfg, extra, recorded[extra])
    elif changed:
        # the rank had to be corrected, so this config was written for a different arm --
        # and without the recorded alpha the scaling alpha/r cannot be reconstructed. Loading
        # anyway would succeed and silently evaluate at the wrong adapter strength.
        raise RuntimeError(
            'LoRA geometry mismatch and no recorded config to resolve it.\n'
            '  checkpoint : %s\n'
            '  inferred   : %s\n'
            '  config says: %s\n'
            '  hps.json   : %s\n'
            'The rank can be read from the checkpoint but lora_alpha cannot, and LoRALinear\n'
            'scales by alpha/r -- guessing it would produce plausible numbers from the wrong\n'
            'model. Point the eval at the arm whose log/hps.json is intact, or set the LoRA\n'
            'fields in the eval config to the values that arm trained with.'
            % (ckpt, ', '.join('%s=%d' % (k, v) for k, v in sorted(ranks.items())),
               ', '.join('%s=%s' % (k, getattr(mcfg, k, None))
                         for _, k in ENCODER_RANK_KEY) + ', lora_alpha=%s'
               % getattr(mcfg, 'lora_alpha', None),
               hps_path or 'not found'))

    if changed and logger is not None:
        logger.info('[LoRA] geometry taken from the checkpoint: %s' % ', '.join(changed))
    elif changed:
        print('[LoRA] geometry taken from the checkpoint: %s' % ', '.join(changed))
