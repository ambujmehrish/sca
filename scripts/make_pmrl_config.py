#!/usr/bin/env python3
"""Evaluate PMRL's RELEASED checkpoint in our environment, on our eval protocol.

    python3 scripts/make_pmrl_config.py --pmrl_root /path/to/pmrl --checkpoint .../pmrl_base.pt

The PMRL authors release both their code (github.com/Xiaohao-Liu/PMRL) and their trained
weights (huggingface.co/xhLiu/PMRL, model_ckpts/pmrl_base.pt), so the PMRL row is produced
exactly the way the GRAM row is: their published checkpoint, our environment, our evaluation.
Nothing is retrained and nothing is reimplemented.

WHAT THIS CHANGES AND WHY.

  val blocks     Taken from OUR benchmark_eval configs, not theirs. Their zero-shot config
                 evaluates VATEX at vision_sample_num 16 and the full VATEX test list; every
                 other row of our table uses 8 frames and our 431-video subset. Evaluating one
                 row on a different protocol is the confound the single-configuration rule
                 exists to prevent, so the protocol is ours for every row alike.

  model_type     Their default_model_cfg.json says "vast"; the released checkpoint is PMRL.
                 This is the ONE model hyperparameter the script sets, because it names which
                 method is being run -- and getting it wrong is invisible: utils/build_model.py
                 loads with strict=False, so a VAST skeleton would take the PMRL weights it
                 recognises, randomly initialise the rest, and evaluate happily.

Everything else -- itm_rerank_num, evaluation_type, contra_dim, the encoder types -- is left
as they ship it and asserted afterwards.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import absolutise, expand, resolve_audio_dir      # noqa: E402

ZEROSHOT_CFG = 'config/pmrl/finetune_cfg/retrieval-all-zeroshot.json'
# Our per-benchmark eval configs -- the same files every other row of the table is produced
# from. gram_* rather than sca_* because those are the released-checkpoint shape.
OUR_EVAL = 'benchmark_eval/configs_e1/gram_%s.json'
BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')

# Left exactly as they ship it, and checked after the rewrite.
FROZEN_MODEL = ('itm_rerank_num', 'evaluation_type', 'contra_dim', 'vision_encoder_type',
                'audio_encoder_type', 'vision_resolution', 'max_caption_len',
                'max_subtitle_len', 'frame_embedding_type')


def argparse_defaults(args_py, names):
    """Read default values straight out of THEIR utils/args.py.

    model/pmrl.py requires tau1, tau2 and lambda_itm, and no config they ship defines them --
    utils/args.py copies them into model_cfg only when passed as --flags, so their own runs
    supplied them there. Rather than typing numbers from a paper, the values come from their
    own parser, and a missing or unparsable one is fatal: an invented hyperparameter in a row
    labelled "authors' released model" is exactly the thing this whole setup exists to avoid.
    """
    src = open(args_py).read()
    out = {}
    for n in names:
        m = re.search(r"add_argument\(\s*['\"]--%s['\"][^)]*default\s*=\s*([0-9.eE+-]+)" % n, src)
        if not m:
            sys.exit('FATAL: no argparse default for --%s in %s. model/pmrl.py requires it and\n'
                     '       no shipped config defines it, so there is no value that is theirs\n'
                     '       rather than ours. Do not guess one.' % (n, args_py))
        out[n] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pmrl_root', required=True, help='checkout of github.com/Xiaohao-Liu/PMRL')
    ap.add_argument('--checkpoint', required=True,
                    help='pmrl_base.pt from huggingface.co/xhLiu/PMRL')
    ap.add_argument('--data_root', default=os.environ.get('DATA_ROOT', ''))
    ap.add_argument('--bench', required=True, choices=BENCHES)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if not args.data_root:
        sys.exit('FATAL: pass --data_root or export DATA_ROOT')
    root = args.pmrl_root
    src = os.path.join(root, ZEROSHOT_CFG)
    if not os.path.exists(src):
        sys.exit('FATAL: %s not found -- is --pmrl_root a PMRL checkout?' % src)
    if not os.path.exists(args.checkpoint):
        sys.exit('FATAL: checkpoint %s not found. Fetch it on a LOGIN node with:\n'
                 '  HF_HUB_OFFLINE=0 huggingface-cli download xhLiu/PMRL '
                 'model_ckpts/pmrl_base.pt \\\n'
                 '      --local-dir $WORK_ROOT/pmrl_weights\n'
                 '  (HF_HUB_OFFLINE=0 is required: $MODELS_DIR/env.sh sets it to 1 for the\n'
                 '   compute nodes, and the download then fails as LocalEntryNotFoundError,\n'
                 '   which reads like a network fault rather than offline mode.)'
                 % args.checkpoint)

    code_dir = os.environ.get('CODE_DIR') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..')

    cfg = json.load(open(src), object_pairs_hook=collections.OrderedDict)
    before_model = dict(cfg['model_cfg'])

    # ---- the val block comes from OUR config for this benchmark
    ours_path = os.path.join(code_dir, OUR_EVAL % args.bench)
    if not os.path.exists(ours_path):
        sys.exit('FATAL: %s not found -- it defines the protocol every other row of the table '
                 'is evaluated on, so there is nothing to match.' % ours_path)
    ours = json.load(open(ours_path))
    val = expand(ours['data_cfg']['val'], args.data_root)
    if not val:
        sys.exit('FATAL: %s has no val block.' % ours_path)
    for d in val:
        d['txt'] = absolutise(d['txt'], code_dir, 'val annotation')
        if 'audio' in d:
            d['audio'] = resolve_audio_dir(d['audio'], d['txt'], d.get('name', args.bench))
        if 'vision' in d and not os.path.isdir(d['vision']):
            sys.exit('FATAL: vision directory %s does not exist for %s.'
                     % (d['vision'], d.get('name')))
    cfg['data_cfg']['val'] = val
    # Their config also carries a train block; testing mode never touches it, but leaving a
    # block that names /home/storage/PMRL/... invites a later run to build it. Point it at the
    # same thing as val, marked non-training, so nothing in the file references a path that
    # does not exist on this machine.
    cfg['data_cfg']['train'] = json.loads(json.dumps(val))

    # ---- run in testing mode against the released checkpoint
    cfg['run_cfg']['mode'] = 'testing'
    cfg['run_cfg']['checkpoint'] = os.path.abspath(args.checkpoint)
    cfg['run_cfg']['zero_shot'] = True
    cfg['run_cfg']['output_dir'] = 'output_pmrl_released_%s' % args.bench
    # run.py:28 reads run_cfg.log_name for wandb.init(name=...), and their released
    # default_run_cfg.json does not define it -- their own experiment configs must have. It is
    # a logging label and nothing else: no code path reads it again. Supplied here so the run
    # does not die at line 28 having loaded a 5.6 GB checkpoint. Every other unset attribute
    # their code touches (valid_steps, beam_size, prompt, ...) is either commented out in
    # utils/args.py or lives on the training path, which mode=testing never enters.
    cfg['run_cfg']['log_name'] = 'pmrl_released_%s' % args.bench
    # Their configs inherit from ./config/vast/, a directory the release does not ship. The
    # launcher links config/vast -> config/pmrl, whose default_model_cfg.json IS the VAST
    # default (it still says model_type "vast"), so the inheritance resolves to what they meant.

    # ---- model_type: the one model key this script sets, and the one that fails silently
    cfg['model_cfg']['model_type'] = 'pmrl'

    # ---- tau1 / tau2 / lambda_itm
    #
    # model/pmrl.py:51-53 reads these and NO shipped config defines them: utils/args.py:59
    # copies them into model_cfg only when they appear as --flags on the command line, so
    # their own runs passed them there. The values are taken from their argparse defaults
    # (utils/args.py:320-322) rather than chosen by us, and re-read from their source each
    # time so this tracks their repo instead of freezing a number we typed.
    #
    # They cannot affect this row's numbers in any case. Every use is inside a loss:
    #   pmrl.py:413,416   eigenvalues / tau1
    #   pmrl.py:422       cross_entropy(... / tau2)
    #   pmrl.py:476       lambda_itm * loss
    # Nothing on the scoring path reads them, and mode=testing computes no loss at all.
    author_defaults = argparse_defaults(os.path.join(root, 'utils/args.py'),
                                        ('tau1', 'tau2', 'lambda_itm'))
    cfg['model_cfg'].update(author_defaults)

    for k in FROZEN_MODEL:
        if k in before_model and before_model[k] != cfg['model_cfg'].get(k):
            sys.exit('FATAL: model_cfg.%s changed (%r -> %r). Only model_type may be set.'
                     % (k, before_model[k], cfg['model_cfg'].get(k)))

    cfg['_repro_note'] = {
        'source_config': ZEROSHOT_CFG,
        'checkpoint': os.path.abspath(args.checkpoint),
        'provenance': ('PMRL as released: their code (github.com/Xiaohao-Liu/PMRL) and their '
                       'published weights (huggingface.co/xhLiu/PMRL, model_ckpts/pmrl_base.pt). '
                       'Nothing retrained, nothing reimplemented.'),
        'evaluation_protocol': ('OUR %s, so this row is measured exactly like every other row '
                                '-- same annotation file, frame count, task string and rerank '
                                'depth. Their own zero-shot config differs (VATEX at 16 frames '
                                'and the full test list); using it would make this the one row '
                                'evaluated differently.' % (OUR_EVAL % args.bench)),
        'model_type': ('set to pmrl; their default_model_cfg says "vast" and build_model loads '
                       'with strict=False, so leaving it would have evaluated a partly random '
                       'model without any error'),
        'benchmark': args.bench,
        'loss_hyperparameters': (
            'tau1/tau2/lambda_itm = %s, read from THEIR utils/args.py argparse defaults. '
            'model/pmrl.py:51-53 requires them and no shipped config defines them (args.py:59 '
            'copies them only from --flags). Every use is inside a loss term '
            '(pmrl.py:413,416,422,476); nothing on the scoring path reads them and mode=testing '
            'computes no loss, so they cannot affect this row.' % author_defaults),
        'log_name': ('supplied because run.py:28 reads run_cfg.log_name for the wandb run '
                     'name and their released default_run_cfg.json omits it. A logging label '
                     'only -- no other code path reads it.'),
    }

    out = args.out or 'config/pmrl/finetune_cfg/repro_released_%s.json' % args.bench
    dst = os.path.join(root, out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')

    print('wrote %s' % dst)
    print('  bench      : %s' % args.bench)
    print('  checkpoint : %s' % cfg['run_cfg']['checkpoint'])
    print('  model_type : %s' % cfg['model_cfg']['model_type'])
    for d in val:
        print('  val %-16s txt=%s' % (d.get('name'), d['txt']))
        print('      %-16s vsn=%s asn=%s task=%s rerank=%s'
              % ('', d.get('vision_sample_num'), d.get('audio_sample_num'), d.get('task'),
                 cfg['model_cfg'].get('itm_rerank_num')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
