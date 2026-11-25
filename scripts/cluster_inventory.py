#!/usr/bin/env python3
"""Read-only inventory of everything the SCALAR experiments need on the cluster.

    source /leonardo_work/AIFAC_S07_041/sca_env.rc
    python3 scripts/cluster_inventory.py          # prints and writes experiments/results/harvest/inventory.txt

Each line is OK, MISSING or WARN. Nothing is created, moved or deleted except the report.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(ROOT, 'experiments/results/harvest/inventory.txt')
BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
ARMS = ('t9_qweight_only', 's1_t9_seed51', 's2_t9_seed52')
VIDEO_EXT = ('.mp4', '.mkv', '.webm', '.avi', '.mov')
lines = []


def say(status, what, detail=''):
    lines.append('%-7s %s%s' % (status, what, ('  ' + detail) if detail else ''))


def ev(p):
    return os.path.expandvars(p)


def check_path(what, p, kind='any'):
    p = ev(p)
    ok = os.path.isdir(p) if kind == 'dir' else os.path.isfile(p) if kind == 'file' else os.path.exists(p)
    size = ''
    if ok and os.path.isfile(p):
        size = '%.2f GB' % (os.path.getsize(p) / 1e9)
    say('OK' if ok else 'MISSING', what, '%s %s' % (p, size))
    return ok


def run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=ROOT)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'


def section(title):
    lines.append('')
    lines.append('== ' + title)


def stems(d):
    try:
        return {os.path.splitext(e.name)[0] for e in os.scandir(d)}
    except OSError:
        return None


def coverage(name, txt, vision, audio):
    """Share of annotation entries whose video / audio file exists."""
    txt, vision, audio = ev(txt), ev(vision) if vision else None, ev(audio) if audio else None
    if not txt.startswith('/'):
        txt = os.path.join(ROOT, txt)
    if not os.path.isfile(txt):
        say('MISSING', '%s annotation' % name, txt)
        return
    annos = json.load(open(txt))
    ids = [str(a.get('video_id', a.get('clip_id', ''))) for a in annos]
    parts = ['%d entries' % len(ids)]
    status = 'OK'
    for label, d in (('video', vision), ('audio', audio)):
        if not d:
            continue
        have = stems(d)
        if have is None:
            parts.append('%s dir MISSING (%s)' % (label, d))
            status = 'MISSING'
            continue
        n = sum(1 for i in ids if i in have)
        parts.append('%s %d/%d' % (label, n, len(ids)))
        if n < len(ids) and label == 'video':
            status = 'WARN' if n else 'MISSING'
    say(status, '%s data' % name, ', '.join(parts))


def main():
    section('code checkout')
    say('INFO', 'branch / HEAD', run('git rev-parse --abbrev-ref HEAD') + ' ' + run('git log -1 --format="%h %ci %s"'))
    dirty = run('git status --porcelain --untracked-files=no')
    say('OK' if not dirty else 'WARN', 'tracked files clean', dirty.replace('\n', ' | ')[:300])

    section('python environment (sca env must be active)')
    say('INFO', 'python3', sys.executable)
    say('OK' if not os.environ.get('VIRTUAL_ENV') else 'WARN', 'no virtualenv active', os.environ.get('VIRTUAL_ENV', ''))
    imp = run('%s -c "import torch, decord, easydict, transformers; print(torch.__version__, transformers.__version__)"' % sys.executable)
    say('OK' if 'Error' not in imp and 'Traceback' not in imp else 'MISSING', 'imports torch/decord/easydict/transformers', imp.splitlines()[-1] if imp else '')
    for tool in ('sbatch', 'squeue', 'git', 'tmux'):
        say('OK' if shutil.which(tool) else 'MISSING', 'command %s' % tool, shutil.which(tool) or '')

    section('environment variables from sca_env.rc')
    for v in ('WORK', 'FAST', 'CODE_DIR', 'WORK_ROOT', 'DATA_ROOT', 'SCA_CACHE_ROOT', 'MODELS_DIR'):
        val = os.environ.get(v)
        say('OK' if val and os.path.isdir(val) else 'MISSING', v, val or 'unset')

    section('run folders (must be on WORK, not scratch)')
    for link in ('workdir', 'workdir_pretrain', 'results', 'slurm_scripts/logs', 'pretrained_weights', 'datasets'):
        p = os.path.join(ROOT, link)
        tgt = os.path.realpath(p)
        exists = os.path.isdir(tgt)
        on_scratch = '/leonardo_scratch/' in tgt
        status = 'MISSING' if not exists else ('WARN' if on_scratch and link.startswith('workdir') else 'OK')
        say(status, link, '-> %s%s' % (tgt, '  (scratch: purged after inactivity)' if on_scratch else ''))

    section('trained SCALAR checkpoints')
    for arm in ARMS:
        ck = sorted(glob.glob(os.path.join(ROOT, 'workdir_pretrain', arm, 'ckpt', 'model_step_*.pt')))
        hps = os.path.isfile(os.path.join(ROOT, 'workdir_pretrain', arm, 'log', 'hps.json'))
        say('OK' if ck and hps else 'MISSING', arm, (os.path.basename(ck[-1]) if ck else 'no model_step_*.pt') + ('' if hps else ', no log/hps.json'))

    section('baseline checkpoints and caches')
    check_path('GRAM', '$WORK/HyperAlign/pretrained_weights/GRAM_pretrained_TVAS/ckpt/model_step_459.pt', 'file')
    check_path('VAST init', '$WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt', 'file')
    check_path('S* cache', '$SCA_CACHE_ROOT/s_star_150k.pt', 'file')
    check_path('offline model env', '$MODELS_DIR/env.sh', 'file')
    for f in ('clip/EVA01_CLIP_g_14_psz14_s11B.pt', 'beats/BEATs_iter3_plus_AS2M.pt', 'bert/bert-base-uncased'):
        check_path('encoder %s' % f.split('/')[0], os.path.join(ROOT, 'pretrained_weights', f))
    check_path('evaluation_tools', os.path.join(ROOT, 'evaluation_tools'), 'dir')
    check_path('data loader (cluster-local)', os.path.join(ROOT, 'data/IndexAnno.py'), 'file')

    section('BaryBind (step 1)')
    check_path('BaryBind code', '$WORK/barybind/run.py', 'file')
    bb = sorted(glob.glob(ev('$WORK/barybind_ckpt/*/log/hps.json')))
    bbck = sorted(glob.glob(ev('$WORK/barybind_ckpt/*/ckpt/model_step_*.pt')))
    say('OK' if bb and bbck else 'MISSING', 'BaryBind checkpoint', (os.path.dirname(os.path.dirname(bb[0])) if bb else '$WORK/barybind_ckpt/<model>/log/hps.json') + (' ' + os.path.basename(bbck[-1]) if bbck else ''))

    section('benchmark data (annotation entries with media on disk)')
    for b in BENCHES:
        cfg = os.path.join(ROOT, 'benchmark_eval/configs_qweight/sca_%s.json' % b)
        v = json.load(open(cfg))['data_cfg']['val'][0]
        coverage(b, v['txt'], v.get('vision'), v.get('audio'))

    section('pretraining data (150k)')
    cfg = json.load(open(os.path.join(ROOT, 'config/sca/pretrain_cfg/sca_paper.json')))
    for t in cfg['data_cfg']['train']:
        coverage('vast27m_150k', t['txt'], t.get('vision'), t.get('audio'))
    say('WARN' if '/leonardo_scratch/' in ev('$DATA_ROOT') else 'OK', 'DATA_ROOT location', ev('$DATA_ROOT') + ('  (scratch: purged after inactivity)' if '/leonardo_scratch/' in ev('$DATA_ROOT') else ''))

    section('storage')
    say('INFO', 'WORK usage', run('df -h "$WORK" | tail -1'))
    q = run('command -v cindata >/dev/null && cindata 2>/dev/null | head -20')
    if q:
        lines.extend('        ' + x for x in q.splitlines())

    section('Slurm')
    say('INFO', 'my jobs', run('squeue --me -h -o "%i %j %T %R" | head -30').replace('\n', ' | '))
    say('INFO', 'runner processes', run('pgrep -af ops_runner.sh').replace('\n', ' | ') or 'none')

    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + '.partial', 'w') as f:
        f.write(text + '\n')
    os.replace(OUT + '.partial', OUT)
    print(text)
    print('\nwritten to %s' % os.path.relpath(OUT, ROOT))
    return 1 if any(x.startswith('MISSING') for x in lines) else 0


if __name__ == '__main__':
    sys.exit(main())
