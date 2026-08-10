#!/usr/bin/env python3
"""A10: calibration-mechanism ablation on k=2 (plan §5, run FIRST -- decides the E6 headline
config before any 4xA100 run).

Three arms, on real Flickr8k features from a frozen CLIP (see a10_prepare_flickr8k.py):

  kl_only       : L_sem = KL only, LEARNABLE tau            (calibration='none')
  kl_regression : KL + cal_w*||S-(2S*-1)||^2, FIXED tau     (calibration='regression')
  fixed_tau     : KL only with tau frozen AT tau*           (calibration='fixed_tau')

Every arm trains the same two projection heads (text/image, the Stage-0 stand-in for the
contra heads) with L_align + alpha*L_sem, using the SHIPPED implementation verbatim:
model/losses_sca.py for the losses (incl. check_calibration_config guard),
model/centroid.py for the gallery representation (|M|=1 branch at k=2),
data/semantic_targets.py for strict S* gathering,
evaluation/eval_calibration.py + eval_missing.py for the metrics.

Metrics per arm (x seeds): S-vs-S* R^2 / Pearson / slope / intercept, graded nDCG@10,
T->I R@1/5/10 (all 5 captions as queries).

    python3 experiments/a10_calibration_sweep.py --workdir experiments/a10_workdir \
        --out experiments/results/a10_flickr8k_k2.json
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.centroid import masked_spherical_mean
from model.losses_sca import l_align, l_sem, check_calibration_config
from data.semantic_targets import SemanticTargets
from evaluation.eval_calibration import calibration_regression, graded_ndcg
from evaluation.eval_missing import recall_at_k

ARMS = {
    # name             calibration    tau_learnable  tau_init            cal_w
    'kl_only':       dict(calibration='none',       tau_learnable=True,  tau_init=0.07, cal_w=0.0),
    'kl_regression': dict(calibration='regression', tau_learnable=False, tau_init=0.07, cal_w=1.0),
    'fixed_tau':     dict(calibration='fixed_tau',  tau_learnable=False, tau_init=None, cal_w=0.0),
}


def train_arm(feats, st_train, arm, seed, tau_star, alpha, steps, warmup, bs, lr, dim, device):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    cfg = ARMS[arm]
    check_calibration_config(cfg['calibration'], cfg['tau_learnable'])   # the shipped guard

    img, txt, ids = feats['img'].to(device), feats['txt'].to(device), feats['ids']
    n, d_in = img.shape
    head_t = torch.nn.Linear(d_in, dim, bias=False).to(device)
    head_v = torch.nn.Linear(d_in, dim, bias=False).to(device)

    tau_init = cfg['tau_init'] if cfg['tau_init'] is not None else tau_star  # fixed_tau: tau == tau*
    params = list(head_t.parameters()) + list(head_v.parameters())
    if cfg['tau_learnable']:
        tau_param = torch.nn.Parameter(torch.tensor(float(tau_init), device=device))
        params.append(tau_param)
    else:
        tau_param = torch.tensor(float(tau_init), device=device)

    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        idx = torch.randperm(n, generator=g)[:bs]
        cap = torch.randint(txt.shape[1], (bs,), generator=g)            # one of the 5 captions
        feat_t = F.normalize(head_t(txt[idx, cap]), dim=-1)
        z = head_v(img[idx]).unsqueeze(1)                                # (B, 1, dim): k=2 gallery
        mu, _, _ = masked_spherical_mean(z, None)                        # real centroid path
        tau = tau_param.clamp(min=1e-3)
        targets = torch.arange(bs, device=device)
        loss = l_align(feat_t, mu, mu, feat_t, tau, targets)
        if step >= warmup and alpha > 0:                                 # L_align-only warmup
            s_star = st_train.gather([ids[i] for i in idx.tolist()], device=device)
            loss = loss + alpha * l_sem(feat_t @ mu.T, s_star, tau, tau_star=tau_star,
                                        calibration=cfg['calibration'], cal_w=cfg['cal_w'])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head_t, head_v, float(tau_param.detach())


@torch.no_grad()
def evaluate_arm(head_t, head_v, feats_test, s_star_test, device):
    img, txt = feats_test['img'].to(device), feats_test['txt'].to(device)
    n, n_cap, _ = txt.shape
    mu, _, _ = masked_spherical_mean(head_v(img).unsqueeze(1), None)
    t0 = F.normalize(head_t(txt[:, 0]), dim=-1)                          # caption_0 = S* rows
    sim0 = t0 @ mu.T
    out = calibration_regression(sim0, s_star_test)['overall']           # k=2: one cardinality
    out['graded_ndcg@10'] = graded_ndcg(sim0, s_star_test, k=10)
    t_all = F.normalize(head_t(txt.reshape(n * n_cap, -1)), dim=-1)      # all 5 captions query
    dist = 1.0 - t_all @ mu.T                                            # rows image-major
    gt = torch.arange(n, device=device).repeat_interleave(n_cap)         # row i*5+c -> image i
    rec, _ = recall_at_k(dist, gt)
    out.update(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default='experiments/a10_workdir')
    ap.add_argument('--out', default='experiments/results/a10_flickr8k_k2.json')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--tau_star', type=float, default=0.5)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feats_tr = torch.load(os.path.join(args.workdir, 'features_train.pt'), map_location=device)
    feats_te = torch.load(os.path.join(args.workdir, 'features_test.pt'), map_location=device)
    st_train = SemanticTargets(os.path.join(args.workdir, 's_star_train.pt'))
    st_test = SemanticTargets(os.path.join(args.workdir, 's_star_test.pt'))
    s_star_test = st_test.gather(feats_te['ids'], device=device)         # dense (1000, 1000)

    results = {}
    for arm in ARMS:
        runs = []
        for seed in args.seeds:
            head_t, head_v, tau_final = train_arm(
                feats_tr, st_train, arm, seed, args.tau_star, args.alpha,
                args.steps, args.warmup, args.bs, args.lr, args.dim, device)
            log = evaluate_arm(head_t, head_v, feats_te, s_star_test, device)
            log['tau_final'] = tau_final
            runs.append(log)
            print(f"[{arm} seed={seed}] R2={log['r2']:.4f} slope={log['slope']:.3f} "
                  f"nDCG={log['graded_ndcg@10']:.4f} R@1={log['R@1']:.2f} "
                  f"tau_end={tau_final:.4f}", flush=True)
        keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
        results[arm] = {
            'runs': runs,
            'mean': {k: sum(r[k] for r in runs) / len(runs) for k in keys},
            'std': {k: (sum((r[k] - sum(x[k] for x in runs) / len(runs)) ** 2
                           for r in runs) / len(runs)) ** 0.5 for k in keys},
        }

    results['setup'] = {'dataset': 'jxie/flickr8k (real)', 'encoder': 'frozen CLIP ViT-B/32',
                        'k': 2, 'seeds': args.seeds, 'steps': args.steps, 'warmup': args.warmup,
                        'bs': args.bs, 'lr': args.lr, 'dim': args.dim, 'alpha': args.alpha,
                        'tau_star': args.tau_star,
                        's_star': {'model': st_train.meta.get('model_name'),
                                   'tau_star': st_train.meta.get('tau_star'),
                                   'topk': st_train.meta.get('topk')}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)

    print('\n| arm | R2 | Pearson | slope | intercept | nDCG@10 | R@1 | R@5 | R@10 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for arm in ARMS:
        m, s = results[arm]['mean'], results[arm]['std']
        print(f"| {arm} | {m['r2']:.4f}±{s['r2']:.4f} | {m['pearson']:.4f} | "
              f"{m['slope']:.3f} | {m['intercept']:.3f} | "
              f"{m['graded_ndcg@10']:.4f}±{s['graded_ndcg@10']:.4f} | "
              f"{m['R@1']:.2f}±{s['R@1']:.2f} | {m['R@5']:.2f} | {m['R@10']:.2f} |")
    print(f'\nresults -> {args.out}')


if __name__ == '__main__':
    main()
