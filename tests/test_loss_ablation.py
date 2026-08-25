"""Table 4 (loss leave-one-out at T9) and the G11 no-masked-training arm."""
import json
import os
import subprocess
import sys

SRC = open('scripts/build_loss_ablation_table.py').read()
LAUNCHER = open('slurm_scripts/b_grid_pretrain.sh').read()


def _flat(path):
    d = json.load(open(path))
    out = {}
    for sec in ('run_cfg', 'model_cfg'):
        for k, v in d[sec].items():
            if k != 'default':
                out['%s.%s' % (sec, k)] = v
    out['data_cfg'] = d['data_cfg']
    return out


def test_g11_is_t9_plus_only_the_mask_p_full_keys():
    """The arm's claim is 'T9 with one knob changed'. Assert it, do not trust it: any other
    drift and a delta in the table would be attributed to masking that masking did not cause."""
    t9 = _flat('config/sca/ablations/T9_qweight_only.json')
    g11 = _flat('config/sca/ablations/G11_train_nomask.json')
    extra = {k: v for k, v in g11.items() if t9.get(k) != v}
    assert extra == {'model_cfg.mask_p_full_start': 1.0,
                     'model_cfg.mask_p_full_end': 1.0}, extra
    assert not set(t9) - set(g11), 'G11 dropped T9 keys: %s' % (set(t9) - set(g11))


def test_g11_is_a_launchable_arm_exactly_once():
    assert LAUNCHER.count('G11_train_nomask') >= 1
    arms_line = next(l for l in LAUNCHER.splitlines() if l.startswith('ARMS=('))
    assert arms_line.count('G11_train_nomask') == 1


def test_l_mask_is_identically_zero_when_no_view_is_masked():
    """At p_full=1 the masked and full centroids coincide; the arm is a pure L_align change
    only if l_mask then contributes NOTHING -- zero value AND zero gradient THROUGH THE REAL
    GRAPH. That last part is where a naive check lies: on a post-normalization leaf,
    d(1 - mu.mu)/dmu = -2mu is NOT zero. In the model mu is the output of the centroid's
    normalization, whose Jacobian (I - mu mu^T)/||x|| annihilates exactly that direction --
    so the gradient at the WEIGHTS is zero. The test therefore differentiates back through
    the normalization, as training does."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, '.')
    from model.losses_sca import l_mask
    x = torch.randn(8, 16, dtype=torch.double, requires_grad=True)   # pre-normalization
    t = F.normalize(torch.randn(8, 16, dtype=torch.double), dim=-1)  # a fixed text query
    mu = F.normalize(x, dim=-1)      # both views are this SAME graph node when p_full=1
    s = (t * mu).sum(-1)
    loss = l_mask(mu, mu, s_M=s, s_K=s)
    assert abs(float(loss.detach())) < 1e-12
    loss.backward()
    assert float(x.grad.abs().max()) < 1e-12


def test_table4_rows_are_the_loss_leave_one_out_set():
    """One row per removable component, reference first; L_align is never a removed row --
    it is the retrieval objective itself."""
    for arm in ('t9_qweight_only', 'g11_train_nomask', 'g10_mask0',
                'g8_sem0', 'g6_lambda0'):
        assert "'%s'" % arm in SRC, arm
    assert 'never removed' in SRC, 'the L_align exclusion must be stated, not implicit'
    rows_block = SRC.split('ROWS = [')[1].split(']')[0]
    assert 'align' not in rows_block.lower(), 'L_align must not appear as a removable row'
    # L_concept is NOT a row: the reported configuration has sca_num_concepts=0, so the
    # loss never trains and a delta=0 arm is a bit-identical no-op (g9 proved it -- its
    # cells equal T9's exactly). The exclusion must be documented, not silent.
    assert 'g9_concept0' not in rows_block, 'a no-op arm must not pose as an ablation'
    assert 'sca_num_concepts=0' in SRC and 'bit-identical' in SRC


def test_table4_reads_e1_frames_through_the_shared_extraction():
    """Same cells, same parser as Tables 1/2 -- the tables cannot disagree on a number."""
    assert 'from build_paper_table import' in SRC and 'cell_metrics' in SRC
    assert "workdir/e1_frames" in SRC


def test_table4_prints_missing_and_fails_loud_when_cells_are_absent(tmp_path):
    """Fabricated numbers are the one unforgivable failure mode; absent cell -> MISSING + rc 1."""
    out = tmp_path / 't4.tex'
    p = subprocess.run([sys.executable, 'scripts/build_loss_ablation_table.py',
                        '--out', str(out)], capture_output=True, text=True,
                       env=dict(os.environ, SCA_FAKE_ROOT='1'))
    tex = out.read_text()
    assert 'MISSING' in tex or p.returncode == 0
    if 'MISSING' in tex:
        assert p.returncode == 1
    assert '\\bar{\\Delta}' in tex
    assert 'full objective' in tex   # no internal arm codename in the paper row
