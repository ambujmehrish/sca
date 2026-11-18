"""Protocol-parity and reranker-swap plumbing: external candidates, frame-set configs, summary."""
import importlib.util
import json
import os
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def _dump(path, task='tvas', ids=('a', 'b', 'c'), ids_txt=('a', 'b', 'c'), dual=None):
    n_t, n_v = len(ids_txt), len(ids)
    torch.save({'task': task, 'dual': dual if dual is not None else torch.randn(n_t, n_v),
                'itm_fwd': torch.zeros(n_t, n_v), 'itm_bwd': torch.zeros(n_t, n_v),
                'ids': list(ids), 'ids_txt': list(ids_txt), 'itm_rerank_num': 2,
                'global_step': 0}, path)


def test_external_candidates_off_by_default(monkeypatch):
    from evaluation.evaluation_mm import _external_candidates
    monkeypatch.delenv('SCA_EXTERNAL_CANDIDATES', raising=False)
    assert _external_candidates('tvas', ['a'], ['a'], torch.zeros(1, 1)) is None


def test_external_candidates_returns_the_dumped_score(tmp_path, monkeypatch):
    from evaluation.evaluation_mm import _external_candidates
    dual = torch.tensor([[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.0, 0.2, 0.7]])
    p = tmp_path / 'rerank_tvas_step0.pt'
    _dump(p, dual=dual)
    monkeypatch.setenv('SCA_EXTERNAL_CANDIDATES', str(p))
    got = _external_candidates('tvas', ['a', 'b', 'c'], ['a', 'b', 'c'], torch.zeros(3, 3))
    assert torch.equal(got, dual)


@pytest.mark.parametrize('kw, ids, ids_txt, task', [
    ({}, ['a', 'c', 'b'], ['a', 'b', 'c'], 'tvas'),        # gallery order differs
    ({}, ['a', 'b', 'c'], ['b', 'a', 'c'], 'tvas'),        # query order differs
    ({}, ['a', 'b', 'c'], ['a', 'b', 'c'], 'tva'),         # different task
])
def test_external_candidates_refuse_to_realign(tmp_path, monkeypatch, kw, ids, ids_txt, task):
    from evaluation.evaluation_mm import _external_candidates
    p = tmp_path / 'd.pt'
    _dump(p, **kw)
    monkeypatch.setenv('SCA_EXTERNAL_CANDIDATES', str(p))
    with pytest.raises(RuntimeError):
        _external_candidates(task, ids, ids_txt, torch.zeros(3, 3))


def test_external_candidates_missing_file_raises(tmp_path, monkeypatch):
    from evaluation.evaluation_mm import _external_candidates
    monkeypatch.setenv('SCA_EXTERNAL_CANDIDATES', str(tmp_path / 'nope.pt'))
    with pytest.raises(FileNotFoundError):
        _external_candidates('tvas', ['a'], ['a'], torch.zeros(1, 1))


def test_external_candidates_choose_which_clips_get_reranked(tmp_path, monkeypatch):
    """The external matrix only decides the top-k; the score written is the reranker's."""
    from evaluation import evaluation_mm as em

    class Cfg:
        sca_itm_qw_gamma = 0.0

    class M:
        config = Cfg()

        def compute_slice_scores(self, cond, ids, mask):
            return cond[:, 0, 0] + ids[:, 0].float() * 0.01

    monkeypatch.setattr(em.dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(em.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(em, 'all_gather_list', lambda x: [x])
    monkeypatch.setattr(em, 'ddp_allgather', lambda x: x)
    cond = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    input_ids = torch.arange(3).view(3, 1)
    mask = torch.ones(3, 1)
    ext = torch.tensor([[0.0, 1.0, 0.9], [1.0, 0.0, 0.9], [1.0, 0.9, 0.0]])
    out = em.refine_score_matrix(cond, input_ids, mask, ext, M(), 2, direction='forward')
    assert (out[0, 0] == 0) and (out[1, 1] == 0) and (out[2, 2] == 0)
    assert torch.isclose(out[0, 1], torch.tensor(2.0)) and torch.isclose(out[2, 0], torch.tensor(1.02))


def _mpc():
    spec = importlib.util.spec_from_file_location('mpc', os.path.join(ROOT, 'scripts/make_parity_configs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('model', ['sca', 'gram'])
@pytest.mark.parametrize('bench', ['msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps'])
def test_fsbb_configs_change_only_test_frames(model, bench):
    mpc = _mpc()
    cfg = mpc.ours_config(model, bench, 'fsbb')
    v = cfg['data_cfg']['val'][0]
    assert v['vision_sample_num'] == mpc.FRAMESETS['fsbb'][bench]
    assert cfg['data_cfg']['train'][0]['vision_sample_num'] == 8
    committed = json.load(open(os.path.join(ROOT, 'benchmark_eval/configs_fsbb/%s_%s.json' % (model, bench))))
    assert committed == cfg, 'committed configs_fsbb is stale; rerun make_parity_configs.py ours'


def test_barybind_configs_use_our_data_and_their_model_defaults(monkeypatch):
    monkeypatch.setenv('DATA_ROOT', '/data')
    monkeypatch.setenv('SCA_CACHE_ROOT', '/cache')
    mpc = _mpc()
    for fs, frames in mpc.FRAMESETS.items():
        for b in mpc.BENCHES:
            cfg = mpc.barybind_config(b, fs)
            ours = mpc.load(mpc.SOURCES['sca'] % b)['data_cfg']['val'][0]
            v = cfg['data_cfg']['val'][0]
            assert cfg['model_cfg']['default'] == './config/barybind/default_model_cfg.json'
            assert cfg['model_cfg']['itm_rerank_num'] == 50
            assert v['task'] == ours['task'] and v['audio_sample_num'] == ours['audio_sample_num']
            assert v['vision_sample_num'] == frames[b]
            assert os.path.isabs(v['txt']) and v['txt'].endswith(ours['txt'])
            assert '$' not in json.dumps(cfg)
            assert 'epoch' in cfg['data_cfg']['train'][0]


def test_barybind_configs_refuse_unset_data_root(monkeypatch):
    monkeypatch.delenv('DATA_ROOT', raising=False)
    with pytest.raises(KeyError):
        _mpc().barybind_config('msrvtt', 'fs8')


def test_summary_prints_missing_for_absent_cells(tmp_path, monkeypatch, capsys):
    import summarize_parity as sp
    monkeypatch.setattr(sp, 'ROOT', str(tmp_path))
    assert sp.main.__call__ and sp.cell(str(tmp_path / 'none')) == {}
    monkeypatch.setattr(sys, 'argv', ['summarize_parity.py'])
    assert sp.main() == 1
    assert 'MISSING' in capsys.readouterr().out


def test_summary_reads_ours_and_barybind_log_dialects(tmp_path):
    import summarize_parity as sp
    ours = tmp_path / 'own' / 'log'
    ours.mkdir(parents=True)
    (ours / 'log.txt').write_text(
        "==== evaluation--ret%tvas--msrvtt_ret_ret_itm_area========\n"
        "{'volume_ITM_T2D_r1': 54.8, 'volume_ITM_T2D_recall': '54.8/78.0/86.0', 'volume_ITM_D2T_r1': 50.0}\n"
        "==== evaluation--ret%tvas--msrvtt_ret_ret_area_forward========\n"
        "{'volume_T2D_r1': 45.2, 'volume_T2D_recall': '45.2/70.0/80.0'}\n"
        "==== evaluation--ret%tvas--msrvtt_ret_ret_itm_ext========\n"
        "{'volume_ITM_T2D_r1': 55.9, 'volume_ITM_T2D_recall': '55.9/79.0/87.0'}\n")
    got = sp.cell(str(tmp_path / 'own'))
    assert got == {'itm': 54.8, 'stage1': 45.2, 'ext': 55.9}
