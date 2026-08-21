"""End-to-end test of the temporal scorer on a real dump file.

The unit tests for softmax_over_frames passed while report() still crashed on the first
real run: the text count is not a multiple of the chunk size (4917 texts, chunk 64 -> a
ragged final chunk of 53), and torch.stack on unequal chunks raises. So this drives the
whole entry point over a written dump, with a deliberately ragged text count."""
import os
import subprocess
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _dump(path, Nt, Ng, F=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    zf = torch.randn(Ng, F, d, generator=g)
    zf = zf / zf.norm(dim=-1, keepdim=True)
    feat_t = torch.randn(Nt, d, generator=g)
    feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
    torch.save({'feat_t': feat_t, 'v_frames': zf,
                'gallery': {'v': zf.mean(1)}, 'ids': ['c%d' % (i % Ng) for i in range(max(Nt, Ng))],
                'meta': {'task': 'ret%tva'}}, path)


def test_report_runs_on_a_ragged_text_count(tmp_path):
    """133 texts against chunk 64 leaves a final chunk of 5 -- the shape that crashed."""
    p = tmp_path / 'sca_ragged.pt'
    _dump(str(p), Nt=133, Ng=133)
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts/try_temporal_centroid.py'),
                          str(p), '--taus', '0.05', '0.5'],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'pooled (current)' in out.stdout
    assert 'max over frames' in out.stdout
    assert 'best tau' in out.stdout


def test_report_runs_when_texts_are_an_exact_multiple_of_the_chunk(tmp_path):
    p = tmp_path / 'sca_exact.pt'
    _dump(str(p), Nt=128, Ng=128)
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts/try_temporal_centroid.py'),
                          str(p), '--taus', '0.1'], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr


def test_missing_frame_features_is_reported_not_crashed(tmp_path):
    p = tmp_path / 'sca_noframes.pt'
    torch.save({'feat_t': torch.randn(8, 16), 'gallery': {'v': torch.randn(8, 16)},
                'ids': ['c%d' % i for i in range(8)], 'meta': {'task': 'ret%tva'}}, p)
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts/try_temporal_centroid.py'),
                          str(p)], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'NO per-frame features' in out.stdout
