"""Guards for the reliability-weighted centroid."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model.centroid import masked_spherical_mean, reliability_weights  # noqa: E402


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


class TestReliabilityWeights(unittest.TestCase):
    def test_absent_modalities_never_get_weight(self):
        z = _norm(torch.randn(8, 4, 16))
        present = torch.tensor([[1., 1., 0., 1.]] * 8)
        w = reliability_weights(z, present)
        self.assertTrue(torch.allclose(w[:, 2], torch.zeros(8), atol=1e-6))
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(8), atol=1e-5))

    def test_noise_modality_is_downweighted(self):
        """Three agreeing modalities and one random: the random one must lose weight."""
        torch.manual_seed(0)
        base = _norm(torch.randn(64, 1, 16))
        agree = _norm(base + 0.05 * torch.randn(64, 3, 16))     # mutually consistent
        noise = _norm(torch.randn(64, 1, 16))                   # unrelated
        z = torch.cat([agree, noise], dim=1)
        w = reliability_weights(z, torch.ones(64, 4), tau=0.1)
        self.assertLess(w[:, 3].mean().item(), w[:, :3].mean().item(),
                        'the noise modality should receive less weight than the agreeing ones')
        self.assertLess(w[:, 3].mean().item(), 0.25,
                        'it should fall below the uniform share of 1/4')

    def test_single_modality_falls_back_to_uniform(self):
        z = _norm(torch.randn(5, 4, 16))
        present = torch.tensor([[1., 0., 0., 0.]] * 5)
        w = reliability_weights(z, present)
        self.assertTrue(torch.allclose(w[:, 0], torch.ones(5), atol=1e-6))

    def test_large_tau_recovers_the_uniform_centroid(self):
        z = _norm(torch.randn(6, 4, 16))
        present = torch.tensor([[1., 1., 1., 0.]] * 6)
        mu_uniform, a_u, n_u = masked_spherical_mean(z, present)
        mu_rel, a_r, n_r = masked_spherical_mean(z, present, weighting='reliability', tau_w=1e6)
        self.assertTrue(torch.allclose(mu_uniform, mu_rel, atol=1e-4))
        # A(M) and |M| are set properties and must not move with the weighting
        self.assertTrue(torch.allclose(a_u, a_r, atol=1e-6))
        self.assertTrue(torch.allclose(n_u, n_r, atol=1e-6))

    def test_centroid_stays_unit_norm_and_arity_invariant(self):
        z = _norm(torch.randn(4, 5, 16))
        for k in (2, 3, 4, 5):
            present = torch.zeros(4, 5)
            present[:, :k] = 1.
            mu, _, n = masked_spherical_mean(z, present, weighting='reliability')
            self.assertTrue(torch.allclose(mu.norm(dim=-1), torch.ones(4), atol=1e-5),
                            'centroid must stay on the unit sphere at every arity')
            self.assertTrue(torch.allclose(n, torch.full((4,), float(k))))


if __name__ == '__main__':
    unittest.main()
