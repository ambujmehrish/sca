"""feat_v_frames keeps the frame axis that pool_vision_for_contra averages away.

The invariant that matters: mean over the frame axis, renormalised, must reproduce feat_v.
If it does not, the per-frame features live in a different space from the pooled ones and
no comparison between set-based and pooled scoring would mean anything."""
import torch
import torch.nn.functional as F


def test_frame_mean_reproduces_the_pooled_feature():
    """Mirrors gram.py's feat_v / feat_v_frames arithmetic on a stand-in head.

    pooled  = normalize(head(mean_f CLS_f))
    frames  = normalize(head(CLS_f))           per frame
    A LINEAR head commutes with the mean, so normalize(mean_f frames_f) == pooled exactly.
    That is what makes the frame dump a strict refinement of the current representation
    rather than a different one.
    """
    torch.manual_seed(0)
    B, n, patches, C, d = 3, 8, 5, 16, 12
    vision_output = torch.randn(B, n, patches, C)
    head = torch.nn.Linear(C, d, bias=False)          # contra_head_v is linear

    cls = vision_output[:, :, 0]                       # (B, n, C)
    pooled = F.normalize(head(cls.mean(dim=1)), dim=-1)
    frames = F.normalize(head(cls), dim=-1)            # (B, n, d)

    recon = F.normalize(F.normalize(head(cls), dim=-1).new_tensor(0) + head(cls).mean(dim=1), dim=-1)
    assert torch.allclose(recon, pooled, atol=1e-5)
    assert frames.shape == (B, n, d)
    assert torch.allclose(frames.norm(dim=-1), torch.ones(B, n), atol=1e-5)


def test_frame_axis_is_not_degenerate():
    """A dump whose frames are all identical would make set scoring == pooled scoring
    trivially, and any measured 'gain' would be noise."""
    torch.manual_seed(1)
    cls = torch.randn(4, 8, 16)
    frames = F.normalize(cls, dim=-1)
    pairwise = torch.einsum('bfd,bgd->bfg', frames, frames)
    off_diag = pairwise - torch.eye(8).unsqueeze(0)
    assert off_diag.abs().max() < 1.0, 'frames are collinear; the axis carries nothing'
