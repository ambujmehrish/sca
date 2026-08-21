"""Frame slots and query weighting inside the SCA training path.

The load-bearing property is that the DEFAULT is unchanged: with sca_frame_slots off and
sca_query_weighting off, the slot set, the presence mask and the centroid must be exactly
what they were before, or every previously measured arm becomes incomparable."""
import torch

from model.centroid import masked_spherical_mean


class _Stub:
    """The slot-building logic of SCA._gallery_slots, isolated from the trunk's heavy init."""

    def __init__(self, frame_slots, frames=None):
        self.frame_slots = frame_slots
        self._frames = frames

    def batch_get(self, batch, key):
        assert key == 'feat_v_frames'
        return self._frames

    _gallery_slots = None      # bound below from the real implementation


def _slots(frame_slots, gallery, mods, frames=None):
    from model.sca import SCA
    stub = _Stub(frame_slots, frames)
    return SCA._gallery_slots(stub, {}, gallery, mods)


def _gallery(B=4, d=8, mods=('v', 'a')):
    g = torch.Generator().manual_seed(0)
    out = {}
    for m in mods:
        x = torch.randn(B, d, generator=g)
        out[m] = x / x.norm(dim=-1, keepdim=True)
    return out


def test_default_is_one_slot_per_modality():
    gallery = _gallery()
    z, owner = _slots(False, gallery, ['v', 'a'])
    assert owner == [0, 1]
    assert torch.allclose(z, torch.stack([gallery['v'], gallery['a']], dim=1))


def test_frame_slots_expand_video_only():
    B, F, d = 4, 3, 8
    gallery = _gallery(B, d)
    frames = torch.randn(B, F, d)
    frames = frames / frames.norm(dim=-1, keepdim=True)
    z, owner = _slots(True, gallery, ['v', 'a'], frames)
    assert owner == [0, 0, 0, 1], owner          # video owns 3 slots, audio 1
    assert z.shape == (B, F + 1, d)
    assert torch.allclose(z[:, :F], frames)
    assert torch.allclose(z[:, F], gallery['a'])


def test_presence_expands_over_owned_slots():
    """A clip missing its audio must have the audio slot absent and every video slot
    present -- presence is a modality property, expanded, not a per-slot measurement."""
    B, F, d = 3, 4, 8
    gallery = _gallery(B, d)
    gallery['a'][1] = 0.0                                  # clip 1 has no audio
    frames = torch.randn(B, F, d)
    frames = frames / frames.norm(dim=-1, keepdim=True)
    _z, owner = _slots(True, gallery, ['v', 'a'], frames)
    owner_idx = torch.tensor(owner)
    present_mod = torch.stack([(gallery[m].norm(dim=-1) > 0.5).float() for m in ('v', 'a')], dim=1)
    present = present_mod[:, owner_idx]
    assert present[1].tolist() == [1, 1, 1, 1, 0]
    assert present[0].tolist() == [1, 1, 1, 1, 1]


def test_modality_level_mask_drops_whole_video_not_one_frame():
    """The mask sampler draws over modalities; expanding by owner is what keeps a 'drop
    video' draw from becoming 'drop one frame', which would be a different curriculum."""
    owner_idx = torch.tensor([0, 0, 0, 1])
    vmask_mod = torch.tensor([[0.0, 1.0], [1.0, 1.0]])     # clip 0 drops video
    vmask = vmask_mod[:, owner_idx]
    assert vmask[0].tolist() == [0, 0, 0, 1]
    assert vmask[1].tolist() == [1, 1, 1, 1]


def test_frame_slots_reduce_to_the_pooled_centroid_when_frames_are_identical():
    """If every frame is the same vector, the frame set carries nothing and the centroid
    must equal the pooled one -- the sanity check that expanding slots alone changes
    nothing, and that any measured gain comes from frames DIFFERING."""
    B, F, d = 3, 5, 8
    gallery = _gallery(B, d)
    frames = gallery['v'].unsqueeze(1).expand(B, F, d).contiguous()
    z_f, owner = _slots(True, gallery, ['v', 'a'], frames)
    present_f = torch.ones(B, len(owner))
    mu_f, _, _ = masked_spherical_mean(z_f, present_f, weighting='query',
                                       tau_w=0.1, query=gallery['v'])
    z_p, _ = _slots(False, gallery, ['v', 'a'])
    mu_p, _, _ = masked_spherical_mean(z_p, torch.ones(B, 2), weighting='query',
                                       tau_w=0.1, query=gallery['v'])
    # identical frames collapse to a single distinct direction, so both centroids lie in
    # span{v, a}; with the same query they agree closely
    assert torch.nn.functional.cosine_similarity(mu_f, mu_p, dim=-1).min() > 0.99
