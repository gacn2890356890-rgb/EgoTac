"""Diffusion head behaviour on small tensors; needs neither GPU nor checkpoints."""
import torch

from egotac.diffusion_head import NUM_POINTS, TactileFlowHead, distance_to_target, samples_to_outputs, target_to_log_distance


def _head():
    torch.manual_seed(0)
    return TactileFlowHead(dim=32, layers=1, heads=2, feature_dim=8)


def test_sampling_is_repeatable_for_a_seed_and_changes_with_it():
    head = _head().eval()
    cond, offsets = torch.randn(2, NUM_POINTS, 8), torch.randn(2, NUM_POINTS, 3)
    first = head.sample(cond, offsets, num_samples=2, steps=2, seed=3)
    assert torch.equal(first, head.sample(cond, offsets, num_samples=2, steps=2, seed=3))
    assert not torch.equal(first, head.sample(cond, offsets, num_samples=2, steps=2, seed=4))


def test_outputs_match_official_field_ranges():
    samples = torch.randn(4, 2, NUM_POINTS, 2)
    out = samples_to_outputs(samples)
    assert out["contact_probability"].shape == (2, NUM_POINTS)
    assert ((out["contact_probability"] > 0) & (out["contact_probability"] < 1)).all()
    assert (out["contact_log_conf"] >= 0).all() and (out["contact_log_conf"] <= 2).all()
    assert (out["contact_log_distance"] >= 0).all()
    assert torch.allclose(torch.sigmoid(out["contact_logits"]), out["contact_probability"], atol=1e-5)


def test_distance_target_round_trips_to_the_paper_log_distance():
    distance = torch.tensor([0.0, 0.005, 0.1])
    assert torch.allclose(target_to_log_distance(distance_to_target(distance)), torch.log1p(1000 * distance), atol=1e-5)


def test_mask_excludes_points_without_labels():
    head = _head()
    cond, offsets = torch.randn(2, NUM_POINTS, 8), torch.zeros(2, NUM_POINTS, 3)
    target = torch.zeros(2, NUM_POINTS, 2)
    mask = torch.zeros(2, NUM_POINTS, dtype=torch.bool)
    assert head.flow_matching_loss(cond, offsets, target, mask).item() == 0.0


def test_flow_overfits_one_batch_to_recover_contact():
    torch.manual_seed(0)
    head = TactileFlowHead(dim=64, layers=2, heads=2, feature_dim=8)
    cond = torch.randn(1, NUM_POINTS, 8)
    offsets = torch.zeros(1, NUM_POINTS, 3)
    contact = (cond[..., 0] > 0).float()
    target = torch.stack([2 * contact - 1, torch.zeros_like(contact)], -1)
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
    mask = torch.ones(1, NUM_POINTS, dtype=torch.bool)
    for _ in range(300):
        optimizer.zero_grad()
        head.flow_matching_loss(cond, offsets, target, mask).backward()
        optimizer.step()
    probability = samples_to_outputs(head.sample(cond, offsets, num_samples=4, steps=10))["contact_probability"]
    accuracy = ((probability > 0.5).float() == contact).float().mean().item()
    assert accuracy > 0.9, accuracy
