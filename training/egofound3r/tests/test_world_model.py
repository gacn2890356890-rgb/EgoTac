"""World-model head on small tensors; needs neither GPU nor checkpoints."""
import torch
from egohandmetric_prompt.world_model import NUM_POINTS, WorldModelHead, distance_to_target, samples_to_outputs, target_to_log_distance


def _head():
    torch.manual_seed(0)
    return WorldModelHead(dim=32, layers=1, heads=2, dynamics_layers=1, past_frames=4, horizon=2, feature_dim=8, visual_dim=16)


def test_imagine_returns_one_latent_per_future_step():
    assert _head().imagine(torch.randn(3, 4, NUM_POINTS, 8)).shape == (3, 2, 32)


def test_future_depends_on_past_input():
    head = _head().eval()
    past = torch.randn(1, 4, NUM_POINTS, 8)
    assert not torch.equal(head.imagine(past), head.imagine(past + 1))


def test_sampling_is_repeatable_and_future_step_changes_it():
    head = _head().eval()
    cond, offsets = torch.randn(2, NUM_POINTS, 8), torch.randn(2, NUM_POINTS, 3)
    first = head.sample(cond, offsets, num_samples=2, steps=2, seed=3)
    assert torch.equal(first, head.sample(cond, offsets, num_samples=2, steps=2, seed=3))
    assert not torch.equal(first, head.sample(cond, offsets, num_samples=2, steps=2, seed=4))
    # Both start at zero by design, so the step has no effect until training moves them.
    torch.nn.init.normal_(head.out.weight)
    torch.nn.init.normal_(head.blocks[0].film.weight)
    now = head.sample(cond, offsets, num_samples=1, steps=2)
    later = head.sample(cond, offsets, torch.ones(2, dtype=torch.long), num_samples=1, steps=2)
    assert not torch.equal(now, later)


def test_outputs_match_official_field_ranges():
    out = samples_to_outputs(torch.randn(4, 2, NUM_POINTS, 2))
    assert out["contact_probability"].shape == (2, NUM_POINTS)
    assert (out["contact_log_conf"] >= 0).all() and (out["contact_log_conf"] <= 2).all()
    assert (out["contact_log_distance"] >= 0).all()


def test_distance_target_round_trips_to_the_paper_log_distance():
    distance = torch.tensor([0.0, 0.005, 0.1])
    assert torch.allclose(target_to_log_distance(distance_to_target(distance)), torch.log1p(1000 * distance), atol=1e-5)


def test_mask_excludes_points_without_labels():
    head = _head()
    cond, offsets, target = torch.randn(2, NUM_POINTS, 8), torch.zeros(2, NUM_POINTS, 3), torch.zeros(2, NUM_POINTS, 2)
    mask = torch.zeros(2, NUM_POINTS, dtype=torch.bool)
    assert head.flow_matching_loss(cond, offsets, target, mask, torch.zeros(2, dtype=torch.long)).item() == 0.0


def test_flow_and_dynamics_overfit_a_future_contact():
    """The future label is readable only from the past sequence, so this needs the dynamics path."""
    torch.manual_seed(0)
    head = WorldModelHead(dim=64, layers=2, heads=2, dynamics_layers=1, past_frames=2, horizon=1, feature_dim=8, visual_dim=4)
    past = torch.randn(2, 2, NUM_POINTS, 8)
    last = past[:, -1]
    contact = (past[:, 0, :, 0].mean(-1, keepdim=True) > past[:, 0, :, 0].mean()).float().expand(-1, NUM_POINTS)
    target = torch.stack([2 * contact - 1, torch.zeros_like(contact)], -1)
    offsets, mask = torch.zeros(2, NUM_POINTS, 3), torch.ones(2, NUM_POINTS, dtype=torch.bool)
    step = torch.ones(2, dtype=torch.long)
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
    for _ in range(200):
        optimizer.zero_grad()
        head.flow_matching_loss(last, offsets, target, mask, step, head.imagine(past)[:, 0]).backward()
        optimizer.step()
    samples = head.sample(last, offsets, step, head.imagine(past)[:, 0], num_samples=4, steps=10)
    accuracy = ((samples_to_outputs(samples)["contact_probability"] > 0.5).float() == contact).float().mean().item()
    assert accuracy > 0.9, accuracy
