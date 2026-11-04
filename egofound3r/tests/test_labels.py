"""Label math checked against brute force, with no GPU, data, or MANO files required."""
import numpy as np
import torch

from egotac.data.labels import point_triangle_sq_distance, point_visibility, points_to_mesh_distance, rasterize_depth


def _brute(points, tris, samples=400):
    w = torch.rand(samples, 3)
    w = w / w.sum(-1, keepdim=True)
    surface = torch.einsum("sv,fvc->fsc", w, tris).reshape(-1, 3)
    vertices = tris.reshape(-1, 3)
    return torch.cdist(points, torch.cat([surface, vertices])).min(1).values


def test_point_triangle_regions_match_brute_force():
    torch.manual_seed(0)
    points, tris = torch.randn(300, 3), torch.randn(40, 3, 3)
    exact = points_to_mesh_distance(points, tris)
    brute = _brute(points, tris, samples=4000)
    assert (exact <= brute + 1e-5).all(), "exact distance can never exceed a sampled upper bound"
    assert (brute - exact).mean() < 0.03


def test_known_distances():
    tri = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    cases = {(0.2, 0.2, 0.5): 0.5, (-1.0, 0, 0): 1.0, (2.0, 0, 0): 1.0, (0.5, -1.0, 0): 1.0, (1.0, 1.0, 0): 2**0.5 / 2}
    for point, expected in cases.items():
        got = point_triangle_sq_distance(torch.tensor([point]), tri).sqrt().item()
        assert abs(got - expected) < 1e-5, (point, got, expected)


def test_zbuffer_occlusion():
    near = np.array([[[-1.0, -1, 1], [1, -1, 1], [0, 1, 1]]], dtype=np.float32)
    depth = rasterize_depth(near, 50, 50, 50, 50, 100, 100)
    assert abs(depth[50, 50] - 1.0) < 1e-4
    assert np.isinf(depth[99, 0])
    points = np.array([[0, 0, 1.0], [0, 0, 2.0], [50, 0, 2.0]], dtype=np.float32)
    assert point_visibility(points, depth, 50, 50, 50, 50).tolist() == [True, False, False]


def test_sampler_refuses_a_configured_dataset_with_no_clips():
    import pytest

    from egotac.data.dataset import MixedInfiniteSampler

    rows = [{"dataset": "taco"}]
    with pytest.raises(ValueError, match="oakink2"):
        MixedInfiniteSampler(rows, {"taco": 1.0, "oakink2": 1.0})
