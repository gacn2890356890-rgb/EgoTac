"""Contact, distance, and visibility labels computed from annotated geometry.

Contact and distance follow EgoFound3R Appendix B and G: point-to-triangle distance from a
hand point to the object or opposite-hand surface, with contact below 18 mm at joints and
14 mm at the 195 surface points.
"""
from __future__ import annotations

import numpy as np
import torch

CONTACT_THRESHOLD_JOINT_M = 0.018
CONTACT_THRESHOLD_VERTEX_M = 0.014


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


def point_triangle_sq_distance(points: torch.Tensor, tris: torch.Tensor) -> torch.Tensor:
    """Squared closest distance, pairwise. points [P, 3], tris [F, 3, 3] -> [P, F].

    Vectorized Ericson closest-point test (Real-Time Collision Detection, section 5.1.5).
    """
    a, b, c = tris[:, 0][None], tris[:, 1][None], tris[:, 2][None]
    p = points[:, None]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    eps = 1e-20

    # Face interior is the default region; every other Voronoi region overrides it.
    denom = (va + vb + vc).clamp_min(eps)
    closest = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]
    mask = vb <= 0
    # Region order matters: later assignments take priority, so vertices come last.
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    closest = torch.where(edge_bc[..., None], b + (c - b) * t[..., None], closest)
    edge_ac = mask & (d2 >= 0) & (d6 <= 0)
    t = d2 / (d2 - d6).clamp_min(eps)
    closest = torch.where(edge_ac[..., None], a + ac * t[..., None], closest)
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / (d1 - d3).clamp_min(eps)
    closest = torch.where(edge_ab[..., None], a + ab * t[..., None], closest)
    closest = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c.expand_as(closest), closest)
    closest = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b.expand_as(closest), closest)
    closest = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a.expand_as(closest), closest)
    return ((p - closest) ** 2).sum(-1)


def points_to_mesh_distance(points: torch.Tensor, tris: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Distance in meters from each point to the closest triangle. points [P, 3], tris [F, 3, 3]."""
    if tris.shape[0] == 0:
        return torch.full((points.shape[0],), float("inf"), device=points.device, dtype=points.dtype)
    best = torch.full((points.shape[0],), float("inf"), device=points.device, dtype=points.dtype)
    for start in range(0, tris.shape[0], chunk):
        best = torch.minimum(best, point_triangle_sq_distance(points, tris[start : start + chunk]).min(dim=1).values)
    return best.sqrt()


def rasterize_depth(tris_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float, height: int, width: int) -> np.ndarray:
    """Nearest-depth buffer of camera-frame triangles [F, 3, 3]. Uncovered pixels are +inf."""
    depth = np.full((height, width), np.inf, dtype=np.float32)
    z = tris_cam[:, :, 2]
    keep = (z > 1e-3).all(axis=1)
    tris_cam, z = tris_cam[keep], z[keep]
    if len(tris_cam) == 0:
        return depth
    u = fx * tris_cam[:, :, 0] / z + cx
    v = fy * tris_cam[:, :, 1] / z + cy
    x0 = np.clip(np.floor(u.min(1)).astype(int), 0, width - 1)
    x1 = np.clip(np.ceil(u.max(1)).astype(int), 0, width - 1)
    y0 = np.clip(np.floor(v.min(1)).astype(int), 0, height - 1)
    y1 = np.clip(np.ceil(v.max(1)).astype(int), 0, height - 1)
    visible = (u.max(1) >= 0) & (u.min(1) < width) & (v.max(1) >= 0) & (v.min(1) < height)
    for i in np.nonzero(visible)[0]:
        xs, ys = np.meshgrid(np.arange(x0[i], x1[i] + 1), np.arange(y0[i], y1[i] + 1))
        (ua, ub, uc), (va, vb, vc) = u[i], v[i]
        area = (ub - ua) * (vc - va) - (uc - ua) * (vb - va)
        if abs(area) < 1e-9:
            continue
        w1 = ((xs - ua) * (vc - va) - (uc - ua) * (ys - va)) / area
        w2 = ((ub - ua) * (ys - va) - (xs - ua) * (vb - va)) / area
        w0 = 1.0 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        # Perspective-correct depth: inverse depth is linear in screen space.
        inv = w0 / z[i, 0] + w1 / z[i, 1] + w2 / z[i, 2]
        candidate = np.where(inside, 1.0 / np.maximum(inv, 1e-6), np.inf)
        patch = depth[ys, xs]
        depth[ys, xs] = np.minimum(patch, candidate)
    return depth


def point_visibility(
    points_cam: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tolerance_m: float = 0.012,
) -> np.ndarray:
    """A point is visible if it projects into the image and no surface sits in front of it.

    The paper does not state its definition, so this is our own: z-buffer test with a tolerance
    that lets a point on the surface pass its own triangle. Joints lie inside the hand, so callers
    should pass a larger tolerance for them.
    """
    height, width = depth.shape
    z = points_cam[:, 2]
    safe_z = np.maximum(z, 1e-6)
    u = np.round(fx * points_cam[:, 0] / safe_z + cx).astype(int)
    v = np.round(fy * points_cam[:, 1] / safe_z + cy).astype(int)
    visible = (z > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    index = np.nonzero(visible)[0]
    visible[index] = z[index] <= depth[v[index], u[index]] + tolerance_m
    return visible
