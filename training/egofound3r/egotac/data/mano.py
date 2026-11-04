"""Dependency-free MANO layer (shared by all dataset converters).

Output joints use the order EgoFound3R expects: OpenPose hand order
(wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4, pinky 1-4), in meters.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

# MANO kinematic order: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb.
_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
# Fingertip vertices in order thumb, index, middle, ring, pinky.
_TIPS = (745, 317, 444, 556, 673)
# Index into [16 kinematic joints + 5 tips] that produces OpenPose order.
_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)


class _ChStub:
    """Stands in for chumpy objects so MANO pickles load without chumpy."""

    def __init__(self, *args, **kwargs):
        self.x = None

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _load_mano_pickle(path: Path) -> dict:
    names = ["chumpy", "chumpy.ch", "chumpy.ch_ops", "chumpy.reordering", "chumpy.logic", "chumpy.linalg"]
    saved = {name: sys.modules.get(name) for name in names}

    class _Module(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _ChStub

    for name in names:
        sys.modules[name] = _Module(name)
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle, encoding="latin1")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _as_array(value) -> np.ndarray:
    if isinstance(value, _ChStub):
        # chumpy "Select": a flat index into a parent Ch, reshaped (this is how shapedirs is stored).
        if "idxs" in value.__dict__:
            parent = _as_array(value.__dict__["a"])
            value = parent.ravel()[value.__dict__["idxs"]].reshape(value.__dict__["preferred_shape"])
        else:
            value = value.__dict__["x"]
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (w, x, y, z) to rotation matrix."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = 0.5 * angle
    return quat_to_rotmat(torch.cat([half.cos(), half.sin() * axis_angle / angle], dim=-1))


class ManoLayer(torch.nn.Module):
    def __init__(self, mano_dir: str | Path, side: str):
        super().__init__()
        if side not in {"left", "right"}:
            raise ValueError(f"side must be left or right, got {side!r}")
        self.side = side
        data = _load_mano_pickle(Path(mano_dir) / f"MANO_{side.upper()}.pkl")
        if tuple(np.asarray(data["kintree_table"])[0, 1:].tolist()) != _PARENTS[1:]:
            raise ValueError("MANO kinematic tree does not match the expected joint order")
        self.register_buffer("v_template", torch.tensor(_as_array(data["v_template"]), dtype=torch.float32))
        self.register_buffer("shapedirs", torch.tensor(_as_array(data["shapedirs"]), dtype=torch.float32))
        self.register_buffer("posedirs", torch.tensor(_as_array(data["posedirs"]), dtype=torch.float32))
        self.register_buffer("j_regressor", torch.tensor(_as_array(data["J_regressor"]), dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(_as_array(data["weights"]), dtype=torch.float32))
        self.register_buffer("faces", torch.tensor(np.asarray(data["f"], dtype=np.int64)))
        self.register_buffer("hands_mean", torch.tensor(_as_array(data["hands_mean"]), dtype=torch.float32))
        self.register_buffer("hands_components", torch.tensor(_as_array(data["hands_components"]), dtype=torch.float32))

    def forward(self, rotmats: torch.Tensor, betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """rotmats [N, 16, 3, 3] (flat-hand-mean convention), betas [N, 10].

        Returns (verts [N, 778, 3], joints [N, 21, 3]) in MANO's own frame, not root-centered.
        """
        batch = rotmats.shape[0]
        v_shaped = self.v_template + torch.einsum("bl,vcl->bvc", betas, self.shapedirs[:, :, : betas.shape[-1]])
        rest_joints = torch.einsum("jv,bvc->bjc", self.j_regressor, v_shaped)
        eye = torch.eye(3, device=rotmats.device, dtype=rotmats.dtype)
        pose_feature = (rotmats[:, 1:] - eye).reshape(batch, -1)
        v_posed = v_shaped + torch.einsum("bp,vcp->bvc", pose_feature, self.posedirs)

        world = [None] * 16
        for joint, parent in enumerate(_PARENTS):
            local = torch.zeros(batch, 4, 4, device=rotmats.device, dtype=rotmats.dtype)
            local[:, :3, :3] = rotmats[:, joint]
            local[:, 3, 3] = 1.0
            local[:, :3, 3] = rest_joints[:, joint] if parent < 0 else rest_joints[:, joint] - rest_joints[:, parent]
            world[joint] = local if parent < 0 else world[parent] @ local
        world = torch.stack(world, dim=1)
        joints = world[:, :, :3, 3]
        skinning = world.clone()
        skinning[:, :, :3, 3] = joints - torch.einsum("bjcd,bjd->bjc", world[:, :, :3, :3], rest_joints)
        blended = torch.einsum("vj,bjcd->bvcd", self.weights, skinning)
        verts = torch.einsum("bvcd,bvd->bvc", blended[:, :, :3, :3], v_posed) + blended[:, :, :3, 3]
        tips = verts[:, list(_TIPS)]
        return verts, torch.cat([joints, tips], dim=1)[:, list(_TO_OPENPOSE)]


def marker_ids_195() -> np.ndarray:
    from ..upstream import ensure_upstream

    ensure_upstream()
    from egohandmetric_prompt.data.marker_vertices import marker_vertex_ids_for_count

    ids = marker_vertex_ids_for_count(195)
    return np.asarray(ids.cpu() if torch.is_tensor(ids) else ids, dtype=np.int64)
