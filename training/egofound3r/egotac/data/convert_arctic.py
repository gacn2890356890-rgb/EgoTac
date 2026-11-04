"""Convert ARCTIC to unified clips, egocentric view. Conventions come from zc-alexfan/arctic:
smplx MANO without flat_hand_mean and with transl, object vertices articulated about z (top part only),
image index = motion index + ioi_offset, and view 0 images carrying the eight-term OpenCV distortion.
Images are undistorted here so the pinhole K applies. object.npy stores translation in millimeters.

Expected layout, as the official download and unzip scripts produce it:
    <root>/raw_seqs/s01/box_grab_01.{mano,object,egocam.dist}.npy
    <root>/meta/misc.json, <root>/meta/object_vtemplates/box/{mesh.obj,parts.json}
    <root>/images/s01/box_grab_01/0/00010.jpg
Not yet run on real ARCTIC files.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from .clip import FrameInput
from .mano import pose_points
from .source import run

FPS = 30.0
MIN_TRANSLATION_MM = 10.0


def discover(root: Path) -> dict[str, str]:
    found = {}
    for path in sorted((root / "raw_seqs").glob("s*/*.mano.npy")):
        sequence = f"{path.parent.name}/{path.name.removesuffix('.mano.npy')}"
        if (root / "images" / sequence / "0").is_dir():
            found[sequence] = sequence
    return found


def object_template(root: Path, name: str, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name not in cache:
        folder = root / "meta" / "object_vtemplates" / name
        mesh = trimesh.load(folder / "mesh.obj", process=False, force="mesh")
        parts = np.asarray(json.loads((folder / "parts.json").read_text()), dtype=bool)
        cache[name] = (np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), parts)
    return cache[name]


def articulated_tris(vertices_mm: np.ndarray, faces: np.ndarray, top: np.ndarray, params: np.ndarray) -> np.ndarray:
    """params = (angle in radians, axis-angle 3, translation in millimeters 3). Returns meters, world frame."""
    posed = vertices_mm.copy()
    posed[top] = Rotation.from_rotvec([0.0, 0.0, -params[0]]).apply(posed[top])
    posed = Rotation.from_rotvec(params[1:4]).apply(posed) + params[4:7]
    return (posed * 0.001)[faces].astype(np.float32)


class Source:
    fps = FPS
    templates: dict = {}

    def __init__(self, sequence: str, context):
        root = context.root
        self.context = context
        self.image_dir = root / "images" / sequence / "0"
        subject, name = sequence.split("/")
        base = root / "raw_seqs" / subject / name
        self.mano = np.load(f"{base}.mano.npy", allow_pickle=True).item()
        self.object = np.load(f"{base}.object.npy", allow_pickle=True)
        if np.median(np.linalg.norm(self.object[:, 4:7], axis=1)) < MIN_TRANSLATION_MM:
            raise ValueError(f"{sequence}: object translation looks like meters, expected millimeters")
        camera = np.load(f"{base}.egocam.dist.npy", allow_pickle=True).item()
        self.R, self.T = np.asarray(camera["R_k_cam_np"]), np.asarray(camera["T_k_cam_np"]).reshape(-1, 3)
        self.K = np.asarray(camera["intrinsics"], dtype=np.float64)
        self.distortion = np.asarray(camera["dist8"], dtype=np.float64)
        misc = json.loads((root / "meta" / "misc.json").read_text())
        self.offset = int(misc[subject]["ioi_offset"])
        self.template = object_template(root, name.split("_")[0], self.templates)
        self.total = len(self.object)

    def prepare(self, indices: set[int]) -> None:
        pass

    def _hand(self, key: str, index: int):
        data = self.mano[key]
        as_tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.context.device).reshape(1, -1)
        return pose_points(self.context.layers[key], as_tensor(data["rot"][index]), as_tensor(data["pose"][index]), as_tensor(data["shape"]), as_tensor(data["trans"][index]), add_mean=True)

    def frame(self, index: int) -> FrameInput:
        path = self.image_dir / f"{index + self.offset:05d}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(cv2.undistort(bgr, self.K, self.distortion), cv2.COLOR_BGR2RGB)
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = self.R[index], self.T[index]
        move = lambda points: points @ w2c[:3, :3].T + w2c[:3, 3]
        left_v, left_j = self._hand("left", index)
        right_v, right_j = self._hand("right", index)
        vertices, faces, top = self.template
        tris = move(articulated_tris(vertices, faces, top, self.object[index]))
        return FrameInput(rgb, self.K, w2c, [move(left_v), move(right_v)], [move(left_j), move(right_j)], tris.astype(np.float32))


def main() -> None:
    run("arctic", __doc__, discover, lambda unit, context: Source(unit, context))


if __name__ == "__main__":
    main()
