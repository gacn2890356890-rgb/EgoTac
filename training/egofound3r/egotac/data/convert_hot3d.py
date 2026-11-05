"""Convert HOT3D Aria to unified clips, from the public HOT3D-Clips release. Conventions come from
facebookresearch/hot3d and hand_tracking_toolkit (not a dependency, as it would add a second OpenCV): MANO with 15 PCA coefficients, no flat_hand_mean, a
corrected left shape basis, wrist_xform as world-frame axis-angle plus translation. The fisheye RGB
stream (214-1) is warped to a pinhole camera with the same focal length, as the toolkit's --undistort does.

Only the 'train' clips carry hand poses, so both our splits come from them, divided by source sequence.

Expected layout:
    <root>/train_aria/clip-001849.tar, <root>/clip_definitions.json, <root>/clip_splits.json
    <root>/object_models_eval/obj_000001.glb
Checked on four real clips: labeled joints land on the hands in the rectified images, which keep Aria's sideways orientation.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .clip import FrameInput
from .mano import pca_to_axis_angle, pose_points
from .source import load_mesh_tris, run

FPS = 30.0
STREAM = "214-1"
_MAPS: dict = {}


def discover(root: Path) -> dict[str, str]:
    definitions = json.loads((root / "clip_definitions.json").read_text())
    train = json.loads((root / "clip_splits.json").read_text())["train"]["Aria"]
    units = {}
    for clip in train:
        name = f"clip-{int(clip):06d}"
        if (root / "train_aria" / f"{name}.tar").exists():
            units[name] = definitions[str(int(clip))]["sequence_id"]
    return units


def se3(pose: dict) -> np.ndarray:
    w, x, y, z = pose["quaternion_wxyz"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    matrix[:3, 3] = pose["translation_xyz"]
    return matrix


def fisheye624_map(width: int, height: int, f: float, cx: float, cy: float, coefficients) -> tuple[np.ndarray, np.ndarray]:
    """Where each pinhole pixel lands in the fisheye image. The same math as the toolkit's OVR624Distortion."""
    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = coefficients
    px, py = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    a, b = (px - cx) / f, (py - cy) / f
    r = np.sqrt(a * a + b * b)
    scale = np.arctan2(r, 1.0) / np.maximum(r, 2.0**-128)
    a, b = a * scale, b * scale
    r2 = np.clip(a * a + b * b, 0.0, np.pi**2)
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * (k4 + r2 * (k5 + r2 * k6)))))
    x, y = a * radial, b * radial
    r2 = x * x + y * y
    dx = 2 * p2 * x * y + p1 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
    dy = 2 * p1 * x * y + p2 * (r2 + 2 * y * y) + s3 * r2 + s4 * r2 * r2
    return (f * (x + dx) + cx).astype(np.float32), (f * (y + dy) + cy).astype(np.float32)


def rectify(camera_json: dict, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calibration = camera_json["calibration"]
    f, cx, cy = calibration["projection_params"][:3]
    key = tuple(calibration["projection_params"])
    if key not in _MAPS:  # one calibration per recording, and computing the map takes most of a frame's time
        _MAPS.clear()
        _MAPS[key] = fisheye624_map(calibration["image_width"], calibration["image_height"], f, cx, cy, calibration["projection_params"][3:])
    map_x, map_y = _MAPS[key]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR), K


class Source:
    fps = FPS
    meshes: dict = {}

    def __init__(self, unit: str, context):
        self.context = context
        self.tar = tarfile.open(context.root / "train_aria" / f"{unit}.tar")
        names = self.tar.getnames()
        self.keys = sorted(name.split(".")[0] for name in names if name.endswith(f".image_{STREAM}.jpg"))
        self.total = len(self.keys)
        self.beta = torch.as_tensor(self._json("__hand_shapes.json__")["mano"], dtype=torch.float32, device=context.device)[None]

    def _json(self, name: str):
        return json.load(self.tar.extractfile(name))

    def prepare(self, indices: set[int]) -> None:
        pass

    def _mesh(self, bop_id: int) -> np.ndarray:
        if bop_id not in self.meshes:
            self.meshes[bop_id] = load_mesh_tris(self.context.root / "object_models_eval" / f"obj_{bop_id:06d}.glb")
        return self.meshes[bop_id]

    def frame(self, index: int) -> FrameInput:
        key = self.keys[index]
        blob = self.tar.extractfile(f"{key}.image_{STREAM}.jpg").read()
        bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        cameras = self._json(f"{key}.cameras.json")
        rgb, K = rectify(cameras[STREAM], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        w2c = np.linalg.inv(se3(cameras[STREAM]["T_world_from_camera"]))
        move = lambda points: points @ w2c[:3, :3].T + w2c[:3, 3]
        hands = self._json(f"{key}.hands.json")
        verts, joints = [None, None], [None, None]
        for side, name in enumerate(("left", "right")):
            pose = hands.get(name, {}).get("mano_pose")
            if pose is None:
                continue
            layer = self.context.layers[name]
            as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.context.device)[None]
            wrist = as_tensor(pose["wrist_xform"])
            v, j = pose_points(layer, wrist[:, :3], pca_to_axis_angle(layer, as_tensor(pose["thetas"])), self.beta, wrist[:, 3:], add_mean=True)
            verts[side], joints[side] = move(v), move(j)
        tris = []
        for instances in self._json(f"{key}.objects.json").values():
            for instance in instances:
                to_camera = w2c @ se3(instance["T_world_from_object"])
                tris.append(self._mesh(int(instance["object_bop_id"])) @ to_camera[:3, :3].T + to_camera[:3, 3])
        return FrameInput(rgb, K, w2c, verts, joints, np.concatenate(tris).astype(np.float32) if tris else None)


def main() -> None:
    run("hot3d_aria", __doc__, discover, lambda unit, context: Source(unit, context), fix_left_shapedirs=True)


if __name__ == "__main__":
    main()
