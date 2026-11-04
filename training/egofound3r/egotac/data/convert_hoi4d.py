"""Convert HOI4D to unified clips. Formats come from the leolyliu/HOI4D-Instructions README: manopth MANO
with flat_hand_mean, translation in the camera frame, 15 fps video.

Expected layout:
    <root>/HOI4D_release/ZY.../H*/C*/N*/S*/s*/T*/{align_rgb/image.mp4, objpose/<frame>.json, 3Dseg/output.log}
    <root>/handpose/refinehandpose_{left,right}/ZY.../H*/C*/N*/S*/s*/T*/<frame>.pickle
    <root>/camera_params/ZY.../intrin.npy
    <root>/HOI4D_CAD_Model_for_release/rigid/<category>/<id>.obj
The hand folders are named from memory of the release, not from its documentation. Articulated objects
(ARTICULATED below) keep hand-hand contact only: their part-to-pose mapping is not documented.
Not yet run on real HOI4D files.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .clip import FrameInput
from .mano import pose_points
from .source import load_mesh_tris, run

FPS = 15.0
CATEGORIES = ["", "ToyCar", "Mug", "Laptop", "StorageFurniture", "Bottle", "Safe", "Bowl", "Bucket", "Scissors", "", "Pliers", "Kettle", "Knife", "TrashCan", "", "", "Lamp", "Stapler", "", "Chair"]
ARTICULATED = {"Laptop", "StorageFurniture", "Safe", "Bucket", "Scissors", "Pliers", "TrashCan", "Lamp", "Stapler", "Chair"}


def discover(root: Path) -> dict[str, str]:
    hands = root / "handpose"
    found = {str(path.relative_to(hands / folder)) for folder in ("refinehandpose_right", "refinehandpose_left") for path in (hands / folder).glob("*/*/*/*/*/*/*") if path.is_dir()}
    videos = {str(path.parent.parent.relative_to(root / "HOI4D_release")) for path in (root / "HOI4D_release").glob("*/*/*/*/*/*/*/align_rgb/image.mp4")}
    return {sequence: sequence for sequence in sorted(found & videos)}


def read_trajectory(path: Path) -> np.ndarray:
    """Open3D .log: one metadata line and four matrix rows per frame, camera to world."""
    rows = np.array([line.split() for line in path.read_text().splitlines() if len(line.split()) == 4], dtype=np.float64)
    return rows.reshape(-1, 4, 4)


def read_object_pose(path: Path) -> np.ndarray | None:
    import json

    content = json.loads(path.read_text())
    entries = content.get("dataList") or content.get("objects")
    if not entries:
        return None
    center, rotation = entries[0]["center"], entries[0]["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("XYZ", [rotation["x"], rotation["y"], rotation["z"]]).as_matrix()
    matrix[:3, 3] = [center["x"], center["y"], center["z"]]
    return matrix


class Source:
    fps = FPS

    def __init__(self, sequence: str, context):
        self.sequence, self.context = sequence, context
        root = context.root
        self.folder = root / "HOI4D_release" / sequence
        parts = Path(sequence).parts
        self.K = np.load(root / "camera_params" / parts[0] / "intrin.npy")
        capture = cv2.VideoCapture(str(self.folder / "align_rgb" / "image.mp4"))
        self.total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        self.cam_to_world = read_trajectory(self.folder / "3Dseg" / "output.log")
        self.total = min(self.total, len(self.cam_to_world))
        category = CATEGORIES[int(parts[2][1:])]
        self.mesh = None
        if category not in ARTICULATED:
            self.mesh = load_mesh_tris(root / "HOI4D_CAD_Model_for_release" / "rigid" / category / f"{int(parts[3][1:]):03d}.obj")
        self.images: dict[int, np.ndarray] = {}

    def prepare(self, indices: set[int]) -> None:
        capture = cv2.VideoCapture(str(self.folder / "align_rgb" / "image.mp4"))
        self.images = {}
        for index in range(max(indices) + 1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"{self.sequence}: video ended at frame {index}")
            if index in indices:
                self.images[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        capture.release()

    def _hand(self, key: str, index: int):
        path = self.context.root / "handpose" / f"refinehandpose_{key}" / self.sequence / f"{index}.pickle"
        if not path.exists():
            return None, None
        with path.open("rb") as handle:
            hand = pickle.load(handle, encoding="latin1")
        as_tensor = lambda name: torch.as_tensor(np.asarray(hand[name]), dtype=torch.float32, device=self.context.device).reshape(1, -1)
        pose = as_tensor("poseCoeff")
        return pose_points(self.context.layers[key], pose[:, :3], pose[:, 3:], as_tensor("beta"), as_tensor("trans"))

    def frame(self, index: int) -> FrameInput:
        left_v, left_j = self._hand("left", index)
        right_v, right_j = self._hand("right", index)
        tris = None
        pose_path = self.folder / "objpose" / f"{index}.json"
        if self.mesh is not None and pose_path.exists():
            matrix = read_object_pose(pose_path)
            if matrix is not None:
                tris = self.mesh @ matrix[:3, :3].T + matrix[:3, 3]
        return FrameInput(self.images[index], self.K, np.linalg.inv(self.cam_to_world[index]), [left_v, right_v], [left_j, right_j], tris)


def main() -> None:
    run("hoi4d", __doc__, discover, lambda unit, context: Source(unit, context))


if __name__ == "__main__":
    main()
