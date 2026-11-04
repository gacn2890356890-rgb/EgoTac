"""Convert H2O to unified clips, egocentric view (cam4). Formats come from the taeinkwon/h2odataset README
and h2oplayer: manopth MANO with flat_hand_mean and no centering, per-frame poses in the camera frame.

Expected layout, as the official download script unpacks it with --mode ego:
    <root>/subject1/h1/0/cam4/{rgb,cam_pose,hand_pose_MANO,obj_pose_RT}/000000.*, cam_intrinsics.txt
    <object-dir>/**/<name>*.{obj,ply}, with name one of OBJECT_NAMES
Not yet run on real H2O files.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .clip import FrameInput
from .mano import pose_points
from .source import load_mesh_tris, run

FPS = 30.0
OBJECT_NAMES = {1: "book", 2: "espresso", 3: "lotion", 4: "spray", 5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino"}
HAND_VALUES = 59  # flag, translation 3, pose 48, shape 10


def discover(root: Path) -> dict[str, str]:
    sequences = [str(path.parent.relative_to(root)) for path in root.glob("subject*/*/*/cam4")]
    return {sequence: sequence for sequence in sorted(sequences)}


def is_test(sequence: str) -> bool:
    return sequence.startswith("subject4/")  # the official test subject


class Source:
    fps = FPS

    def __init__(self, folder: Path, context):
        self.folder, self.context = folder, context
        self.total = len(list((folder / "rgb").glob("*.png")))
        fx, fy, cx, cy = np.loadtxt(folder / "cam_intrinsics.txt")[:4]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.meshes: dict[int, np.ndarray] = {}

    def prepare(self, indices: set[int]) -> None:
        pass

    def _mesh(self, object_class: int) -> np.ndarray:
        if object_class not in self.meshes:
            name = OBJECT_NAMES[object_class]
            found = sorted(p for p in Path(self.context.args.object_dir).rglob(f"{name}*") if p.suffix in {".obj", ".ply"})
            if not found:
                raise FileNotFoundError(f"no mesh named {name} under {self.context.args.object_dir}")
            self.meshes[object_class] = load_mesh_tris(found[0])
        return self.meshes[object_class]

    def frame(self, index: int) -> FrameInput:
        name = f"{index:06d}"
        bgr = cv2.imread(str(self.folder / "rgb" / f"{name}.png"))
        if bgr is None:
            raise FileNotFoundError(self.folder / "rgb" / f"{name}.png")
        cam_to_world = np.loadtxt(self.folder / "cam_pose" / f"{name}.txt").reshape(4, 4)
        values = np.loadtxt(self.folder / "hand_pose_MANO" / f"{name}.txt")
        verts, joints = [None, None], [None, None]
        for side, key in enumerate(("left", "right")):
            hand = values[side * HAND_VALUES : (side + 1) * HAND_VALUES]
            if hand[0] > 0:
                as_tensor = lambda part: torch.as_tensor(part, dtype=torch.float32, device=self.context.device)[None]
                verts[side], joints[side] = pose_points(self.context.layers[key], as_tensor(hand[4:7]), as_tensor(hand[7:52]), as_tensor(hand[52:62]), as_tensor(hand[1:4]))
        tris = None
        pose = np.loadtxt(self.folder / "obj_pose_RT" / f"{name}.txt")
        if int(pose[0]) != 0:
            matrix = pose[1:].reshape(4, 4)
            tris = self._mesh(int(pose[0])) @ matrix[:3, :3].T + matrix[:3, 3]
        return FrameInput(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.K, np.linalg.inv(cam_to_world), verts, joints, tris)


def main() -> None:
    run(
        "h2o",
        __doc__,
        discover,
        lambda unit, context: Source(context.root / unit / "cam4", context),
        is_fixed_test=is_test,
        add_arguments=lambda parser: parser.add_argument("--object-dir", type=Path, required=True),
    )


if __name__ == "__main__":
    main()
