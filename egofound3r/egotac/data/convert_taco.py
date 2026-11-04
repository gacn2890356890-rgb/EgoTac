"""Convert TACO to unified clips. Conventions come from the official leolyliu/TACO-Instructions code:
axis-angle MANO (flat_hand_mean, center_idx=0) with hand_trans in meters, object meshes in centimeters,
per-frame egocentric extrinsics mapping world to camera.

The archives can be read in place: --fetch pulls only the listed members over HTTP range requests.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from .clip import CLIP_FRAMES, HAND_RATE_HZ, ClipMeta, FrameInput, build_clip, save_clip, write_manifest
from .convert_oakink2 import MAX_OBJECT_FACES, split_of, transform
from .mano import ManoLayer, axis_angle_to_rotmat, marker_ids_195

HF_BASE = "https://huggingface.co/datasets/leolyliu/TACO/resolve/main/"
MODEL_DIR = "object_models_released"


def fetch(root: Path, count: int | None, seed: int) -> None:
    """Download only what conversion needs: no allocentric videos, depth, or segmentation."""
    import remotezip

    root.mkdir(parents=True, exist_ok=True)
    with remotezip.RemoteZip(HF_BASE + "Hand_Poses.zip") as archive:
        names = [n for n in archive.namelist() if n.endswith("right_hand.pkl")]
        sequences = sorted(str(Path(n).parent.relative_to("Hand_Poses")) for n in names)
        if count is not None:
            random.Random(seed).shuffle(sequences)
            sequences = sorted(sequences[:count])
        wanted = set(sequences)
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) >= 3 and str(Path(*parts[1:3])) in wanted and not name.endswith("/") and not (root / name).exists():
                archive.extract(name, root)
    needed_models: set[str] = set()
    for zip_name, top in (("Egocentric_RGB_Videos", "Egocentric_RGB_Videos"), ("Object_Poses", "Object_Poses"), ("Egocentric_Camera_Parameters", "Egocentric_Camera_Parameters")):
        with remotezip.RemoteZip(HF_BASE + zip_name + ".zip") as archive:
            for name in archive.namelist():
                parts = Path(name).parts
                if len(parts) >= 3 and parts[0] == top and str(Path(*parts[1:3])) in wanted and not name.endswith("/"):
                    if not (root / name).exists():
                        archive.extract(name, root)
                    if top == "Object_Poses":
                        needed_models.add(Path(name).stem.split("_", 1)[1])
    with remotezip.RemoteZip(HF_BASE + "Object_Models.zip") as archive:
        for model in sorted(needed_models):
            name = f"{MODEL_DIR}/{model}_cm.obj"
            if not (root / name).exists():
                archive.extract(name, root)
    print(f"fetched {len(sequences)} sequences, {len(needed_models)} object models", flush=True)


def load_object_tris(root: Path, model: str, cache: dict) -> np.ndarray:
    if model in cache:
        return cache[model]
    mesh = trimesh.load(root / MODEL_DIR / f"{model}_cm.obj", process=False, force="mesh")
    if len(mesh.faces) > MAX_OBJECT_FACES:
        import fast_simplification

        points, faces = fast_simplification.simplify(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces), target_count=MAX_OBJECT_FACES)
        mesh = trimesh.Trimesh(points, faces, process=False)
    tris = np.asarray(mesh.vertices, dtype=np.float32)[np.asarray(mesh.faces)] * 0.01  # cm to m
    cache[model] = tris
    return tris


def hand_world(pose_dir: Path, side: str, layer: ManoLayer, device: str) -> tuple[np.ndarray, np.ndarray]:
    with open(pose_dir / f"{side}_hand.pkl", "rb") as handle:
        data = pickle.load(handle)
    with open(pose_dir / f"{side}_hand_shape.pkl", "rb") as handle:
        beta = pickle.load(handle)["hand_shape"].reshape(10).detach().cpu()
    keys = sorted(data)
    theta = torch.stack([data[k]["hand_pose"].detach().cpu().reshape(48) for k in keys]).float().to(device)
    trans = torch.stack([data[k]["hand_trans"].detach().cpu().reshape(3) for k in keys]).float().to(device)
    verts, joints = [], []
    for start in range(0, len(keys), 512):
        rot = axis_angle_to_rotmat(theta[start : start + 512].reshape(-1, 16, 3))
        v, j = layer(rot, beta.to(device).expand(rot.shape[0], -1))
        wrist = j[:, :1]
        offset = trans[start : start + 512, None]
        verts.append((v - wrist + offset).cpu().numpy())
        joints.append((j - wrist + offset).cpu().numpy())
    return np.concatenate(verts), np.concatenate(joints)


def read_frames(path: Path, indices: set[int]) -> tuple[dict[int, np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames: dict[int, np.ndarray] = {}
    last = max(indices)
    index = 0
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in indices:
            frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    return frames, fps


def convert_sequence(root: Path, output: Path, sequence: str, layers: dict, markers: np.ndarray, args, cache: dict) -> int:
    camera_dir = root / "Egocentric_Camera_Parameters" / sequence
    w2c = np.load(camera_dir / "egocentric_frame_extrinsic.npy")
    K = np.loadtxt(camera_dir / "egocentric_intrinsic.txt")
    pose_dir = root / "Object_Poses" / sequence
    objects = [(path.stem.split("_", 1)[1], np.load(path)) for path in sorted(pose_dir.glob("*.npy"))]
    left_v, left_j = hand_world(root / "Hand_Poses" / sequence, "left", layers["left"], args.device)
    right_v, right_j = hand_world(root / "Hand_Poses" / sequence, "right", layers["right"], args.device)
    total = min([len(w2c), len(left_v), len(right_v)] + [len(poses) for _, poses in objects])

    capture = cv2.VideoCapture(str(root / "Egocentric_RGB_Videos" / sequence / "color.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps <= 0:
        raise ValueError(f"{sequence}: video reports fps={fps}")
    step = max(1, round(fps / HAND_RATE_HZ))
    span = (CLIP_FRAMES - 1) * step
    split = "test" if sequence in args.test_set else split_of(sequence, args.eval_fraction)
    hop = CLIP_FRAMES if split == "test" else args.train_hop
    safe = hashlib.sha1(sequence.encode()).hexdigest()[:16]
    starts = list(range(0, total - span, hop * step))
    pending = [s for s in starts if not (output / "taco" / safe / f"{s:06d}.npz").exists()]
    if not pending:
        return len(starts)
    wanted = {s + i * step for s in pending for i in range(CLIP_FRAMES)}
    images, _ = read_frames(root / "Egocentric_RGB_Videos" / sequence / "color.mp4", wanted)
    if len(images) != len(wanted):
        raise ValueError(f"{sequence}: decoded {len(images)} of {len(wanted)} frames")
    faces = layers["left"].faces.cpu().numpy()
    for start in pending:
        frames = []
        window = [start + i * step for i in range(CLIP_FRAMES)]
        for f in window:
            extrinsic = w2c[f]
            tris = [transform(extrinsic @ poses[f], load_object_tris(root, model, cache)) for model, poses in objects]
            frames.append(
                FrameInput(
                    rgb=images[f],
                    K=K,
                    w2c=extrinsic,
                    verts_cam=[transform(extrinsic, left_v[f]), transform(extrinsic, right_v[f])],
                    joints_cam=[transform(extrinsic, left_j[f]), transform(extrinsic, right_j[f])],
                    object_tris_cam=np.concatenate(tris) if tris else None,
                )
            )
        arrays = build_clip(frames, faces, markers, device=args.device)
        save_clip(output / "taco" / safe / f"{start:06d}.npz", arrays, ClipMeta("taco", sequence, split, window))
    return len(starts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--train-hop", type=int, default=10)
    parser.add_argument("--fetch", action="store_true", help="download sequences first")
    parser.add_argument("--fetch-count", type=int, default=None, help="random subset size; default all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=0, help="force this many sequences into the test split")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        print(f"manifest: {write_manifest(args.output)}")
        return
    if args.fetch:
        fetch(args.root, args.fetch_count, args.seed)
        return
    layers = {side: ManoLayer(args.mano_dir, side).to(args.device) for side in ("left", "right")}
    markers = marker_ids_195()
    sequences = sorted(str(p.parent.relative_to(args.root / "Hand_Poses")) for p in (args.root / "Hand_Poses").rglob("right_hand.pkl"))
    args.test_set = set(random.Random(args.seed).sample(sequences, args.test_count))
    index, count = (int(part) for part in args.shard.split("/"))
    cache: dict = {}
    for sequence in sequences[index::count]:
        print(f"{sequence}: {convert_sequence(args.root, args.output, sequence, layers, markers, args, cache)} clips", flush=True)
    if count == 1:
        print(f"manifest: {write_manifest(args.output)}")


if __name__ == "__main__":
    main()
