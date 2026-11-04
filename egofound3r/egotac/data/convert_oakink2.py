"""Convert OakInk-v2 to unified clips. Conventions come from the official oakink/OakInk2 toolkit:
quaternion MANO with flat_hand_mean=True and center_idx=0, cam_extr maps world to camera, color at
30 Hz on every fourth 120 Hz mocap frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from .clip import CLIP_FRAMES, ClipMeta, FrameInput, build_clip, save_clip, write_manifest
from .mano import ManoLayer, marker_ids_195, quat_to_rotmat

COLOR_FPS = 30
HAND_STEP = 3  # 30 Hz color down to the 10 Hz hand rate
MAX_OBJECT_FACES = 4000
HF_REPO = "kelvin34501/OakInk-v2"


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def load_object_tris(root: Path, obj_id: str, cache: dict) -> np.ndarray:
    if obj_id in cache:
        return cache[obj_id]
    directory = root / "object_repair" / "align_ds" / obj_id
    candidates = [p for p in directory.iterdir() if p.suffix in {".obj", ".ply"}] if directory.is_dir() else []
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one mesh in {directory}, found {candidates}")
    mesh = trimesh.load(candidates[0], process=False, force="mesh")
    if len(mesh.faces) > MAX_OBJECT_FACES:
        import fast_simplification

        points, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces), target_count=MAX_OBJECT_FACES
        )
        mesh = trimesh.Trimesh(points, faces, process=False)
    tris = np.asarray(mesh.vertices, dtype=np.float32)[np.asarray(mesh.faces)]
    cache[obj_id] = tris
    return tris


def hand_world(anno: dict, mocap_ids: list[int], prefix: str, layer: ManoLayer, device: str):
    frames = [anno["raw_mano"][fid] for fid in mocap_ids]
    quats = torch.cat([f[f"{prefix}__pose_coeffs"] for f in frames]).to(device)
    betas = torch.cat([f[f"{prefix}__betas"] for f in frames]).to(device)
    tsl = torch.cat([f[f"{prefix}__tsl"] for f in frames]).to(device)
    verts, joints = layer(quat_to_rotmat(quats), betas)
    wrist = joints[:, :1]
    return (verts - wrist + tsl[:, None]).cpu().numpy(), (joints - wrist + tsl[:, None]).cpu().numpy()


def split_of(sequence: str, eval_fraction: float) -> str:
    # Stable across machines and independent of which sequences are present.
    bucket = int(hashlib.sha1(sequence.encode()).hexdigest(), 16) % 1000
    return "test" if bucket < eval_fraction * 1000 else "train"


def convert_sequence(root: Path, output: Path, seq: str, layers: dict, markers: np.ndarray, args, cache: dict) -> int:
    with open(root / "anno_preview" / f"{seq}.pkl", "rb") as handle:
        anno = pickle.load(handle)
    serial = {v: k for k, v in anno["cam_def"].items()}["egocentric"]
    frame_ids = anno["frame_id_list"]
    split = args.force_split.get(seq) if args.force_split else None
    split = split or split_of(seq, args.eval_fraction)
    faces = layers["left"].faces.cpu().numpy()
    hop = CLIP_FRAMES if split == "test" else args.train_hop
    written = 0
    # A full window spans 21 hand-rate frames, or 61 frames of 30 Hz color.
    for start in range(0, len(frame_ids) - (CLIP_FRAMES - 1) * HAND_STEP, hop * HAND_STEP):
        target = output / "oakink2" / seq / f"{start:06d}.npz"
        if target.exists():
            written += 1
            continue
        window = [frame_ids[start + i * HAND_STEP] for i in range(CLIP_FRAMES)]
        left_v, left_j = hand_world(anno, window, "lh", layers["left"], args.device)
        right_v, right_j = hand_world(anno, window, "rh", layers["right"], args.device)
        frames = []
        for t, fid in enumerate(window):
            bgr = cv2.imread(str(root / "data" / seq / serial / f"{fid:06d}.png"))
            if bgr is None:
                raise FileNotFoundError(f"missing color frame {seq}/{serial}/{fid:06d}.png")
            w2c = np.asarray(anno["cam_extr"]["egocentric"][fid], dtype=np.float64)
            object_tris = []
            for obj in anno["obj_list"]:
                tris = load_object_tris(root, obj, cache)
                world_to_cam_obj = w2c @ np.asarray(anno["obj_transf"][obj][fid], dtype=np.float64)
                object_tris.append(transform(world_to_cam_obj, tris))
            frames.append(
                FrameInput(
                    rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    K=np.asarray(anno["cam_intr"]["egocentric"][fid]),
                    w2c=w2c,
                    verts_cam=[transform(w2c, left_v[t]), transform(w2c, right_v[t])],
                    joints_cam=[transform(w2c, left_j[t]), transform(w2c, right_j[t])],
                    object_tris_cam=np.concatenate(object_tris) if object_tris else None,
                )
            )
        arrays = build_clip(frames, faces, markers, device=args.device)
        save_clip(target, arrays, ClipMeta("oakink2", seq, split, [int(f) for f in window]))
        written += 1
    return written


def remote_sequences() -> list[str]:
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data?limit=1000"
    with urllib.request.urlopen(url, timeout=60) as response:
        return sorted(Path(item["path"]).stem for item in json.load(response))


def _download(url: str, target: Path) -> None:
    subprocess.run(["curl", "--http1.1", "-sSLf", "--retry", "8", "--retry-all-errors", "-C", "-", "-o", str(target), url], check=True)


def fetch_egocentric(root: Path, seq: str) -> None:
    """Download one sequence, keep only the egocentric camera. The full dataset is 1.95 TB."""
    base = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
    anno_path = root / "anno_preview" / f"{seq}.pkl"
    anno_path.parent.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    if not anno_path.exists():
        temporary = anno_path.with_suffix(".part")
        _download(f"{base}/anno_preview/{seq}.pkl", temporary)
        temporary.replace(anno_path)
    with open(anno_path, "rb") as handle:
        cam_def = pickle.load(handle)["cam_def"]
    serial = {v: k for k, v in cam_def.items()}["egocentric"]
    tar_path = root / "data" / f"{seq}.tar"
    _download(f"{base}/data/{seq}.tar", tar_path)
    subprocess.run(["tar", "xf", str(tar_path), "-C", str(root / "data"), "--wildcards", f"*/{serial}/*"], check=True)
    tar_path.unlink()


def remove_fetched(root: Path, seq: str) -> None:
    shutil.rmtree(root / "data" / seq, ignore_errors=True)
    (root / "anno_preview" / f"{seq}.pkl").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--train-hop", type=int, default=10, help="window hop in hand-rate frames for the train split")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--test-sequences", nargs="*", default=[], help="force these sequences into the test split")
    parser.add_argument("--shard", default="0/1", help="i/n: convert every n-th sequence, for parallel workers")
    parser.add_argument("--from-hf", action="store_true", help="stream sequences from Hugging Face, deleting source frames after conversion")
    parser.add_argument("--manifest-only", action="store_true", help="only rebuild the manifest, after all shards finish")
    args = parser.parse_args()
    if args.manifest_only:
        print(f"manifest: {write_manifest(args.output)}")
        return
    args.force_split = {seq: "test" for seq in args.test_sequences}
    shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    layers = {side: ManoLayer(args.mano_dir, side).to(args.device) for side in ("left", "right")}
    markers = marker_ids_195()
    if args.from_hf:
        sequences = remote_sequences()
    else:
        sequences = sorted(p.stem for p in (args.root / "anno_preview").glob("*.pkl") if (args.root / "data" / p.stem).is_dir())
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    cache: dict = {}
    for seq in sequences[shard_index::shard_count]:
        finished = args.output / "oakink2" / seq / "DONE"
        if finished.exists():
            continue
        if args.from_hf:
            fetch_egocentric(args.root, seq)
        count = convert_sequence(args.root, args.output, seq, layers, markers, args, cache)
        finished.parent.mkdir(parents=True, exist_ok=True)
        finished.touch()
        if args.from_hf:
            remove_fetched(args.root, seq)
        print(f"{seq}: {count} clips", flush=True)
    if shard_count == 1:
        print(f"manifest: {write_manifest(args.output)}")


if __name__ == "__main__":
    main()
