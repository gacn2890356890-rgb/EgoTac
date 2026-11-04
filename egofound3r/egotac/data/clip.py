"""Unified on-disk clip format and the label builder every dataset converter shares.

One clip is one 2 s window at the 10 Hz hand rate (21 frames), side 0 = left hand, side 1 = right
hand (checked against the official model's output), all geometry in meters in the camera frame.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .labels import point_visibility, points_to_mesh_distance, rasterize_depth

CLIP_FRAMES = 21
HAND_RATE_HZ = 10.0
STORED_LONG_SIDE = 512
JOINT_VISIBILITY_TOLERANCE_M = 0.025
VERTEX_VISIBILITY_TOLERANCE_M = 0.012
VISIBILITY_RASTER_LONG_SIDE = 256


@dataclass
class FrameInput:
    """One frame. Hand arrays are None for a side with no annotation."""

    rgb: np.ndarray  # H, W, 3 uint8 RGB
    K: np.ndarray  # 3, 3 for this rgb
    w2c: np.ndarray  # 4, 4, world to camera
    verts_cam: list  # [left, right], each [778, 3] or None
    joints_cam: list  # [left, right], each [21, 3] or None
    object_tris_cam: np.ndarray | None = None  # [F, 3, 3]; None when objects are unannotated
    depth: np.ndarray | None = None  # optional metric depth aligned with rgb


@dataclass
class ClipMeta:
    dataset: str
    sequence: str
    split: str
    source_frames: list = field(default_factory=list)


def _encode_rgb(rgb: np.ndarray, K: np.ndarray) -> tuple[bytes, np.ndarray, tuple[int, int]]:
    height, width = rgb.shape[:2]
    scale = STORED_LONG_SIDE / max(height, width)
    K = K.astype(np.float64).copy()
    if scale < 1.0:
        new_w, new_h = round(width * scale), round(height * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        K[0] *= new_w / width
        K[1] *= new_h / height
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes(), K, rgb.shape[:2]


def build_clip(frames: list[FrameInput], faces: np.ndarray, marker_ids: np.ndarray, device: str = "cpu") -> dict[str, np.ndarray]:
    """Compute labels for one clip. faces is the shared 778-vertex MANO topology."""
    num_frames = len(frames)
    arrays = {
        "joints_cam": np.full((num_frames, 2, 21, 3), np.nan, dtype=np.float32),
        "verts_cam": np.full((num_frames, 2, 195, 3), np.nan, dtype=np.float32),
        "hand_valid": np.zeros((num_frames, 2), dtype=bool),
        "dist_joints": np.full((num_frames, 2, 21), np.nan, dtype=np.float32),
        "dist_verts": np.full((num_frames, 2, 195), np.nan, dtype=np.float32),
        "vis_joints": np.zeros((num_frames, 2, 21), dtype=bool),
        "vis_verts": np.zeros((num_frames, 2, 195), dtype=bool),
        "K": np.zeros((num_frames, 3, 3), dtype=np.float32),
        "w2c": np.zeros((num_frames, 4, 4), dtype=np.float32),
    }
    encoded: list[bytes] = []
    faces_t = torch.as_tensor(faces, device=device, dtype=torch.long)
    for t, frame in enumerate(frames):
        blob, K, (height, width) = _encode_rgb(frame.rgb, frame.K)
        encoded.append(blob)
        arrays["K"][t] = K
        arrays["w2c"][t] = frame.w2c
        scale = VISIBILITY_RASTER_LONG_SIDE / max(height, width)
        raster_h, raster_w = max(1, round(height * scale)), max(1, round(width * scale))
        fx, fy = K[0, 0] * raster_w / width, K[1, 1] * raster_h / height
        cx, cy = K[0, 2] * raster_w / width, K[1, 2] * raster_h / height

        hand_tris = [
            None if verts is None else verts[faces].astype(np.float32) for verts in frame.verts_cam
        ]
        occluders = [tris for tris in hand_tris if tris is not None]
        if frame.object_tris_cam is not None and len(frame.object_tris_cam):
            occluders.append(frame.object_tris_cam.astype(np.float32))
        depth = rasterize_depth(np.concatenate(occluders), fx, fy, cx, cy, raster_h, raster_w) if occluders else None

        for side in range(2):
            verts, joints = frame.verts_cam[side], frame.joints_cam[side]
            if verts is None or joints is None:
                continue
            markers = verts[marker_ids]
            arrays["joints_cam"][t, side] = joints
            arrays["verts_cam"][t, side] = markers
            arrays["hand_valid"][t, side] = True
            arrays["vis_joints"][t, side] = point_visibility(joints, depth, fx, fy, cx, cy, JOINT_VISIBILITY_TOLERANCE_M)
            arrays["vis_verts"][t, side] = point_visibility(markers, depth, fx, fy, cx, cy, VERTEX_VISIBILITY_TOLERANCE_M)

            # Contact partners are the objects plus the opposite hand, never named separately.
            partners = []
            if frame.object_tris_cam is not None and len(frame.object_tris_cam):
                partners.append(torch.as_tensor(frame.object_tris_cam, device=device, dtype=torch.float32))
            if frame.verts_cam[1 - side] is not None:
                other = torch.as_tensor(frame.verts_cam[1 - side], device=device, dtype=torch.float32)
                partners.append(other[faces_t])
            if not partners:
                continue
            tris = torch.cat(partners)
            query = torch.as_tensor(np.concatenate([joints, markers]), device=device, dtype=torch.float32)
            distance = points_to_mesh_distance(query, tris).cpu().numpy()
            arrays["dist_joints"][t, side] = distance[:21]
            arrays["dist_verts"][t, side] = distance[21:]
    offsets = np.cumsum([0] + [len(blob) for blob in encoded], dtype=np.int64)
    arrays["jpeg_bytes"] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    arrays["jpeg_offsets"] = offsets
    return arrays


def save_clip(path: Path, arrays: dict[str, np.ndarray], meta: ClipMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**arrays, "meta_json": np.frombuffer(json.dumps(meta.__dict__).encode(), dtype=np.uint8)}
    temporary = path.with_suffix(".tmp.npz")
    # Stored rather than compressed: the JPEGs already are, and reading stays fast.
    np.savez(temporary, **payload)
    temporary.replace(path)


def decode_frame(jpeg_bytes: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    blob = jpeg_bytes[int(offsets[index]) : int(offsets[index + 1])]
    bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"corrupt JPEG at frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_manifest(output_root: Path) -> Path:
    """Index every clip under output_root so the loader never has to scan the tree."""
    rows = []
    for path in sorted(output_root.rglob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as data:
            meta = json.loads(bytes(data["meta_json"]).decode())
        rows.append({"path": str(path.relative_to(output_root)), **{k: meta[k] for k in ("dataset", "sequence", "split")}})
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    return manifest
