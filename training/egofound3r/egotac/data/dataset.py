"""PyTorch side of the unified clip format, plus the multi-dataset sampling the paper trains with."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .clip import decode_frame
from .labels import CONTACT_THRESHOLD_JOINT_M, CONTACT_THRESHOLD_VERTEX_M

IMAGE_MODES = ("pad", "center_crop", "aspect")
PATCH_MULTIPLE = 16


def _geometry(height: int, width: int, size: int, mode: str, rng: np.random.Generator | None):
    """Return (scale, x_shift, y_shift, out_h, out_w) mapping stored pixels to network pixels."""
    if mode == "aspect":
        scale = size / max(height, width)
        out_h = max(PATCH_MULTIPLE, round(height * scale / PATCH_MULTIPLE) * PATCH_MULTIPLE)
        out_w = max(PATCH_MULTIPLE, round(width * scale / PATCH_MULTIPLE) * PATCH_MULTIPLE)
        return (out_w / width, out_h / height, 0.0, 0.0, out_h, out_w)
    if mode == "pad":
        scale = size / max(height, width)
        x_shift = (size - width * scale) / 2
        y_shift = (size - height * scale) / 2
        return (scale, scale, x_shift, y_shift, size, size)
    if mode == "center_crop":
        crop = min(height, width)
        if rng is not None:
            crop = int(crop * rng.uniform(0.85, 1.0))
            x0 = rng.integers(0, width - crop + 1)
            y0 = rng.integers(0, height - crop + 1)
        else:
            x0, y0 = (width - crop) // 2, (height - crop) // 2
        scale = size / crop
        return (scale, scale, -x0 * scale, -y0 * scale, size, size)
    raise ValueError(f"image mode must be one of {IMAGE_MODES}, got {mode!r}")


def load_manifest(root: Path, split: str, datasets: set[str] | None = None) -> list[dict]:
    manifest = Path(root) / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run a converter first")
    rows = [json.loads(line) for line in manifest.open()]
    return [r for r in rows if r["split"] == split and (datasets is None or r["dataset"] in datasets)]


class ClipDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int = 256, image_mode: str = "pad", datasets: set[str] | None = None, augment: bool = False, seed: int = 0):
        self.root = Path(root)
        self.rows = load_manifest(self.root, split, datasets)
        if not self.rows:
            raise ValueError(f"no clips for split={split!r}, datasets={datasets} under {root}")
        self.image_size = image_size
        self.image_mode = image_mode
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        # Damaged clips must not end a long run, but each one is reported rather than hidden.
        for attempt in range(5):
            position = (index + attempt) % len(self.rows)
            try:
                return self._load(position)
            except (ValueError, OSError, zipfile.BadZipFile, cv2.error) as error:
                print(f"skipping damaged clip {self.rows[position]['path']}: {error}", flush=True)
        raise RuntimeError(f"five consecutive damaged clips starting at index {index}")

    def _load(self, index: int) -> dict:
        row = self.rows[index]
        with np.load(self.root / row["path"]) as npz:
            data = {key: npz[key] for key in npz.files if key != "meta_json"}
        num_frames = data["joints_cam"].shape[0]
        first = decode_frame(data["jpeg_bytes"], data["jpeg_offsets"], 0)
        height, width = first.shape[:2]
        rng = np.random.default_rng() if self.augment else None
        sx, sy, x_shift, y_shift, out_h, out_w = _geometry(height, width, self.image_size, self.image_mode, rng)
        matrix = np.array([[sx, 0.0, x_shift], [0.0, sy, y_shift]], dtype=np.float64)
        images = np.empty((num_frames, out_h, out_w, 3), dtype=np.uint8)
        for t in range(num_frames):
            frame = first if t == 0 else decode_frame(data["jpeg_bytes"], data["jpeg_offsets"], t)
            images[t] = cv2.warpAffine(frame, matrix, (out_w, out_h), flags=cv2.INTER_AREA if max(sx, sy) < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if self.augment:
            gain = rng.uniform(0.8, 1.2)
            images = np.clip(images.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        K = data["K"].astype(np.float64).copy()
        K[:, 0, 0] *= sx
        K[:, 1, 1] *= sy
        K[:, 0, 2] = K[:, 0, 2] * sx + x_shift
        K[:, 1, 2] = K[:, 1, 2] * sy + y_shift
        joints = data["joints_cam"]
        verts = data["verts_cam"]
        hand_valid = data["hand_valid"]

        # Pixel coordinates at network resolution, with the upstream normalization u = x / (W - 1).
        with np.errstate(invalid="ignore", divide="ignore"):
            px = K[:, None, None, 0, 0] * joints[..., 0] / joints[..., 2] + K[:, None, None, 0, 2]
            py = K[:, None, None, 1, 1] * joints[..., 1] / joints[..., 2] + K[:, None, None, 1, 2]
        in_image = (px >= 0) & (px <= out_w - 1) & (py >= 0) & (py <= out_h - 1) & (joints[..., 2] > 1e-3)
        in_image &= hand_valid[..., None]
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = K[:, None, None, 0, 0] * verts[..., 0] / verts[..., 2] + K[:, None, None, 0, 2]
            vy = K[:, None, None, 1, 1] * verts[..., 1] / verts[..., 2] + K[:, None, None, 1, 2]
        vert_in_image = (vx >= 0) & (vx <= out_w - 1) & (vy >= 0) & (vy <= out_h - 1) & (verts[..., 2] > 1e-3)

        dist_j, dist_v = data["dist_joints"], data["dist_verts"]
        out = {
            "images": torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0,
            "K": torch.from_numpy(K).float(),
            "w2c": torch.from_numpy(data["w2c"]).float(),
            "joints": torch.from_numpy(np.nan_to_num(joints)).float(),
            "verts": torch.from_numpy(np.nan_to_num(verts)).float(),
            "hand_valid": torch.from_numpy(hand_valid),
            "in_view": torch.from_numpy(in_image.any(-1)),
            "uv": torch.from_numpy(np.nan_to_num(np.stack([px / max(out_w - 1, 1), py / max(out_h - 1, 1)], -1))).float(),
            "uv_valid": torch.from_numpy(in_image),
            "dist_joints": torch.from_numpy(np.nan_to_num(dist_j, posinf=0.0)).float(),
            "dist_verts": torch.from_numpy(np.nan_to_num(dist_v, posinf=0.0)).float(),
            "dist_joints_valid": torch.from_numpy(np.isfinite(dist_j) & hand_valid[..., None]),
            "dist_verts_valid": torch.from_numpy(np.isfinite(dist_v) & hand_valid[..., None]),
            "contact_joints": torch.from_numpy(np.nan_to_num(dist_j, nan=1.0) < CONTACT_THRESHOLD_JOINT_M).float(),
            "contact_verts": torch.from_numpy(np.nan_to_num(dist_v, nan=1.0) < CONTACT_THRESHOLD_VERTEX_M).float(),
            # A point outside the network crop cannot be seen, whatever the stored label says.
            "vis_joints": torch.from_numpy(data["vis_joints"] & in_image).float(),
            "vis_verts": torch.from_numpy(data["vis_verts"] & vert_in_image & hand_valid[..., None]).float(),
            "image_hw": torch.tensor([out_h, out_w]),
            "dataset": row["dataset"],
            "clip": row["path"],
        }
        return out


class MixedInfiniteSampler(Sampler):
    """Endless sampling with per-dataset proportions. Each rank draws its own independent stream."""

    def __init__(self, rows: list[dict], weights: dict[str, float] | None, rank: int = 0, seed: int = 0):
        self.seed = seed + 1000 * rank
        names = sorted({row["dataset"] for row in rows})
        wanted = {name for name, weight in (weights or {}).items() if weight > 0}
        missing = sorted(wanted - set(names))
        if missing:
            # A silently absent dataset would change the training mix without anyone noticing.
            raise ValueError(f"data.weights lists datasets with no training clips: {missing}; found {names}. Set a weight to 0 to train without it")
        self.indices = {name: np.array([i for i, r in enumerate(rows) if r["dataset"] == name]) for name in names}
        raw = np.array([(weights or {}).get(name, 1.0) for name in names], dtype=np.float64)
        if raw.sum() <= 0:
            raise ValueError("dataset weights must have a positive sum")
        self.names, self.probabilities = names, raw / raw.sum()

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            name = self.names[rng.choice(len(self.names), p=self.probabilities)]
            yield int(rng.choice(self.indices[name]))
