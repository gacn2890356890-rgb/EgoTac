"""Portable descriptors for externally stored v15 feature bundles."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

INPUT_KEYS = ("grid", "geometry", "uv", "geometry_valid", "local_grid", "local_uv", "local_valid")
TARGET_KEYS = ("target", "target_valid")


def load_manifest(path, data_root=None):
    path = Path(path).resolve()
    dataset = json.loads(path.read_text())
    root = Path(data_root).resolve() if data_root else path.parent
    if dataset.get("task", "same_frame_window_estimation") != "same_frame_window_estimation":
        raise ValueError("This release implements same-frame estimation, not future world modeling")
    rows = dataset.get("rows", [])
    if not rows:
        raise ValueError("Manifest rows are empty; supply your external dataset manifest")
    names = set()
    for row in rows:
        if row["name"] in names:
            raise ValueError(f"Duplicate episode/split identity: {row['name']}")
        names.add(row["name"])
        candidate = Path(row["file"])
        row["file"] = str(candidate if candidate.is_absolute() else root / candidate)
        if row.get("split") not in {"train", "val", "test", "diagnostic", "confirmation"}:
            raise ValueError(f"Unknown split for {row['name']}")
    stats = dataset["stats"]
    for key in ("contact_prior", "positive_log_median", "response_scale"):
        values = np.asarray(stats[key], dtype=float)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError(f"Expected 12 finite train-only values for {key}")
    if np.any(np.asarray(stats["response_scale"]) <= 0):
        raise ValueError("response_scale must be positive")
    if np.any(np.asarray(stats["contact_prior"]) < 0) or np.any(np.asarray(stats["contact_prior"]) > 1):
        raise ValueError("contact_prior must be a probability")
    if np.asarray(stats["geometry_mean"]).shape != (14,):
        raise ValueError("Expected geometry_mean with shape [14]")
    if np.asarray(stats["geometry_std"]).shape != (14,):
        raise ValueError("Expected geometry_std with shape [14]")
    if not np.isfinite(stats["geometry_mean"]).all() or not np.isfinite(stats["geometry_std"]).all():
        raise ValueError("Geometry normalization must be finite")
    if np.any(np.asarray(stats["geometry_std"]) <= 0):
        raise ValueError("geometry_std must be positive")
    return dataset


def load_row(path, require_targets=True):
    row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = INPUT_KEYS + (TARGET_KEYS if require_targets else ()) + ("frame_indices", "timestamps")
    missing = set(required) - set(row)
    if missing:
        raise ValueError(f"Bundle missing fields: {sorted(missing)}")
    n = len(row["grid"])
    shapes = {
        "grid": (n, 64, 1024), "geometry": (n, 12, 14), "uv": (n, 12, 2),
        "geometry_valid": (n, 12), "local_grid": (n, 32, 1024),
        "local_uv": (n, 32, 2), "local_valid": (n, 32),
        "frame_indices": (n,), "timestamps": (n,),
    }
    if require_targets:
        shapes.update(target=(n, 12), target_valid=(n, 12))
    for key, shape in shapes.items():
        if tuple(row[key].shape) != shape:
            raise ValueError(f"{key}: expected {shape}, got {tuple(row[key].shape)}")
    if n < 2 or not torch.all(row["timestamps"][1:] > row["timestamps"][:-1]):
        raise ValueError("Need at least two frames with strictly increasing timestamps")
    for key in ("geometry_valid", "local_valid") + (("target_valid",) if require_targets else ()):
        if row[key].dtype != torch.bool:
            raise ValueError(f"{key} must be bool")
    if require_targets and not row["target_valid"].any():
        raise ValueError("Episode has no valid tactile labels")
    for key in INPUT_KEYS + (("target",) if require_targets else ()):
        value = row[key]
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite {key}; sanitize values while preserving validity masks")
    if require_targets and (row["target"][row["target_valid"]] < 0).any():
        raise ValueError("Valid response values must be nonnegative")
    peak_keys = {"native_bin_peak", "native_bin_peak_valid"}
    if peak_keys & row.keys():
        if not peak_keys <= row.keys():
            raise ValueError("Supply both native peak values and their valid mask")
        if row["native_bin_peak"].shape != (n, 12) or row["native_bin_peak_valid"].shape != (n, 12):
            raise ValueError("Native peak arrays must have shape [T,12]")
        if row["native_bin_peak_valid"].dtype != torch.bool:
            raise ValueError("Native peak validity must be bool")
        if not torch.isfinite(row["native_bin_peak"]).all():
            raise ValueError("Non-finite native peak values")
    return row


def model_inputs(row, indices, stats, device):
    result = {k: row[k][indices].to(device) for k in INPUT_KEYS}
    mean = torch.as_tensor(stats["geometry_mean"], device=device)
    std = torch.as_tensor(stats["geometry_std"], device=device)
    result["geometry"] = ((result["geometry"] - mean) / std).clamp(-5, 5)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 checks all episodes in the selected split")
    args = parser.parse_args()
    dataset = load_manifest(args.manifest, args.data_root)
    selected = [r for r in dataset["rows"] if r["split"] == args.split]
    if not selected:
        parser.error("Selected split is empty")
    checked = selected[:args.limit] if args.limit > 0 else selected
    for row in checked:
        value = load_row(row["file"])
        print(json.dumps({"episode": row["name"], "frames": len(value["grid"]), "valid_values": int(value["target_valid"].sum())}))
    print(json.dumps({"state": "data_checked", "split": args.split, "episodes_checked": len(checked), "neural_forward_calls": 0}))


if __name__ == "__main__":
    main()
