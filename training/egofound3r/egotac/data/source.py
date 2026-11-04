"""What the H2O, HOI4D, ARCTIC, and HOT3D converters share: window layout, the conversion loop, the CLI.

A converter supplies discover(root) -> {unit: sequence} and open_source(unit, ctx) -> a source with
total, fps, prepare(indices), and frame(index) -> FrameInput. A unit is what gets converted, usually
the sequence itself; HOT3D splits one source sequence over many clip archives.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from .clip import CLIP_FRAMES, HAND_RATE_HZ, ClipMeta, build_clip, save_clip, write_manifest
from .mano import ManoLayer, marker_ids_195
from .splits import plan_splits

MAX_OBJECT_FACES = 4000


def hand_rate_indices(start: int, fps: float) -> list[int]:
    """Source frames of one window, rounded as the official TemporalRateContract.high_indices does."""
    return [start + int(round(i * fps / HAND_RATE_HZ)) for i in range(CLIP_FRAMES)]


def window_starts(total: int, fps: float, hop: int) -> list[int]:
    """hop is in hand-rate frames. Test windows use CLIP_FRAMES, so they do not overlap (Appendix D)."""
    span = hand_rate_indices(0, fps)[-1]
    step = max(1, int(round(fps / HAND_RATE_HZ)))
    return list(range(0, total - span, hop * step))


def load_mesh_tris(path: Path, to_meters: float | None = None) -> np.ndarray:
    """Triangles [F, 3, 3] in meters. With to_meters unset, units are inferred from the mesh extent."""
    import fast_simplification

    mesh = trimesh.load(path, process=False, force="mesh")
    points, faces = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces)
    if len(faces) > MAX_OBJECT_FACES:
        points, faces = fast_simplification.simplify(points, faces, target_count=MAX_OBJECT_FACES)
    if to_meters is None:
        # No hand-held object here is over 5 m across, and none is under 5 mm.
        to_meters = 0.001 if float(np.ptp(points, axis=0).max()) > 5.0 else 1.0
    return points[faces] * to_meters


@dataclass
class Context:
    root: Path
    layers: dict
    device: str
    args: argparse.Namespace


def convert_unit(dataset: str, unit: str, sequence: str, split: str, source, output: Path, faces: np.ndarray, markers: np.ndarray, train_hop: int, device: str) -> int:
    hop = CLIP_FRAMES if split == "test" else train_hop
    starts = window_starts(source.total, source.fps, hop)
    folder = output / dataset / hashlib.sha1(unit.encode()).hexdigest()[:16]
    pending = [start for start in starts if not (folder / f"{start:06d}.npz").exists()]
    if pending:
        source.prepare({index for start in pending for index in hand_rate_indices(start, source.fps)})
    for start in pending:
        window = hand_rate_indices(start, source.fps)
        arrays = build_clip([source.frame(index) for index in window], faces, markers, device=device)
        save_clip(folder / f"{start:06d}.npz", arrays, ClipMeta(dataset, sequence, split, window))
    return len(starts)


def run(dataset: str, description: str, discover: Callable, open_source: Callable, is_fixed_test: Callable | None = None, fix_left_shapedirs: bool = False, add_arguments: Callable | None = None) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-hop", type=int, default=10)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int, default=None, help="convert only this many units, to check a few sequences first")
    if add_arguments:
        add_arguments(parser)
    args = parser.parse_args()
    if args.mano_dir is None:
        parser.error("--mano-dir is required")
    layers = {side: ManoLayer(args.mano_dir, side).to(args.device) for side in ("left", "right")}
    if fix_left_shapedirs and (layers["left"].shapedirs[:, 0] - layers["right"].shapedirs[:, 0]).abs().sum() < 1:
        # The published left-hand file has this sign flipped; the dataset's own toolkit corrects it.
        layers["left"].shapedirs[:, 0] *= -1
    context = Context(args.root, layers, args.device, args)
    units = discover(args.root)
    if not units:
        raise SystemExit(f"no {dataset} sequences found under {args.root}; see the layout at the top of the converter")
    plan = plan_splits(dataset, sorted(set(units.values())), is_fixed_test)
    selected = sorted(unit for unit, sequence in units.items() if sequence in plan)
    if args.limit is not None:
        selected = selected[: args.limit]
    index, count = (int(part) for part in args.shard.split("/"))
    faces = layers["left"].faces.cpu().numpy()
    markers = marker_ids_195()
    for unit in selected[index::count]:
        sequence = units[unit]
        made = convert_unit(dataset, unit, sequence, plan[sequence], open_source(unit, context), args.output, faces, markers, args.train_hop, args.device)
        print(f"{unit}: {made} clips", flush=True)
    if count == 1:
        print(f"manifest: {write_manifest(args.output)}")
