"""Check the world version keeps the official input/output contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

TACTILE_WORDS = ("contact",)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}/{key}")
    elif torch.is_tensor(value):
        yield prefix, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--same-weights", action="store_true", help="also require non-tactile outputs to be bit-identical")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    official = dict(flatten(torch.load(args.official / "inference_output.pt", map_location="cpu", weights_only=False)))
    world = dict(flatten(torch.load(args.world / "inference_output.pt", map_location="cpu", weights_only=False)))
    only_official, only_world = sorted(set(official) - set(world)), sorted(set(world) - set(official))
    mismatched = [
        f"{key}: {tuple(official[key].shape)}/{official[key].dtype} vs {tuple(world[key].shape)}/{world[key].dtype}"
        for key in official.keys() & world.keys()
        if official[key].shape != world[key].shape or official[key].dtype != world[key].dtype
    ]
    changed = sorted(k for k in official.keys() & world.keys() if official[k].shape == world[k].shape and not torch.equal(official[k], world[k]))
    tactile_changed = [k for k in changed if any(word in k for word in TACTILE_WORDS)]
    other_changed = [k for k in changed if k not in tactile_changed]
    files_official = sorted(p.name for p in args.official.iterdir() if p.is_file())
    files_world = sorted(p.name for p in args.world.iterdir() if p.is_file())
    passed = not only_official and not only_world and not mismatched and files_official == files_world
    if args.same_weights:
        passed = passed and not other_changed and bool(tactile_changed)
    report = {
        "passed": passed,
        "tensors_compared": len(official),
        "only_in_official": only_official,
        "only_in_world": only_world,
        "shape_or_dtype_mismatch": mismatched,
        "tactile_fields_changed": len(tactile_changed),
        "non_tactile_fields_changed": other_changed,
        "same_files": files_official == files_world,
        "files": files_world,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
