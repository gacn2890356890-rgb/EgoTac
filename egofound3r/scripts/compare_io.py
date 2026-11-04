"""Check the diffusion version keeps the official input/output contract."""
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
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--same-weights", action="store_true", help="also require non-tactile outputs to be bit-identical")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    official = dict(flatten(torch.load(args.official / "inference_output.pt", map_location="cpu", weights_only=False)))
    diffusion = dict(flatten(torch.load(args.diffusion / "inference_output.pt", map_location="cpu", weights_only=False)))
    only_official, only_diffusion = sorted(set(official) - set(diffusion)), sorted(set(diffusion) - set(official))
    mismatched = [
        f"{key}: {tuple(official[key].shape)}/{official[key].dtype} vs {tuple(diffusion[key].shape)}/{diffusion[key].dtype}"
        for key in official.keys() & diffusion.keys()
        if official[key].shape != diffusion[key].shape or official[key].dtype != diffusion[key].dtype
    ]
    changed = sorted(k for k in official.keys() & diffusion.keys() if official[k].shape == diffusion[k].shape and not torch.equal(official[k], diffusion[k]))
    tactile_changed = [k for k in changed if any(word in k for word in TACTILE_WORDS)]
    other_changed = [k for k in changed if k not in tactile_changed]
    files_official = sorted(p.name for p in args.official.iterdir() if p.is_file())
    files_diffusion = sorted(p.name for p in args.diffusion.iterdir() if p.is_file())
    passed = not only_official and not only_diffusion and not mismatched and files_official == files_diffusion
    if args.same_weights:
        passed = passed and not other_changed and bool(tactile_changed)
    report = {
        "passed": passed,
        "tensors_compared": len(official),
        "only_in_official": only_official,
        "only_in_diffusion": only_diffusion,
        "shape_or_dtype_mismatch": mismatched,
        "tactile_fields_changed": len(tactile_changed),
        "non_tactile_fields_changed": other_changed,
        "same_files": files_official == files_diffusion,
        "files": files_diffusion,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
