"""Prove the installed environment can run this repository. Exits nonzero on any problem."""
from __future__ import annotations

import importlib
import sys
from importlib import metadata
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
IMPORTS = ["cv2", "fast_simplification", "numpy", "remotezip", "scipy", "PIL", "tomli", "torch", "torchvision", "trimesh", "pytest", "huggingface_hub"]
OWN_MODULES = [
    "egotac.config", "egotac.data.clip", "egotac.data.dataset", "egotac.data.labels", "egotac.data.mano", "egotac.data.splits",
    "egotac.data.convert_oakink2", "egotac.data.convert_taco", "egohandmetric_prompt.world_model", "egotac.evaluate", "egotac.integrity",
    "egotac.losses", "egotac.metrics", "egotac.model", "egotac.train",
]


def main() -> None:
    from egotac.upstream import ensure_upstream

    ensure_upstream()
    problems = []
    if sys.version_info[:2] != (3, 10):
        problems.append(f"Python {sys.version.split()[0]}, expected 3.10")
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, wanted = line.split("==")
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed")
            continue
        if found.split("+")[0] != wanted:
            problems.append(f"{name} is {found}, requirements.txt pins {wanted}")
    for name in IMPORTS + OWN_MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:  # any import failure is the thing being reported
            problems.append(f"import {name}: {type(error).__name__}: {error}")

    import torch

    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        torch.ones(4, 4, device="cuda").sum().item()
    else:
        print("note: no GPU on this node; that is fine on a login node, not on a training node")
    if problems:
        print("\n".join(f"FAILED: {problem}" for problem in problems))
        raise SystemExit(1)
    print("environment OK")


if __name__ == "__main__":
    main()
