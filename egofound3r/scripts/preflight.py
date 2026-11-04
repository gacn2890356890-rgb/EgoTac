"""Check everything a full run depends on, before it queues for GPUs. Exits nonzero on any problem."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from egotac.config import accumulation_steps, load_config
from egotac.data.dataset import ClipDataset
from egotac.data.splits import PAPER_TABLE_7

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    problems: list[str] = []
    warnings: list[str] = []

    upstream = Path(config["model"]["upstream_config"])
    paths = {"upstream config": upstream}
    if config["model"].get("checkpoint"):
        paths["EgoFound3R checkpoint"] = Path(config["model"]["checkpoint"])
    if upstream.exists():
        with upstream.open("rb") as handle:
            paths["VGGT-Omega checkpoint"] = Path(tomllib.load(handle)["marker_runtime"]["vggt_checkpoint_path"])
    for name, path in paths.items():
        if not path.is_file():
            problems.append(f"{name} missing: {path}")

    gpus = torch.cuda.device_count()
    train = config["train"]
    steps = accumulation_steps(train, max(gpus, 1))
    print(f"GPUs on this node: {gpus}; batch {train['batch_size']} per GPU, accumulation {steps}, target total batch {train.get('total_batch_size')}")

    clips = Path(config["data"]["clips"])
    manifest = clips / "manifest.jsonl"
    if not manifest.exists():
        problems.append(f"manifest missing: {manifest}; run scripts/convert_all.sh")
    else:
        rows = [json.loads(line) for line in manifest.open()]
        sequences: dict[str, set] = defaultdict(set)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows:
            counts[row["dataset"], row["split"]] += 1
            if row["split"] == "train":
                sequences[row["dataset"]].add(row["sequence"])
        print("dataset      train seqs (paper)  train clips  test clips (paper)")
        for dataset in config["data"].get("weights", {}):
            paper_train, paper_test = PAPER_TABLE_7.get(dataset, (0, 0))
            found = len(sequences[dataset])
            print(f"{dataset:12s} {found:6d} ({paper_train:5d}) {counts[dataset, 'train']:12d} {counts[dataset, 'test']:7d} ({paper_test:4d})")
            if not counts[dataset, "train"]:
                problems.append(f"{dataset}: no training clips")
            elif found != paper_train:
                warnings.append(f"{dataset}: {found} training sequences, the paper uses {paper_train}")
            if not counts[dataset, "test"]:
                problems.append(f"{dataset}: no test clips")
        if not problems:
            data = config["data"]
            sample = ClipDataset(clips, "train", data["image_size"], data["image_mode"], set(data.get("weights") or []) or None)[0]
            if not np.isfinite(sample["images"].numpy()).all():
                problems.append("first training clip has non-finite pixels")

    for warning in warnings:
        print("WARNING:", warning)
    for problem in problems:
        print("FAILED:", problem)
    if problems:
        raise SystemExit(1)
    print("preflight OK")


if __name__ == "__main__":
    main()
