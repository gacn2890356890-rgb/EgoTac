"""Evaluate a checkpoint, including the official weights (zero-shot), with the paper's metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import ClipDataset
from .metrics import MetricAccumulator
from .model import EgoTacModel


@torch.inference_mode()
def run_evaluation(model: EgoTacModel, loader: DataLoader, device: torch.device, stride: int = 5, max_batches: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    accumulator = MetricAccumulator()
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        tensors = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(tensors["images"], stride=stride)
        accumulator.update(outputs, tensors)
    model.train(was_training)
    return accumulator.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-config", required=True)
    parser.add_argument("--checkpoint", required=True, help="official EgoFound3R.pt or an EgoTac training checkpoint")
    parser.add_argument("--variant", default="baseline", choices=["baseline", "diffusion"])
    parser.add_argument("--clips", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-mode", default="pad")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    is_egotac = "egotac_variant" in payload
    options = payload.get("egotac_diffusion_options")
    del payload
    model = EgoTacModel(args.upstream_config, None if is_egotac else args.checkpoint, args.variant, options).to(device)
    if is_egotac:
        model.load_egotac_checkpoint(args.checkpoint)
    print("weight integrity:", json.dumps(model.verify_weights()), flush=True)
    dataset = ClipDataset(args.clips, args.split, args.image_size, args.image_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    report = run_evaluation(model, loader, device, args.stride, args.max_batches)
    text = json.dumps({"checkpoint": args.checkpoint, "variant": args.variant, "image_size": args.image_size, "image_mode": args.image_mode, "metrics": report}, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
