"""Evaluate a fixed checkpoint and its validation-selected contact threshold."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_manifest
from .inference import predict
from .metrics import constant_records, metrics
from .model import JointTactileHead
from .runtime import peak_memory, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test", "diagnostic", "confirmation"], default="val")
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto")
    parser.add_argument("--output", required=True, help="New local output directory")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-only", action="store_true", help="Do not use tactile targets or calculate metrics")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new evaluation directory")
    dataset = load_manifest(args.manifest, args.data_root)
    selected = [row for row in dataset["rows"] if row["split"] == args.split]
    if not selected:
        parser.error("Selected split is empty")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" not in checkpoint:
        parser.error("Use a checkpoint written by this portable trainer; legacy weights can warm-start training")
    config, stats = checkpoint["config"], checkpoint["stats"]
    if not args.prediction_only and "threshold" not in checkpoint:
        parser.error("Metrics require best.pt with its validation-selected threshold")
    if stats != dataset["stats"]:
        parser.error("Manifest statistics differ from checkpoint training statistics")
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = JointTactileHead(stats["contact_prior"], stats["positive_log_median"], config["condition"], width=config["width"], context=config["context"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    records, forward_s = predict(model, selected, stats, device, config["batch_size"], require_targets=not args.prediction_only)
    output.mkdir(parents=True, exist_ok=False)
    result = {"task": config["task"], "split": args.split, "episodes": len(selected), "checkpoint_step": checkpoint["step"],
              "prediction_forward_s": forward_s, "torch_peak_bytes": peak_memory(device),
              "input": "cached main-RGB-derived features; no GT wrist/tactile/pose input", "causal_future_prediction": False}
    if not args.prediction_only:
        result["metrics"] = metrics(records, checkpoint["threshold"])
        result["constant_baselines"] = {"zero": constant_records(records, np.zeros(12))}
        for name, key in [("train_mean", "train_patch_mean"), ("train_median", "train_patch_median")]:
            if key in stats:
                result["constant_baselines"][name] = constant_records(records, stats[key])
    if args.save_predictions:
        folder = output / "predictions"
        folder.mkdir()
        for row in records:
            # Episode IDs are metadata, never paths supplied for writing.
            name = row["name"]
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Unsafe episode filename")
            np.savez_compressed(folder / (name + ".npz"), **{k: v for k, v in row.items() if k not in {"name", "task"}})
    write_json(output / "metrics.json", result)
    print(json.dumps({"state": "evaluation_complete", "split": args.split, "episodes": len(selected), "metrics_written": not args.prediction_only}))


if __name__ == "__main__":
    main()
