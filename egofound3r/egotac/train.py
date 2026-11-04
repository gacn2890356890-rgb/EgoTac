"""Train EgoFound3R, original or diffusion variant, from one config.

Single GPU:  python -m egotac.train --config configs/smoke.toml
Multi GPU:   torchrun --nproc_per_node=8 -m egotac.train --config configs/full.toml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which the official requirements also support
    import tomli as tomllib

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data.dataset import ClipDataset, MixedInfiniteSampler
from .evaluate import run_evaluation
from .losses import DEFAULT_WEIGHTS, total_loss
from .model import EgoTacModel


def load_config(path: Path, overrides: list[str]) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not raw:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        section, _, name = key.partition(".")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        config.setdefault(section, {})[name] = value
    return config


def learning_rate(step: int, train: dict) -> float:
    """Table 5: linear warmup, then cosine decay to the floor."""
    base, floor, warmup, total = train["lr"], train["lr_min"], train["warmup_steps"], train["steps"]
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * progress))


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="override, e.g. train.steps=300")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    train_cfg, data_cfg, model_cfg = config["train"], config["data"], config["model"]

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = train_cfg.get("seed", 0) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output = Path(train_cfg["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2))

    variant = train_cfg["variant"]
    model = EgoTacModel(model_cfg["upstream_config"], model_cfg["checkpoint"], variant, config.get("diffusion")).to(device)
    model.train()
    start_step = 0
    resume = output / "latest.pt"
    if resume.exists():
        model.load_egotac_checkpoint(resume)
    # Before DDP wraps the model: DDP copies rank 0's weights to every other rank at init.
    integrity = model.verify_weights() if rank == 0 else None
    if rank == 0:
        print("weight integrity:", json.dumps(integrity), flush=True)
        (output / "weight_integrity.json").write_text(json.dumps(integrity, indent=2))
    wrapped = (
        DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True, broadcast_buffers=False)
        if distributed
        else model
    )
    parameters = model.trainable_parameters()
    trainable_names = {name for name, parameter in model.marker.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=train_cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=train_cfg["weight_decay"])

    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        del state
        if rank == 0:
            print(f"resumed from step {start_step}", flush=True)

    datasets = set(data_cfg["weights"]) if data_cfg.get("weights") else None
    train_set = ClipDataset(data_cfg["clips"], "train", data_cfg["image_size"], data_cfg["image_mode"], datasets, augment=data_cfg.get("augment", True))
    sampler = MixedInfiniteSampler(train_set.rows, data_cfg.get("weights"), rank=rank, seed=train_cfg.get("seed", 0) + start_step)
    loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], sampler=sampler, num_workers=data_cfg["workers"], pin_memory=True, drop_last=True, persistent_workers=data_cfg["workers"] > 0)
    iterator = iter(loader)
    eval_loader = None
    if rank == 0 and train_cfg.get("eval_every"):
        eval_set = ClipDataset(data_cfg["clips"], "test", data_cfg["image_size"], data_cfg["image_mode"], datasets)
        eval_loader = DataLoader(eval_set, batch_size=train_cfg.get("eval_batch_size", 2), num_workers=2, pin_memory=True)

    weights = {**DEFAULT_WEIGHTS, **config.get("loss", {}).get("weights", {})}
    alpha = config.get("loss", {}).get("alpha", 0.2)
    strides, stride_weights = train_cfg["strides"], train_cfg["stride_weights"]
    log_path = output / "train_log.jsonl"
    stride_rng = np.random.default_rng(train_cfg.get("seed", 0) + 7 * rank + start_step)
    began = time.time()
    skipped = 0

    def save(step: int, name: str) -> None:
        if rank != 0:
            return
        payload = model.checkpoint_payload()
        payload["step"] = step
        if name == "latest.pt":
            payload["optimizer"] = optimizer.state_dict()
            payload["resume_marker_fp32"] = {
                key: value.detach().cpu() for key, value in model.marker.state_dict().items() if key in trainable_names
            }
        temporary = output / f".{name}.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / name)

    for step in range(start_step, train_cfg["steps"]):
        batch = to_device(next(iterator), device)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, train_cfg)
        stride = int(stride_rng.choice(strides, p=np.array(stride_weights) / sum(stride_weights)))
        phase = int(stride_rng.integers(0, stride))
        outputs = wrapped(batch["images"], stride=stride, phase=phase, batch=batch)
        loss, terms = total_loss(outputs, batch, weights, variant, alpha, outputs["global_anchor_indices"].to(device))
        finite = torch.tensor(float(torch.isfinite(loss)), device=device)
        if distributed:
            # Every rank must agree, or the ones that skip would leave the others hanging in backward.
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        optimizer.zero_grad(set_to_none=True)
        if finite.item() < 1.0:
            skipped += 1
            print(f"rank {rank}: skipped step {step}, non-finite loss, {skipped} so far: {terms}", flush=True)
            if skipped > train_cfg.get("max_skipped_steps", 20):
                raise FloatingPointError(f"more than {train_cfg.get('max_skipped_steps', 20)} non-finite steps; stopping")
            del outputs, loss
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, train_cfg["grad_clip"])
        if not torch.isfinite(grad_norm):
            # clip_grad_norm_ would have scaled every gradient by NaN; updating here destroys the model.
            skipped += 1
            print(f"rank {rank}: skipped step {step}, non-finite gradient, {skipped} so far", flush=True)
            if skipped > train_cfg.get("max_skipped_steps", 20):
                raise FloatingPointError("too many non-finite steps; stopping")
            continue
        optimizer.step()

        if rank == 0 and (step % train_cfg["log_every"] == 0 or step == train_cfg["steps"] - 1):
            row = {
                "step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"],
                "stride": stride, "seconds": round(time.time() - began, 1), **terms,
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0.0,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        done = step + 1
        if train_cfg.get("save_every") and done % train_cfg["save_every"] == 0:
            save(done, "latest.pt")
        if eval_loader is not None and done % train_cfg["eval_every"] == 0:
            report = run_evaluation(model, eval_loader, device, stride=5)
            if rank == 0:
                with (output / "eval_log.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": done, "metrics": report}) + "\n")
                print(json.dumps({"eval_step": done, "metrics": report}), flush=True)
        if distributed:
            dist.barrier() if done % train_cfg["log_every"] == 0 else None
    save(train_cfg["steps"], "final.pt")
    if rank == 0:
        print(f"saved {output / 'final.pt'}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
