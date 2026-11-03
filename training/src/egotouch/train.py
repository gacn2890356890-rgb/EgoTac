"""Train the complete v15 tactile decoder on frozen main-view feature bundles."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .data import INPUT_KEYS, load_manifest, load_row, model_inputs
from .inference import predict
from .losses import training_loss
from .metrics import metrics, threshold_from_val
from .model import JointTactileHead
from .runtime import git_revision, peak_memory, save_checkpoint, select_device, synchronize, write_json


@contextmanager
def accelerator_lock(root, device):
    root.mkdir(parents=True, exist_ok=True)
    with (root / (".egotouch-" + device.type + ".lock")).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another EgoTouch job uses this output root; run experiments serially") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_config(config):
    if config["task"] != "same_frame_window_estimation":
        raise ValueError("Future visual/tactile forecasting is planned, not implemented in this release")
    if config["condition"] not in {"global", "hand"}:
        raise ValueError("condition must be global or hand")
    if not 1 <= config["steps"] <= 10000 or not 1 <= config["max_seconds"] <= 14400:
        raise ValueError("Per-fit budget is at most 10000 updates and 14400 seconds")
    for key in ("context", "batch_size", "validate_every", "log_every", "warmup_steps"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 2 <= config["context"] <= 40 or config["width"] % 4:
        raise ValueError("Need context 2..40 and width divisible by four")
    if config.get("contact_response_threshold", 1.) != 1.:
        raise ValueError("v15 loss/metrics use response>=1 as contact; changing this requires an explicit new protocol")
    for key in ("learning_rate", "gradient_clip"):
        if config[key] <= 0 or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite and positive")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--init-checkpoint", help="Optional explicit warm start; NOT bundled")
    parser.add_argument("--resume", help="Resume this output directory's last.pt, including optimizer/RNG")
    return parser


def run(args):
    config = json.loads(Path(args.config).read_text())
    for name in ("seed", "steps"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    validate_config(config)
    if args.resume and args.init_checkpoint:
        raise ValueError("Choose resume or initialization, not both")
    device = select_device(args.device)
    output = Path(args.output).resolve()
    started = time.monotonic()
    with accelerator_lock(output.parent, device):
        if args.resume:
            resume_path = Path(args.resume).resolve()
            if resume_path.parent != output or not resume_path.exists():
                raise ValueError("Resume the last checkpoint inside the same existing output directory")
        else:
            output.mkdir(parents=True, exist_ok=False)
        dataset = load_manifest(args.manifest, args.data_root)
        stats = dataset["stats"]
        train = [row for row in dataset["rows"] if row["split"] == "train"]
        val = [row for row in dataset["rows"] if row["split"] == "val"]
        if not train or not val:
            raise ValueError("Training and validation splits must both be nonempty")
        torch.manual_seed(config["seed"])
        rng = np.random.default_rng(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
        rows = {row["name"]: load_row(row["file"]) for row in train}
        tasks = {}
        peak_mode = {"native_bin_peak" in rows[row["name"]] for row in train}
        if len(peak_mode) != 1:
            raise ValueError("All training bundles must consistently include or omit native peak supervision")
        for row in train:
            if len(rows[row["name"]]["grid"]) < config["context"]:
                raise ValueError(f"Episode too short for context: {row['name']}")
            tasks.setdefault(row["task"], []).append(row)
        task_list = sorted(tasks)
        model = JointTactileHead(stats["contact_prior"], stats["positive_log_median"], config["condition"],
                                 width=config["width"], context=config["context"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        completed, previous_elapsed, best_score = 0, 0., None
        best_step, best_threshold, seen = None, None, set()
        history = []
        initialization = {"kind": "random"}
        if args.init_checkpoint:
            initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(initial["state_dict"], strict=True)
            with torch.no_grad():
                prior = torch.tensor(stats["contact_prior"], device=device).clamp(.001, .999)
                median = torch.tensor(stats["positive_log_median"], device=device).clamp_min(.1)
                model.prior_logit.copy_(torch.logit(prior))
                model.amplitude_offset.copy_(torch.log(torch.expm1(median)))
            initialization = {"kind": "warm_start", "checkpoint": str(Path(args.init_checkpoint).resolve()), "previous_step": initial.get("step"), "optimizer_reset": True, "prior_reset": "current_train_only"}
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            if checkpoint["config"] != config or checkpoint["stats"] != stats:
                raise ValueError("Resume config/statistics differ from the saved run")
            expected_rows = [(r["name"], r["task"], r["split"]) for r in dataset["rows"]]
            if checkpoint["row_identity"] != expected_rows or checkpoint["data_revision"] != dataset.get("data_revision"):
                raise ValueError("Resume dataset identity differs from the saved run")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            completed = checkpoint["step"]
            previous_elapsed = checkpoint["elapsed_s"]
            best_score = tuple(checkpoint["best_score"]) if checkpoint["best_score"] is not None else None
            best_step, best_threshold = checkpoint["best_step"], checkpoint["best_threshold"]
            initialization = checkpoint["initialization"]
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("accelerator_rng_state") is not None:
                if device.type == "cuda":
                    torch.cuda.set_rng_state(checkpoint["accelerator_rng_state"], device)
                elif hasattr(torch.mps, "set_rng_state"):
                    torch.mps.set_rng_state(checkpoint["accelerator_rng_state"])
            seen = set(checkpoint["seen_episodes"])
            if (output / "history.json").exists():
                history = json.loads((output / "history.json").read_text())
        deadline = started + max(0., config["max_seconds"] - previous_elapsed)
        scale = torch.as_tensor(stats["response_scale"], device=device)
        initial_parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        metadata = {"task": config["task"], "pid": os.getpid(), "command": sys.argv,
                    "code_revision": git_revision(), "data_revision": dataset.get("data_revision"),
                    "config": config, "initialization": initialization, "resume": bool(args.resume),
                    "device": str(device), "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                    "train_episodes": len(train), "val_episodes": len(val), "tasks": len(tasks),
                    "trainable_parameters": sum(p.numel() for p in model.parameters()),
                    "precision": "float32 decoder, external cached visual features", "GT_forward_inputs": [],
                    "native_peak_supervision": next(iter(peak_mode)), "test_targets_read": False}
        write_json(output / ("resume_run.json" if args.resume else "run.json"), metadata)

        def elapsed():
            return previous_elapsed + time.monotonic() - started

        def last_checkpoint():
            if device.type == "cuda":
                accelerator_rng = torch.cuda.get_rng_state(device)
            elif hasattr(torch.mps, "get_rng_state"):
                accelerator_rng = torch.mps.get_rng_state()
            else:
                accelerator_rng = None
            save_checkpoint(output / "last.pt", {
                "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "step": completed, "config": config, "stats": stats, "data_revision": dataset.get("data_revision"),
                "row_identity": [(r["name"], r["task"], r["split"]) for r in dataset["rows"]],
                "elapsed_s": elapsed(), "best_score": best_score, "best_step": best_step,
                "best_threshold": best_threshold, "initialization": initialization,
                "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                "accelerator_rng_state": accelerator_rng, "seen_episodes": sorted(seen),
            })

        def validate(loss=None):
            nonlocal best_score, best_step, best_threshold
            records, forward_s = predict(model, val, stats, device, config["batch_size"], deadline=deadline)
            threshold = threshold_from_val(records)
            result = metrics(records, threshold)
            score = (result["task_macro"]["soft_MAE"], -result["task_macro"]["AP"])
            entry = {"step": completed, "loss": loss, "threshold": threshold,
                     "task_macro": result["task_macro"], "elapsed_s": elapsed(), "validation_forward_s": forward_s}
            history.append(entry)
            write_json(output / "history.json", history)
            print("VALIDATION " + json.dumps(entry), flush=True)
            if best_score is None or score < best_score:
                best_score, best_step, best_threshold = score, completed, threshold
                save_checkpoint(output / "best.pt", {"state_dict": model.state_dict(), "step": completed,
                    "config": config, "stats": stats, "threshold": threshold, "validation": result,
                    "selection": "val_task_macro_soft_MAE_then_AP", "data_revision": dataset.get("data_revision"),
                    "initialization": initialization})

        print("TRAINING_STARTED " + json.dumps(metadata), flush=True)
        state, exit_code = "fit_complete", 0
        try:
            if not args.resume:
                validate()
                last_checkpoint()
            for step in range(completed + 1, config["steps"] + 1):
                if time.monotonic() >= deadline:
                    state = "budget_stopped"
                    break
                model.train()
                samples = []
                for _ in range(config["batch_size"]):
                    task = task_list[int(rng.integers(len(task_list)))]
                    descriptor = tasks[task][int(rng.integers(len(tasks[task])))]
                    row = rows[descriptor["name"]]
                    left = int(rng.integers(len(row["grid"]) - config["context"] + 1))
                    samples.append((row, torch.arange(left, left + config["context"])))
                    seen.add(descriptor["name"])
                keys = INPUT_KEYS + ("target", "target_valid")
                if next(iter(peak_mode)):
                    keys += ("native_bin_peak", "native_bin_peak_valid")
                batch = {key: torch.stack([row[key][index] for row, index in samples]) for key in keys}
                prediction = model(**model_inputs(batch, torch.arange(config["batch_size"]), stats, device))
                loss, parts = training_loss(prediction, batch, scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                rate = config["learning_rate"] * min(1., step / config["warmup_steps"]) * (.1 + .9 * (1 + math.cos(math.pi * step / config["steps"])) / 2)
                optimizer.param_groups[0]["lr"] = rate
                optimizer.step()
                completed = step
                if step == 1 or step % config["log_every"] == 0:
                    synchronize(device)
                    progress = {"pid": os.getpid(), "step": step, "total_steps": config["steps"], "loss": float(loss.detach()),
                                **{k: float(v.detach()) for k, v in parts.items()}, "gradient_norm": float(gradient),
                                "learning_rate": rate, "elapsed_s": elapsed(), "train_episodes_seen": len(seen),
                                "torch_peak_bytes": peak_memory(device)}
                    write_json(output / "progress.json", progress)
                    print("TRAIN_PROGRESS " + json.dumps(progress), flush=True)
                if step % config["validate_every"] == 0 or step == config["steps"]:
                    last_checkpoint()
                    validate(float(loss.detach()))
                    last_checkpoint()
        except TimeoutError:
            state = "budget_stopped"
        except KeyboardInterrupt:
            state, exit_code = "interrupted", 130
        except Exception as error:
            write_json(output / "failure.json", {"state": "failed", "step": completed, "error_type": type(error).__name__, "message": str(error), "elapsed_s": elapsed(), "exit_code": 1})
            raise
        last_checkpoint()
        change = max(float((value.detach().cpu() - initial_parameters[name]).abs().max()) for name, value in model.named_parameters())
        status = {"state": state, "pid": os.getpid(), "optimizer_steps": completed, "best_step": best_step,
                  "best_validation_score": best_score, "elapsed_s": elapsed(), "exit_code": exit_code,
                  "train_episodes_seen": len(seen), "train_episodes_total": len(train), "parameter_change_this_session_max": change,
                  "torch_peak_bytes": peak_memory(device), "test_targets_read": False}
        write_json(output / "status.json", status)
        print("FIT_FINISHED " + json.dumps(status), flush=True)
        return exit_code


def main():
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
