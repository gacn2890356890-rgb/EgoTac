"""Explicit accelerator selection and atomic local training artifacts."""

import json
import os
from pathlib import Path
import subprocess

import torch


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=<GPU-UUID>")
        device = torch.device("cuda:0")
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; choose an available accelerator explicitly")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before launching Python")
        torch.backends.mha.set_fastpath_enabled(False)
        device = torch.device("mps")
    else:
        raise ValueError("Training/inference device must be cuda or mps; no CPU fallback")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory(device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(cpu_state(value), temporary)
    temporary.replace(path)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
