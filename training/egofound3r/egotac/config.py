"""Config loading and batch planning, free of heavy imports so both are cheap to test."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        unresolved = re.search(r"\$\{?\w+", expanded)
        if unresolved:
            raise ValueError(f"environment variable {unresolved.group()} in {value!r} is not set; source scripts/env.sh")
        return expanded
    return value


def load_config(path: Path, overrides: list[str]) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not raw:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        section, *path, name = key.split(".")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        target = config.setdefault(section, {})
        for part in path:
            target = target.setdefault(part, {})
        target[name] = value
    return _expand(config)


def accumulation_steps(train: dict, world_size: int) -> int:
    """Gradient accumulation that keeps total_batch_size whatever the GPU count or memory."""
    per_step = train["batch_size"] * world_size
    return max(1, math.ceil(train.get("total_batch_size", per_step) / per_step))
