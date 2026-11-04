"""Guards against silent memory corruption while loading weights.

Why this exists: the msc-vision server (ECC disabled) returned different bytes for the same file on
repeated reads, and a user-space pattern test found flipped bits in RAM. A flipped exponent bit in a
frozen weight turns the whole forward pass into NaN. The file on disk is fine, so reading it several
times and voting recovers the true values. Weights on the GPU are then compared against the vote.
"""
from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_READS = 3


def _read_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return {key: value for key, value in state.items() if torch.is_tensor(value)}


def voted_state(path: str | Path, reads: int = DEFAULT_READS) -> tuple[dict[str, torch.Tensor], int]:
    """Per-tensor majority across reads. Returns (state, number of tensors that disagreed)."""
    states = [_read_state(Path(path)) for _ in range(reads)]
    voted: dict[str, torch.Tensor] = {}
    disagreed = 0
    for key, first in states[0].items():
        candidates = [state[key] for state in states]
        if all(torch.equal(first, other) for other in candidates[1:]):
            voted[key] = first
            continue
        disagreed += 1
        for index, candidate in enumerate(candidates):
            if sum(torch.equal(candidate, other) for other in candidates) >= 2:
                voted[key] = candidate
                break
        else:
            raise RuntimeError(f"{path}: all {reads} reads of {key} differ; memory is too unreliable to continue")
    return voted, disagreed


@torch.no_grad()
def repair_module(module: torch.nn.Module, sources: list[tuple[str | Path, str]], reads: int = DEFAULT_READS) -> dict:
    """Compare live weights to the vote and overwrite any that differ.

    sources are (file, name prefix inside module), e.g. the backbone file with prefix "backbone.".
    """
    live = dict(module.state_dict())
    report = {"tensors_checked": 0, "tensors_repaired": [], "read_disagreements": 0}
    for path, prefix in sources:
        state, disagreed = voted_state(path, reads)
        report["read_disagreements"] += disagreed
        for key, value in state.items():
            name = prefix + key
            if name not in live:
                continue
            target = live[name]
            expected = value.to(device=target.device, dtype=target.dtype)
            report["tensors_checked"] += 1
            if not torch.equal(target, expected):
                target.copy_(expected)
                report["tensors_repaired"].append(name)
        del state
    return report
