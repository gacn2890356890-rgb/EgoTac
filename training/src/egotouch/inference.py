"""Overlapping same-frame windows; this is not a causal future predictor."""

import time

import torch

from .data import load_row, model_inputs
from .runtime import synchronize


def predict(model, descriptors, stats, device, batch_size=8, require_targets=True, deadline=None):
    model.eval()
    records = []
    elapsed = 0.
    with torch.inference_mode():
        for descriptor in descriptors:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Validation budget expired before completing all episodes")
            row = load_row(descriptor["file"], require_targets=require_targets)
            n = len(row["grid"])
            context = min(model.context, n)
            stride = max(context // 2, 1)
            starts = sorted(set(list(range(0, n - context + 1, stride)) + [n - context]))
            total = {key: torch.zeros(n, 12) for key in ("probability", "amplitude", "response")}
            den = torch.zeros(n, 1)
            weight = torch.minimum(torch.arange(1, context + 1), torch.arange(context, 0, -1)).float()[:, None]
            for first in range(0, len(starts), batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Validation budget expired; partial metrics are not selected")
                chosen = starts[first:first + batch_size]
                indices = torch.tensor(chosen)[:, None] + torch.arange(context)[None]
                inputs = model_inputs(row, indices, stats, device)
                synchronize(device)
                stamp = time.monotonic()
                output = model(**inputs)
                synchronize(device)
                elapsed += time.monotonic() - stamp
                values = {"probability": output["contact_logits"].sigmoid().cpu(),
                          "amplitude": output["conditional_response"].cpu(), "response": output["response"].cpu()}
                if not all(torch.isfinite(value).all() for value in values.values()):
                    raise RuntimeError("Non-finite model prediction")
                for i, left in enumerate(chosen):
                    for key in total:
                        total[key][left:left + context] += values[key][i] * weight
                    den[left:left + context] += weight
            if not (den > 0).all():
                raise RuntimeError("Some frames received no prediction")
            record = {"name": descriptor["name"], "task": descriptor["task"],
                      "frame_indices": row["frame_indices"].numpy(), "timestamps": row["timestamps"].numpy(),
                      **{key: (value / den).numpy() for key, value in total.items()}}
            if require_targets:
                record.update(target=row["target"].numpy(), valid=row["target_valid"].numpy())
            records.append(record)
    return records, elapsed
