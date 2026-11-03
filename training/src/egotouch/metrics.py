"""Task/episode aggregation; AP is average precision, not trapezoidal PR area."""

import numpy as np

from .contact_metrics import evaluate


def average(entries):
    keys = [k for k in entries[0] if k not in {"counts", "valid_values"}]
    return {k: float(np.mean([x[k] for x in entries if x.get(k) is not None]))
            if any(x.get(k) is not None for x in entries) else None for k in keys}


def metrics(records, threshold):
    if not records:
        raise ValueError("No evaluation records")
    result = evaluate(records, threshold)
    tasks = {}
    for row in records:
        y, valid = row["target"], row["valid"]
        error = (row["response"] - y)[valid]
        positive = valid & (y >= 1)
        entry = result["episodes"][row["name"]]
        entry["soft_MAE"] = float(np.abs(error).mean())
        entry["soft_RMSE"] = float(np.sqrt(np.square(error).mean()))
        entry["soft_positive_MAE"] = float(np.abs((row["response"] - y)[positive]).mean()) if positive.any() else None
        entry["positive_rate"] = float((y[valid] >= 1).mean())
        entry["AP"] = entry["AUPRC"]
        tasks.setdefault(row["task"], []).append(entry)
    result["episode_macro"] = average(list(result["episodes"].values()))
    result["per_task"] = {task: average(rows) for task, rows in tasks.items()}
    result["task_macro"] = average(list(result["per_task"].values()))
    result["metric_definitions"] = {
        "AP": "average precision with tied-score grouping; AUPRC is the legacy alias",
        "soft_MAE": "continuous probability-times-amplitude response error",
        "MAE": "legacy response error after binary probability threshold",
        "aggregation": "valid values per episode, episodes per task, equal task mean",
        "unit": "relative tactile sensor response, not Newtons",
    }
    return result


def threshold_from_val(records):
    candidates = []
    for threshold in np.arange(.05, .951, .05):
        tasks = {}
        for row in records:
            valid = row["valid"]
            pred = row["probability"][valid] > threshold
            gt = row["target"][valid] >= 1
            tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
            tasks.setdefault(row["task"], []).append(2 * tp / max(2 * tp + fp + fn, 1))
        score = float(np.mean([np.mean(values) for values in tasks.values()]))
        candidates.append((score, float(threshold)))
    return max(candidates)[1]


def constant_records(records, values):
    """Response-only constant baselines; no fabricated contact probabilities."""
    tasks = {}
    for row in records:
        y, valid = row["target"], row["valid"]
        response = np.broadcast_to(np.asarray(values), y.shape)
        error = (response - y)[valid]
        tasks.setdefault(row["task"], []).append({"soft_MAE": float(np.abs(error).mean()), "soft_RMSE": float(np.sqrt(np.square(error).mean()))})
    per_task = {task: average(rows) for task, rows in tasks.items()}
    return {"per_task": per_task, "task_macro": average(list(per_task.values()))}
