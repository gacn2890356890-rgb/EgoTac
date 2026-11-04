"""Metrics from EgoFound3R Appendix D. Window scores are averaged over windows, as in the paper."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Similarity transform per item. pred and gt [N, P, 3]."""
    mu_p, mu_g = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    x, y = pred - mu_p, gt - mu_g
    u, s, vh = torch.linalg.svd(x.transpose(1, 2) @ y)
    d = torch.sign(torch.det(vh.transpose(1, 2) @ u.transpose(1, 2)))
    flip = torch.ones_like(s)
    flip[:, -1] = d
    rotation = vh.transpose(1, 2) @ torch.diag_embed(flip) @ u.transpose(1, 2)
    scale = (s * flip).sum(1) / x.pow(2).sum((1, 2)).clamp_min(1e-9)
    return scale[:, None, None] * x @ rotation.transpose(1, 2) + mu_g


def _prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float] | None:
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    if tp + fp + fn == 0:
        return None  # nothing to score in this window; the paper reports these as undefined
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


class MetricAccumulator:
    def __init__(self):
        self.windows = defaultdict(lambda: defaultdict(list))
        self.pooled = defaultdict(lambda: defaultdict(list))

    def update(self, outputs: dict, batch: dict) -> None:
        for b, dataset in enumerate(batch["dataset"]):
            self._update_window(dataset, {k: v[b] for k, v in outputs.items() if torch.is_tensor(v) and v.ndim and v.shape[0] == len(batch["dataset"])}, {k: v[b] for k, v in batch.items() if torch.is_tensor(v)})

    def _update_window(self, dataset: str, out: dict, gt: dict) -> None:
        store = self.windows[dataset]
        # Scored frames: annotated hands that are actually in the image.
        valid = (gt["hand_valid"] & gt["in_view"]).cpu()
        if not valid.any():
            return
        joints_gt = gt["joints"].cpu()[valid]
        verts_gt = gt["verts"].cpu()[valid]
        joints_pred = out["dense_joint_xyz"].float().cpu()[valid]
        verts_pred = out["dense_vertex_xyz"].float().cpu()[valid]
        root_gt, root_pred = joints_gt[:, :1], joints_pred[:, :1]
        store["mpjpe_raw"].append((joints_pred - joints_gt).norm(dim=-1).mean().item() * 1000)
        store["mpjpe_rr"].append(((joints_pred - root_pred) - (joints_gt - root_gt)).norm(dim=-1).mean().item() * 1000)
        store["mpvpe_rr"].append(((verts_pred - root_pred) - (verts_gt - root_gt)).norm(dim=-1).mean().item() * 1000)
        store["mpjpe_pa"].append((procrustes_align(joints_pred, joints_gt) - joints_gt).norm(dim=-1).mean().item() * 1000)
        store["mpvpe_pa"].append((procrustes_align(verts_pred, verts_gt) - verts_gt).norm(dim=-1).mean().item() * 1000)

        # Swapped hands score far worse on the raw metric, which exposes a wrong side convention.
        full = out["dense_joint_xyz"].float().cpu().flip(1)[valid]
        store["mpjpe_raw_sides_swapped"].append((full - joints_gt).norm(dim=-1).mean().item() * 1000)

        for level, point_key, label_key, valid_key, vis_key in (
            ("joint", "joint", "contact_joints", "dist_joints_valid", "vis_joints"),
            ("vertex", "vertex", "contact_verts", "dist_verts_valid", "vis_verts"),
        ):
            mask = gt[valid_key].cpu() & valid[..., None]
            if mask.any():
                probability = torch.sigmoid(out[f"dense_{point_key}_contact_logits"].float()).cpu()[mask].numpy()
                label = gt[label_key].cpu()[mask].numpy() > 0.5
                result = _prf(probability > 0.5, label)
                if result is not None:
                    store[f"contact_{level}_p"].append(result[0])
                    store[f"contact_{level}_r"].append(result[1])
                    store[f"contact_{level}_f1"].append(result[2])
                self.pooled[dataset][f"contact_{level}_scores"].append(probability)
                self.pooled[dataset][f"contact_{level}_labels"].append(label)
                label_key_distance = "dist_joints" if level == "joint" else "dist_verts"
                predicted_mm = torch.expm1(out[f"dense_{point_key}_contact_log_distance"].float()).cpu()[mask].numpy()
                actual_mm = gt[label_key_distance].cpu()[mask].numpy() * 1000
                error = np.abs(predicted_mm - actual_mm)
                store[f"distance_{level}_mae_all"].append(float(error.mean()))
                if (probability > 0.5).any():
                    store[f"distance_{level}_mae_pred"].append(float(error[probability > 0.5].mean()))
                if label.any():
                    store[f"distance_{level}_mae_gt"].append(float(error[label].mean()))
            vis_mask = valid[..., None].expand_as(gt[vis_key].cpu())
            vis_pred = out[f"dense_{point_key}_visibility_logits"].float().cpu()[vis_mask].numpy() > 0
            result = _prf(vis_pred, gt[vis_key].cpu()[vis_mask].numpy() > 0.5)
            if result is not None:
                store[f"visibility_{level}_f1"].append(result[2])

    def summary(self) -> dict:
        report = {}
        for dataset, store in self.windows.items():
            row = {name: float(np.mean(values)) for name, values in store.items()}
            row["windows"] = max((len(v) for v in store.values()), default=0)
            pooled = self.pooled[dataset]
            for level in ("joint", "vertex"):
                if f"contact_{level}_scores" in pooled:
                    row[f"contact_{level}_ap"] = average_precision(
                        np.concatenate(pooled[f"contact_{level}_scores"]), np.concatenate(pooled[f"contact_{level}_labels"])
                    )
                    row[f"contact_{level}_positive_rate"] = float(np.concatenate(pooled[f"contact_{level}_labels"]).mean())
            report[dataset] = row
        return report


class FutureAccumulator:
    """Future contact at each horizon, against two references that use no future information:
    the model's own prediction at the last observed frame, and the ground truth at that frame."""

    def __init__(self):
        self.pooled = defaultdict(lambda: defaultdict(list))

    def update(self, prediction: dict, batch: dict, past_frames: int) -> None:
        labels = torch.cat([batch["contact_joints"], batch["contact_verts"]], dim=-1).transpose(1, 2).cpu() > 0.5
        valid = torch.cat([batch["dist_joints_valid"], batch["dist_verts_valid"]], dim=-1).transpose(1, 2).cpu()
        distance_mm = torch.cat([batch["dist_joints"], batch["dist_verts"]], dim=-1).transpose(1, 2).cpu() * 1000
        probability = prediction["probability"].float().cpu()
        predicted_mm = torch.expm1(prediction["log_distance"].float()).cpu()
        last_model = prediction["last_probability"].float().cpu()
        last_truth = labels[:, :, past_frames - 1]
        for b, dataset in enumerate(batch["dataset"]):
            store = self.pooled[dataset]
            for step in range(probability.shape[2]):
                t = past_frames + step
                mask = valid[b, :, t] & valid[b, :, past_frames - 1]
                if not mask.any():
                    continue
                label = labels[b, :, t][mask].numpy()
                store[f"{step + 1}_label"].append(label)
                store[f"{step + 1}_score"].append(probability[b, :, step][mask].numpy())
                store[f"{step + 1}_hold_model"].append(last_model[b][mask].numpy())
                store[f"{step + 1}_hold_truth"].append(last_truth[b][mask].numpy())
                store[f"{step + 1}_abs_mm"].append(np.abs(predicted_mm[b, :, step][mask].numpy() - distance_mm[b, :, t][mask].numpy()))

    def summary(self) -> dict:
        report = {}
        for dataset, store in self.pooled.items():
            row = {}
            for step in sorted({int(key.split("_")[0]) for key in store}):
                label = np.concatenate(store[f"{step}_label"])
                score = np.concatenate(store[f"{step}_score"])
                model = _prf(score > 0.5, label)
                hold_model = _prf(np.concatenate(store[f"{step}_hold_model"]) > 0.5, label)
                hold_truth = _prf(np.concatenate(store[f"{step}_hold_truth"]), label)
                row[f"future_{step}_contact_f1"] = model[2] if model else float("nan")
                row[f"future_{step}_contact_ap"] = average_precision(score, label)
                row[f"future_{step}_hold_last_model_f1"] = hold_model[2] if hold_model else float("nan")
                row[f"future_{step}_hold_last_truth_f1"] = hold_truth[2] if hold_truth else float("nan")
                row[f"future_{step}_distance_mae_mm"] = float(np.concatenate(store[f"{step}_abs_mm"]).mean())
            report[dataset] = row
        return report
