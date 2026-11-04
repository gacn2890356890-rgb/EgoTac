"""EgoFound3R training objective, Eq. 3-8 of Appendix B.

The paper leaves the balancing weights, heatmap details, and scene-term details unpublished; those
values are ours and live in the config. The scale, root, and in-view terms are also ours: Eq. 3
normalizes scale away and the paper describes root and in-view supervision only in words.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CONTACT_LOG_CONF_RANGE = (0.0, 2.0)
GEOM_LOG_CONF_MAX = 4.0
HEATMAP_SIGMA_CELLS = 1.0
HEATMAP_POSITIVE_WEIGHT = 20.0
KEYPOINT_OFFSET_RANGE_CELLS = 1.5
REPROJECTION_CLAMP = 0.2
MIN_CAMERA_BASELINE_M = 0.01  # below this, normalized camera translation is pure noise

DEFAULT_WEIGHTS = {
    "geom": 1.0,  # the paper fixes this one to unit weight
    "scale": 1.0,
    "vis": 0.2,
    "con": 0.5,
    "dist": 0.2,
    "heatmap": 0.5,
    "offset": 0.5,
    "reproj": 0.5,
    "root": 1.0,
    "in_view": 0.1,
    "cam_rot": 0.1,
    "cam_trans": 0.1,
    "tactile_flow": 1.0,
}


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def geometry_loss(pred: torch.Tensor, log_conf: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha: float):
    """Eq. 3, on offsets from the wrist. pred and target [..., P, 3], mask [..., P]."""
    weight = mask.to(pred.dtype)
    count = weight.sum(-1, keepdim=True).clamp_min(1.0)
    target_scale = ((target.norm(dim=-1) * weight).sum(-1, keepdim=True) / count).clamp_min(1e-3)
    pred_scale = ((pred.norm(dim=-1) * weight).sum(-1, keepdim=True) / count).clamp_min(1e-3)
    residual = (pred / pred_scale[..., None] - target / target_scale[..., None]).abs().sum(-1)
    s = log_conf.clamp_max(GEOM_LOG_CONF_MAX)
    has_points = mask.any(-1)
    scale_loss = _masked_mean((pred_scale.log() - target_scale.log()).abs().squeeze(-1), has_points)
    return _masked_mean(torch.exp(s) * residual - alpha * s, mask), scale_loss


def heatmap_targets(uv: torch.Tensor, image_hw: torch.Tensor, grid_hw: tuple[int, int]):
    """Gaussian targets on the heatmap grid, using the inverse of upstream's cell-to-uv decoding."""
    grid_h, grid_w = grid_hw
    height, width = image_hw[:, 0].float(), image_hw[:, 1].float()
    view = (-1,) + (1,) * (uv.ndim - 2)
    x = (uv[..., 0] * (width - 1).view(view) + 0.5) * grid_w / width.view(view) - 0.5
    y = (uv[..., 1] * (height - 1).view(view) + 0.5) * grid_h / height.view(view) - 0.5
    xs = torch.arange(grid_w, device=uv.device, dtype=uv.dtype)
    ys = torch.arange(grid_h, device=uv.device, dtype=uv.dtype)
    gauss_x = torch.exp(-((xs - x[..., None]) ** 2) / (2 * HEATMAP_SIGMA_CELLS**2))
    gauss_y = torch.exp(-((ys - y[..., None]) ** 2) / (2 * HEATMAP_SIGMA_CELLS**2))
    return gauss_y[..., :, None] * gauss_x[..., None, :], x, y


def two_d_losses(outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
    logits = outputs["keypoint_heatmap_logits"].float()
    offsets = outputs["keypoint_offset"].float()
    mask = batch["uv_valid"]
    grid_h, grid_w = logits.shape[-2:]
    target, x, y = heatmap_targets(batch["uv"], batch["image_hw"], (grid_h, grid_w))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    heatmap = _masked_mean((bce * (1 + HEATMAP_POSITIVE_WEIGHT * target)).mean((-1, -2)), mask)

    cell_x = x.round().long().clamp(0, grid_w - 1)
    cell_y = y.round().long().clamp(0, grid_h - 1)
    index = (cell_y * grid_w + cell_x)[..., None, None].expand(*cell_x.shape, 2, 1)
    at_cell = offsets.flatten(-2).gather(-1, index).squeeze(-1)
    predicted = KEYPOINT_OFFSET_RANGE_CELLS * torch.tanh(at_cell)
    wanted = torch.stack([x - cell_x, y - cell_y], dim=-1)
    offset = _masked_mean(F.smooth_l1_loss(predicted, wanted, reduction="none").sum(-1), mask)
    return {"heatmap": heatmap, "offset": offset}


def reprojection_loss(outputs: dict, batch: dict) -> torch.Tensor:
    points = outputs["dense_joint_xyz"].float()
    K = batch["K"]
    z = points[..., 2].clamp_min(1e-2)
    height, width = batch["image_hw"][:, 0].float(), batch["image_hw"][:, 1].float()
    view = (-1, 1, 1, 1)
    fx, fy = K[:, :, 0, 0][:, :, None, None], K[:, :, 1, 1][:, :, None, None]
    cx, cy = K[:, :, 0, 2][:, :, None, None], K[:, :, 1, 2][:, :, None, None]
    u = (fx * points[..., 0] / z + cx) / (width.view(view) - 1)
    v = (fy * points[..., 1] / z + cy) / (height.view(view) - 1)
    error = (torch.stack([u, v], -1) - batch["uv"]).abs().sum(-1).clamp_max(REPROJECTION_CLAMP)
    mask = batch["uv_valid"] & outputs["root_translation_valid"][..., None] & (points[..., 2] > 1e-2)
    return _masked_mean(error, mask)


def camera_losses(outputs: dict, batch: dict, anchor_indices: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Part of L_3R: rotation geodesic and scene-scale-normalized translation, relative to anchor 0."""
    pred = outputs["camera_pose_global"].float()
    if anchor_indices is None:
        raise ValueError("camera losses need the anchor frame indices")
    gt = batch["w2c"].gather(1, anchor_indices[:, :, None, None].expand(-1, -1, 4, 4))
    relative_gt = gt @ torch.linalg.inv(gt[:, :1])
    relative_pred = pred @ torch.linalg.inv(pred[:, :1])
    rotation = relative_pred[..., :3, :3] @ relative_gt[..., :3, :3].transpose(-1, -2)
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    geodesic = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
    t_pred, t_gt = relative_pred[..., :3, 3], relative_gt[..., :3, 3]
    scale_pred = t_pred.norm(dim=-1).mean(1, keepdim=True).clamp_min(1e-4)
    scale_gt = t_gt.norm(dim=-1).mean(1, keepdim=True).clamp_min(1e-4)
    translation = (t_pred / scale_pred[..., None] - t_gt / scale_gt[..., None]).abs().sum(-1)
    moving = (scale_gt.squeeze(1) > MIN_CAMERA_BASELINE_M)[:, None].expand_as(translation[:, 1:])
    return {"cam_rot": geodesic[:, 1:].mean(), "cam_trans": _masked_mean(translation[:, 1:], moving)}


def total_loss(outputs: dict, batch: dict, weights: dict, variant: str, alpha: float, anchor_indices: torch.Tensor | None = None):
    terms: dict[str, torch.Tensor] = {}
    hand_valid = batch["hand_valid"]
    joints_gt = batch["joints"] - batch["joints"][..., :1, :]
    verts_gt = batch["verts"] - batch["joints"][..., :1, :]
    joint_mask = hand_valid[..., None].expand(*hand_valid.shape, 20)
    vert_mask = hand_valid[..., None].expand(*hand_valid.shape, 195)

    geom_j, scale_j = geometry_loss(
        outputs["dense_joint_offset"][..., 1:, :].float(), outputs["dense_joint_log_conf"][..., 1:].float(), joints_gt[..., 1:, :], joint_mask, alpha
    )
    geom_v, scale_v = geometry_loss(outputs["dense_vertex_offset"].float(), outputs["dense_vertex_log_conf"].float(), verts_gt, vert_mask, alpha)
    terms["geom"] = geom_j + geom_v
    terms["scale"] = scale_j + scale_v

    vis_j = F.binary_cross_entropy_with_logits(outputs["dense_joint_visibility_logits"].float(), batch["vis_joints"], reduction="none")
    vis_v = F.binary_cross_entropy_with_logits(outputs["dense_vertex_visibility_logits"].float(), batch["vis_verts"], reduction="none")
    terms["vis"] = _masked_mean(vis_j, hand_valid[..., None].expand_as(vis_j)) + _masked_mean(vis_v, vert_mask)

    if variant == "baseline":
        for key, label, valid in (("joint", "joints", "dist_joints_valid"), ("vertex", "verts", "dist_verts_valid")):
            c = outputs[f"dense_{key}_contact_log_conf"].float().clamp(*CONTACT_LOG_CONF_RANGE)
            bce = F.binary_cross_entropy_with_logits(outputs[f"dense_{key}_contact_logits"].float(), batch[f"contact_{label}"], reduction="none")
            terms["con"] = terms.get("con", 0.0) + _masked_mean(torch.exp(c) * bce - alpha * c, batch[valid])
            log_distance = torch.log1p(1000.0 * batch[f"dist_{label}"])
            smooth = F.smooth_l1_loss(outputs[f"dense_{key}_contact_log_distance"].float(), log_distance, reduction="none")
            terms["dist"] = terms.get("dist", 0.0) + _masked_mean(smooth, batch[valid])
    else:
        # The only difference between the two variants: these replace Eq. 5 and Eq. 6.
        terms["tactile_flow"] = outputs["tactile_flow_loss"]

    terms.update(two_d_losses(outputs, batch))
    terms["reproj"] = reprojection_loss(outputs, batch)

    root_mask = hand_valid & outputs["root_translation_valid"]
    root_error = F.smooth_l1_loss(outputs["dense_hand_root_xyz"].float(), batch["joints"][..., 0, :], beta=0.05, reduction="none").sum(-1)
    terms["root"] = _masked_mean(root_error, root_mask)
    terms["in_view"] = F.binary_cross_entropy_with_logits(outputs["hand_in_view_logits"].float(), batch["in_view"].float())
    terms.update(camera_losses(outputs, batch, anchor_indices))

    total = sum(weights[name] * value for name, value in terms.items())
    return total, {name: float(value.detach()) for name, value in terms.items()}
