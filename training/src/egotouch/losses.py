"""v15 response/contact objective with explicit masks and device handling."""

import math

import torch
from torch.nn import functional as F


def training_loss(z, batch, response_scale):
    device = z["response"].device
    y = batch["target"].to(device)
    valid = batch["target_valid"].to(device)
    if not valid.any():
        raise ValueError("Sampled batch has no valid tactile supervision")
    binary = (y >= 1).float()
    bce = F.binary_cross_entropy_with_logits(z["contact_logits"], binary, reduction="none")
    natural = bce[valid].mean()
    negatives = valid & (binary == 0)
    hard = bce[negatives].topk(max(1, math.ceil(.2 * int(negatives.sum())))).values.mean() if negatives.any() else natural * 0
    response = ((z["response"] - y).abs() / response_scale)[valid].mean()
    positive = valid & (binary == 1)
    amplitude = F.smooth_l1_loss(z["amplitude_log"][positive], torch.log1p(y[positive])) if positive.any() else response * 0
    pairs = valid[:, 1:] & valid[:, :-1]
    error_delta = ((z["response"][:, 1:] - z["response"][:, :-1]) - (y[:, 1:] - y[:, :-1])).abs() / response_scale
    delta = error_delta[pairs].mean() if pairs.any() else response * 0
    peak = response * 0
    if "native_bin_peak" in batch and "native_bin_peak_valid" in batch:
        native = batch["native_bin_peak"].to(device)
        native_valid = batch["native_bin_peak_valid"].to(device)
        window_valid = native_valid.any(1)
        if window_valid.any():
            true_peak = native.masked_fill(~native_valid, float("-inf")).amax(1).clamp_min(0)
            pred_peak = z["response"].amax(1)
            peak = F.smooth_l1_loss(torch.log1p(pred_peak[window_valid]), torch.log1p(true_peak[window_valid]))
    total = natural + response + .25 * amplitude + .25 * hard + .05 * delta + .05 * peak
    return total, {"BCE": natural, "scaled_response_MAE": response, "amplitude_loss": amplitude, "hard_negative_loss": hard, "delta_loss": delta, "native_peak_loss": peak}
