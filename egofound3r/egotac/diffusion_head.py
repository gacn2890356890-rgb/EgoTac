"""Rectified-flow tactile head.

Jointly generates (contact state, log-distance) over the 216 points of one hand (root, 20 joints,
195 vertices), conditioned on the point features EgoFound3R already computes. Attached with PyTorch
hooks, so the official code is untouched and removing the hooks restores the original model exactly.
"""
from __future__ import annotations

import math

import torch
from torch import nn

NUM_POINTS = 216
FEATURE_DIM = 768
LOG_DISTANCE_MEAN = 2.5
LOG_DISTANCE_STD = 1.5
PROBABILITY_SHARPNESS = 6.0


def distance_to_target(distance_m: torch.Tensor) -> torch.Tensor:
    """Same target as the paper's Eq. 6, log(1 + 1000 d), then standardized for the flow."""
    return (torch.log1p(1000.0 * distance_m) - LOG_DISTANCE_MEAN) / LOG_DISTANCE_STD


def target_to_log_distance(value: torch.Tensor) -> torch.Tensor:
    return (value * LOG_DISTANCE_STD + LOG_DISTANCE_MEAN).clamp_min(0.0)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.film = nn.Linear(dim, 4 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        s1, b1, s2, b2 = self.film(time_emb).unsqueeze(1).chunk(4, dim=-1)
        y = self.norm1(x) * (1 + s1) + b1
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x) * (1 + s2) + b2)


class TactileFlowHead(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 4, heads: int = 8, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.dim = dim
        self.x_proj = nn.Linear(2, dim)
        self.cond_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim))
        self.geom_proj = nn.Linear(3, dim)
        self.point_embed = nn.Embedding(NUM_POINTS, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1)))
        angles = t[:, None] * 1000.0 * freqs
        return self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))

    def velocity(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # Offsets are in meters; dividing by a hand-sized length keeps them near unit scale.
        tokens = self.x_proj(x) + self.cond_proj(cond) + self.geom_proj(offsets / 0.1) + self.point_embed.weight
        time_emb = self._time_embedding(t)
        for block in self.blocks:
            tokens = block(tokens, time_emb)
        return self.out(self.out_norm(tokens))

    def flow_matching_loss(self, cond, offsets, target, mask) -> torch.Tensor:
        """target [N, 216, 2], mask [N, 216] with True where the label exists."""
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        error = (self.velocity(x_t, t, cond, offsets) - (target - noise)) ** 2
        weight = mask.to(error.dtype)[..., None]
        return (error * weight).sum() / (weight.sum() * 2).clamp_min(1.0)

    @torch.no_grad()
    def sample(self, cond, offsets, num_samples: int = 8, steps: int = 10, seed: int = 0) -> torch.Tensor:
        """Euler integration of the flow. Returns [S, N, 216, 2]; seeded so evaluation repeats exactly."""
        generator = torch.Generator(device=cond.device).manual_seed(seed)
        results = []
        for _ in range(num_samples):
            x = torch.randn(cond.shape[0], NUM_POINTS, 2, device=cond.device, dtype=cond.dtype, generator=generator)
            for step in range(steps):
                t = torch.full((cond.shape[0],), step / steps, device=cond.device, dtype=cond.dtype)
                x = x + self.velocity(x, t, cond, offsets) / steps
            results.append(x)
        return torch.stack(results)


def samples_to_outputs(samples: torch.Tensor) -> dict[str, torch.Tensor]:
    """Collapse samples into the fields the original head returns, so downstream code cannot tell."""
    probability = torch.sigmoid(PROBABILITY_SHARPNESS * samples[..., 0]).mean(0).clamp(1e-4, 1 - 1e-4)
    agreement = (2 * probability - 1).abs()
    return {
        "contact_probability": probability,
        "contact_logits": torch.logit(probability),
        "contact_log_distance": target_to_log_distance(samples[..., 1]).mean(0),
        # Log-confidence in the original head's [0, 2] range, from agreement between samples.
        "contact_log_conf": 2.0 * agreement,
    }


class TactileHooks:
    """Installs the head on an official MarkerModel without editing its code. remove() undoes it."""

    FIELDS = ("contact_probability", "contact_logits", "contact_log_distance", "contact_log_conf")

    def __init__(self, marker: nn.Module, tactile: TactileFlowHead, options: dict | None = None):
        self.marker = marker
        self.tactile = tactile
        self.options = {"eval_samples": 8, "eval_steps": 10, "eval_seed": 0, **(options or {})}
        self.features: dict[str, torch.Tensor] = {}
        head = marker.hand_head
        self._handles = [
            getattr(head, f"{name}_output_norm").register_forward_hook(self._capture(name)) for name in ("root", "joint", "vertex")
        ]
        self._handles.append(head.register_forward_hook(self._replace))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            self.features[name] = output

        return hook

    def conditioning(self) -> torch.Tensor:
        return torch.cat([self.features["root"][:, None], self.features["joint"], self.features["vertex"]], dim=1)

    def _replace(self, _module, _inputs, outputs):
        if self.marker.training:
            return outputs
        from egohandmetric_prompt.contact_confidence import contact_confidence_from_log

        offsets = torch.cat([outputs["joint_offset"], outputs["vertex_offset"]], dim=1)
        samples = self.tactile.sample(
            self.conditioning().float(),
            offsets.float(),
            num_samples=self.options["eval_samples"],
            steps=self.options["eval_steps"],
            seed=self.options["eval_seed"],
        )
        tactile = samples_to_outputs(samples)
        replaced = dict(outputs)
        for prefix, points in (("joint", slice(0, 21)), ("vertex", slice(21, NUM_POINTS))):
            for field in self.FIELDS:
                key = f"{prefix}_{field}"
                replaced[key] = tactile[field][:, points].to(dtype=outputs[key].dtype)
            log_conf = replaced[f"{prefix}_contact_log_conf"]
            replaced[f"{prefix}_contact_conf"] = contact_confidence_from_log(log_conf).to(dtype=outputs[f"{prefix}_contact_conf"].dtype)
        return replaced
