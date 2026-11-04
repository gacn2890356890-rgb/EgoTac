"""The official EgoFound3R model, plus the world-model head when running the modified tree."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .integrity import repair_module
from .upstream import ensure_upstream

ensure_upstream()

from egohandmetric_prompt import build_runtime_marker_model, load_marker_model_weights  # noqa: E402
from egohandmetric_prompt.checkpoints import (  # noqa: E402
    MARKER_CHECKPOINT_NAMESPACE,
    MARKER_CHECKPOINT_SCHEMA_VERSION,
)
from egohandmetric_prompt.configs import load_project_config, marker_architecture_descriptor, marker_architecture_hash  # noqa: E402
from egohandmetric_prompt.inference_forward import _build_prediction_marker_forward_contract, _call_marker_model  # noqa: E402

VARIANTS = ("baseline", "world")
TIME_KEYS = (
    "images", "K", "w2c", "joints", "verts", "hand_valid", "in_view", "uv", "uv_valid", "dist_joints", "dist_verts",
    "dist_joints_valid", "dist_verts_valid", "contact_joints", "contact_verts", "vis_joints", "vis_verts",
)


def slice_time(batch: dict, start: int, stop: int | None) -> dict:
    return {key: value[:, start:stop] if key in TIME_KEYS else value for key, value in batch.items()}


class EgoTacModel(nn.Module):
    def __init__(self, config_path: str | Path, checkpoint: str | Path | None, variant: str, world: dict | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.project_config = load_project_config(config_path)
        self.marker = build_runtime_marker_model(self.project_config)
        self.weight_sources: list[tuple[str, str]] = [(self.project_config.marker_runtime.vggt_checkpoint_path, "backbone.")]
        if checkpoint is not None:
            self.weight_sources.append((str(checkpoint), ""))
        if checkpoint is not None:
            load_marker_model_weights(self.marker, checkpoint, model_config=self.project_config.marker_model, align_model_floating_dtype=True)
        self.world_options = {"eval_samples": 8, "eval_steps": 10, "eval_seed": 0, "past_frames": 16, "horizon": 5, **(world or {})}
        self.world = None
        if variant == "world":
            try:
                from egohandmetric_prompt.world_model import WorldModelHead, WorldModelHooks
            except ModuleNotFoundError as error:
                raise RuntimeError("the world variant needs EGOFOUND3R_ROOT=EgoFound3R_world; scripts/train_full.sh sets it") from error
            self.world = WorldModelHead(
                dim=self.world_options.get("dim", 512),
                layers=self.world_options.get("layers", 8),
                heads=self.world_options.get("heads", 8),
                dynamics_layers=self.world_options.get("dynamics_layers", 4),
                past_frames=self.world_options["past_frames"],
                horizon=self.world_options["horizon"],
            )
            self.hooks = WorldModelHooks(self.marker, self.world, self.world_options)
        self._prepare_dtypes()

    def _prepare_dtypes(self) -> None:
        # Updates below bfloat16 resolution vanish, so trainable weights live in fp32 under autocast.
        for parameter in self.marker.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        for name, buffer in self.marker.named_buffers():
            if not name.startswith("backbone.") and buffer.is_floating_point():
                buffer.data = buffer.data.float()

    @staticmethod
    def _touch_target(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        from egohandmetric_prompt.world_model import distance_to_target

        distance = torch.cat([batch["dist_joints"], batch["dist_verts"]], dim=-1)
        contact = torch.cat([batch["contact_joints"], batch["contact_verts"]], dim=-1)
        mask = torch.cat([batch["dist_joints_valid"], batch["dist_verts_valid"]], dim=-1)
        return torch.stack([2 * contact - 1, distance_to_target(distance)], dim=-1), mask

    def _past_state(self, offsets_flat: torch.Tensor, batch_size: int, frames: int) -> dict[str, torch.Tensor]:
        """Hook-captured tensors are flattened [B * T * 2, ...]; regroup them to one row per hand."""
        cond = self.hooks.conditioning()
        if cond.shape[0] != batch_size * frames * 2:
            raise ValueError(f"expected {batch_size * frames * 2} hand features, got {cond.shape[0]}")
        offsets = offsets_flat.detach().to(cond.dtype)
        by_hand = cond.reshape(batch_size, frames, 2, *cond.shape[1:]).transpose(1, 2).flatten(0, 1)
        last_offsets = offsets.reshape(batch_size, frames, 2, *offsets.shape[1:]).transpose(1, 2).flatten(0, 1)[:, -1]
        return {"cond": cond, "offsets": offsets, "by_hand": by_hand, "last_offsets": last_offsets}

    @torch.no_grad()
    def _visual_target(self, future_images: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder target with fixed pooling, so prediction and target cannot collapse together."""
        tokens = self.marker.backbone.encode_dino_high_rate(future_images).tokens
        batch, frames = future_images.shape[:2]
        pooled = tokens.reshape(batch, frames, -1, tokens.shape[-1]).mean(2)
        return F.normalize(pooled.float(), dim=-1)

    def _world_losses(self, batch: dict, offsets_flat: torch.Tensor, future_images: torch.Tensor) -> dict[str, torch.Tensor]:
        past = slice_time(batch, 0, self.world.past_frames)
        future = slice_time(batch, self.world.past_frames, None)
        batch_size, frames = past["images"].shape[:2]
        state = self._past_state(offsets_flat, batch_size, frames)

        # Current frame: replaces the paper's Eq. 5-6, as in the same-frame tactile head.
        target, mask = self._touch_target(past)
        step = torch.zeros(state["cond"].shape[0], dtype=torch.long, device=target.device)
        current = self.world.flow_matching_loss(state["cond"], state["offsets"], target.flatten(0, 2), mask.flatten(0, 2), step)

        # Future: same flow, conditioned on the last observed hand state and the imagined latent.
        latents = self.world.imagine(state["by_hand"])
        horizon = latents.shape[1]
        target, mask = self._touch_target(future)
        target = target.transpose(1, 2).flatten(0, 2)
        mask = mask.transpose(1, 2).flatten(0, 2)
        step = torch.arange(1, horizon + 1, device=target.device).repeat(batch_size * 2)
        future_flow = self.world.flow_matching_loss(
            state["by_hand"][:, -1].repeat_interleave(horizon, 0),
            state["last_offsets"].repeat_interleave(horizon, 0),
            target,
            mask,
            step,
            latents.flatten(0, 1),
        )
        predicted = self.world.visual_readout(latents.reshape(batch_size, 2, horizon, -1).mean(1))
        visual = (1 - F.cosine_similarity(predicted.float(), self._visual_target(future_images), dim=-1)).mean()
        return {"tactile_flow_loss": current, "future_flow_loss": future_flow, "future_visual_loss": visual}

    def forward(self, images: torch.Tensor, stride: int = 5, phase: int | None = None, batch: dict | None = None) -> dict:
        train_world = self.variant == "world" and self.training and batch is not None
        observed = images[:, : self.world.past_frames] if train_world else images
        contract = _build_prediction_marker_forward_contract(
            project_config=self.project_config,
            batch={},
            images=observed,
            global_stride=stride,
            global_anchor_phase=min(2, stride - 1) if phase is None else phase,
        )
        # Requested-time interpolation is a post-process that training never uses.
        contract["temporal_interpolation_query_map"] = None
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
            raw = _call_marker_model(self.marker, observed.to(dtype=torch.bfloat16), **contract)
        outputs = raw.to_dict()
        outputs["global_anchor_indices"] = contract["multirate_frame_map"].global_anchor_indices
        if train_world:
            offsets = torch.cat([outputs["dense_joint_offset"], outputs["dense_vertex_offset"]], dim=-2).flatten(0, 2)
            with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
                outputs.update(self._world_losses(batch, offsets, images[:, self.world.past_frames :].to(dtype=torch.bfloat16)))
        return outputs

    def loss_targets(self, batch: dict) -> dict:
        """The world variant is supervised on observed frames here; future frames go through its head."""
        return slice_time(batch, 0, self.world.past_frames) if self.variant == "world" else batch

    @torch.no_grad()
    def predict_future(self, images: torch.Tensor, stride: int = 5) -> dict[str, torch.Tensor]:
        """From past frames alone. Returns [B, 2, horizon, 216] contact probability and log distance,
        plus the same-frame prediction at the last observed frame, for the persistence reference."""
        from egohandmetric_prompt.world_model import samples_to_outputs

        past = images[:, : self.world.past_frames]
        batch_size, frames = past.shape[:2]
        outputs = self.forward(past, stride=stride)
        offsets = torch.cat([outputs["dense_joint_offset"], outputs["dense_vertex_offset"]], dim=-2).flatten(0, 2)
        options = self.world_options
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
            state = self._past_state(offsets, batch_size, frames)
            latents = self.world.imagine(state["by_hand"].float())
            horizon = latents.shape[1]
            step = torch.arange(1, horizon + 1, device=images.device).repeat(batch_size * 2)
            samples = self.world.sample(
                state["by_hand"][:, -1].float().repeat_interleave(horizon, 0),
                state["last_offsets"].float().repeat_interleave(horizon, 0),
                step,
                latents.flatten(0, 1),
                num_samples=options["eval_samples"],
                steps=options["eval_steps"],
                seed=options["eval_seed"],
            )
        touch = samples_to_outputs(samples)
        shape = (batch_size, 2, horizon, -1)
        joint = torch.sigmoid(outputs["dense_joint_contact_logits"].float())[:, -1]
        vertex = torch.sigmoid(outputs["dense_vertex_contact_logits"].float())[:, -1]
        return {
            "probability": touch["contact_probability"].reshape(shape),
            "log_distance": touch["contact_log_distance"].reshape(shape),
            "last_probability": torch.cat([joint, vertex], dim=-1),
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def verify_weights(self) -> dict:
        """Call after moving to the final device. See integrity.py for why this is needed."""
        return repair_module(self.marker, self.weight_sources)

    def checkpoint_payload(self) -> dict:
        """Same layout as the official checkpoint, so its inference scripts load the file unchanged."""
        state = {
            name: value.detach().cpu().to(torch.bfloat16) if value.is_floating_point() else value.detach().cpu()
            for name, value in self.marker.state_dict().items()
            if not name.startswith("backbone.")
        }
        model_config = self.project_config.marker_model
        payload = {
            "model": state,
            "checkpoint_schema_version": MARKER_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_namespace": MARKER_CHECKPOINT_NAMESPACE,
            "temporal_architecture": marker_architecture_descriptor(model_config),
            "temporal_architecture_hash": marker_architecture_hash(model_config),
            "egotac_variant": self.variant,
        }
        if self.world is not None:
            payload["egotac_world_head"] = {k: v.detach().cpu() for k, v in self.world.state_dict().items()}
            payload["egotac_world_options"] = dict(self.world_options)
        return payload

    def load_egotac_checkpoint(self, path: str | Path) -> None:
        """Load one of our training checkpoints, after __init__ built the matching architecture."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current = self.marker.state_dict()
        for name, value in payload["model"].items():
            current[name] = value.to(current[name].dtype)
        # bfloat16 "model" is for official tools; resume files also keep exact fp32 trainable weights.
        current.update(payload.get("resume_marker_fp32", {}))
        self.marker.load_state_dict(current)
        self.weight_sources = [item for item in self.weight_sources if item[1] != ""]
        if "resume_marker_fp32" not in payload:
            self.weight_sources.append((str(path), ""))
        if self.world is not None:
            if "egotac_world_head" not in payload:
                raise ValueError(f"{path} has no world-model head; it is a baseline checkpoint")
            self.world.load_state_dict(payload["egotac_world_head"])
