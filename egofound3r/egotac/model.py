"""The official EgoFound3R model, plus an optional diffusion tactile head attached by hooks."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .diffusion_head import TactileFlowHead, TactileHooks, distance_to_target
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

VARIANTS = ("baseline", "diffusion")


class EgoTacModel(nn.Module):
    def __init__(self, config_path: str | Path, checkpoint: str | Path | None, variant: str, diffusion: dict | None = None):
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
        self.diffusion_options = {"eval_samples": 8, "eval_steps": 10, "eval_seed": 0, **(diffusion or {})}
        self.tactile = None
        if variant == "diffusion":
            self.tactile = TactileFlowHead(
                dim=self.diffusion_options.get("dim", 256),
                layers=self.diffusion_options.get("layers", 4),
                heads=self.diffusion_options.get("heads", 8),
            )
            self.hooks = TactileHooks(self.marker, self.tactile, self.diffusion_options)
        self._prepare_dtypes()

    def _prepare_dtypes(self) -> None:
        # Updates below bfloat16 resolution vanish, so trainable weights live in fp32 under autocast.
        for parameter in self.marker.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        for name, buffer in self.marker.named_buffers():
            if not name.startswith("backbone.") and buffer.is_floating_point():
                buffer.data = buffer.data.float()

    def _tactile_loss(self, batch: dict) -> torch.Tensor:
        # Hook-captured tensors are flattened [B * T * 2, ...]; targets are reshaped to match.
        cond = self.hooks.conditioning()
        offsets = torch.cat([self._flat_offsets["joint"], self._flat_offsets["vertex"]], dim=1).detach().to(cond.dtype)
        distance = torch.cat([batch["dist_joints"], batch["dist_verts"]], dim=-1).flatten(0, 2)
        contact = torch.cat([batch["contact_joints"], batch["contact_verts"]], dim=-1).flatten(0, 2)
        mask = torch.cat([batch["dist_joints_valid"], batch["dist_verts_valid"]], dim=-1).flatten(0, 2)
        target = torch.stack([2 * contact - 1, distance_to_target(distance)], dim=-1)
        return self.tactile.flow_matching_loss(cond, offsets, target, mask)

    def forward(self, images: torch.Tensor, stride: int = 5, phase: int | None = None, batch: dict | None = None) -> dict:
        contract = _build_prediction_marker_forward_contract(
            project_config=self.project_config,
            batch={},
            images=images,
            global_stride=stride,
            global_anchor_phase=min(2, stride - 1) if phase is None else phase,
        )
        # Requested-time interpolation is a post-process that training never uses.
        contract["temporal_interpolation_query_map"] = None
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
            raw = _call_marker_model(self.marker, images.to(dtype=torch.bfloat16), **contract)
        outputs = raw.to_dict()
        outputs["global_anchor_indices"] = contract["multirate_frame_map"].global_anchor_indices
        if self.variant == "diffusion" and self.training and batch is not None:
            self._flat_offsets = {
                "joint": outputs["dense_joint_offset"].flatten(0, 2),
                "vertex": outputs["dense_vertex_offset"].flatten(0, 2),
            }
            with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
                outputs["tactile_flow_loss"] = self._tactile_loss(batch)
        return outputs

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
        if self.tactile is not None:
            payload["egotac_tactile_head"] = {k: v.detach().cpu() for k, v in self.tactile.state_dict().items()}
            payload["egotac_diffusion_options"] = dict(self.diffusion_options)
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
        if self.tactile is not None:
            if "egotac_tactile_head" not in payload:
                raise ValueError(f"{path} has no diffusion head; it is a baseline checkpoint")
            self.tactile.load_state_dict(payload["egotac_tactile_head"])
