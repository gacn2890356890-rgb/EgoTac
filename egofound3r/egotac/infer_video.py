"""Run the official infer_marker_video.py with the world-model head installed.

Arguments are the official script's, plus --world-checkpoint. Marker weights still come from
--checkpoint, so official weights plus a head isolates exactly what the head changes.
"""
from __future__ import annotations

import argparse
import sys

import torch

from .upstream import ensure_upstream


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--world-checkpoint", required=True)
    known, remaining = parser.parse_known_args()
    root = ensure_upstream()
    sys.path.insert(0, str(root))
    import infer_marker_video as official
    from egohandmetric_prompt.world_model import WorldModelHead, WorldModelHooks

    payload = torch.load(known.world_checkpoint, map_location="cpu", weights_only=False)
    if "egotac_world_head" not in payload:
        raise ValueError(f"{known.world_checkpoint} has no world-model head")
    options = payload["egotac_world_options"]
    original_load = official.load_marker_model_weights

    def load_and_attach(model, *args, **kwargs):
        result = original_load(model, *args, **kwargs)
        device = next(model.parameters()).device
        head = WorldModelHead(options.get("dim", 512), options.get("layers", 8), options.get("heads", 8), options.get("dynamics_layers", 4), options["past_frames"], options["horizon"])
        head.load_state_dict(payload["egotac_world_head"])
        # Held by the hooks, not registered on the model: official code requires one dtype inside it.
        WorldModelHooks(model, head.float().to(device).eval(), options)
        return result

    official.load_marker_model_weights = load_and_attach
    sys.argv = [official.__file__, *remaining]
    official.main()


if __name__ == "__main__":
    main()
