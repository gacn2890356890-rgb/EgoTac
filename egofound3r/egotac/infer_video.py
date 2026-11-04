"""Run the official infer_marker_video.py with the diffusion head installed.

Arguments are the official script's, plus --tactile-checkpoint. Marker weights still come from
--checkpoint, so official weights plus a head isolates exactly what the head changes.
"""
from __future__ import annotations

import argparse
import sys

import torch

from .diffusion_head import TactileFlowHead, TactileHooks
from .upstream import ensure_upstream


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tactile-checkpoint", required=True)
    known, remaining = parser.parse_known_args()
    root = ensure_upstream()
    sys.path.insert(0, str(root))
    import infer_marker_video as official

    payload = torch.load(known.tactile_checkpoint, map_location="cpu", weights_only=False)
    if "egotac_tactile_head" not in payload:
        raise ValueError(f"{known.tactile_checkpoint} has no diffusion head")
    options = payload["egotac_diffusion_options"]
    original_load = official.load_marker_model_weights

    def load_and_attach(model, *args, **kwargs):
        result = original_load(model, *args, **kwargs)
        device = next(model.parameters()).device
        tactile = TactileFlowHead(options.get("dim", 256), options.get("layers", 4), options.get("heads", 8))
        tactile.load_state_dict(payload["egotac_tactile_head"])
        # Held by the hooks, not registered on the model: official code requires one dtype inside it.
        TactileHooks(model, tactile.float().to(device).eval(), options)
        return result

    official.load_marker_model_weights = load_and_attach
    sys.argv = [official.__file__, *remaining]
    official.main()


if __name__ == "__main__":
    main()
