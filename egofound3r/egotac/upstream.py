"""Single entry point to the unmodified official EgoFound3R code."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def upstream_root() -> Path:
    env = os.environ.get("EGOFOUND3R_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[1] / "third_party" / "EgoFound3R"
    if not (root / "egohandmetric_prompt").is_dir():
        raise FileNotFoundError(
            f"official EgoFound3R code not found at {root}; set EGOFOUND3R_ROOT or run scripts/setup.sh"
        )
    return root


def ensure_upstream() -> Path:
    root = upstream_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
