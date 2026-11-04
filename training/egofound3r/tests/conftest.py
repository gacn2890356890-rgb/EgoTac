import os
from pathlib import Path

# Tests of the world-model head run against the modified tree.
os.environ["EGOFOUND3R_ROOT"] = str(Path(__file__).resolve().parents[1] / "EgoFound3R_world")

from egotac.upstream import ensure_upstream  # noqa: E402

ensure_upstream()
