"""Write portable metadata outside Git; feature/label tensors remain external."""

import argparse
import json
from pathlib import Path

from .runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original local v15 dataset.json")
    parser.add_argument("--output", required=True, help="Ignored external metadata path")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    source = json.loads(Path(args.source).read_text())
    rows = [{key: row[key] for key in ("name", "task", "split", "frames")}
            | {"file": "bundles/" + Path(row["file"]).name}
            for row in source["rows"] if row["split"] in {"train", "val"}]
    stats = {key: source["stats"][key] for key in ("geometry_mean", "geometry_std", "contact_prior", "positive_log_median", "response_scale", "train_patch_mean", "train_patch_median")}
    result = {"schema_version": 1, "task": "same_frame_window_estimation",
              "data_revision": source.get("data_revision"), "stats_source": "train_only",
              "feature_causality": "not_verified_for_forecasting", "rows": rows, "stats": stats}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(json.dumps({"state": "portable_manifest_written", "episodes": len(rows), "data_copied": False, "weights_copied": False}))


if __name__ == "__main__":
    main()
