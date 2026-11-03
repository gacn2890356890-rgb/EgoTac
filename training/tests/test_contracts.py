"""Small synthetic interface/metric tests; no training dataset or model fitting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from egotouch.contact_metrics import ap
from egotouch.data import load_manifest, load_row
from egotouch.metrics import metrics, threshold_from_val
from egotouch.train import validate_config


def stats():
    return {"geometry_mean": [0.] * 14, "geometry_std": [1.] * 14,
            "contact_prior": [.1] * 12, "positive_log_median": [1.] * 12, "response_scale": [1.] * 12}


class Contracts(unittest.TestCase):
    def test_ap_ties_and_empty_positives(self):
        self.assertAlmostEqual(ap(np.ones(4), np.array([True, False, True, False])), .5)
        self.assertEqual(ap(np.array([.1, .8]), np.array([False, False])), 0.)
        self.assertEqual(ap(np.array([.9, .8, .2]), np.array([True, True, False])), 1.)

    def test_macro_is_not_frame_weighted(self):
        records = []
        for name, task, n, response in [("a", "one", 100, 1.), ("b", "one", 2, 3.), ("c", "two", 2, 10.)]:
            records.append({"name": name, "task": task, "target": np.zeros((n, 12)), "valid": np.ones((n, 12), bool),
                            "probability": np.full((n, 12), .1), "amplitude": np.full((n, 12), response),
                            "response": np.full((n, 12), response), "timestamps": np.arange(n) / 10.})
        result = metrics(records, .5)
        self.assertAlmostEqual(result["task_macro"]["soft_MAE"], 6.)
        self.assertEqual(result["task_macro"]["AP"], result["task_macro"]["AUPRC"])
        json.dumps(result, allow_nan=False)
        self.assertGreaterEqual(threshold_from_val(records), .05)

    def test_relative_paths_and_split_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            data = {"stats": stats(), "rows": [{"name": "synthetic", "task": "test", "split": "train", "file": "bundles/synthetic.pt"}]}
            path.write_text(json.dumps(data))
            self.assertEqual(load_manifest(path)["rows"][0]["file"], str(Path(folder).resolve() / "bundles/synthetic.pt"))
            data["rows"].append({**data["rows"][0], "split": "val"})
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_manifest(path)

    def test_no_invalid_future_task_claim(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/v15_hand.json").read_text())
        validate_config(config)
        wrong = copy.deepcopy(config)
        wrong["task"] = "future_joint_prediction"
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_config(wrong)
        wrong = copy.deepcopy(config)
        wrong["steps"] = 10001
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_config(wrong)

    def test_bundle_shapes_and_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            n = 2
            row = {"grid": torch.zeros(n, 64, 1024), "local_grid": torch.zeros(n, 32, 1024),
                   "geometry": torch.zeros(n, 12, 14), "uv": torch.zeros(n, 12, 2), "local_uv": torch.zeros(n, 32, 2),
                   "geometry_valid": torch.ones(n, 12, dtype=torch.bool), "local_valid": torch.ones(n, 32, dtype=torch.bool),
                   "target": torch.zeros(n, 12), "target_valid": torch.ones(n, 12, dtype=torch.bool),
                   "frame_indices": torch.arange(n), "timestamps": torch.arange(n) / 10.}
            torch.save(row, path)
            self.assertEqual(len(load_row(path)["grid"]), n)
            row["target_valid"].zero_()
            torch.save(row, path)
            with self.assertRaisesRegex(ValueError, "no valid"):
                load_row(path)


if __name__ == "__main__":
    unittest.main()
