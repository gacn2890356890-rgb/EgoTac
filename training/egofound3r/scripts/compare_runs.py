"""Side-by-side table of evaluation reports from egotac.evaluate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("mpjpe_rr", "RR-MPJPE (mm)", "low"),
    ("mpjpe_pa", "PA-MPJPE (mm)", "low"),
    ("mpvpe_rr", "RR-MPVPE (mm)", "low"),
    ("mpjpe_raw", "Raw MPJPE (mm)", "low"),
    ("contact_joint_f1", "Contact F1, joints", "high"),
    ("contact_joint_ap", "Contact AP, joints", "high"),
    ("contact_vertex_f1", "Contact F1, vertices", "high"),
    ("contact_vertex_ap", "Contact AP, vertices", "high"),
    ("distance_joint_mae_all", "Distance MAE all, joints (mm)", "low"),
    ("distance_joint_mae_gt", "Distance MAE GT-contact, joints (mm)", "low"),
    ("distance_vertex_mae_all", "Distance MAE all, vertices (mm)", "low"),
    ("distance_vertex_mae_gt", "Distance MAE GT-contact, vertices (mm)", "low"),
    ("visibility_joint_f1", "Visibility F1, joints", "high"),
    ("visibility_vertex_f1", "Visibility F1, vertices", "high"),
    ("future_5_contact_f1", "Future 0.5 s contact F1", "high"),
    ("future_5_contact_ap", "Future 0.5 s contact AP", "high"),
    ("future_5_hold_last_model_f1", "  reference: hold model's last frame", None),
    ("future_5_hold_last_truth_f1", "  reference: hold true last frame", None),
    ("future_5_distance_mae_mm", "Future 0.5 s distance MAE (mm)", "low"),
    ("windows", "Windows", None),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="label=path/to/report.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    runs = []
    for item in args.runs:
        label, _, path = item.partition("=")
        runs.append((label, json.loads(Path(path).read_text())["metrics"]))
    datasets = sorted({name for _, metrics in runs for name in metrics})
    lines = []
    for dataset in datasets:
        lines += [f"### {dataset}", "", "| Metric | " + " | ".join(label for label, _ in runs) + " |", "|---|" + "---|" * len(runs)]
        for key, title, _direction in ROWS:
            cells = []
            for _, metrics in runs:
                value = metrics.get(dataset, {}).get(key)
                cells.append("–" if value is None else (f"{value:.0f}" if key == "windows" else f"{value:.3f}" if value < 1.5 else f"{value:.2f}"))
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
