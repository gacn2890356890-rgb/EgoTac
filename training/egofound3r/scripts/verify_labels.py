"""Draw labels on real frames and report label statistics, so a human can check a converter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from egotac.data.clip import decode_frame
from egotac.data.labels import CONTACT_THRESHOLD_JOINT_M, CONTACT_THRESHOLD_VERTEX_M


def project(points, K):
    return np.stack([K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2], K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]], -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.clips / "manifest.jsonl").open()]
    stats: dict = {}
    for i, row in enumerate(rows):
        data = np.load(args.clips / row["path"])
        ds = stats.setdefault(row["dataset"], {"clips": 0, "hand_valid": [], "contact_j": [], "contact_v": [], "vis_j": [], "vis_v": [], "dist_v_mm": [], "bone_mm": []})
        ds["clips"] += 1
        valid = data["hand_valid"]
        ds["hand_valid"].append(valid.mean())
        ds["contact_j"].append(np.nanmean((data["dist_joints"][valid] < CONTACT_THRESHOLD_JOINT_M)))
        ds["contact_v"].append(np.nanmean((data["dist_verts"][valid] < CONTACT_THRESHOLD_VERTEX_M)))
        ds["vis_j"].append(data["vis_joints"][valid].mean())
        ds["vis_v"].append(data["vis_verts"][valid].mean())
        ds["dist_v_mm"].append(np.nanmedian(data["dist_verts"][valid]) * 1000)
        joints = data["joints_cam"][valid]
        ds["bone_mm"].append(np.linalg.norm(joints[:, 12] - joints[:, 0], axis=-1).mean() * 1000)
        if i >= args.num_images * max(1, len(rows) // args.num_images):
            continue
        if i % max(1, len(rows) // args.num_images):
            continue
        t = 10
        image = cv2.cvtColor(decode_frame(data["jpeg_bytes"], data["jpeg_offsets"], t), cv2.COLOR_RGB2BGR)
        K = data["K"][t]
        panel_vis, panel_con = image.copy(), image.copy()
        for side in range(2):
            if not valid[t, side]:
                continue
            for points, dist, vis, thr in (
                (data["verts_cam"][t, side], data["dist_verts"][t, side], data["vis_verts"][t, side], CONTACT_THRESHOLD_VERTEX_M),
                (data["joints_cam"][t, side], data["dist_joints"][t, side], data["vis_joints"][t, side], CONTACT_THRESHOLD_JOINT_M),
            ):
                uv = project(points, K)
                for (u, v), d, seen in zip(uv, dist, vis):
                    if not (np.isfinite(u) and np.isfinite(v)):
                        continue
                    center = (int(u), int(v))
                    cv2.circle(panel_vis, center, 2, (0, 220, 0) if seen else (0, 0, 230), -1)
                    cv2.circle(panel_con, center, 2, (0, 0, 255) if d < thr else (230, 170, 80), -1)
        cv2.putText(panel_vis, f"{row['dataset']} green=visible red=occluded", (8, 18), 0, 0.45, (255, 255, 255), 1)
        cv2.putText(panel_con, "red=contact blue=none; left hand side 0", (8, 18), 0, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(args.output / f"{row['dataset']}_{i:04d}.jpg"), np.hstack([panel_vis, panel_con]))
    summary = {
        name: {k: (v if k == "clips" else round(float(np.nanmean(v)), 4)) for k, v in ds.items()} for name, ds in stats.items()
    }
    (args.output / "label_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
