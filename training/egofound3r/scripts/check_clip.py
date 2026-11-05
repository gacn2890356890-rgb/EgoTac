"""Draw a clip's labeled joints on its frames, to check a new dataset by eye before converting all of it.
Usage: python scripts/check_clip.py <clip.npz> <output.jpg>"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from egotac.data.clip import decode_frame

COLORS = ((255, 80, 80), (80, 200, 255))


def main() -> None:
    clip, output = sys.argv[1], sys.argv[2]
    with np.load(clip) as data:
        arrays = {key: data[key] for key in data.files}
    panels = []
    for t in (0, arrays["joints_cam"].shape[0] // 2, arrays["joints_cam"].shape[0] - 1):
        image = cv2.cvtColor(decode_frame(arrays["jpeg_bytes"], arrays["jpeg_offsets"], t), cv2.COLOR_RGB2BGR)
        K = arrays["K"][t]
        for side in range(2):
            if not arrays["hand_valid"][t, side]:
                continue
            for x, y, z in arrays["joints_cam"][t, side]:
                center = (int(K[0, 0] * x / z + K[0, 2]), int(K[1, 1] * y / z + K[1, 2]))
                cv2.circle(image, center, 3, COLORS[side], -1)
        panels.append(image)
    cv2.imwrite(output, np.concatenate(panels, axis=1))
    print(f"wrote {output}; the dots should sit on the hands (blue left, orange right)")


if __name__ == "__main__":
    main()
