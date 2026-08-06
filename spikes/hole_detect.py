"""Spike: can we reliably detect breadboard hole centres in real photos?

This is exploratory, not production. It answers one question: does a simple,
deterministic blob detector find enough of the ~830 holes, accurately enough,
to fit a lattice against? If yes, rectification is viable and the architecture
holds. See ARCHITECTURE.md section 3[2].

Usage:  ./venv/bin/python spikes/hole_detect.py data/cache/sample/*.jpg
"""

from __future__ import annotations

import sys

import cv2
import numpy as np


def detect_holes(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (Nx2 array of hole centres in pixels, estimated pitch in pixels).

    Holes are small dark squares on bright plastic. We threshold locally so
    that lighting gradients across the board do not matter, then keep blobs
    that look like holes: roughly square, roughly the right size, solid.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Local threshold: holes are darker than the plastic immediately around
    # them. Block size must be a few hole pitches wide so the local mean is
    # dominated by plastic rather than by the hole itself.
    dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=8,
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)

    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6 or area > 400:
            continue
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > 1.9:
            continue
        # Solidity: a hole is a filled square, not a thin or ragged shape.
        if area / (w * h) < 0.45:
            continue
        keep.append(centroids[i])

    pts = np.array(keep, dtype=np.float32) if keep else np.zeros((0, 2), np.float32)
    return pts, estimate_pitch(pts)


def estimate_pitch(pts: np.ndarray) -> float:
    """Median nearest-neighbour distance, which for a dense grid is the pitch."""
    if len(pts) < 20:
        return float("nan")
    # Brute-force nearest neighbour is fine at these point counts.
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def main(paths: list[str]) -> None:
    tiles = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"skip {p}")
            continue
        # Work at a consistent width so pitch estimates are comparable.
        scale = 2000 / bgr.shape[1]
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        pts, pitch = detect_holes(bgr)
        expected = 830
        print(f"{p.split('/')[-1]:>16}  holes={len(pts):4d} "
              f"({100 * len(pts) / expected:5.1f}% of 830)  pitch={pitch:5.1f}px")

        vis = bgr.copy()
        for x, y in pts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        tiles.append(cv2.resize(vis, (700, int(700 * vis.shape[0] / vis.shape[1]))))

    if tiles:
        h = min(t.shape[0] for t in tiles)
        tiles = [t[:h] for t in tiles]
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
        cv2.imwrite("data/cache/holes_overlay.jpg", np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("\nwrote data/cache/holes_overlay.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
