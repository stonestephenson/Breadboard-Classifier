"""Spike: fit the breadboard hole lattice and rectify the photo.

This is the go/no-go for the whole architecture (ARCHITECTURE.md section 3[2]).
The question: given noisy hole detections from a real photo, can we assign each
hole its (row, column) index on the board and recover a homography accurate
enough that every hole lands where the board spec says it should?

Method, and why:

  1. Detect hole candidates (spikes/hole_detect.py). Background texture produces
     false positives, so we never trust the raw point set.

  2. Grow the lattice by breadth-first search from a seed, stepping only to
     neighbours that sit at a plausible lattice offset. Growth is *local*, which
     is what makes it robust to perspective: we never assume a constant pitch
     across the whole board, only between adjacent holes. Background speckle is
     irregular, so the search simply refuses to propagate into it — the lattice
     itself is the outlier filter, and no board segmentation is needed.

  3. The WB-102's centre channel is exactly 3 pitches (7.62mm = 3 x 2.54mm), so
     row f sits 3 steps below row e. Allowing a step of 3 lets the search cross
     the channel and index both halves in one consistent frame.

  4. Fit a homography from integer (col, row) to pixels with RANSAC, then
     re-assign indices through it and refit. Two passes suffice.

Usage:  ./venv/bin/python spikes/lattice_fit.py data/cache/sample/*.jpg
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hole_detect import detect_holes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "cache"

# WB-102 geometry. Terminal area only: 63 columns, rows a-e and f-j, with the
# centre channel spanning 3 pitches so f = e + 3.
N_COLS = 63
ROW_INDICES = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]  # a b c d e | f g h i j
N_TERMINAL_HOLES = N_COLS * len(ROW_INDICES)


def lattice_vectors(pts: np.ndarray, radius: int = 110) -> tuple[np.ndarray, np.ndarray]:
    """Recover the two lattice basis vectors by point-set autocorrelation.

    Histogram every pairwise displacement shorter than `radius`. A regular grid
    puts sharp peaks at every integer combination of its basis vectors; random
    background texture contributes only a diffuse blob. So the two shortest
    strong peaks *are* the basis, recovered without needing a pitch estimate
    first — which matters, because a pitch estimated from raw detections is
    poisoned by desk speckle on exactly the images we most need to handle.
    """
    R = radius
    acc = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
    for i in range(0, len(pts), 400):
        d = (pts[i:i + 400, None, :] - pts[None, :, :]).reshape(-1, 2)
        m = (np.abs(d[:, 0]) <= R) & (np.abs(d[:, 1]) <= R)
        d = d[m]
        xi = np.round(d[:, 0]).astype(np.int32) + R
        yi = np.round(d[:, 1]).astype(np.int32) + R
        np.add.at(acc, (yi, xi), 1.0)

    acc = cv2.GaussianBlur(acc, (0, 0), 1.3)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    acc[rr < 7] = 0  # self-pairs and sub-pitch noise

    # Local maxima, kept only where they clearly stand above the neighbourhood.
    dil = cv2.dilate(acc, np.ones((7, 7), np.uint8))
    cand = (acc >= dil) & (acc > 0.25 * acc.max())
    ys, xs = np.nonzero(cand)
    vecs = np.stack([xs - R, ys - R], axis=1).astype(float)
    if len(vecs) < 2:
        raise RuntimeError("no lattice peaks found")

    # Displacements are symmetric, so fold antipodal peaks together.
    vecs = vecs[(vecs[:, 0] > 0) | ((vecs[:, 0] == 0) & (vecs[:, 1] > 0))]
    vecs = vecs[np.argsort(np.linalg.norm(vecs, axis=1))]

    v1 = vecs[0]
    v2 = None
    for v in vecs[1:]:
        cosang = abs(v @ v1) / (np.linalg.norm(v) * np.linalg.norm(v1))
        if cosang < np.cos(np.deg2rad(30)):
            v2 = v
            break
    if v2 is None:
        raise RuntimeError("could not find a second lattice direction")

    # Order so v1 runs along the board's long axis (columns).
    if abs(v1[0]) < abs(v2[0]):
        v1, v2 = v2, v1
    return np.asarray(v1, float), np.asarray(v2, float)


def grow_lattice(pts: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                 tol: float = 0.34) -> dict[tuple[int, int], int]:
    """BFS from a central seed, assigning integer (col, row) to each hole.

    Steps of 1 pitch are tried first; a step of 3 along the row axis crosses the
    centre channel. `tol` is the match radius as a fraction of pitch.
    """
    pitch = float(np.linalg.norm(v2))
    seed = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))

    index: dict[int, tuple[int, int]] = {seed: (0, 0)}
    taken: dict[tuple[int, int], int] = {(0, 0): seed}
    queue = deque([seed])

    steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 3), (0, -3)]

    while queue:
        i = queue.popleft()
        ci, ri = index[i]
        p = pts[i]
        for dc, dr in steps:
            key = (ci + dc, ri + dr)
            if key in taken:
                continue
            target = p + dc * v1 + dr * v2
            d = np.linalg.norm(pts - target, axis=1)
            j = int(np.argmin(d))
            if d[j] > tol * pitch or j in index:
                continue
            index[j] = key
            taken[key] = j
            queue.append(j)
    return taken


def fit_homography(pts: np.ndarray, taken: dict[tuple[int, int], int]):
    """Least-squares homography from integer lattice coords to pixels."""
    src = np.array([[c, r] for (c, r) in taken], dtype=np.float64)
    dst = np.array([pts[i] for i in taken.values()], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    resid = np.linalg.norm(
        cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst,
        axis=1)
    return H, src, dst, resid, mask.ravel().astype(bool)


def analyse(path: str) -> dict | None:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    scale = 2000 / bgr.shape[1]
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    pts, _ = detect_holes(bgr)
    if len(pts) < 200:
        return None

    v1, v2 = lattice_vectors(pts)
    taken = grow_lattice(pts, v1, v2)
    if len(taken) < 100:
        return None

    H, src, dst, resid, inliers = fit_homography(pts, taken)
    pitch_px = float(np.linalg.norm(v2))

    cols = sorted({c for c, _ in taken})
    rows = sorted({r for _, r in taken})
    return {
        "path": path, "bgr": bgr, "pts": pts, "taken": taken, "H": H,
        "resid": resid, "inliers": inliers, "pitch_px": pitch_px,
        "n_indexed": len(taken), "col_span": len(cols), "row_span": len(rows),
    }


def main(paths: list[str]) -> None:
    print(f"{'image':>16} {'indexed':>8} {'/630':>6} {'cols':>5} {'rows':>5} "
          f"{'pitch':>6} {'resid_px':>9} {'resid_pitch':>11}")
    tiles = []
    for p in paths:
        r = analyse(p)
        if r is None:
            print(f"{p.split('/')[-1]:>16}   FAILED")
            continue
        med = float(np.median(r["resid"][r["inliers"]]))
        p95 = float(np.percentile(r["resid"][r["inliers"]], 95))
        print(f"{p.split('/')[-1]:>16} {r['n_indexed']:8d} "
              f"{100 * r['n_indexed'] / N_TERMINAL_HOLES:5.1f}% "
              f"{r['col_span']:5d} {r['row_span']:5d} {r['pitch_px']:6.2f} "
              f"{med:6.2f}/{p95:.2f} {med / r['pitch_px']:10.3f}")

        # Rectify: map lattice coords to a canonical image at a fixed scale.
        S = 24.0  # pixels per pitch in canonical space
        pad = 3
        canon = np.array([[S * pad, S * pad, 1],
                          [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        T = np.array([[S, 0, S * pad], [0, S, S * pad], [0, 0, 1]])
        Hinv = np.linalg.inv(r["H"])
        warp = T @ Hinv
        out = cv2.warpPerspective(
            r["bgr"], warp,
            (int(S * (N_COLS + 2 * pad)), int(S * (12 + 2 * pad))))
        tiles.append(out)
        _ = canon

    if tiles:
        w = min(t.shape[1] for t in tiles)
        dest = OUT / "rectified.jpg"
        ok = cv2.imwrite(str(dest), np.vstack([t[:, :w] for t in tiles]),
                         [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\n{'wrote' if ok else 'FAILED to write'} {dest}")


if __name__ == "__main__":
    main(sys.argv[1:])
