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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hole_detect import detect_holes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "cache"

# WB-102 geometry. Terminal area only: 63 columns, rows a-e and f-j, with the
# centre channel spanning 3 pitches so f = e + 3.
N_COLS = 63
ROW_INDICES = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]  # a b c d e | f g h i j
N_TERMINAL_HOLES = N_COLS * len(ROW_INDICES)

# Largest lattice the board can possibly produce, with slack. 63 terminal
# columns; 12 row-slots for a-j including the 3-pitch centre channel, plus the
# power rails a few pitches beyond each edge.
MAX_COL_SPAN = 72
MAX_ROW_SPAN = 22


def lattice_vectors(pts: np.ndarray, radius: int = 110, n_peaks: int = 6
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Recover the two lattice basis vectors by point-set autocorrelation.

    Histogram every pairwise displacement shorter than `radius`. A regular grid
    puts sharp peaks at every integer combination of its basis vectors; random
    background texture contributes only a diffuse blob. Peaks therefore give us
    the basis without needing a pitch estimate first — which matters, because a
    pitch estimated from raw detections is poisoned by desk speckle on exactly
    the images we most need to handle.

    Returns *all* plausible basis pairs rather than one, strongest first.
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
    strength = acc[(vecs[:, 1] + R).astype(int), (vecs[:, 0] + R).astype(int)]
    vecs = vecs[np.argsort(strength)[::-1]][:n_peaks]

    # Every non-degenerate pair is a candidate basis. Picking the shortest peak
    # outright is wrong: clutter (desk texture, keyboards, carpet) can produce a
    # strong short spurious peak, and locking onto a half-pitch sub-lattice
    # silently ruins every index. The caller decides between candidates by which
    # one actually explains the most points.
    cands = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a, b = vecs[i], vecs[j]
            cosang = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if cosang < np.cos(np.deg2rad(30)):
                # Order so the first vector runs along the board's long axis.
                cands.append((b, a) if abs(a[0]) < abs(b[0]) else (a, b))
    if not cands:
        raise RuntimeError("no valid lattice basis candidates")
    return [(np.asarray(u, float), np.asarray(v, float)) for u, v in cands]


def grow_lattice(pts: np.ndarray, v1: np.ndarray, v2: np.ndarray, seed: int,
                 tree: cKDTree, tol: float = 0.34) -> dict[tuple[int, int], int]:
    """BFS from `seed`, assigning integer (col, row) to each hole.

    Steps of 1 pitch are tried first; a step of 3 along the row axis crosses the
    centre channel. `tol` is the match radius as a fraction of pitch. Growth is
    local, so perspective costs nothing and irregular clutter is never reached.
    """
    pitch = float(np.linalg.norm(v2))
    index: dict[int, tuple[int, int]] = {seed: (0, 0)}
    taken: dict[tuple[int, int], int] = {(0, 0): seed}
    queue = deque([seed])

    steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 3), (0, -3)]

    # Growth is bounded to the board's own dimensions. Without this, a lattice
    # can escape into background clutter (desk texture, a keyboard, carpet) and
    # run for dozens of extra rows, which inflates its coverage score and lets a
    # wrong basis beat the right one. Bounding every candidate the same way makes
    # coverage a fair comparison.
    lo_c = hi_c = lo_r = hi_r = 0

    while queue:
        i = queue.popleft()
        ci, ri = index[i]
        p = pts[i]
        for dc, dr in steps:
            c, r = ci + dc, ri + dr
            key = (c, r)
            if key in taken:
                continue
            if (max(hi_c, c) - min(lo_c, c) + 1 > MAX_COL_SPAN
                    or max(hi_r, r) - min(lo_r, r) + 1 > MAX_ROW_SPAN):
                continue
            d, j = tree.query(p + dc * v1 + dr * v2, k=1,
                              distance_upper_bound=tol * pitch)
            if not np.isfinite(d) or j in index:
                continue
            index[j] = key
            taken[key] = j
            lo_c, hi_c = min(lo_c, c), max(hi_c, c)
            lo_r, hi_r = min(lo_r, r), max(hi_r, r)
            queue.append(j)
    return taken


def best_lattice(pts: np.ndarray, n_seeds: int = 4
                 ) -> tuple[dict[tuple[int, int], int], np.ndarray, np.ndarray]:
    """Try every candidate basis from several seeds; keep the largest lattice.

    Selecting the basis by *how many points it explains* — rather than by peak
    strength or shortest length — is what stops a spurious sub-pitch peak from
    being adopted. On cluttered scenes (keyboard, carpet, desk edge) the naive
    choice locked onto a half-pitch lattice and indexed a single point.
    """
    tree = cKDTree(pts)
    # Seed from the densest neighbourhoods: the board is the one place in the
    # frame with hundreds of regularly spaced blobs, so density finds it even
    # when it is a minority of the detections.
    density = tree.query_ball_point(pts, r=60, return_length=True)
    seeds = [int(i) for i in np.argsort(density)[::-1][:n_seeds * 40:40]]

    def spans(t: dict[tuple[int, int], int]) -> tuple[int, int]:
        cs = [c for c, _ in t]
        rs = [r for _, r in t]
        return max(cs) - min(cs) + 1, max(rs) - min(rs) + 1

    def plausible(t: dict[tuple[int, int], int]) -> bool:
        # Hard geometric prior from the board spec: a WB-102 is 63 columns wide
        # and 12 row-slots deep (a-e, a 3-pitch centre channel, f-j), plus power
        # rails a few pitches beyond. Anything claiming far more than that is
        # not the board — it is growth that has escaped into background clutter,
        # or a diagonal basis inflating both spans. Rejecting on span is what
        # stops those candidates from winning on coverage alone.
        c, r = spans(t)
        return c <= MAX_COL_SPAN and r <= MAX_ROW_SPAN

    scored = []
    for v1, v2 in lattice_vectors(pts):
        best_for_basis: dict[tuple[int, int], int] = {}
        for seed in seeds:
            taken = grow_lattice(pts, v1, v2, seed, tree)
            if len(taken) > len(best_for_basis):
                best_for_basis = taken
        if best_for_basis:
            scored.append((best_for_basis, v1, v2))
    if not scored:
        raise RuntimeError("no basis produced a lattice")

    # Two-stage choice, and both stages are needed.
    #
    # Coverage alone is not enough: the diagonals of a square lattice form
    # another square, perpendicular lattice with sqrt(2) times the spacing. It
    # fits a homography perfectly while indexing every *other* hole, and it can
    # reach further into background clutter, so it often scores higher. Its
    # cells have twice the area.
    #
    # Shortest-basis alone is not enough either: spurious sub-pitch peaks from
    # desk texture are shorter still, but explain almost nothing.
    #
    # So: keep candidates that explain a comparable share of the points, then
    # among those take the finest lattice — the true basis is the shortest one
    # that actually works.
    top = max(len(t) for t, _, _ in scored)
    viable = [s for s in scored if len(s[0]) >= 0.5 * top]
    def cell_area(s: tuple) -> float:
        (ax, ay), (bx, by) = s[1], s[2]
        return abs(ax * by - ay * bx)

    taken, v1, v2 = min(viable, key=cell_area)
    return taken, v1, v2


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

    taken, v1, v2 = best_lattice(pts)
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
