"""Rectification: find the board in a photo, and map holes between photo and board.

Every later stage reads the board in *canonical* space, a fixed top-down view in
which each hole always sits at the same pixel. That removes perspective, scale and
rotation before anything tries to recognise a part (ARCHITECTURE.md section 3[2]).
It also works in reverse: anything found in canonical space can be drawn back
onto the student's own photo.

The fit is breadboard-normalizer, vendored with a few small
changes: optional imports, and a pluggable corner model
(credits and changes: breadboard/_vendor/breadboard_normalizer/__init__.py). It
works coarse to fine:

1. A pretrained document-corner model (DocAligner) finds the board's four
   corners. It needs no training from us, and it only has to be roughly right.
2. Pinholes are detected in the rough warp and snapped onto a hole template,
   which corrects the corner model's error.
3. The red and blue rail stripes decide which end is column 1.

This module adds four things:

- our hole names on the template;
- the corner model run on onnxruntime directly, because DocAligner's own package
  does not install on this project's Python;
- two self-checks the normaliser's own grade lacks. The rail stripes must
  actually have been read, and the fitted column must explain the holes clearly
  better than the same fit shifted by a column or two. The hole grid repeats, so
  an off-by-one fit can otherwise still grade as perfect;
- a result type that answers "where is hole j37 in this photo?" and "which hole
  is under this pixel?".
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
import urllib.request
from collections.abc import Generator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pillow_heif
from PIL import Image, ImageOps

from breadboard.board import ALL_HOLES, ROWS_LOWER, ROWS_UPPER


@contextlib.contextmanager
def _quiet() -> Generator[None, None, None]:
    """The vendored normaliser reports progress with print(); keep that off stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


with _quiet():
    from breadboard._vendor.breadboard_normalizer.normalizer import Normalizer, PinGrid

pillow_heif.register_heif_opener()

# The normaliser's default output, and the only size it has been tuned and tested at.
# About 16 px per hole pitch.
CANONICAL_SIZE = (1024, 340)

# How much better, in fraction of detected holes landing on the template, the fitted
# column registration must be than the best shift by one or two columns. Shifting
# costs the rails their gaps and the grid its ends, so a true fit wins clearly: on
# the 13 test photos the margin was 0.038-0.062. Below this, the fit cannot tell
# column 30 from column 31, so the result must not be trusted.
MIN_COLUMN_MARGIN = 0.02
_SHIFTS = (-2, -1, 1, 2)

# Photos are shrunk to this before fitting, matching how upstream was tested.
# Results are always reported in the original photo's pixels.
WORKING_MAX_SIDE = 1920

# DocAligner's heatmap corner model (Apache 2.0, github.com/DocsaidLab/DocAligner),
# fetched on first use from the same place DocAligner fetches it.
MODEL_NAME = "fastvit_sa24_h_e_bifpn_256_fp32.onnx"
MODEL_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl&export=download&confirm=t"
)
MODEL_SHA256 = "7f9f5a8935b2eb22b3ee0245d34996063f54562df390d34714af2d76928695bc"
MODEL_DIR = Path(
    os.environ.get("BREADBOARD_MODEL_DIR")
    or Path(__file__).resolve().parent.parent / "data" / "models"
)


def hole_name(label: tuple[str, int, int]) -> str:
    """Our name for one of the normaliser's template labels.

    Its labels are (grid, x, y). x counts from column 1, or from rail hole 1,
    which sits at the column-1 end. y counts outward from the top edge for the
    upper grids, and the lower grids are mirror images of the upper ones, so
    their y counts outward from the bottom edge. On the real board the red (+)
    stripe is the outer rail row along row a and the inner rail row along row j.
    """
    grid, x, y = label
    if grid == "base_top":
        return f"{ROWS_UPPER[y]}{x + 1}"
    if grid == "base_bot":
        return f"{ROWS_LOWER[len(ROWS_LOWER) - 1 - y]}{x + 1}"
    if grid == "rail_top":
        return f"p1{'+-'[y]}:{x + 1}"
    if grid == "rail_bot":
        return f"p2{'-+'[y]}:{x + 1}"
    raise ValueError(f"unknown template label {label!r}")


@cache
def _grid() -> PinGrid:
    """The normaliser's hole template at CANONICAL_SIZE (no corner model needed)."""
    with _quiet():
        return PinGrid(np.array(CANONICAL_SIZE))


@cache
def _template() -> tuple[tuple[str, ...], np.ndarray]:
    grid = _grid()
    names = tuple(hole_name(label) for label in grid.labels or ())
    return names, np.asarray(grid.points, dtype=np.float64)


def template_holes() -> dict[str, tuple[float, float]]:
    """Every hole's centre in canonical pixels, keyed by our hole name."""
    names, points = _template()
    return {n: (float(x), float(y)) for n, (x, y) in zip(names, points, strict=True)}


def ensure_model(directory: Path = MODEL_DIR) -> Path:
    """Path to the corner model, downloading and verifying it on first use (83 MB)."""
    path = directory / MODEL_NAME
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
        partial = Path(tmp.name)
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    tmp.write(chunk)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    if digest.hexdigest() != MODEL_SHA256:
        partial.unlink()
        raise RuntimeError(
            f"downloaded corner model failed its checksum; fetch {MODEL_URL} by hand "
            f"into {path} and check its sha256 is {MODEL_SHA256}"
        )
    partial.replace(path)
    return path


class CornerModel:
    """The board's four corners, via DocAligner's heatmap model.

    A port of DocAligner's inference (docaligner/heatmap_reg/infer.py, v1.1.1)
    onto onnxruntime and OpenCV. It gave identical corners on 13 real photos. It
    takes the image as the normaliser passes it, RGB, and returns up to four (x, y)
    points in that image's pixels.
    """

    INPUT_SIZE = 256
    HEATMAP_THRESHOLD = 0.3

    def __init__(self, path: Path) -> None:
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        side = self.INPUT_SIZE
        x = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
        x = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        (heatmaps,) = self._session.run(["heatmap"], {"img": x})
        points = []
        for heat in np.asarray(heatmaps)[0][:4]:
            m = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
            m[m < self.HEATMAP_THRESHOLD] = 0
            _, mask = cv2.threshold(
                (m * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            moments = [cv2.moments(c) for c in contours if len(c) >= 3]
            if not moments:
                continue
            best = max(moments, key=lambda mm: mm["m00"])
            area = best["m00"] + 1e-5
            points.append((best["m10"] / area, best["m01"] / area))
        return np.array(points, dtype=np.float32)


@cache
def _normalizer() -> Normalizer:
    with _quiet():
        return Normalizer(
            output_resolution=CANONICAL_SIZE, corner_model=CornerModel(ensure_model())
        )


def column_margin(holes: np.ndarray) -> float:
    """How much better detected holes fit the template than the same holes shifted.

    holes are detected hole centres in canonical pixels. Returns the fraction that
    land on a template hole, minus the best fraction under a shift of one or two
    columns either way. Positive means this column registration is the best
    explanation of what was detected.
    """
    grid = _grid()
    pitch = float(grid.pitch[0])

    def landed(dx: float) -> float:
        _, inlier_ratio, _ = grid.evaluate_fit(holes + np.array([dx, 0.0], np.float32))
        return float(inlier_ratio)

    return landed(0.0) - max(landed(k * pitch) for k in _SHIFTS)


@dataclass(frozen=True, eq=False)
class Rectification:
    """A board found in a photo.

    photo_to_canonical maps original photo pixels to canonical pixels.
    canonical is the top-down RGB view, CANONICAL_SIZE, column 1 on the right and
    row a at the top.

    confidence is the normaliser's own grade of how well the detected holes
    matched the template: 1.0 good, 0.75 close but imperfect, and 0.0 when the
    hole fit was rejected and only the rough corner fit remains. It does not cover
    which end is column 1, or an off-by-one column fit; oriented and
    column_margin do.
    """

    photo_to_canonical: np.ndarray
    canonical: np.ndarray
    confidence: float
    oriented: bool
    column_margin: float

    @property
    def ok(self) -> bool:
        """Whether holes may be read from this fit. If not, abstain.

        Requires an accepted hole fit, rail stripes that confirm which end is
        column 1, and a clear column registration. This is the minimum bar.
        Anything that tells a student their wiring is *correct* should also
        require confidence == 1.0 (CLAUDE.md: "correct" is a higher bar than
        "incorrect").
        """
        return (
            self.confidence > 0.0
            and self.oriented
            and self.column_margin >= MIN_COLUMN_MARGIN
        )

    def to_photo(self, hole: str) -> tuple[float, float]:
        """Where a hole's centre is in the original photo, in pixels."""
        return self.photo_holes()[hole]

    def photo_holes(self) -> dict[str, tuple[float, float]]:
        """Every hole's centre in the original photo, keyed by hole name."""
        names, points = _template()
        back = np.linalg.inv(self.photo_to_canonical)
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), back).reshape(-1, 2)
        return {n: (float(x), float(y)) for n, (x, y) in zip(names, mapped, strict=True)}

    def hole_at(self, xy: tuple[float, float]) -> str:
        """The hole nearest a point in the original photo, even one off the board."""
        names, points = _template()
        p = np.array([[xy]], dtype=np.float64)
        cx, cy = cv2.perspectiveTransform(p, self.photo_to_canonical)[0, 0]
        return names[int(np.argmin(np.hypot(points[:, 0] - cx, points[:, 1] - cy)))]


def load_photo(path: str | Path) -> np.ndarray:
    """An RGB array of a photo on disk, upright per its EXIF tag. HEIC is fine."""
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


def rectify(image: np.ndarray) -> Rectification | None:
    """Find the board in an RGB photo. None if nothing board-shaped was found.

    Callers should abstain rather than read holes from a result with ok == False.
    """
    h, w = image.shape[:2]
    scale = min(1.0, WORKING_MAX_SIDE / max(h, w))
    if scale < 1.0:
        size = (round(w * scale), round(h * scale))
        work = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        to_work = np.diag([size[0] / w, size[1] / h, 1.0])
    else:
        work, to_work = image, np.eye(3)

    normalizer = _normalizer()
    with _quiet():
        canonical, _, score = normalizer.normalize_image(work)
    if canonical is None or score is None:
        return None
    homography = np.asarray(normalizer.last_homography, dtype=np.float64) @ to_work
    canonical = np.ascontiguousarray(canonical)

    # The normaliser treats rail stripes it could not read as "the right way up".
    # Re-read them on the final view: only "correct" confirms column 1's end.
    with _quiet():
        oriented = normalizer.breadboard_orientation_cv(canonical) == "correct"
    # After an accepted fit the normaliser leaves the detected holes in canonical
    # pixels; after a rejected one they are not comparable, and ok is False anyway.
    raw = normalizer.last_pinhole_detections
    detected = np.asarray([] if raw is None else raw, dtype=np.float32).reshape(-1, 2)
    margin = column_margin(detected) if score > 0.0 and len(detected) else 0.0
    return Rectification(
        photo_to_canonical=homography / homography[2, 2],
        canonical=canonical,
        confidence=float(score),
        oriented=oriented,
        column_margin=margin,
    )


def draw_holes(
    image: np.ndarray,
    rect: Rectification,
    labels: tuple[str, ...] = ("a1", "j1", "a63", "j63", "p1+:1", "p2-:50"),
) -> np.ndarray:
    """A copy of the photo with every hole marked and a few of them named.

    Green means the hole grid was confirmed, orange that only a rough fit exists.
    """
    out = image.copy()
    colour = (40, 200, 40) if rect.ok else (255, 140, 0)
    holes = rect.photo_holes()
    radius = max(2, round(min(image.shape[:2]) / 400))
    for x, y in holes.values():
        cv2.circle(out, (round(x), round(y)), radius, colour, -1)
    scale = max(0.5, min(image.shape[:2]) / 1200)
    for name in labels:
        x, y = holes[name]
        org = (round(x) + 2 * radius, round(y) - 2 * radius)
        for thickness, ink in ((4, (0, 0, 0)), (1, (255, 255, 0))):
            cv2.putText(out, name, org, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thickness)
    return out


__all__ = [
    "ALL_HOLES",
    "CANONICAL_SIZE",
    "CornerModel",
    "Rectification",
    "column_margin",
    "draw_holes",
    "ensure_model",
    "hole_name",
    "load_photo",
    "rectify",
    "template_holes",
]
