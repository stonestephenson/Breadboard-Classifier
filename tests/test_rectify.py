"""Rectification: from a photo to board coordinates, and back onto the photo.

Two things must hold for anything downstream to mean anything:

- Every template point carries the right hole name. A mislabelled rectifier is
  worse than a failed one, because it looks fine while every reading is wrong.
- Named holes land on the real holes in real photos, the right way round.

The photo tests need the local sample photos (data/cache, not in git) and skip
without them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from breadboard.board import ALL_HOLES, PITCH_MM, hole_position
from breadboard.rectify import (
    MIN_COLUMN_MARGIN,
    column_margin,
    load_photo,
    rectify,
    template_holes,
)

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
BASICBOARD = CACHE / "kit" / "basicboard.jpg"
SAMPLES = sorted((CACHE / "sample").glob("*.jpg"))

needs_photos = pytest.mark.skipif(
    not BASICBOARD.exists() or not SAMPLES, reason="sample photos not present"
)


class TestTemplate:
    def test_names_every_hole_exactly_once(self):
        holes = template_holes()
        assert len(holes) == len(ALL_HOLES)
        assert set(holes) == set(ALL_HOLES)

    def test_column_1_is_on_the_right_and_row_a_on_top(self):
        h = template_holes()
        assert h["a1"][0] > h["a63"][0]
        assert h["a1"][1] < h["j1"][1]

    def test_rail_p1_runs_along_row_a_and_p2_along_row_j(self):
        h = template_holes()
        assert h["p1+:1"][1] < h["p1-:1"][1] < h["a1"][1]
        assert h["j1"][1] < h["p2+:1"][1] < h["p2-:1"][1]

    def test_agrees_with_board_geometry(self):
        # One affine map must carry board.py's millimetres onto every template
        # point. If it cannot, the rectifier and the checker disagree about where
        # holes are, and "move it one column left" would point at the wrong hole.
        holes = template_holes()
        mm = np.array([hole_position(h) for h in ALL_HOLES])
        px = np.array([holes[h] for h in ALL_HOLES])
        design = np.hstack([mm, np.ones((len(mm), 1))])
        fit, *_ = np.linalg.lstsq(design, px, rcond=None)
        px_per_pitch = np.abs(fit[0, 0]) * PITCH_MM
        residual = np.linalg.norm(design @ fit - px, axis=1) / px_per_pitch
        worst = ALL_HOLES[int(residual.argmax())]
        assert residual.max() < 0.1, f"{worst} is {residual.max():.2f} pitch out"


class TestColumnMargin:
    # The hole grid repeats, so a fit one column off still lands most holes on
    # template holes and the normaliser grades it perfect. What gives it away is
    # the rails' gaps and the grid's ends, which is what column_margin measures.

    def _holes(self) -> np.ndarray:
        return np.array(list(template_holes().values()), dtype=np.float32)

    def test_the_true_registration_wins_clearly(self):
        assert column_margin(self._holes()) >= MIN_COLUMN_MARGIN

    @pytest.mark.parametrize("columns", [-2, -1, 1, 2])
    def test_a_registration_off_by_a_column_or_two_loses(self, columns):
        holes = template_holes()
        pitch = (holes["a1"][0] - holes["a63"][0]) / 62
        shifted = self._holes() + np.array([columns * pitch, 0], dtype=np.float32)
        assert column_margin(shifted) < 0


@pytest.fixture(scope="module")
def basicboard():
    result = rectify(load_photo(BASICBOARD))
    assert result is not None
    return result


@needs_photos
class TestPhotos:
    @pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
    def test_sample_photo_rectifies_confidently(self, path):
        result = rectify(load_photo(path))
        assert result is not None
        assert result.ok

    def test_basicboard_photo_rectifies_confidently(self, basicboard):
        assert basicboard.ok
        assert basicboard.confidence == 1.0

    def test_unreadable_rail_stripes_mean_no_verdict(self):
        # Without colour the rail stripes cannot say which end is column 1. The
        # fit itself still grades as perfect, so only the orientation check stops
        # a board read backwards from being reported as usable.
        photo = load_photo(BASICBOARD)
        grey = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        result = rectify(grey)
        assert result is not None
        assert not result.oriented
        assert not result.ok

    @pytest.mark.parametrize(
        ("hole", "expected"),
        [
            # data/cache/kit/basicboard.jpg is portrait, column 1 at the top and
            # row a on the left. These positions are the rectifier's output,
            # confirmed by eye on 2026-09-27 to sit on those exact holes against
            # the printed row letters, column numbers and rail stripes. One pitch
            # is 16-20 px here, so the tolerance is about half a hole: any more
            # and it would be a different hole.
            ("a1", (657, 333)),
            ("j1", (855, 331)),
            ("a63", (583, 1545)),
            ("j63", (830, 1560)),
        ],
    )
    def test_named_holes_land_on_the_real_holes(self, basicboard, hole, expected):
        x, y = basicboard.to_photo(hole)
        assert np.hypot(x - expected[0], y - expected[1]) < 8

    def test_to_photo_and_hole_at_are_inverses(self, basicboard):
        # A consistency check between the two directions of the mapping, not a
        # check on the fit itself.
        for hole in ALL_HOLES:
            assert basicboard.hole_at(basicboard.to_photo(hole)) == hole


def test_onnxruntime_telemetry_is_off_before_onnxruntime_loads():
    # Its upload thread can abort the process at exit. The switch only works if
    # it is set before onnxruntime is imported, so check in a fresh interpreter.
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    probe = (
        "import sys, breadboard, os\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "set_first = os.environ.get('ORT_DISABLE_TELEMETRY')\n"
        "import breadboard.rectify\n"
        "assert 'onnxruntime' in sys.modules\n"
        "print(set_first)\n"
    )
    out = subprocess.run(  # noqa: S603 -- this interpreter and a fixed string
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"
