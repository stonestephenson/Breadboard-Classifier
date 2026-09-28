"""Blink and watch: finding an LED by switching its pin and seeing what lights.

The behaviours that matter:

- A lit LED is found at its own column, and named by its colour.
- Nothing else counts as a lit LED. When the evidence is not a single clean LED,
  the answer is "unclear", never a guess. "Dark" claims a broken branch, so it
  is only given when the board barely changed.
- The board is left safe: every pin low however the run ends, and a pin tied to
  ground is released at once.
- Real recorded runs give the answer we saw with our own eyes.

The recorded-run tests need local recordings (data/cache, not in git) and skip
without them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from breadboard.blink import analyse, find_glow, glow_colour, run_sequence
from breadboard.rectify import CANONICAL_SIZE, Rectification, template_holes

W, H = CANONICAL_SIZE

# A fit whose photo is the canonical view itself, so photo pixels are board
# pixels. One hole pitch is about 16 px here.
FLAT = Rectification(
    photo_to_canonical=np.eye(3),
    canonical=np.zeros((H, W, 3), np.uint8),
    confidence=1.0,
    oriented=True,
    column_margin=0.05,
)


def _board(shape=(H, W), level=(150, 140, 125)) -> np.ndarray:
    return np.full((*shape, 3), level, np.uint8)


def _light(image: np.ndarray, xy, spill_rgb, core_radius=10, spill_radius=45):
    """Add an LED as the camera sees it: a coloured spill, then a white core."""
    out = image.astype(np.int16)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    d = np.hypot(xx - xy[0], yy - xy[1])
    falloff = np.clip(1 - d / spill_radius, 0, 1)[..., None]
    out += (falloff * np.array(spill_rgb)).astype(np.int16)
    if core_radius:
        out[d <= core_radius] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


HOLES = template_holes()


class TestFindGlow:
    def test_a_lit_led_is_found_at_its_column_and_named_by_colour(self):
        off = _board()
        on = _light(off, HOLES["c30"], spill_rgb=(0, 90, 60))
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit"
        assert glow.column == 30
        assert glow.colour == "green"

    def test_no_change_is_dark(self):
        off = _board()
        assert find_glow(2, off.copy(), off, FLAT).status == "dark"

    def test_a_tiny_glint_is_not_an_led(self):
        off = _board()
        on = off.copy()
        cv2.circle(on, (400, 150), 2, (255, 255, 255), -1)
        assert find_glow(2, on, off, FLAT).status == "dark"

    def test_a_light_off_the_board_is_ignored(self):
        # A screen or lamp in view can flare up at any moment. Only what happens
        # inside the board's outline counts.
        off = _board((H + 200, W + 200))
        on = _light(off, (W + 100, H + 100), spill_rgb=(90, 90, 90))
        assert find_glow(2, on, off, FLAT).status == "dark"

    def test_the_camera_re_adjusting_exposure_is_unclear_not_an_led(self):
        # Auto-exposure, or a cloud, brightens the whole picture without an LED.
        off = _board()
        on = np.clip(off.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        assert find_glow(2, on, off, FLAT).status == "unclear"

    def test_a_near_white_board_saturating_is_unclear_not_an_led(self):
        # A white board just under saturation, then a shadow lifts: every pixel
        # turns newly white at once. That is not an LED.
        off = _board(level=(196, 196, 196))
        on = _board(level=(252, 252, 252))
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "much of the board" in (glow.note or "")

    def test_a_dim_led_is_unclear_not_dark(self):
        # A glow with no saturated core (dim LED, big resistor, short exposure)
        # must not be reported as a broken branch.
        off = _board()
        on = _light(
            off, HOLES["c30"], spill_rgb=(0, 90, 60), core_radius=0, spill_radius=300
        )
        assert find_glow(2, on, off, FLAT).status == "unclear"

    def test_light_in_two_places_is_unclear(self):
        # Two LEDs on one pin, or a reflection as bright as the LED: reporting
        # one of them would silently drop the other.
        off = _board()
        on = _light(off, HOLES["c30"], spill_rgb=(0, 90, 60))
        on = _light(on, HOLES["c15"], spill_rgb=(90, 20, 10))
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "two places" in (glow.note or "")


class TestGlowColour:
    # The mean spill increase measured around each LED in the two real runs: a
    # dim room filmed from low, then a lit room filmed from above. These are the
    # measurements the thresholds were set from, so this guards against
    # regressions; it does not prove the thresholds generalise.
    @pytest.mark.parametrize(
        ("increase", "name"),
        [
            ((3, 181, 127), "green"),
            ((2, 144, 196), "blue"),
            ((117, 168, 209), "white"),
            ((168, 58, 31), "red"),
            ((5, 183, 154), "green"),
            ((2, 141, 216), "blue"),
            ((47, 136, 216), "white"),
            ((143, 19, 7), "red"),
        ],
    )
    def test_measured_spills_get_their_led_colour(self, increase, name):
        assert glow_colour(np.array(increase)) == name

    def test_no_spill_has_no_colour(self):
        assert glow_colour(np.zeros(3)) == "unknown"


class FakeBoard:
    """Records what run_sequence does to the pins. `grounded` pins read low."""

    def __init__(self, grounded=()):
        self.high: set[int] = set()
        self.grounded = set(grounded)
        self.log: list[tuple[int, bool]] = []

    def digital_write(self, pin: int, high: bool) -> None:
        self.log.append((pin, high))
        if high:
            self.high.add(pin)
        else:
            self.high.discard(pin)

    def pin_level(self, pin: int) -> int:
        return 0 if pin in self.grounded else int(pin in self.high)


def _frame():
    return np.zeros((4, 4, 3), np.uint8)


def _no_wait(_seconds):
    pass


class TestRunSequence:
    def test_photographs_each_pin_on_and_off_and_leaves_all_low(self):
        board = FakeBoard()
        frames, shorted = run_sequence(board, [2, 3], _frame, sleep=_no_wait)
        assert set(frames) == {"base", "pin2_on", "pin2_off", "pin3_on", "pin3_off"}
        assert shorted == []
        assert not board.high

    def test_a_pin_tied_to_ground_is_released_at_once_and_not_photographed(self):
        board = FakeBoard(grounded={3})
        frames, shorted = run_sequence(board, [2, 3, 4], _frame, sleep=_no_wait)
        assert shorted == [3]
        assert "pin3_on" not in frames
        # The very next command after driving it high is driving it low again.
        i = board.log.index((3, True))
        assert board.log[i + 1] == (3, False)
        assert not board.high

    def test_every_pin_is_left_low_when_the_camera_fails_mid_run(self):
        board = FakeBoard()
        shots = iter([_frame(), _frame()])

        def capture():
            photo = next(shots, None)
            if photo is None:
                raise OSError("the camera stopped sending frames")
            return photo

        with pytest.raises(OSError, match="camera stopped"):
            run_sequence(board, [2, 3, 4], capture, sleep=_no_wait)
        assert board.log[-3:] == [(2, False), (3, False), (4, False)]
        assert not board.high

    def test_refuses_pins_that_are_not_outputs(self):
        board = FakeBoard()
        with pytest.raises(ValueError, match="not output pins"):
            run_sequence(board, [2, 11], _frame, sleep=_no_wait)
        assert board.log == []


RUNS = Path(__file__).resolve().parent.parent / "data" / "cache" / "blink"

# Two runs of the BasicBoard as rewired by 2026-09-27. First, a dim room filmed
# from low across the desk. Second, a lit room with the phone higher, the board
# filling the frame. What lit was checked by eye in the frames: pin 2 green, 3
# blue, 4 white, 5 red, nothing on the other pins. The glow centres were checked
# to sit on each LED's body. The columns differ by one between runs because the
# LED bodies stand above the board and the viewpoint changed.
LEDS = {
    "basicboard-2026-09-27": {
        2: ("green", 30),
        3: ("blue", 26),
        4: ("white", 20),
        5: ("red", 15),
    },
    "basicboard-2026-09-27-overhead": {
        2: ("green", 29),
        3: ("blue", 25),
        4: ("white", 19),
        5: ("red", 15),
    },
}


@pytest.fixture(scope="module", params=sorted(LEDS))
def recorded(request):
    run = RUNS / request.param
    if not run.exists():
        pytest.skip("recorded blink run not present")
    meta = json.loads((run / "meta.json").read_text())
    frames = {}
    for path in sorted(run.glob("*.png")):
        image = cv2.imread(str(path))
        assert image is not None
        frames[path.stem] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return analyse(frames, meta["pins"]), meta["pins"], LEDS[request.param]


class TestRecordedRuns:
    def test_the_board_is_found(self, recorded):
        session, _, _ = recorded
        assert session.rect is not None

    def test_each_led_is_found_by_its_pin_with_its_colour(self, recorded):
        session, _, leds = recorded
        for pin, (colour, column) in leds.items():
            glow = session.glows[pin]
            assert glow.status == "lit", f"pin {pin}: {glow.note}"
            assert glow.colour == colour, f"pin {pin}"
            # The LED's body spans about two columns; the claim is "to a column".
            assert glow.column is not None
            assert abs(glow.column - column) <= 1, f"pin {pin} at {glow.hole}"

    def test_pins_without_an_led_are_dark(self, recorded):
        session, pins, leds = recorded
        assert all(session.glows[p].status == "dark" for p in pins if p not in leds)
        assert not session.moved
