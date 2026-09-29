"""Blink and watch: finding an LED by switching its pin and seeing what lights.

The behaviours that matter:

- A lit LED is found at its own column, and named by its colour.
- Nothing else counts as a lit LED. When the evidence is not a single clean LED,
  the answer is "unclear", never a guess. "Dark" claims a broken branch, so it
  is only given when the board barely changed.
- A board that drifts, held in a hand, is lined up before photos are compared,
  and an LED's glow does not drag that alignment. Photos that cannot be lined
  up are not judged.
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

import breadboard.blink
from breadboard.blink import (
    align_off,
    align_on,
    analyse,
    find_glow,
    glow_colour,
    run_sequence,
)
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


def _light(image: np.ndarray, xy, spill_rgb, core_radius=10, spill_radius=130):
    """Add an LED as the camera sees it: a coloured spill, then a white core.

    Real spills reach well past the 1.5-3 pitches where colour is read.
    """
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
        on = _light(off, HOLES["c30"], spill_rgb=(0, 200, 120))
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
            off, HOLES["c30"], spill_rgb=(0, 200, 120), core_radius=0, spill_radius=300
        )
        assert find_glow(2, on, off, FLAT).status == "unclear"

    def test_light_in_two_places_is_unclear(self):
        # Two LEDs on one pin, or a reflection as bright as the LED: reporting
        # one of them would silently drop the other.
        off = _board()
        on = _light(off, HOLES["c30"], spill_rgb=(0, 200, 120))
        on = _light(on, HOLES["c15"], spill_rgb=(90, 20, 10))
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "two places" in (glow.note or "")

    def test_a_white_spot_with_no_glow_is_a_glint_not_an_led(self):
        # A tilting board catches the light on a lead or a lens: saturated, and
        # big enough to pass for a core, but it lights nothing around it.
        off = _board()
        on = off.copy()
        x, y = HOLES["c30"]
        cv2.circle(on, (round(x), round(y)), 10, (255, 255, 255), -1)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "no glow" in (glow.note or "")


class TestGlowColour:
    # The mean spill increase 1.5-3 pitches from each LED in five real runs: a
    # dim room, two lit rooms, and two with the board held up to the camera.
    # These are the measurements the thresholds were set from, so this guards
    # against regressions; it does not prove the thresholds generalise. The
    # third white (7, 88, 187) is the closest call: 3.7% red, against 2% to be
    # called white and 1% to be called blue.
    @pytest.mark.parametrize(
        ("increase", "name"),
        [
            ((1, 183, 100), "green"),
            ((0, 120, 196), "blue"),
            ((109, 164, 209), "white"),
            ((167, 14, 6), "red"),
            ((0, 175, 68), "green"),
            ((0, 57, 210), "blue"),
            ((19, 104, 201), "white"),
            ((142, 0, 0), "red"),
            ((0, 178, 70), "green"),
            ((0, 55, 206), "blue"),
            ((7, 88, 187), "white"),
            ((136, 0, 0), "red"),
            ((1, 158, 14), "green"),
            ((0, 16, 168), "blue"),
            ((75, 108, 145), "white"),
            ((141, 6, 0), "red"),
            ((1, 158, 17), "green"),
            ((0, 28, 171), "blue"),
            ((83, 107, 152), "white"),
            ((146, 9, 0), "red"),
        ],
    )
    def test_measured_spills_get_their_led_colour(self, increase, name):
        assert glow_colour(np.array(increase)) == name

    @pytest.mark.parametrize(
        "increase",
        [
            (10, 50, 200),  # some red, but not the green a white LED adds
            (3, 100, 200),  # too much red for blue, too little for white
        ],
    )
    def test_between_blue_and_white_it_does_not_guess(self, increase):
        assert glow_colour(np.array(increase)) == "unknown"

    def test_no_spill_has_no_colour(self):
        assert glow_colour(np.zeros(3)) == "unknown"


# A photo with a margin around the board, so it can drift without leaving the
# frame. The board sits MARGIN pixels in from the photo's corner.
MARGIN = 100
PLACED = Rectification(
    photo_to_canonical=np.array([[1, 0, -MARGIN], [0, 1, -MARGIN], [0, 0, 1]], float),
    canonical=np.zeros((H, W, 3), np.uint8),
    confidence=1.0,
    oriented=True,
    column_margin=0.05,
)


def _held_board() -> np.ndarray:
    """A board photo with the detail alignment works from: every hole, and a few
    wires, so the pattern is not purely a repeating grid. Seeded."""
    rng = np.random.default_rng(0)
    image = _board((H + 2 * MARGIN, W + 2 * MARGIN), level=(90, 80, 70))
    cv2.rectangle(image, (MARGIN, MARGIN), (MARGIN + W, MARGIN + H), (150, 140, 125), -1)
    for x, y in HOLES.values():
        cv2.circle(image, (round(x) + MARGIN, round(y) + MARGIN), 3, (60, 55, 50), -1)
    for _ in range(12):
        a, b = rng.uniform((MARGIN, MARGIN), (MARGIN + W, MARGIN + H), (2, 2))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), colour, 3)
    return cv2.GaussianBlur(image, (0, 0), 1.0)


def _moved(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The same photo with the board slid by (dx, dy) pixels."""
    shift = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(image, shift, (image.shape[1], image.shape[0]))


def _where(warp: np.ndarray, xy) -> np.ndarray:
    return cv2.perspectiveTransform(np.array([[xy]], float), warp)[0, 0]


def _placed(hole: str) -> tuple[int, int]:
    x, y = HOLES[hole]
    return round(x) + MARGIN, round(y) + MARGIN


def _crowd(seed: int = 1):
    """A board held by someone who fills most of the picture: returns a
    function giving the picture with the board slid by board_shift and the
    person by person_shift."""
    rng = np.random.default_rng(seed)
    size = (H + 2 * MARGIN + 700, W + 2 * MARGIN + 800)
    person = rng.integers(0, 255, (size[0] // 4, size[1] // 4, 3)).astype(np.uint8)
    person = cv2.resize(person, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    board = np.zeros_like(person)
    board[: H + 2 * MARGIN, : W + 2 * MARGIN] = _held_board()
    inside = np.zeros(size, np.uint8)
    inside[MARGIN : MARGIN + H, MARGIN : MARGIN + W] = 1

    def scene(board_shift, person_shift):
        out = _moved(person, *person_shift)
        shown = _moved(inside, *board_shift) > 0
        out[shown] = _moved(board, *board_shift)[shown]
        return out

    return scene


class TestAlignment:
    def test_a_drifting_board_is_lined_up(self):
        fitted = _held_board()
        off = _moved(fitted, 17, -9)
        warp = align_off(off, fitted, PLACED)
        assert warp is not None
        for xy in [(200, 200), (1000, 400)]:
            assert np.allclose(_where(warp, xy), np.add(xy, (17, -9)), atol=0.5)

    def test_a_glow_does_not_drag_the_alignment(self):
        # A white LED can flood half the board. Lining up on brightness, the
        # alignment bent the photo thousands of pixels to explain it away.
        fitted = _held_board()
        off = _moved(fitted, 17, -9)
        on = _moved(fitted, 22, -6)
        on = _light(on, _placed("c30"), (120, 160, 200), core_radius=40, spill_radius=400)
        off_warp = align_off(off, fitted, PLACED)
        assert off_warp is not None
        warp = align_on(on, off, off_warp, PLACED)
        assert warp is not None
        for xy in [(200, 200), (1000, 400)]:
            assert np.allclose(_where(warp, xy), np.add(xy, (22, -6)), atol=0.5)

    def test_too_far_to_trust_is_not_lined_up(self):
        fitted = _held_board()
        assert align_off(_moved(fitted, 6 * 16, 0), fitted, PLACED) is None

    def test_a_jump_between_on_and_off_is_not_lined_up(self):
        # 0.6 s apart, a hand moved the board 1.6 pitches at most. Over two
        # pitches is too fast to trust.
        fitted = _held_board()
        off = _moved(fitted, 17, -9)
        on = _moved(fitted, 17 + 36, -9)
        off_warp = align_off(off, fitted, PLACED)
        assert off_warp is not None
        assert align_on(on, off, off_warp, PLACED) is None

    def test_the_person_holding_the_board_does_not_mislead_it(self):
        # Filmed from further away, the person holding the board fills most of
        # the picture, and moves differently. Between a pin's on and off photos,
        # a first guess from the whole picture followed the person; this follows
        # the board.
        scene = _crowd()
        off, on = scene((10, -4), (0, 0)), scene((20, 0), (-30, 12))
        off_warp = np.array([[1, 0, 10], [0, 1, -4], [0, 0, 1]], float)
        warp = align_on(on, off, off_warp, PLACED)
        assert warp is not None
        assert np.allclose(_where(warp, (700, 300)), (720, 300), atol=0.5)

    def test_nothing_but_repeating_holes_is_a_tie_and_not_lined_up(self):
        # Every hole looks like its neighbour, so one hole over matches as well.
        fitted = np.full((H + 2 * MARGIN, W + 2 * MARGIN, 3), 150, np.uint8)
        for x in range(MARGIN, MARGIN + W, 16):
            for y in range(MARGIN, MARGIN + H, 16):
                cv2.circle(fitted, (x, y), 3, (60, 55, 50), -1)
        fitted = cv2.GaussianBlur(fitted, (0, 0), 1.0)
        off, on = _moved(fitted, 3, 2), _moved(fitted, 5, 2)
        off_warp = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]], float)
        assert align_on(on, off, off_warp, PLACED) is None

    def test_too_little_unlit_board_to_line_up_on_is_not_lined_up(self):
        fitted = _held_board()
        off = _moved(fitted, 17, -9)
        on = off.copy()
        on[MARGIN : MARGIN + H, MARGIN : MARGIN + int(0.9 * W)] = 255
        off_warp = align_off(off, fitted, PLACED)
        assert off_warp is not None
        assert align_on(on, off, off_warp, PLACED) is None

    def test_a_different_picture_is_not_lined_up(self):
        fitted = _held_board()
        other = np.ascontiguousarray(fitted[::-1, ::-1])
        assert align_off(other, fitted, PLACED) is None

    def test_a_held_board_is_judged_on_the_board(self, monkeypatch):
        # The board drifts over the run, and more between one pin's photos. The
        # LED is still found at its own column, and empty pins stay dark.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        frames = {
            "base": fitted,
            "pin2_on": _light(
                _moved(fitted, 9, 4), np.add(_placed("c30"), (9, 4)), (0, 200, 120)
            ),
            "pin2_off": _moved(fitted, 12, 2),
            "pin3_on": _moved(fitted, 14, -3),
            "pin3_off": _moved(fitted, 11, -5),
        }
        session = analyse(frames, [2, 3])
        assert not session.moved
        assert session.glows[2].status == "lit"
        assert session.glows[2].column == 30
        assert session.glows[3].status == "dark"

    def test_where_a_photo_saw_nothing_does_not_count(self, monkeypatch):
        # The board fills the photo. Slid left for the off photo, its left end
        # left the frame, so lined up, that strip is black. Something always
        # bright there must not read as newly lit.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: FLAT)
        fitted = _held_board()[MARGIN : MARGIN + H, MARGIN : MARGIN + W].copy()
        fitted[150:190, 4:18] = 255
        frames = {
            "base": fitted,
            "pin2_on": _moved(fitted, -2, 0),
            "pin2_off": _moved(fitted, -10, 0),
        }
        session = analyse(frames, [2])
        assert session.glows[2].status == "dark"

    def test_a_photo_lined_up_one_hole_over_is_not_judged(self, monkeypatch):
        # The holes repeat, so an alignment can lock one hole over and pass
        # every other check. Each lit photo is lined up twice, via the unlit
        # photos on either side; here one of the two slips, and they disagree.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        real = breadboard.blink.align_on
        calls = []

        def one_slips(on, off, off_warp, rect):
            warp = real(on, off, off_warp, rect)
            calls.append(warp)
            hole = np.array([[1, 0, 16], [0, 1, 0], [0, 0, 1]], float)
            return hole @ warp if len(calls) == 1 and warp is not None else warp

        monkeypatch.setattr(breadboard.blink, "align_on", one_slips)
        fitted = _held_board()
        frames = {
            "base": fitted,
            "pin2_on": _light(
                _moved(fitted, 9, 4), np.add(_placed("c30"), (9, 4)), (0, 200, 120)
            ),
            "pin2_off": _moved(fitted, 12, 2),
        }
        session = analyse(frames, [2])
        assert len(calls) == 2
        assert session.moved == {2}

    def test_a_slip_is_not_carried_to_the_next_pin(self, monkeypatch):
        # Pin 2's off photo is lined up one hole over. Pin 2 is set aside; pin
        # 3 must not build on that alignment, or both of its own would inherit
        # the slip, agree, and put its LED a column off.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        frames = {
            "base": fitted,
            "pin2_on": _moved(fitted, 2, 1),
            "pin2_off": _moved(fitted, 4, 1),
            "pin3_on": _light(
                _moved(fitted, 6, 2), np.add(_placed("c20"), (6, 2)), (0, 200, 120)
            ),
            "pin3_off": _moved(fitted, 8, 2),
        }
        real = breadboard.blink.align_off

        def slips_on_pin2(off, fitted_, rect, guess=None):
            warp = real(off, fitted_, rect, guess)
            if off is frames["pin2_off"] and warp is not None:
                return np.array([[1, 0, 16], [0, 1, 0], [0, 0, 1]], float) @ warp
            return warp

        monkeypatch.setattr(breadboard.blink, "align_off", slips_on_pin2)
        session = analyse(frames, [2, 3])
        assert session.moved == {2}
        assert session.glows[3].status == "lit"
        assert session.glows[3].column == 20

    def test_a_board_found_mid_run_is_followed_both_ways(self, monkeypatch):
        # In a dim room the board may only be found on a later photo. Photos
        # before it are lined up working backwards from it.
        board = _held_board()
        frames = {
            "base": _moved(board, -6, 2),
            "pin2_on": _light(
                _moved(board, -4, 1), np.add(_placed("c30"), (-4, 1)), (0, 200, 120)
            ),
            "pin2_off": _moved(board, -2, 1),
            "pin3_on": _moved(board, -1, 0),
            "pin3_off": board,
            "pin4_on": _light(
                _moved(board, 3, -1), np.add(_placed("c20"), (3, -1)), (0, 200, 120)
            ),
            "pin4_off": _moved(board, 5, -2),
        }
        monkeypatch.setattr(
            breadboard.blink,
            "rectify",
            lambda image: PLACED if image is frames["pin3_off"] else None,
        )
        session = analyse(frames, [2, 3, 4])
        assert not session.moved
        assert (session.glows[2].column, session.glows[4].column) == (30, 20)
        assert session.glows[3].status == "dark"

    def test_a_board_held_by_a_moving_person_is_followed(self, monkeypatch):
        # The person moves differently from the board in every photo. A start
        # from the whole picture follows the person; following the board from
        # photo to photo does not.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        scene = _crowd()
        steps = [
            ((3, 1), (-20, 8)),
            ((6, 2), (15, -10)),
            ((9, 2), (-25, 5)),
            ((12, 3), (20, 12)),
            ((14, 3), (-10, -15)),
            ((16, 4), (25, 0)),
        ]
        shots = [scene(b, p) for b, p in steps]
        board_at = [b for b, _ in steps]
        shots[2] = _light(shots[2], np.add(_placed("c25"), board_at[2]), (0, 200, 120))
        frames = {"base": scene((0, 0), (0, 0))}
        for n, pin in enumerate([2, 3, 4]):
            frames[f"pin{pin}_on"], frames[f"pin{pin}_off"] = shots[2 * n : 2 * n + 2]
        session = analyse(frames, [2, 3, 4])
        assert not session.moved
        assert session.glows[3].status == "lit"
        assert session.glows[3].column == 25
        assert session.glows[2].status == session.glows[4].status == "dark"

    def test_photos_that_cannot_be_lined_up_are_not_judged(self, monkeypatch):
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        frames = {
            "base": fitted,
            "pin2_on": _moved(fitted, 9 * 16, 0),
            "pin2_off": _moved(fitted, 9 * 16, 0),
        }
        session = analyse(frames, [2])
        assert session.moved == {2}
        assert 2 not in session.glows


class FakeBoard:
    """Records the order of run_sequence's pin commands. `grounded` pins read low.

    This checks the sequence only. The electrical guarantees of the real
    drive_only are tested against an emulated chip in test_probe_board.py.
    """

    def __init__(self, grounded=(), fail_release=False):
        self.driven: int | None = None
        self.grounded = set(grounded)
        self.fail_release = fail_release
        self.log: list[int | None] = []

    def drive_only(self, pin: int | None) -> int | None:
        self.log.append(pin)
        if pin is None and self.fail_release:
            raise OSError("serial fault")
        if pin in self.grounded:
            self.driven = None
            return 0
        self.driven = pin
        return None if pin is None else 1


def _frame():
    return np.zeros((4, 4, 3), np.uint8)


def _no_wait(_seconds):
    pass


class TestRunSequence:
    def test_photographs_each_pin_on_and_off_and_ends_with_all_released(self):
        board = FakeBoard()
        frames, shorted = run_sequence(board, [2, 3], _frame, sleep=_no_wait)
        assert set(frames) == {"base", "pin2_on", "pin2_off", "pin3_on", "pin3_off"}
        assert shorted == []
        assert board.driven is None

    def test_only_one_pin_is_ever_driven_and_always_from_all_released(self):
        # Driving a pin only ever follows "all released", so two pins are never
        # driven together, and nothing else is held low to act as ground.
        board = FakeBoard()
        run_sequence(board, [2, 3, 4], _frame, sleep=_no_wait)
        assert board.log[0] is None
        for before, after in zip(board.log, board.log[1:], strict=False):
            if after is not None:
                assert before is None

    def test_a_pin_tied_to_ground_is_released_at_once_and_not_photographed(self):
        board = FakeBoard(grounded={3})
        frames, shorted = run_sequence(board, [2, 3, 4], _frame, sleep=_no_wait)
        assert shorted == [3]
        assert "pin3_on" not in frames
        # The very next command after driving it is releasing it.
        i = board.log.index(3)
        assert board.log[i + 1] is None
        assert board.driven is None

    def test_every_pin_is_released_when_the_camera_fails_mid_run(self):
        board = FakeBoard()
        shots = iter([_frame(), _frame()])

        def capture():
            photo = next(shots, None)
            if photo is None:
                raise OSError("the camera stopped sending frames")
            return photo

        with pytest.raises(OSError, match="camera stopped"):
            run_sequence(board, [2, 3, 4], capture, sleep=_no_wait)
        assert board.log[-1] is None
        assert board.driven is None

    def test_an_interrupt_still_stops_the_run_when_releasing_fails_too(self):
        # A Ctrl-C mid-run must stop the program, even if the serial link is in
        # a state where releasing the pins fails as well.
        board = FakeBoard(fail_release=True)
        board.fail_release = False

        def capture():
            board.fail_release = True
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_sequence(board, [2, 3], capture, sleep=_no_wait)

    def test_refuses_pins_that_are_not_outputs(self):
        board = FakeBoard()
        with pytest.raises(ValueError, match="not output pins"):
            run_sequence(board, [2, 11], _frame, sleep=_no_wait)
        assert board.log == []


RUNS = Path(__file__).resolve().parent.parent / "data" / "cache" / "blink"

# Runs of the BasicBoard as rewired by 2026-09-27. First, a dim room filmed
# from low across the desk. Second, a lit room with the phone higher, the board
# filling the frame. Then four with the phone fixed and the board held by hand
# (2026-09-28); it drifted up to 3.4 pitches over a run. The third was held in
# front of the body, small in the picture; the fourth over a dark desk, where it
# jumped over a pitch between one pin's photos. Both failed live before the
# board search in align_on. What lit was checked by eye in the frames: pin 2
# green, 3 blue, 4 white, 5 red, nothing on the other pins. The glow centres were checked to sit on each LED's body.
# The columns differ by one between runs because the LED bodies stand above the
# board and the viewpoint changed.
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
    "basicboard-2026-09-28-handheld-1": {
        2: ("green", 29),
        3: ("blue", 25),
        4: ("white", 19),
        5: ("red", 15),
    },
    "basicboard-2026-09-28-handheld-2": {
        2: ("green", 29),
        3: ("blue", 25),
        4: ("white", 19),
        5: ("red", 15),
    },
    "basicboard-2026-09-28-handheld-3": {
        2: ("green", 30),
        3: ("blue", 25),
        4: ("white", 19),
        5: ("red", 15),
    },
    "basicboard-2026-09-28-handheld-4": {
        2: ("green", 29),
        3: ("blue", 25),
        4: ("white", 19),
        5: ("red", 15),
    },
}
# Pins that may be set aside rather than judged. In handheld-4, the white LED
# lit most of the board while it jumped 1.6 pitches, and its photo could only be
# lined up one way, so there was no second alignment to check it against. Pin 5
# then has to be checked against the photo before pin 4's, across the jump.
MAY_BE_SET_ASIDE = {"basicboard-2026-09-28-handheld-4": {4, 5}}


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
    set_aside = MAY_BE_SET_ASIDE.get(request.param, set())
    return analyse(frames, meta["pins"]), meta["pins"], LEDS[request.param], set_aside


class TestRecordedRuns:
    def test_the_board_is_found(self, recorded):
        session, _, _, _ = recorded
        assert session.rect is not None

    def test_each_led_is_found_by_its_pin_with_its_colour(self, recorded):
        session, _, leds, set_aside = recorded
        for pin, (colour, column) in leds.items():
            if pin in set_aside and pin in session.moved:
                continue
            glow = session.glows[pin]
            assert glow.status == "lit", f"pin {pin}: {glow.note}"
            assert glow.colour == colour, f"pin {pin}"
            # The column exactly as seen by eye, so a photo lined up one hole
            # over would show here.
            assert glow.column == column, f"pin {pin} at {glow.hole}"

    def test_pins_without_an_led_are_dark(self, recorded):
        session, pins, leds, set_aside = recorded
        assert all(session.glows[p].status == "dark" for p in pins if p not in leds)
        assert session.moved <= set_aside
