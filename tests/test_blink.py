"""Blink and watch: finding an LED by switching its pin and seeing what lights.

The behaviours that matter:

- A lit LED is found at its own column, and named by its colour.
- Nothing else counts as a lit LED. When the evidence is not a single clean LED,
  the answer is "unclear", never a guess. "Dark" claims a broken branch, so it
  is only given when no part of the board changed more than the rest.
- No fixed amount of light is asked for. A glow is judged against what this
  run's own quiet pins did, and white from blue by the colour of the lit board,
  so a bright room or another camera does not change the answer. A board too
  bright to judge is said to be so.
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
    MAX_RETRIES,
    Attempt,
    Glow,
    Session,
    align_off,
    align_on,
    analyse,
    analyse_attempts,
    blink_and_watch,
    board_change,
    find_glow,
    glow_colour,
    retried,
    run_sequence,
    too_bright,
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
PITCH = 16  # pixels between holes in FLAT

# What a laptop camera showed on a bright board (2026-10-05, the board at about
# 190 of 255 before any LED lit): the unlit board, then the lit board's colour
# 1.5-3, 3-6 and 6-10 pitches from the LED. That camera does not add an LED's
# light to the picture so much as replace the board's colour with the LED's.
BRIGHT_BOARD = (193, 180, 165)
SEEN_BLUE = ((88, 115, 244), (46, 37, 245), (60, 38, 219))
SEEN_WHITE = ((130, 179, 228), (113, 155, 206), (120, 147, 179))
SEEN_GREEN = ((14, 216, 136), (1, 185, 89), (37, 161, 90))


def _seen(image: np.ndarray, xy, rings, core_radius=10) -> np.ndarray:
    """A lit LED as a camera showed it: the board in three rings around it, out
    to 3, 6 and 10 pitches, takes the colours given, then the white core."""
    out = image.copy()
    centre = (round(xy[0]), round(xy[1]))
    for reach, colour in reversed(list(zip((3, 6, 10), rings, strict=True))):
        cv2.circle(out, centre, core_radius + reach * PITCH, colour, -1)
    cv2.circle(out, centre, core_radius, (255, 255, 255), -1)
    return out


class TestFindGlow:
    def test_a_lit_led_is_found_at_its_column_and_named_by_colour(self):
        off = _board()
        on = _light(off, HOLES["c30"], spill_rgb=(0, 200, 120))
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit"
        assert glow.column == 30
        assert glow.colour == "green"

    def test_a_blue_led_is_blue_though_it_bleeds_pale_close_in(self):
        # Close to a blue LED the overloaded sensor bleeds into every channel,
        # and the board there looks as pale as beside a white one. Further out
        # it is deep blue, which a white LED never makes it.
        off = _board(level=BRIGHT_BOARD)
        on = _seen(off, HOLES["c30"], SEEN_BLUE)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit"
        assert glow.colour == "blue"

    def test_a_white_led_is_white_though_no_channel_rose_much(self):
        # On a bright board that camera let the white LED's red fall and its
        # green stand still. The lit board is pale blue in every ring.
        off = _board(level=BRIGHT_BOARD)
        on = _seen(off, HOLES["c30"], SEEN_WHITE)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit"
        assert glow.colour == "white"

    def test_a_red_streak_beside_a_blue_led_does_not_make_it_white(self):
        # A red wire's edge, where the two photos are a little out of line. Red
        # once decided blue from white, and a streak like this had to be caught.
        off = _board(level=BRIGHT_BOARD)
        blue = _seen(off, HOLES["c30"], SEEN_BLUE)
        x, y = (round(v) for v in HOLES["c30"])
        streaked = blue.copy()
        streaked[y - 46 : y - 43, x - 50 : x + 50, 0] = 255
        assert find_glow(2, streaked, off, FLAT).colour == "blue"

    def test_on_a_bright_board_an_led_shows_by_the_colour_it_takes_away(self):
        # A green LED on that board raised green by only 36 of 255, less than a
        # fixed bar once asked for. Red around it fell from 193 to 14.
        off = _board(level=BRIGHT_BOARD)
        on = _seen(off, HOLES["c30"], SEEN_GREEN)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit"
        assert glow.colour == "green"
        assert glow.column == 30

    def test_a_glow_must_stand_well_above_what_quiet_pins_did(self):
        # A faint glow: the board beside the core rose about 40 levels more
        # than the board far away. In a still run that is plainly an LED. In a
        # run where the same patch rose 18 more than the far board on a pin
        # that lit nothing (a lamp swinging, a screen playing beside it), it is
        # not enough to tell.
        off = _board()
        on = _light(off, HOLES["c30"], spill_rgb=(0, 60, 25), spill_radius=400)
        still = board_change(off, off, PITCH)
        patchy = _light(off, HOLES["c30"], (25, 25, 25), core_radius=0, spill_radius=400)
        assert find_glow(2, on, off, FLAT, quiet=[still]).status == "lit"
        glow = find_glow(
            2, on, off, FLAT, quiet=[still, board_change(patchy, off, PITCH)]
        )
        assert glow.status == "unclear"
        assert "kept changing" in (glow.note or "")

    def test_light_that_changed_all_over_the_board_does_not_hide_a_glow(self):
        # The room got 40 levels darker between the two photos. Beside the LED
        # the board still changed far more than it did far away.
        off = _board()
        dimmer = np.clip(off.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        on = _light(dimmer, HOLES["c30"], spill_rgb=(0, 200, 120))
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit"
        assert glow.column == 30

    def test_a_piece_of_an_leds_flare_cut_off_by_a_wire_is_not_a_second_light(self):
        # A white LED's core can be twelve pitches long. A wire across its edge
        # splits a piece off, far from the core's centre but touching its side.
        off = _board(level=BRIGHT_BOARD)
        on = _seen(off, HOLES["c30"], SEEN_WHITE)
        x, y = (round(v) for v in HOLES["c30"])
        cv2.ellipse(on, (x, y), (30, 100), 0, 0, 360, (255, 255, 255), -1)
        cv2.line(on, (x - 40, y + 86), (x + 40, y + 86), (120, 150, 200), 3)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "lit", glow.note

    def test_a_blue_led_on_a_camera_that_does_not_darken_is_not_called_white(self):
        # Both cameras recorded darken and recolour their picture when a blue
        # LED lights, so the board turns deep blue around it. One that only
        # added the LED's light on top of a bright room's would leave the board
        # pale, and paler the further out. That is not a white LED's glow, with
        # or without some of the blue leaking into green.
        off = _board()
        for spill in ((0, 0, 200), (0, 60, 200)):
            glow = find_glow(2, _light(off, HOLES["c30"], spill_rgb=spill), off, FLAT)
            assert glow.status == "lit"
            assert glow.colour == "unknown", spill

    def test_two_leds_a_few_columns_apart_are_light_in_two_places(self):
        # Two LEDs lighting from one pin is a fault in itself (two LEDs in
        # series). Here they are three columns apart with cores a pitch across:
        # close, but with two pitches of unlit board between them.
        off = _board()
        on = _light(off, HOLES["c27"], spill_rgb=(0, 200, 120), core_radius=8)
        on = _light(on, HOLES["c30"], spill_rgb=(90, 20, 10), core_radius=8)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "two places" in (glow.note or "")

    def test_two_leds_with_wide_cores_nearly_touching_are_still_two(self):
        # Five columns apart, cores four pitches across: one pitch of board
        # between them. Not a piece of one LED's flare, which is a small thing
        # beside a large one.
        off = _board()
        on = _light(off, HOLES["c27"], (0, 200, 120), core_radius=2 * PITCH)
        on = _light(on, HOLES["c32"], (90, 20, 10), core_radius=2 * PITCH)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "two places" in (glow.note or "")

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

    def test_light_that_changes_all_over_the_board_alike_is_not_an_led(self):
        # Auto-exposure, a cloud, or a screen beside the board brightens the
        # whole picture without an LED. Nothing lit, and nothing uneven hints
        # at an LED too dim to show a core, so the pin is dark.
        off = _board()
        on = np.clip(off.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        assert find_glow(2, on, off, FLAT).status == "dark"

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

    def test_a_glint_while_the_rooms_light_changed_is_still_a_glint(self):
        # The whole board got 60 levels brighter between the photos, twice what
        # a glow must reach. But the board beside the spot changed no more than
        # the board far from it, so nothing glowed there.
        off = _board()
        on = np.clip(off.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        x, y = HOLES["c30"]
        cv2.circle(on, (round(x), round(y)), 10, (255, 255, 255), -1)
        glow = find_glow(2, on, off, FLAT)
        assert glow.status == "unclear"
        assert "no glow" in (glow.note or "")

    def test_a_glint_with_a_little_change_around_it_is_still_a_glint(self):
        # Two photos half a pitch out of line change the board around a faked
        # core by up to 15 levels more than the board far away. A real LED's
        # weakest glow was 62.
        off = _board()
        on = _light(off, HOLES["c30"], spill_rgb=(18, 18, 18), spill_radius=150)
        assert find_glow(2, on, off, FLAT).status == "unclear"

    def test_a_faint_glow_in_one_place_with_no_core_is_unclear_not_dark(self):
        # An LED too dim to saturate, lighting a small patch: the board as a
        # whole barely changed, but one place did. The same when the room also
        # got darker between the photos.
        off = _board()
        faint = _light(off, HOLES["c30"], (0, 90, 50), core_radius=0, spill_radius=100)
        darker = np.clip(faint.astype(np.int16) - 20, 0, 255).astype(np.uint8)
        assert find_glow(2, faint, off, FLAT).status == "unclear"
        assert find_glow(2, darker, off, FLAT).status == "unclear"


class TestTooBright:
    def test_a_washed_out_board_is_too_bright(self):
        assert too_bright(_board(level=(252, 250, 247)), FLAT)

    def test_a_board_too_near_white_for_a_core_to_show_is_too_bright(self):
        # Not clipped, but nothing on it could turn "newly" white.
        assert too_bright(_board(level=(236, 232, 228)), FLAT)

    def test_a_bright_board_that_is_not_washed_out_is_fine(self):
        # The brightest board recorded, on which every LED was still found.
        assert not too_bright(_board(level=BRIGHT_BOARD), FLAT)

    def test_a_few_white_things_on_the_board_do_not_make_it_too_bright(self):
        board = _board()
        board[100:140, 300:700] = 255  # a white label, a bright reflection
        assert not too_bright(board, FLAT)

    def test_nothing_is_judged_on_a_washed_out_board(self, monkeypatch):
        # Not even "dark": an LED could light there and show nothing new.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: FLAT)
        washed = _board(level=(252, 250, 247))
        frames = {"base": washed, "pin2_on": washed.copy(), "pin2_off": washed.copy()}
        session = analyse(frames, [2])
        assert session.too_bright
        assert not session.glows
        assert not session.moved


class TestGlowColour:
    # What ten real runs measured for each LED: the mean rise 1.5-3 pitches out,
    # and for the blue-peaked ones the lit board's green as a share of its blue
    # in each ring (1.5-3, 3-6, 6-10 pitches). A dim room, a lit room, the board
    # held up to a phone, then a laptop camera: held, and still on a bright
    # board. These are the measurements the cut-offs were set from, so this
    # guards against regressions; it does not prove they generalise. The closest
    # calls are a blue LED whose purest ring read 0.27 against 0.36 to be blue,
    # and a white one at 0.64 against 0.48 to be white.
    @pytest.mark.parametrize(
        ("increase", "pale", "name"),
        [
            ((1, 172, 94), (), "green"),
            ((0, 120, 197), (0.75, 0.44, 0.23), "blue"),
            ((125, 169, 198), (0.94, 0.88, 0.79), "white"),
            ((149, 14, 12), (), "red"),
            ((0, 175, 68), (), "green"),
            ((0, 57, 210), (0.51, 0.21, 0.07), "blue"),
            ((19, 104, 202), (0.74, 0.71, 0.77), "white"),
            ((142, 0, 0), (), "red"),
            ((1, 158, 14), (), "green"),
            ((0, 16, 168), (0.4, 0.13, 0.13), "blue"),
            ((74, 108, 145), (0.93, 0.91, 0.91), "white"),
            ((141, 6, 0), (), "red"),
            ((0, 137, 13), (), "green"),
            ((0, 23, 157), (0.53, 0.19, 0.15), "blue"),
            ((55, 98, 142), (0.9, 0.9, 0.92), "white"),
            ((124, 3, 0), (), "red"),
            ((1, 90, 9), (), "green"),
            ((1, 8, 118), (0.57, 0.27, 0.27), "blue"),
            ((73, 4, 0), (), "red"),
            ((10, 92, 172), (0.85, 0.76, 0.71), "white"),
            ((0, 21, 178), (0.5, 0.2, 0.09), "blue"),
            ((0, 141, 24), (), "green"),
            ((116, 0, 0), (), "red"),
            ((5, 36, 103), (0.75, 0.72, 0.71), "white"),
            ((0, 36, 118), (0.67, 0.2, 0.06), "blue"),
            ((0, 76, 10), (), "green"),
            ((136, 16, 3), (), "red"),
            ((2, 21, 80), (0.8, 0.67, 0.65), "white"),
            ((0, 22, 76), (0.7, 0.28, 0.09), "blue"),
            ((0, 48, 7), (), "green"),
            ((96, 3, 0), (), "red"),
            ((3, 19, 71), (0.79, 0.66, 0.64), "white"),
            ((1, 12, 77), (0.45, 0.13, 0.07), "blue"),
            ((0, 44, 6), (), "green"),
            ((88, 1, 0), (), "red"),
            ((1, 14, 65), (0.79, 0.76, 0.82), "white"),
            ((0, 7, 78), (0.47, 0.15, 0.17), "blue"),
            ((0, 33, 4), (), "green"),
            ((78, 0, 0), (), "red"),
        ],
    )
    def test_measured_glows_get_their_led_colour(self, increase, pale, name):
        assert glow_colour(np.array(increase), pale) == name

    def test_between_blue_and_white_it_does_not_guess(self):
        # Too pale for blue, too blue for white.
        assert glow_colour(np.array((3, 100, 200)), (0.55, 0.42, 0.41)) == "unknown"

    def test_how_much_each_channel_rose_does_not_decide_blue_from_white(self):
        # A laptop camera gave a white LED and a blue one nearly the same rise.
        # The colour of the lit board is what tells them.
        rise = np.array((1, 14, 70))
        assert glow_colour(rise, (0.79, 0.76, 0.82)) == "white"
        assert glow_colour(rise, (0.47, 0.15, 0.17)) == "blue"

    def test_pale_far_out_but_bluer_close_in_is_not_called_white(self):
        # What a blue LED too weak to dominate the room's light would show. A
        # white LED's rings agree with each other.
        assert glow_colour(np.array((1, 14, 70)), (0.5, 0.7, 0.95)) == "unknown"

    def test_white_needs_every_ring_read(self):
        # With a ring missing, nothing shows that the pale rings are not a blue
        # LED's bleed, or blue clipping close in. Deep blue needs only one.
        assert glow_colour(np.array((1, 14, 70)), (0.8,)) == "unknown"
        assert glow_colour(np.array((1, 14, 70)), (0.8, 0.78)) == "unknown"
        assert glow_colour(np.array((1, 14, 70)), (0.8, 0.78, 0.8)) == "white"
        assert glow_colour(np.array((1, 14, 70)), (0.2,)) == "blue"

    def test_rings_that_pale_with_distance_are_not_called_white(self):
        # Less of the light is the LED's the further out, so the room's own
        # colour shows through: a blue LED that does not dominate. The other
        # way round, a little bluer far out, is what white LEDs showed.
        assert glow_colour(np.array((1, 14, 70)), (0.6, 0.7, 0.78)) == "unknown"
        assert glow_colour(np.array((1, 14, 70)), (0.78, 0.7, 0.6)) == "white"

    def test_rings_that_differ_too_much_are_not_called_white(self):
        # Pale close in and much bluer far out, but never deep blue: a blue
        # LED's bleed over a glow too weak to crush the room's light.
        assert glow_colour(np.array((1, 14, 70)), (0.9, 0.62, 0.6)) == "unknown"

    def test_a_blue_peak_with_no_ring_read_has_no_colour(self):
        assert glow_colour(np.array((1, 14, 70))) == "unknown"

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

    @staticmethod
    def _under(fitted: np.ndarray, tint) -> np.ndarray:
        """fitted with a tint of light across it, growing from left to right."""
        ramp = np.linspace(0, 1, fitted.shape[1], dtype=np.float32)[None, :, None]
        lit = fitted + ramp * np.array(tint, np.float32)
        return np.clip(lit, 0, 255).astype(np.uint8)

    def test_changing_light_neither_hides_an_led_nor_fakes_one(self, monkeypatch):
        # A screen beside the board plays a film: every photo is lit a little
        # differently, more at one end of the board than the other. The LED is
        # still found. The pins that lit nothing changed too unevenly to be
        # called dark, but only as much as each other, so they are calm. And a
        # glint on one of them is not taken for an LED.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        under = self._under
        glint = under(fitted, (60, 10, 0))
        cv2.circle(glint, _placed("c50"), 10, (255, 255, 255), -1)
        frames = {
            "base": fitted,
            "pin2_on": _light(under(fitted, (50, 0, 0)), _placed("c30"), (0, 200, 120)),
            "pin2_off": under(fitted, (0, 55, 0)),
            "pin3_on": under(fitted, (0, 0, 60)),
            "pin3_off": under(fitted, (45, 0, 0)),
            "pin4_on": under(fitted, (0, 60, 10)),
            "pin4_off": under(fitted, (0, 0, 50)),
            "pin5_on": under(fitted, (55, 10, 0)),
            "pin5_off": under(fitted, (0, 50, 0)),
            "pin6_on": glint,
            "pin6_off": under(fitted, (0, 0, 50)),
        }
        session = analyse(frames, [2, 3, 4, 5, 6])
        assert not session.moved
        assert session.glows[2].status == "lit"
        assert session.glows[2].column == 30
        assert [session.glows[p].status for p in (3, 4, 5)] == ["unclear"] * 3
        assert session.calm == {3, 4, 5}
        assert session.glows[6].status == "unclear"

    def test_a_pin_that_changed_unlike_the_quiet_ones_is_not_calm(self, monkeypatch):
        # The light drifts a little on three pins. On the fourth the board
        # brightened unevenly with no core, several times more than on those:
        # an LED too dim to saturate, perhaps.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        under = self._under
        dim = _light(
            fitted, _placed("c30"), (0, 200, 120), core_radius=0, spill_radius=300
        )
        frames = {"base": fitted}
        for pin, tint in ((2, (16, 0, 0)), (3, (0, 16, 0)), (4, (0, 0, 16))):
            frames[f"pin{pin}_on"] = under(fitted, tint)
            frames[f"pin{pin}_off"] = fitted
        frames["pin5_on"], frames["pin5_off"] = dim, fitted
        session = analyse(frames, [2, 3, 4, 5])
        assert [session.glows[p].status for p in (2, 3, 4)] == ["dark"] * 3
        assert session.glows[5].status == "unclear"
        assert not session.calm

    @staticmethod
    def _patch(image: np.ndarray, hole: str, levels: int) -> np.ndarray:
        """image with one patch of the board lit by levels more, as by a lamp
        or a screen that reaches only there."""
        return _light(
            image, _placed(hole), (levels,) * 3, core_radius=0, spill_radius=120
        )

    def test_a_glint_is_judged_against_the_unlit_photos_either_side(self, monkeypatch):
        # One pin, checked on its own. Light on one patch of the board comes
        # and goes: brighter with the pin on, where a glint also shows, and
        # still a little brighter after it. Beside the glint the board rose
        # about 40 more than the far board, enough to pass for a glow. But the
        # two unlit photos differ by about 20 there with nothing lit, so 40
        # proves nothing.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        on = self._patch(fitted, "c40", 100)
        cv2.circle(on, _placed("c40"), 10, (255, 255, 255), -1)
        frames = {
            "base": fitted,
            "pin2_on": on,
            "pin2_off": self._patch(fitted, "c40", 30),
        }
        glow = analyse(frames, [2]).glows[2]
        assert glow.status == "unclear"
        assert "kept changing" in (glow.note or "")

    def test_a_glint_is_judged_against_the_pins_that_lit_nothing(self, monkeypatch):
        # The same patch brightens by about 20 whenever any pin is on (a
        # screen showing the camera's own picture does this), and every unlit
        # photo is alike. On pin 5 a glint shows there too, and the board
        # beside it rose about 45 above the far board.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        frames = {"base": fitted}
        for pin in (2, 3, 4):
            frames[f"pin{pin}_on"] = self._patch(fitted, "c40", 30)
            frames[f"pin{pin}_off"] = fitted
        glinting = self._patch(fitted, "c40", 75)
        cv2.circle(glinting, _placed("c40"), 10, (255, 255, 255), -1)
        frames["pin5_on"], frames["pin5_off"] = glinting, fitted
        glow = analyse(frames, [2, 3, 4, 5]).glows[5]
        assert glow.status == "unclear"
        assert "kept changing" in (glow.note or "")

    def test_two_pins_are_too_few_to_say_what_is_typical(self, monkeypatch):
        # Two pins, both changed unevenly with no core. Each is like the other,
        # which is no evidence that it is only the room's light.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        frames = {"base": fitted}
        for pin, hole in ((2, "c15"), (3, "c45")):
            frames[f"pin{pin}_on"] = _light(
                fitted, _placed(hole), (0, 120, 70), core_radius=0, spill_radius=200
            )
            frames[f"pin{pin}_off"] = fitted
        session = analyse(frames, [2, 3])
        assert [session.glows[p].status for p in (2, 3)] == ["unclear"] * 2
        assert not session.calm

    def test_dim_leds_are_never_called_dark_however_many_there_are(self, monkeypatch):
        # Three pins, each with an LED too dim to show a core. They look like
        # each other, which says nothing about whether they lit.
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        fitted = _held_board()
        frames = {"base": fitted}
        for pin, hole in ((2, "c15"), (3, "c30"), (4, "c45")):
            frames[f"pin{pin}_on"] = _light(
                fitted, _placed(hole), (0, 120, 70), core_radius=0, spill_radius=200
            )
            frames[f"pin{pin}_off"] = fitted
        session = analyse(frames, [2, 3, 4])
        assert [session.glows[p].status for p in (2, 3, 4)] == ["unclear"] * 3

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


def _camera(board: FakeBoard, moving: set[int], lit=(2, "c30")):
    """A camera over a held board: the photos of a pin in moving show the board
    jumped too far to line up; pin lit[0]'s LED sits at hole lit[1]."""
    still = _held_board()

    def capture() -> np.ndarray:
        pin = board.driven
        if pin is None:
            return still
        if pin in moving:
            return _moved(still, 9 * 16, 0)
        if pin == lit[0]:
            return _light(still.copy(), _placed(lit[1]), (0, 200, 120))
        return still

    return capture


class TestRetries:
    """Pins set aside because the board moved are checked again at once."""

    def test_a_pin_set_aside_is_checked_again_and_judged(self, monkeypatch):
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        board = FakeBoard()
        moving = {2}
        capture = _camera(board, moving)
        said = []

        def say(message: str) -> None:
            said.append(message)
            moving.clear()  # the person holds still once asked

        attempts, session = blink_and_watch(
            board, [2, 3], capture, sleep=_no_wait, say=say
        )
        assert [a.pins for a in attempts] == [(2, 3), (2,)]
        assert not session.moved
        assert session.glows[2].status == "lit"
        assert session.glows[2].column == 30
        assert session.glows[3].status == "dark"
        assert "Pin 2 again: hold still..." in said
        assert board.driven is None

    def test_retries_stop_and_the_pin_stays_set_aside(self, monkeypatch):
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        board = FakeBoard()
        attempts, session = blink_and_watch(
            board, [2, 3], _camera(board, {2}), sleep=_no_wait
        )
        assert len(attempts) == 1 + MAX_RETRIES
        assert session.moved == {2}
        assert board.log.count(2) == 1 + MAX_RETRIES

    def test_no_retry_when_the_board_was_not_found(self, monkeypatch):
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: None)
        board = FakeBoard()
        attempts, session = blink_and_watch(
            board, [2, 3], _camera(board, {2}), sleep=_no_wait
        )
        assert len(attempts) == 1
        assert session.rect is None

    def test_saved_attempts_judge_the_same_again(self, monkeypatch):
        monkeypatch.setattr(breadboard.blink, "rectify", lambda _image: PLACED)
        board = FakeBoard()
        moving = {2}
        attempts, session = blink_and_watch(
            board,
            [2, 3],
            _camera(board, moving),
            sleep=_no_wait,
            say=lambda _m: moving.clear(),
        )
        again = analyse_attempts(attempts)
        assert again.glows == session.glows
        assert again.moved == session.moved

    def test_a_run_saves_with_its_retries_and_loads_back(self, monkeypatch, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        import blink as tool

        monkeypatch.setattr(tool, "SESSIONS", tmp_path)
        frame = np.full((8, 8, 3), 7, np.uint8)
        attempts = [
            Attempt({"base": frame, "pin2_on": frame, "pin2_off": frame}, (2, 3), (3,)),
            Attempt({"base": frame, "pin2_on": frame, "pin2_off": frame}, (2,)),
        ]
        run = tool.save_run(attempts, camera=1)
        loaded = tool.load_attempts(run)
        assert [(a.pins, a.shorted, sorted(a.frames)) for a in loaded] == [
            (a.pins, a.shorted, sorted(a.frames)) for a in attempts
        ]
        assert np.array_equal(loaded[1].frames["pin2_on"], frame)

    def test_a_retry_fitted_less_well_than_the_first_run_is_not_trusted(self):
        # "Works" needs a top-grade fit, and only the first run's is graded.
        rough = Rectification(
            photo_to_canonical=PLACED.photo_to_canonical,
            canonical=PLACED.canonical,
            confidence=0.75,
            oriented=True,
            column_margin=0.05,
        )
        first = Session(PLACED, {3: Glow(3, "dark")}, moved=frozenset({2}))
        glow = Glow(2, "lit", hole="c30", photo_xy=(300.0, 120.0), colour="green")
        merged = retried(first, Session(rough, {2: glow}))
        assert merged.moved == {2}
        assert 2 not in merged.glows

    def test_a_washed_out_run_is_not_filled_in_by_a_retry_nor_a_retry_by_one(self):
        lit = Glow(2, "lit", hole="c30", photo_xy=(1.0, 1.0), colour="green")
        washed = Session(FLAT, moved=frozenset({2}), too_bright=True)
        assert retried(washed, Session(FLAT, {2: lit})) == washed
        first = Session(FLAT, moved=frozenset({2}))
        assert retried(first, Session(FLAT, too_bright=True)) == first

    def test_a_retried_pin_keeps_being_calm(self):
        unclear = Glow(2, "unclear", "the board changed, but no LED core showed")
        first = Session(FLAT, moved=frozenset({2}), calm=frozenset({7}))
        again = Session(FLAT, {2: unclear}, calm=frozenset({2, 9}))
        merged = retried(first, again)
        assert merged.calm == {2, 7}  # 9 was not this run's to judge
        assert merged.glows[2] == unclear

    def test_a_retried_glow_is_drawn_where_it_is_in_the_first_photo(self):
        # The retry's photo saw the board 50 px further right.
        shifted = Rectification(
            photo_to_canonical=np.array(
                [[1, 0, -MARGIN - 50], [0, 1, -MARGIN], [0, 0, 1]], float
            ),
            canonical=PLACED.canonical,
            confidence=1.0,
            oriented=True,
            column_margin=0.05,
        )
        first = Session(PLACED, {3: Glow(3, "dark")}, moved=frozenset({2}))
        glow = Glow(2, "lit", hole="c30", photo_xy=(300.0, 120.0), colour="green")
        merged = retried(first, Session(shifted, {2: glow}))
        assert merged.glows[2].photo_xy == pytest.approx((250.0, 120.0))
        assert merged.glows[2].hole == "c30"
        assert not merged.moved


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
    # The demo board (rebuilt 2026-09-29: pin 3 white, 4 blue, 5 green, 6 red),
    # held up to a laptop's own camera, in front of a red shirt, with red wires
    # beside the LEDs. That camera darkens much more for a lit LED than the
    # phone, and its white LED was first read as "unknown".
    "basicboard-2026-10-02-laptop": {
        3: ("white", 27),
        4: ("blue", 31),
        5: ("green", 38),
        6: ("red", 44),
    },
    # The same board and camera in brighter light (2026-10-05), held in a hand.
    # Both said "Not sure yet" live: the green LED's glow rose by 44-48 levels
    # where 60 were asked for, and in the second, shaken hard enough to blur its
    # first photo, the blue LED's colour was read as "unknown".
    "basicboard-2026-10-05-laptop-held-1": {
        3: ("white", 27),
        4: ("blue", 32),
        5: ("green", 38),
        6: ("red", 44),
    },
    "basicboard-2026-10-05-laptop-held-2": {
        3: ("white", 27),
        4: ("blue", 32),
        5: ("green", 38),
        6: ("red", 44),
    },
    # Then propped still, close to that camera, the board at about 190 of 255
    # before any LED lit. It failed five checks out of five: green rose by 33.
    "basicboard-2026-10-05-laptop-bright": {
        3: ("white", 27),
        4: ("blue", 31),
        5: ("green", 38),
        6: ("red", 44),
    },
    # The same, still, while the laptop's screen saver played: its screen faces
    # the board and threw red, then green, then blue light over it, a new colour
    # about every second. Between two photos of a pin that lit nothing the
    # board's colour moved by up to 68 levels of 255, unevenly.
    "basicboard-2026-10-05-laptop-cycling-1": {
        3: ("white", 27),
        4: ("blue", 31),
        5: ("green", 38),
        6: ("red", 44),
    },
    "basicboard-2026-10-05-laptop-cycling-2": {
        3: ("white", 27),
        4: ("blue", 31),
        5: ("green", 38),
        6: ("red", 44),
    },
    # And in steady light again: the white LED's light caught a wire's end just
    # beside its core, 6.5 pitches from the core's centre, which read as light
    # in two places.
    "basicboard-2026-10-05-laptop-flare": {
        3: ("white", 27),
        4: ("blue", 31),
        5: ("green", 38),
        6: ("red", 44),
    },
}
# Pins that may be set aside rather than judged. In handheld-4, the white LED
# lit most of the board while it jumped 1.6 pitches, and its photo could only be
# lined up one way, so there was no second alignment to check it against. Pin 5
# then has to be checked against the photo before pin 4's, across the jump.
# In one of the 2026-10-05 runs the board moved during pin 2, the first pin,
# which has no LED; live, it was blinked again and came back dark.
MAY_BE_SET_ASIDE = {
    "basicboard-2026-09-28-handheld-4": {4, 5},
    "basicboard-2026-10-05-laptop-held-2": {2},
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
        assert session.moved <= set_aside
        # Under changing light a pin with no core is not called dark, only calm.
        for pin in (p for p in pins if p not in leds and p not in session.moved):
            assert session.glows[pin].status == "dark" or pin in session.calm, pin
