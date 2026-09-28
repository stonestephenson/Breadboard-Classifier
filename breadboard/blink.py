"""Blink and watch: find each LED by switching its pin on and seeing what lights.

This is measurement standing in for a perception model (ARCHITECTURE.md section
3[4], "stimulus-response capture"). The Arduino drives one pin at a time while a
camera watches the board. Whatever lights up is, by construction, the LED on
that pin, so the camera learns three things a trained model would otherwise have
to infer:

- which pin drives which LED;
- which column that LED sits in;
- which pins light nothing, meaning that LED's branch is broken or absent.

How a glow is found. An LED's centre overloads the camera sensor to pure white,
while its glow spills coloured light over the board around it. The spill moves
with the camera's automatic exposure and can cover half the board, so the
brightness change as a whole says little about where the LED is. The *newly
white core* does: pixels saturated in every channel with the pin on, but not with
it off, inside the board's outline. Its centre marks the LED, and the colour of
the spill around it names the LED's colour.

When unsure, it says so. Each pin comes out "lit", "dark" or "unclear". Unclear
means something changed that does not look like a single LED:
- the board brightened with no white core (a dim LED, or the room light changed);
- light appeared in two places;
- a "core" covered much of the board.
A dark pin claims its branch is broken, so it is only reported when the board
barely changed at all.

Limits. The LED's body stands several millimetres above the board. Where it
appears, projected onto the board, is accurate along the board (the column), but
not across it (the row). So results give the column, and the row is only
approximate. That is harmless in rows a-e, which share one strip, but it cannot
say which side of the centre gap the LED is on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import cv2
import numpy as np

from breadboard.rectify import CANONICAL_SIZE, Rectification, rectify

# A pixel is saturated when every channel is at least this bright...
SATURATED = 245
# ...and "newly" so when, with the pin off, at least one channel was below this.
WAS_BELOW = 200
# Core sizes are measured in square hole pitches, so they hold at any distance.
# Real LED cores in two recorded runs were 2.1-215; stray glints were under 0.1.
MIN_CORE_PITCH2 = 0.5
# A "core" larger than this share of the board is not an LED (the largest real
# one, a white LED in a dim room, was 15%). A lifting shadow can saturate a whole
# near-white board.
MAX_CORE_SHARE = 0.3
# Another core at least this many pitches from the main one means light in two
# places. The white LED's own bright patches sat 2.8 pitches from its centre.
SEPARATE_PITCHES = 4.0
# Mean brightness change over the board, 0-255, above which "no core" does not
# mean dark. Pins with an LED changed it by 33-142; empty pins by under 3.
SPILL_LEVELS = 10.0
# The spill is measured in a ring this many pixels beyond the core.
RING_PX = 30
# A frame shifted by more than this, in photo pixels, relative to the frame the
# board was fitted on, came from a moved camera and is not judged.
MAX_SHIFT_PX = 2.0
# The pins the shipped LbyM sketch sets up as outputs, and so can light an LED.
# 11 and 12 are inputs with pull-ups; "switching them off" would disable those.
OUTPUT_PINS = tuple(range(2, 11))

Status = Literal["lit", "dark", "unclear"]


@dataclass(frozen=True)
class Glow:
    """What switching one pin on showed.

    status is "lit", "dark" or "unclear" (see the module docstring), and note
    says why when unclear. For a lit pin: hole is the hole nearest the centre of
    the glow, reliable to about a column (the row is approximate); photo_xy is
    that centre in photo pixels; colour is a coarse name for the spill:
    red, orange, yellow, green, blue or white. core_px is the size of the main
    saturated core.
    """

    pin: int
    status: Status
    note: str | None = None
    hole: str | None = None
    photo_xy: tuple[float, float] | None = None
    colour: str | None = None
    core_px: int = 0

    @property
    def lit(self) -> bool:
        return self.status == "lit"

    @property
    def column(self) -> int | None:
        """The column of the glow, which is the reliable part of its position."""
        if self.hole is None or ":" in self.hole:
            return None
        return int(self.hole[1:])


def board_mask(shape: tuple[int, ...], rect: Rectification) -> np.ndarray:
    """1 inside the board's outline in the photo, 0 elsewhere.

    It keeps out changes that are not on the board: a screen in view, a lamp,
    someone walking past.
    """
    w, h = CANONICAL_SIZE
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float64)
    outline = cv2.perspectiveTransform(corners, np.linalg.inv(rect.photo_to_canonical))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(outline[0]).astype(np.int32)], 1)
    return mask


def photo_pitch(rect: Rectification) -> float:
    """The mean spacing between holes in the photo, in pixels."""
    (x1, y1), (x2, y2) = rect.to_photo("a1"), rect.to_photo("a63")
    return float(np.hypot(x2 - x1, y2 - y1)) / 62


def glow_colour(increase: np.ndarray) -> str:
    """A coarse colour name for the mean RGB brightness increase in the spill.

    Single-colour LEDs give almost nothing in the channel opposite their own,
    while white LEDs light all three. White LEDs are blue LEDs with a phosphor
    coat, so their spill still peaks in blue. What sets them apart from a real
    blue LED is red: a blue LED added 1% as much red as blue in both recorded
    runs, and a white one 22-56%. On camera, green LEDs spill cyan-ish light that
    peaks in green. Only red, green, blue and white have been seen on camera; the
    yellow and orange cut-offs are guesses.
    """
    r, g, b = (float(v) for v in increase)
    top = max(r, g, b)
    if top < 1.0:
        return "unknown"
    if top == b:
        return "white" if r / b >= 0.1 else "blue"
    if top == g:
        return "white" if min(r, b) / g >= 0.5 else "green"
    if min(g, b) / r >= 0.5:
        return "white"
    if g / r > 0.75:
        return "yellow"
    return "orange" if g / r > 0.45 else "red"


def find_glow(pin: int, on: np.ndarray, off: np.ndarray, rect: Rectification) -> Glow:
    """Compare RGB frames taken with the pin on and off, from the same viewpoint."""
    inside = board_mask(on.shape, rect)
    pitch = photo_pitch(rect)
    core = (
        (on.min(axis=2) >= SATURATED) & (off.min(axis=2) < WAS_BELOW) & (inside > 0)
    ).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(core)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(0, int)
    cores = [i + 1 for i in np.argsort(-areas) if areas[i] >= MIN_CORE_PITCH2 * pitch**2]

    if not cores:
        on_v, off_v = on.max(axis=2).astype(np.int16), off.max(axis=2).astype(np.int16)
        if np.abs(on_v - off_v)[inside > 0].mean() >= SPILL_LEVELS:
            return Glow(pin, "unclear", "the board brightened, but no LED core showed")
        return Glow(pin, "dark")

    main = cores[0]
    size = int(stats[main, cv2.CC_STAT_AREA])
    x, y = (float(v) for v in centres[main])
    if size > MAX_CORE_SHARE * inside.sum():
        return Glow(pin, "unclear", "much of the board lit up at once", core_px=size)
    far = [
        i
        for i in cores[1:]
        if np.hypot(*(centres[i] - centres[main])) > SEPARATE_PITCHES * pitch
    ]
    if far:
        return Glow(pin, "unclear", "light appeared in two places", core_px=size)

    blob = (labels == main).astype(np.uint8)
    outer = cv2.dilate(blob, np.ones((2 * RING_PX + 1,) * 2, np.uint8))
    near = cv2.dilate(blob, np.ones((9, 9), np.uint8))
    ring = (outer > 0) & (near == 0) & (inside > 0)
    increase = np.clip(on.astype(np.int16) - off.astype(np.int16), 0, None)[ring]
    return Glow(
        pin,
        "lit",
        hole=rect.hole_at((x, y)),
        photo_xy=(x, y),
        colour=glow_colour(increase.mean(axis=0)) if len(increase) else "unknown",
        core_px=size,
    )


class PinDriver(Protocol):
    """What run_sequence needs from a board (tools/probe.py's Board)."""

    def drive_only(self, pin: int | None) -> int | None:
        """Drive this pin high and disconnect every other pin; None: all off.

        Returns what the pin reads the moment it starts driving (0 means held at
        ground, and it has already been released), or None for no pin.
        """
        ...


def run_sequence(
    board: PinDriver,
    pins: Sequence[int],
    capture: Callable[[], np.ndarray],
    settle_s: float = 0.6,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Switch each pin on and then off, photographing the board both ways.

    capture() returns the current RGB frame. Returns the frames, named as
    analyse() expects, and the pins that did not go high when driven.

    Safety. Only one pin ever drives at a time. Every other pin is disconnected
    (an input with no pull-up), not held low. So a wire between two pins cannot
    make them fight, and no other pin can act as a ground. An LED whose short leg
    goes to another pin's strip, instead of to ground, gets no help from that
    pin. It stays dark, or, if that strip has its own LED to ground, both glow
    faintly in series, which normally reads as light in two places. It no longer
    passes here only to fail in the student's program.
    Every pin is disconnected before starting and again on the way out, even if
    something fails part-way.

    Each pin is read back the moment it is driven high. If it reads low, the
    circuit is holding it at ground, so it is released at once and not
    photographed. That catches a pin wired straight to ground. It cannot catch an
    LED with no resistor: the pin still reads high while over-driven, and is held
    for about a second, much as the student's own program would hold it.
    """
    unsafe = [p for p in pins if p not in OUTPUT_PINS]
    if unsafe:
        raise ValueError(f"not output pins on the shipped sketch: {unsafe}")
    frames: dict[str, np.ndarray] = {}
    shorted: list[int] = []
    try:
        board.drive_only(None)
        sleep(settle_s)
        frames["base"] = capture()
        for pin in pins:
            if board.drive_only(pin) == 0:
                board.drive_only(None)
                shorted.append(pin)
                continue
            sleep(settle_s)
            frames[f"pin{pin}_on"] = capture()
            board.drive_only(None)
            sleep(settle_s)
            frames[f"pin{pin}_off"] = capture()
    except BaseException as error:
        # Release everything, but never let a failure here hide why the run
        # stopped: a Ctrl-C or signal must still stop the program.
        try:
            board.drive_only(None)
        except OSError as cleanup:
            if hasattr(error, "add_note"):
                error.add_note(f"releasing the pins also failed: {cleanup}")
        raise
    board.drive_only(None)
    return frames, shorted


@dataclass(frozen=True)
class Session:
    """Everything one blink run showed.

    rect is the board fit used for every frame. It is None when no frame gave a
    usable fit, and then nothing else is judged. glows holds a result for every
    pin that was judged. moved holds pins not judged because the camera moved.
    shorted holds pins that did not go high when switched on, and so were
    switched straight back off.
    """

    rect: Rectification | None
    glows: dict[int, Glow] = field(default_factory=dict)
    moved: frozenset[int] = frozenset()
    shorted: frozenset[int] = frozenset()


def frame_shift(a: np.ndarray, b: np.ndarray) -> float:
    """How far, in pixels, the camera moved between two RGB frames."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, window)
    return float(np.hypot(dx, dy))


def analyse(
    frames: Mapping[str, np.ndarray],
    pins: Sequence[int],
    shorted: Collection[int] = (),
) -> Session:
    """Judge a blink run.

    frames holds RGB frames: "base" with every pin off, then "pin<N>_on" and
    "pin<N>_off" for each pin that was photographed, in the order of pins.

    The camera is assumed still, so one board fit serves every frame, and a
    fixed camera needs only one frame that fits. Movement is checked against the
    frame the fit came from, using only frames with every pin off, since a lit
    LED's glow disturbs the shift estimate. Each on-frame sits between two such
    frames: the base or the previous pin's off-frame, and its own off-frame. If
    either of those moved, the pin is set aside rather than misread.
    """
    order = ["base"] + [f"pin{p}_off" for p in pins] + [f"pin{p}_on" for p in pins]
    rect, fitted_on = None, None
    for name in order:
        if name in frames:
            candidate = rectify(frames[name])
            if candidate is not None and candidate.ok:
                rect, fitted_on = candidate, frames[name]
                break
    if rect is None or fitted_on is None:
        return Session(rect=None, shorted=frozenset(shorted))

    def still(name: str) -> bool:
        return frame_shift(fitted_on, frames[name]) <= MAX_SHIFT_PX

    glows: dict[int, Glow] = {}
    moved: set[int] = set()
    before = "base"
    for pin in pins:
        on, off = f"pin{pin}_on", f"pin{pin}_off"
        if on not in frames or off not in frames:
            continue
        if not (still(before) and still(off)):
            moved.add(pin)
        else:
            glows[pin] = find_glow(pin, frames[on], frames[off], rect)
        before = off
    return Session(rect, glows, frozenset(moved), frozenset(shorted))
