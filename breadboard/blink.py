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

The board may be held in a hand, so it drifts between photos. Before comparing
them, each photo is warped onto the one the board was fitted on, lining them up
on the board itself (align). The LED's own glow is left out when lining up a
photo with the LED on, or the alignment would try to explain the glow as
movement. A pin whose photos cannot be lined up is not judged.

When unsure, it says so. Each pin comes out "lit", "dark" or "unclear". Unclear
means something changed that does not look like a single LED:
- the board brightened with no white core (a dim LED, or the room light changed);
- a white spot appeared with no glow around it (a glint as the board tilts);
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
# mean dark. Measured on photos blurred by a pitch, so the edges left by a
# slightly imperfect alignment do not count. Pins with an LED changed it by
# 30-142; empty pins by under 3, hand-held too.
SPILL_LEVELS = 10.0
# The spill's colour is read this many pitches out from the core. Nearer in, the
# overloaded sensor bleeds into every channel: a red LED looks orange there, and
# a blue one picks up red and green. Seven recorded runs, still and hand-held,
# read cleanly at this distance. Pixels still saturated there are left out.
SPILL_BAND = (1.5, 3.0)
# A real LED raises the board around it by 136-210 levels in its strongest
# channel, at that distance. A white spot with no such glow is a glint; cores
# faked by a deliberately misaligned photo had at most 14. (This is not relative
# to the board as a whole, because in a dim room an LED lights the whole board
# about as much as its surroundings. So a glint that coincides with the whole
# board brightening by this much would pass.)
MIN_SPILL = 60.0
# Photos are lined up at this scale, for speed; the result is sub-pixel anyway.
ALIGN_SCALE = 0.5
ALIGN_STOP = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
# Lining up is trusted only when the photos then match this well (correlation of
# the board's brightness, 0-1). Hand-held photos matched 0.85-0.99, still ones
# 0.99. One dragged off by an LED's glow matched 0.4-0.7.
MIN_ALIGN_MATCH = 0.8
# How far the board may drift, in pitches: over a whole check, and between the
# photos with a pin on and off (0.6 s apart). Hand-held, it drifted up to 2.3
# pitches over a check, and 0.4 between on and off.
MAX_DRIFT_PITCHES = 5.0
MAX_STEP_PITCHES = 0.6
# The holes repeat every pitch, so a fine alignment can lock on one hole over.
# The rough first estimate, from the whole picture, cannot; between on and off
# the two agreed to within 0.22 pitch. Further apart than this, it slipped.
MAX_DISAGREE_PITCHES = 0.5
# The share of the board that must be left to align on, once pixels a glow has
# saturated are left out.
MIN_ALIGN_SHARE = 0.15
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

    Measured SPILL_BAND pitches from the core. Single-colour LEDs give almost
    nothing in the channel opposite their own. White LEDs are blue LEDs with a
    phosphor coat, so their spill still peaks in blue. In seven recorded runs a
    blue LED added 0-0.1% as much red as blue, and a white one 3.5-54% (the camera
    darkening its exposure for the white LED can hide most of its red). White
    also adds more green: 47-79% as much as blue, where a blue LED added 10-29%
    in a lit room but 61% in a dim one. So white needs some red and plenty of
    green; blue needs no red; anything else is "unknown" rather than a guess. On
    camera, green LEDs spill cyan-ish light that peaks in green. Only red, green,
    blue and white have been seen on camera; the yellow and orange cut-offs are
    guesses.
    """
    r, g, b = (float(v) for v in increase)
    top = max(r, g, b)
    if top < 1.0:
        return "unknown"
    if top == b:
        if r / b <= 0.01:
            return "blue"
        return "white" if r / b >= 0.02 and g / b >= 0.42 else "unknown"
    if top == g:
        return "white" if min(r, b) / g >= 0.5 else "green"
    if min(g, b) / r >= 0.5:
        return "white"
    if g / r > 0.75:
        return "yellow"
    return "orange" if g / r > 0.45 else "red"


def find_glow(
    pin: int,
    on: np.ndarray,
    off: np.ndarray,
    rect: Rectification,
    seen: np.ndarray | None = None,
) -> Glow:
    """Compare RGB frames taken with the pin on and off, lined up on the board.

    seen, if given, is 1 where both photos actually showed the board. A photo
    warped into line is black where it saw nothing, and black would read as
    "dark before" next to anything bright.
    """
    inside = board_mask(on.shape, rect)
    if seen is not None:
        inside &= seen
    pitch = photo_pitch(rect)
    core = (
        (on.min(axis=2) >= SATURATED) & (off.min(axis=2) < WAS_BELOW) & (inside > 0)
    ).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(core)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(0, int)
    cores = [i + 1 for i in np.argsort(-areas) if areas[i] >= MIN_CORE_PITCH2 * pitch**2]

    if not cores:
        on_v, off_v = (
            cv2.GaussianBlur(im.max(axis=2).astype(np.float32), (0, 0), pitch)
            for im in (on, off)
        )
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

    away = cv2.distanceTransform((labels != main).astype(np.uint8), cv2.DIST_L2, 5)
    near, far_edge = (d * pitch for d in SPILL_BAND)
    band = (away > near) & (away <= far_edge) & (inside > 0)
    band &= on.min(axis=2) < SATURATED
    increase = np.clip(on.astype(np.int16) - off.astype(np.int16), 0, None)[band]
    spill = increase.mean(axis=0) if len(increase) else np.zeros(3)
    if spill.max() < MIN_SPILL:
        return Glow(
            pin, "unclear", "a white spot appeared with no glow around it", core_px=size
        )
    return Glow(
        pin,
        "lit",
        hole=rect.hole_at((x, y)),
        photo_xy=(x, y),
        colour=glow_colour(spill),
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
    pin that was judged. moved holds pins not judged because their photos could
    not be lined up: the board or camera moved too much, or too fast. shorted
    holds pins that did not go high when switched on, and so were switched
    straight back off.
    """

    rect: Rectification | None
    glows: dict[int, Glow] = field(default_factory=dict)
    moved: frozenset[int] = frozenset()
    shorted: frozenset[int] = frozenset()


def _small_grey(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.resize(
        grey, None, fx=ALIGN_SCALE, fy=ALIGN_SCALE, interpolation=cv2.INTER_AREA
    )


def _detail(grey: np.ndarray, pitch: float) -> np.ndarray:
    """The board's fine detail (holes, wire edges), without smooth light.

    A glow, the camera's exposure and the board tilting under a lamp all change
    the light smoothly and multiplicatively. On a log scale that is a smooth
    offset, which subtracting a blur of about a hole pitch removes.
    """
    log = np.log1p(grey)
    return log - cv2.GaussianBlur(log, (0, 0), pitch * ALIGN_SCALE)


def _shrink(mask: np.ndarray, like: np.ndarray) -> np.ndarray:
    return cv2.resize(
        mask, (like.shape[1], like.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def _shift(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """A rough start: the translation that best lines moving up with fixed."""
    window = cv2.createHanningWindow((fixed.shape[1], fixed.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(fixed, moving, window)
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32)


def _refine(
    fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray, model: int, start: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Enhanced-correlation alignment at ALIGN_SCALE. Returns the warp from
    fixed's pixels to moving's, full size, and how well they then match (0-1).
    None if it diverged."""
    rows = 3 if model == cv2.MOTION_HOMOGRAPHY else 2
    try:
        match, warp = cv2.findTransformECC(
            fixed, moving, start[:rows].copy(), model, ALIGN_STOP, mask, 5
        )
    except cv2.error:
        return None
    full = np.vstack([warp, [0, 0, 1]]) if rows == 2 else warp
    scale = np.diag([ALIGN_SCALE, ALIGN_SCALE, 1.0])
    return np.linalg.inv(scale) @ full.astype(np.float64) @ scale, float(match)


def _corners(rect: Rectification, warp: np.ndarray | None = None) -> np.ndarray:
    """The board's four corners in the fitted photo, or where warp takes them."""
    w, h = CANONICAL_SIZE
    square = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float64)
    photo = cv2.perspectiveTransform(square, np.linalg.inv(rect.photo_to_canonical))
    return photo[0] if warp is None else cv2.perspectiveTransform(photo, warp)[0]


def _farthest(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(a - b).T).max())


def align_off(
    off: np.ndarray, fitted: np.ndarray, rect: Rectification
) -> np.ndarray | None:
    """The warp taking each pixel of the fitted photo to the same point of the
    board in off, a photo with every pin off. None if they cannot be lined up.

    Both photos are unlit, so the whole board's brightness can be used to line
    them up. (Its fine detail is less reliable here: a shaking hand blurs each
    averaged photo differently.)
    """
    pitch = photo_pitch(rect)
    fixed, moving = _small_grey(fitted), _small_grey(off)
    mask = _shrink(board_mask(fitted.shape, rect), fixed)
    found = _refine(fixed, moving, mask, cv2.MOTION_HOMOGRAPHY, _shift(fixed, moving))
    if found is None or found[1] < MIN_ALIGN_MATCH:
        return None
    warp = found[0]
    if _farthest(_corners(rect, warp), _corners(rect)) > MAX_DRIFT_PITCHES * pitch:
        return None
    return warp


def align_on(
    on: np.ndarray, off: np.ndarray, off_warp: np.ndarray, rect: Rectification
) -> np.ndarray | None:
    """The warp from the fitted photo to on, a photo with one pin on, found via
    off, the same pin's photo with it off. off_warp is align_off's for off.

    Lined up on fine detail, which the LED's smooth glow barely touches. Its
    saturated centre has no detail at all, and its edge would look like movement,
    so pixels saturated in either photo are left out. Only a shift is allowed:
    in the 0.6 s between the photos a hand slides the board rather than turning
    it, and a freer warp could bend the photo to fit a glow that floods the
    board. How well the two then match says little, since a glow over most of
    the board dims the detail it does not hide. So the check is that the board
    moved no more than a hand can in that time.
    """
    pitch = photo_pitch(rect)
    grey_fixed, grey_moving = _small_grey(off), _small_grey(on)
    fixed, moving = _detail(grey_fixed, pitch), _detail(grey_moving, pitch)
    start = _shift(grey_fixed, grey_moving)
    size = (fixed.shape[1], fixed.shape[0])
    board = cv2.warpPerspective(
        board_mask(off.shape, rect), off_warp, (off.shape[1], off.shape[0])
    )
    board = _shrink(board, fixed)
    lit = (
        cv2.warpPerspective(grey_moving, start, size, flags=cv2.WARP_INVERSE_MAP)
        >= SATURATED
    )
    white = (grey_fixed >= SATURATED) | lit
    white = cv2.dilate(
        white.astype(np.uint8), np.ones((int(2 * pitch * ALIGN_SCALE) | 1,) * 2, np.uint8)
    )
    keep = board & (1 - white)
    if keep.sum() < MIN_ALIGN_SHARE * max(int(board.sum()), 1):
        return None
    found = _refine(fixed, moving, keep, cv2.MOTION_TRANSLATION, start)
    if found is None:
        return None
    slide, rough = found[0][:2, 2], start[:2, 2] / ALIGN_SCALE
    if np.hypot(*(slide - rough)) > MAX_DISAGREE_PITCHES * pitch:
        return None
    on_warp = found[0] @ off_warp
    if (
        _farthest(_corners(rect, on_warp), _corners(rect, off_warp))
        > MAX_STEP_PITCHES * pitch
    ):
        return None
    return on_warp


def seen_by(warp: np.ndarray, image: np.ndarray, like: np.ndarray) -> np.ndarray:
    """1 where warp_onto(image, warp, like) shows part of image, not the black
    fill beyond its edge. A 2 px border is dropped, where the fill blends in."""
    ones = np.ones(image.shape[:2], np.uint8)
    size = (like.shape[1], like.shape[0])
    mask = cv2.warpPerspective(
        ones, warp, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def warp_onto(image: np.ndarray, warp: np.ndarray, like: np.ndarray) -> np.ndarray:
    """image as seen from the photo that like is, given the warp from align_*."""
    size = (like.shape[1], like.shape[0])
    return cv2.warpPerspective(
        image, warp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )


def analyse(
    frames: Mapping[str, np.ndarray],
    pins: Sequence[int],
    shorted: Collection[int] = (),
) -> Session:
    """Judge a blink run.

    frames holds RGB frames: "base" with every pin off, then "pin<N>_on" and
    "pin<N>_off" for each pin that was photographed, in the order of pins.

    The board is fitted once, on the first frame that fits (a lit LED can spoil
    the fit, so unlit frames go first). Every other photo is lined up with that
    one on the board itself, so the board or camera may drift during the run. A
    pin whose photos cannot be lined up is set aside rather than misread.
    """
    order = ["base"] + [f"pin{p}_off" for p in pins] + [f"pin{p}_on" for p in pins]
    rect, fitted = None, None
    for name in order:
        if name in frames:
            candidate = rectify(frames[name])
            if candidate is not None and candidate.ok:
                rect, fitted = candidate, frames[name]
                break
    if rect is None or fitted is None:
        return Session(rect=None, shorted=frozenset(shorted))

    glows: dict[int, Glow] = {}
    moved: set[int] = set()
    for pin in pins:
        on, off = frames.get(f"pin{pin}_on"), frames.get(f"pin{pin}_off")
        if on is None or off is None:
            continue
        off_warp = align_off(off, fitted, rect)
        on_warp = None if off_warp is None else align_on(on, off, off_warp, rect)
        if off_warp is None or on_warp is None:
            moved.add(pin)
            continue
        seen = seen_by(on_warp, on, fitted) & seen_by(off_warp, off, fitted)
        glows[pin] = find_glow(
            pin,
            warp_onto(on, on_warp, fitted),
            warp_onto(off, off_warp, fitted),
            rect,
            seen,
        )
    return Session(rect, glows, frozenset(moved), frozenset(shorted))
