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

Little here asks for a fixed amount of light. A room, a camera and a hand all
change how bright a glow reads, so each check is judged against itself: the
board around a core must change more than the board far from it, and far more
than the same patch did in this run's photos that hold no glow (find_glow).
White is told from blue by the colour of the lit board, not by how much any
channel rose (glow_colour). What stays fixed are floors set between the most
that nothing has ever read and the least that an LED has.

The board may be held in a hand, so it drifts between photos. Before comparing
them, each photo is warped onto the one the board was fitted on, lining them up
on the board itself (align). The LED's own glow is left out when lining up a
photo with the LED on, or the alignment would try to explain the glow as
movement. A pin whose photos cannot be lined up is set aside, and checked again
on its own straight after (blink_and_watch), up to MAX_RETRIES times.

When unsure, it says so. Each pin comes out "lit", "dark" or "unclear". Unclear
means something changed that does not look like a single LED:
- the board changed unevenly with no white core (a dim LED, or the room's
  light changing);
- a white spot appeared with no glow around it (a glint as the board tilts);
- the light kept changing, too much to be sure of a glow;
- light appeared in two places;
- a "core" covered much of the board.
A dark pin claims its branch is broken, so it is only reported when no core
showed and no part of the board changed much more than the rest: light that
shifted all over alike is the room's, not an LED's. And when the unlit board
is washed out, so bright that no core could show on most of it, nothing is
judged at all (too_bright).

Limits. An LED is found by its white core. One that lit without saturating
the camera, and whose glow covered under about a hundredth of the board, would
read as dark; none has been recorded on either camera tried.

The LED's body stands several millimetres above the board. Where it
appears, projected onto the board, is accurate along the board (the column), but
not across it (the row). So results give the column, and the row is only
approximate. That is harmless in rows a-e, which share one strip, but it cannot
say which side of the centre gap the LED is on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
# Another core means light in two places when its centre is more than
# SEPARATE_PITCHES from the main core's centre, or when more than SEPARATE_GAP
# pitches of board lie between the two. Unless it is a piece of the main one: a
# wire or a lead across an LED's flare splits pieces off it, and catches its
# light just beside it. In three runs the five such pieces sat within half a
# pitch of the main core and were at most 1.4% of its size where their centres
# were far from its centre (up to 7.7 pitches, since a white LED's core can be
# twelve pitches long). So a core that near, and under FRAGMENT_SHARE of the
# main one's size, is not counted. A second LED's core is not that small beside
# another LED's, except beside a white one's, where it is then missed.
SEPARATE_PITCHES = 4.0
SEPARATE_GAP = 1.5
FRAGMENT_SHARE = 0.04
# With no core, the pin is dark only if the board did not change unevenly: how
# far each point's change is from the board's median change, 0-255, on photos
# blurred by a pitch (_uneven). SPILL_LEVELS is for its mean over the board and
# SPILL_PEAK for its peak. A change that is the same all over is the room's
# light or the camera's exposure, and says nothing about an LED.
# In steady light, pins that lit nothing had a mean of 0.2-2.8 and a peak of
# 0.6-17. Pins with an LED: 27-91 and 82-239 (they had cores too; no LED too dim
# to show a core has been recorded, and this is what would catch one).
SPILL_LEVELS = 10.0
SPILL_PEAK = 35.0
# A pin with no core that changed unevenly is still "unclear". But when a
# screen saver threw a new colour of light over the board every second, pins
# that lit nothing reached a mean of 12 and a peak of 61. Such a pin is listed
# as calm (Session.calm) when it changed no more than this many times the
# typical no-core pin of its run (their median, mean and peak both, given at
# least MIN_CALM of them): there, the largest was 1.7 times the median.
CALM_OVER_TYPICAL = 2.5
MIN_CALM = 3
# The spill's colour is read this many pitches out from the core. Nearer in, the
# overloaded sensor bleeds into every channel: a red LED looks orange there, and
# a blue one picks up red and green. Seven recorded runs, still and hand-held,
# read cleanly at this distance. Pixels still saturated there are left out.
SPILL_BAND = (1.5, 3.0)
# Is there a glow around the core, or is it a glint? The board around a real LED
# changes, and changes more than the board far from it: with the pin on, minus
# off, in the ring SPILL_BAND, minus the same beyond FAR_PITCHES, in whichever
# channel differs most (_strength). Both photos are blurred by a pitch first and
# the difference keeps its sign, so the edges two photos a little out of line
# leave at every hole cancel instead of adding up. A fall counts as much as a
# rise: a laptop camera crushed red from 196 to 15 around a green LED on a
# bright board, while green itself rose by only 38. And taking away the far
# board leaves out light that changed all over between the two photos.
# Over 39 recorded runs (a phone and a laptop camera, still and hand-held, a
# dim room to a board at 190 of 255, and five under a screen saver's cycling
# colours) a real LED scored 62-186. The same ring scored at most 15 on what is
# known to hold no glow: a pin that lit nothing, or the two unlit photos either
# side of a lit one (under 5 for most; 15 after a white LED on a laptop camera,
# which takes a second to recover, with the board held in a hand).
# So a glow must clear a floor between the two (GLOW_FLOOR), and stand
# GLOW_OVER_QUIET times above the most the same ring scored on this run's own
# glowless photos (find_glow's quiet): its noise from the camera, the light and
# the hand, measured rather than assumed. In steady light the floor is the
# higher bar, or 3 x 15 = 45 in that one run, against 96 for its LED. Under the
# cycling colours the glowless photos scored up to 14, so the bar rose to 42,
# against 81 for the weakest LED there. That second bar earns its place: 855
# faked cores (a pin that lit nothing, its lit photo slid a third, a half or a
# whole pitch out of line, or a white blob painted on) were never called lit.
# On the floor alone 15 were, all under the cycling colours with the photo a
# whole pitch out, at the Arduino's own always-lit LED, scoring 30-62.
GLOW_FLOOR = 30.0
GLOW_OVER_QUIET = 3.0
# "Far from the core" starts this many pitches out.
FAR_PITCHES = 12.0
# The change is measured at this scale: it is blurred by a pitch anyway.
CHANGE_SCALE = 0.25
# Blue or white is told by the colour of the lit board itself: its green as a
# share of its blue, in SPILL_BAND and in these two rings further out (pitches
# from the core). A blue LED turns the board deep blue somewhere: its purest
# ring read 0.06-0.27 in every run. A white LED turns it pale blue everywhere:
# 0.64-0.94 in every ring, and its rings agreed within a factor of 1.23.
# The rings further out matter because near a blue LED the overloaded sensor
# bleeds into every channel: on the laptop camera SPILL_BAND read 0.45-0.71
# for a blue LED, as pale as a white one, and 0.06-0.17 at 6-10 pitches.
# (How much each channel *rose* cannot tell them. That camera crushes red and
# green wherever a lit LED's colour dominates, a white one's too, so a white
# LED's red rose by 1.8-5% of its blue and a blue one's by up to 1.7%.)
# White needs all three rings read, agreeing within FLAT, and none further out
# paler than the first by more than PALER_OUT (white LEDs: 1.04 at most). Where
# a blue LED does not dominate the room's light, or a glow does not reach, the
# board reads paler the further out it is, since less of the light there is
# the LED's. Both cameras darkened and recoloured their picture enough that
# this never showed. A camera that did neither would show it, and then the
# answer is "unknown". It could still call a blue LED white, if blue clipped
# evenly through all three rings; that has not been seen, and is not ruled out.
COLOUR_ZONES = ((3.0, 6.0), (6.0, 10.0))
# A ring with less unsaturated board than this, in square pitches, is not read
# (an LED at the board's end, or a glow that saturates the ring).
MIN_ZONE_PITCH2 = 8.0
BLUE_MAX_GREEN = 0.36
WHITE_MIN_GREEN = 0.48
FLAT = 1.35
PALER_OUT = 1.1
# The unlit board is washed out when more than this share of it is at WAS_BELOW
# or above in every channel: nothing there can turn "newly" white, so an LED
# could light and show no core. The brightest boards recorded had 18%, and
# every LED on them was found. The limit itself is a judgement; no washed-out
# board has been recorded.
MAX_BRIGHT_SHARE = 0.5
# Photos are lined up at this scale, for speed; the result is sub-pixel anyway.
ALIGN_SCALE = 0.5
# Refining stops when the match improves by less than this per step. At 1e-4 it
# often ran all 60 steps, eight times slower, for results identical on the
# recorded runs.
ALIGN_STOP = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-3)
# Lining up is trusted only when the photos then match this well (correlation of
# the board's brightness, 0-1). Hand-held photos matched 0.85-0.99, still ones
# 0.99. One dragged off by an LED's glow matched 0.4-0.7.
MIN_ALIGN_MATCH = 0.8
# How far the board may drift, in pitches: over a whole check, and between the
# photos with a pin on and off (0.6 s apart). Hand-held, it drifted up to 2.3
# pitches over a check (3.4 with the board further away), and up to 1.6
# between on and off.
MAX_DRIFT_PITCHES = 5.0
MAX_STEP_PITCHES = 2.0
# The first estimate tries every slide of the board's detail within reach. The
# holes repeat every pitch, so a slide one hole over can match nearly as well.
# The best slide must beat the next-best separate match by this much
# (correlation, 0-1), or it is a near tie. On nine recorded runs the best led by
# 0.13-0.69. Blur from a moving hand makes neighbouring holes alike, so the lead
# is often small, and this only catches near ties. A slide one hole over that
# passes it would put an LED one column off while every other check passes, so
# analyse also lines each photo with a pin on up twice, via the unlit photos on
# either side, and keeps a chain of trusted alignments (see there).
MIN_SEARCH_LEAD = 0.1
# Two estimates of the same position further apart than this, and one of them
# has slipped toward a neighbouring hole: the fine alignment against its first
# estimate, or analyse's two alignments of one photo. Real pairs of estimates
# agreed within 0.25 pitch; a slip is about one.
MAX_DISAGREE_PITCHES = 0.5
# The share of the board that must be left to align on, once pixels a glow has
# saturated are left out.
MIN_ALIGN_SHARE = 0.15
# The pins the shipped LbyM sketch sets up as outputs, and so can light an LED.
# 11 and 12 are inputs with pull-ups; "switching them off" would disable those.
OUTPUT_PINS = tuple(range(2, 11))

# How many times pins set aside (their photos could not be lined up) are checked
# again. Each time takes about two seconds a pin. A board that keeps moving is
# better told to hold still than retried for ever.
MAX_RETRIES = 2

Status = Literal["lit", "dark", "unclear"]
# A pin's two photos as the fitted photo sees them, on then off, and a mask of
# where both saw the board.
Views = tuple[np.ndarray, np.ndarray, np.ndarray]


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


def glow_colour(increase: np.ndarray, pale: Sequence[float] = ()) -> str:
    """A coarse colour name for an LED, from the light around it.

    increase is the mean RGB brightness increase in the spill, SPILL_BAND
    pitches from the core. Single-colour LEDs give almost nothing in the channel
    opposite their own, so the channel that rose most names red and green ones.

    White LEDs are blue LEDs with a phosphor coat, so their spill peaks in blue
    too. What tells the two apart is the colour of the lit board, not how much
    any channel rose. pale holds its green as a share of its blue, in each ring
    that could be read (find_glow measures them, SPILL_BAND first, then
    COLOUR_ZONES, nearest first). Somewhere a blue LED turns the board deep
    blue (BLUE_MAX_GREEN). A white one leaves it pale in every ring
    (WHITE_MIN_GREEN), about equally (FLAT), and no paler far out than close in
    (PALER_OUT), and all the rings must have been read to say so. Anything
    else is "unknown" rather than a guess.

    On camera, green LEDs spill cyan-ish light that peaks in green. Only red,
    green, blue and white have been seen on camera; the yellow and orange
    cut-offs are guesses.
    """
    r, g, b = (float(v) for v in increase)
    top = max(r, g, b)
    if top < 1.0:
        return "unknown"
    if top == b:
        if not len(pale):
            return "unknown"
        if min(pale) <= BLUE_MAX_GREEN:
            return "blue"
        if len(pale) <= len(COLOUR_ZONES) or min(pale) < WHITE_MIN_GREEN:
            return "unknown"
        even = max(pale) <= FLAT * min(pale)
        fades = max(pale[1:]) > PALER_OUT * pale[0]
        return "white" if even and not fades else "unknown"
    if top == g:
        return "white" if min(r, b) / g >= 0.5 else "green"
    if min(g, b) / r >= 0.5:
        return "white"
    if g / r > 0.75:
        return "yellow"
    return "orange" if g / r > 0.45 else "red"


def board_change(on: np.ndarray, off: np.ndarray, pitch: float) -> np.ndarray:
    """How the picture changed from off to on, two photos lined up on the board:
    the signed difference per channel, blurred by a pitch, at CHANGE_SCALE.

    Blurred and signed, what two photos a little out of line leave at every
    hole's edge cancels, and what a glow adds or takes away does not.
    """
    small = [
        cv2.resize(
            im.astype(np.float32),
            None,
            fx=CHANGE_SCALE,
            fy=CHANGE_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        for im in (on, off)
    ]
    return cv2.GaussianBlur(small[0] - small[1], (0, 0), pitch * CHANGE_SCALE)


def _strength(change: np.ndarray, near: np.ndarray, far: np.ndarray) -> float:
    """How much more the board changed in near than in far (masks the size of
    change), in whichever channel differs most, up or down. 0 if near is empty.

    Taking far away leaves out what the room's light did between the two
    photos, which reaches the whole board alike. An LED's glow does not.
    """
    if not near.any():
        return 0.0
    around = change[near].mean(axis=0)
    beyond = change[far].mean(axis=0) if far.any() else 0.0
    return float(np.abs(around - beyond).max())


def _uneven(change: np.ndarray, board: np.ndarray) -> tuple[float, float]:
    """How unevenly the board changed: how far each point's change is from the
    board's median change, in whichever channel is furthest, as the mean over
    board (a mask the size of change) and as its peak (the 99th percentile). A
    change that is the same all over, as when the room's light or the camera's
    exposure shifts, scores nothing. A faint glow in one place shows in the
    peak before it moves the mean."""
    if not board.any():
        return 0.0, 0.0
    over = change[board]
    apart = np.abs(over - np.median(over, axis=0)).max(axis=1)
    return float(apart.mean()), float(np.percentile(apart, 99))


def too_bright(image: np.ndarray, rect: Rectification) -> bool:
    """Whether the unlit board is washed out in this photo (MAX_BRIGHT_SHARE).

    Then no pin is judged: a lit LED could show no new white core and no glow,
    and would be reported as a broken branch.
    """
    inside = board_mask(image.shape, rect) > 0
    if not inside.any():
        return False
    bright = image.min(axis=2)[inside] >= WAS_BELOW
    return float(bright.mean()) > MAX_BRIGHT_SHARE


def find_glow(
    pin: int,
    on: np.ndarray,
    off: np.ndarray,
    rect: Rectification,
    seen: np.ndarray | None = None,
    quiet: Sequence[np.ndarray] = (),
) -> Glow:
    """Compare RGB frames taken with the pin on and off, lined up on the board.

    seen, if given, is 1 where both photos actually showed the board. A photo
    warped into line is black where it saw nothing, and black would read as
    "dark before" next to anything bright.

    quiet holds board_change() for pins of the same run that lit nothing, lined
    up the same way. A glow must stand well above what the same patch of board
    did in those (GLOW_OVER_QUIET), as well as above GLOW_FLOOR.

    The glow is how much more the ring around the core changed than the board
    far from it did (_strength), so light that changed over the whole board
    between the two photos does not count for or against it.
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

    change = board_change(on, off, pitch)

    if not cores:
        mean, peak = _uneven(change, _shrink(inside, change) > 0)
        if mean >= SPILL_LEVELS or peak >= SPILL_PEAK:
            return Glow(pin, "unclear", "the board changed, but no LED core showed")
        return Glow(pin, "dark")

    main = cores[0]
    size = int(stats[main, cv2.CC_STAT_AREA])
    x, y = (float(v) for v in centres[main])
    if size > MAX_CORE_SHARE * inside.sum():
        return Glow(pin, "unclear", "much of the board lit up at once", core_px=size)
    away = cv2.distanceTransform((labels != main).astype(np.uint8), cv2.DIST_L2, 5)

    def second_light(i: int) -> bool:
        if away[labels == i].min() > SEPARATE_GAP * pitch:
            return True
        apart = np.hypot(*(centres[i] - centres[main])) > SEPARATE_PITCHES * pitch
        return bool(apart) and stats[i, cv2.CC_STAT_AREA] >= FRAGMENT_SHARE * size

    if any(second_light(int(i)) for i in cores[1:]):
        return Glow(pin, "unclear", "light appeared in two places", core_px=size)

    unclipped = (inside > 0) & (on.min(axis=2) < SATURATED)

    def ring(band: tuple[float, float]) -> np.ndarray:
        return (away > band[0] * pitch) & (away <= band[1] * pitch) & unclipped

    band = ring(SPILL_BAND)
    near = _shrink(band.astype(np.uint8), change) > 0
    beyond = (away > FAR_PITCHES * pitch) & (inside > 0)
    far = _shrink(beyond.astype(np.uint8), change) > 0
    strength = _strength(change, near, far)
    if strength < GLOW_FLOOR:
        return Glow(
            pin, "unclear", "a white spot appeared with no glow around it", core_px=size
        )
    if any(strength < GLOW_OVER_QUIET * _strength(q, near, far) for q in quiet):
        return Glow(
            pin,
            "unclear",
            "the light kept changing, too much to be sure of a glow",
            core_px=size,
        )
    increase = np.clip(on.astype(np.int16) - off.astype(np.int16), 0, None)[band]
    spill = increase.mean(axis=0) if len(increase) else np.zeros(3)
    pale = []
    for where in [band] + [ring(zone) for zone in COLOUR_ZONES]:
        if where.sum() >= MIN_ZONE_PITCH2 * pitch**2:
            lit = on[where].mean(axis=0)
            pale.append(float(lit[1] / max(lit[2], 1.0)))
    return Glow(
        pin,
        "lit",
        hole=rect.hole_at((x, y)),
        photo_xy=(x, y),
        colour=glow_colour(spill, pale),
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
    not be lined up with confidence: the board or camera moved too much or too
    fast, the two alignments of its lit photo disagreed, or there was no trusted
    unlit photo near enough to check against (see analyse). shorted
    holds pins that did not go high when switched on, and so were switched
    straight back off. too_bright means the unlit board was washed out in the
    photo it was fitted on (too_bright), and then no pin is judged either.

    calm holds pins that are "unclear" only because the board changed unevenly
    with no core, and no more than the run's other pins with no core did
    (CALM_OVER_TYPICAL): the room's light kept changing. Nothing says an LED lit
    there and nothing rules out a faint one, so they are not called dark. A
    caller that only needs such a pin to be quiet, because nothing should be on
    it, can take it as quiet (verify does).
    """

    rect: Rectification | None
    glows: dict[int, Glow] = field(default_factory=dict)
    moved: frozenset[int] = frozenset()
    shorted: frozenset[int] = frozenset()
    too_bright: bool = False
    calm: frozenset[int] = frozenset()


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


def _search(
    fixed: np.ndarray, moving: np.ndarray, corners: np.ndarray, reach: float, pitch: float
) -> np.ndarray | None:
    """The slide, in small pixels, that best lines up the board in two detail
    images, tried at every position within reach pitches. corners are the
    board's in fixed, full size. None if the best slide does not clearly beat
    one a hole over (MIN_SEARCH_LEAD).

    Only the board is compared, inset by a pitch. The whole picture would also
    weigh the person holding it, who does not move with the board.
    """
    x0, y0 = np.ceil((corners.min(axis=0) + pitch) * ALIGN_SCALE).astype(int)
    x1, y1 = np.floor((corners.max(axis=0) - pitch) * ALIGN_SCALE).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, fixed.shape[1]), min(y1, fixed.shape[0])
    near = round(reach * pitch * ALIGN_SCALE)
    sx0, sy0 = max(x0 - near, 0), max(y0 - near, 0)
    sx1, sy1 = min(x1 + near, moving.shape[1]), min(y1 + near, moving.shape[0])
    template, window = fixed[y0:y1, x0:x1], moving[sy0:sy1, sx0:sx1]
    if template.size == 0 or any(
        w < t for w, t in zip(window.shape, template.shape, strict=True)
    ):
        return None
    scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (bx, by) = cv2.minMaxLoc(scores)
    yy, xx = np.mgrid[: scores.shape[0], : scores.shape[1]]
    apart = np.hypot(xx - bx, yy - by) >= 0.5 * pitch * ALIGN_SCALE
    peaks = scores >= cv2.dilate(scores, np.ones((3, 3), np.uint8))
    others = peaks & apart  # separate matches, not the best one's own slope
    runner_up = float(scores[others].max()) if others.any() else -1.0
    if best - runner_up < MIN_SEARCH_LEAD:
        return None
    dx, dy = bx + sx0 - x0, by + sy0 - y0
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
    off: np.ndarray,
    fitted: np.ndarray,
    rect: Rectification,
    guess: np.ndarray | None = None,
) -> np.ndarray | None:
    """The warp taking each pixel of the fitted photo to the same point of the
    board in off, a photo with every pin off. None if they cannot be lined up.

    guess, if given, is a warp close to the answer, from following the board
    photo by photo (track). Without it, the start comes from the whole picture.

    Both photos are unlit, so the whole board's brightness can be used to line
    them up. (Its fine detail is less reliable here: a shaking hand blurs each
    averaged photo differently.) A start from the whole picture can be a hole or
    so out, since the person holding the board moves too, and on one hand-held
    run the alignment then locked one hole over. A search for the best slide
    straight to the fitted photo did worse: over a whole check the board also
    turns, and no single slide stood out. Following it photo by photo, 1.2 s
    apart, it barely turns between any two.
    """
    pitch = photo_pitch(rect)
    fixed, moving = _small_grey(fitted), _small_grey(off)
    if guess is None:
        size = (fixed.shape[1], fixed.shape[0])
        window = cv2.createHanningWindow(size, cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(fixed, moving, window)
        start = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32)
    else:
        scale = np.diag([ALIGN_SCALE, ALIGN_SCALE, 1.0])
        start = (scale @ guess @ np.linalg.inv(scale)).astype(np.float32)
    mask = _shrink(board_mask(fitted.shape, rect), fixed)
    found = _refine(fixed, moving, mask, cv2.MOTION_HOMOGRAPHY, start)
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
    off, a photo with every pin off taken just before or after it. off_warp is
    align_off's for off.

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
    grow = np.ones((int(2 * pitch * ALIGN_SCALE) | 1,) * 2, np.uint8)
    blank = [  # a saturated patch has no detail, and its edge is new
        cv2.dilate((g >= SATURATED).astype(np.uint8), grow) > 0
        for g in (grey_fixed, grey_moving)
    ]
    start = _search(
        np.where(blank[0], 0, fixed),
        np.where(blank[1], 0, moving),
        _corners(rect, off_warp),
        MAX_STEP_PITCHES + 0.5,
        pitch,
    )
    if start is None:
        return None
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
    white = cv2.dilate(white.astype(np.uint8), grow)
    keep = board & (1 - white)
    if keep.sum() < MIN_ALIGN_SHARE * max(int(board.sum()), 1):
        return None
    found = _refine(fixed, moving, keep, cv2.MOTION_TRANSLATION, start)
    if found is None:
        return None
    slide, first = found[0][:2, 2], start[:2, 2] / ALIGN_SCALE
    if np.hypot(*(slide - first)) > MAX_DISAGREE_PITCHES * pitch:
        return None
    on_warp = found[0] @ off_warp
    if (
        _farthest(_corners(rect, on_warp), _corners(rect, off_warp))
        > MAX_STEP_PITCHES * pitch
    ):
        return None
    return on_warp


def track(
    before: np.ndarray, before_warp: np.ndarray, after: np.ndarray, rect: Rectification
) -> np.ndarray | None:
    """A guess at align_off's warp for after, from a trusted unlit photo near it
    in time (before, usually 1.2 s away) and that photo's warp: the best slide
    of the board between the two. None if no slide clearly wins."""
    pitch = photo_pitch(rect)
    found = _search(
        _detail(_small_grey(before), pitch),
        _detail(_small_grey(after), pitch),
        _corners(rect, before_warp),
        2 * MAX_STEP_PITCHES + 0.5,
        pitch,
    )
    if found is None:
        return None
    scale = np.diag([ALIGN_SCALE, ALIGN_SCALE, 1.0])
    return np.linalg.inv(scale) @ found.astype(np.float64) @ scale @ before_warp


def _gap(a: np.ndarray, b: np.ndarray, xy: np.ndarray | tuple[float, float]) -> float:
    """How far apart two warps put one point of the fitted photo, in pixels."""
    point = np.asarray(xy, np.float64).reshape(1, 1, 2)
    pa, pb = (cv2.perspectiveTransform(point, w)[0, 0] for w in (a, b))
    return float(np.hypot(*(pa - pb)))


def _step(
    pin: int,
    on: np.ndarray,
    anchor: np.ndarray,
    anchor_warp: np.ndarray,
    new: np.ndarray,
    fitted: np.ndarray,
    rect: Rectification,
) -> tuple[np.ndarray, Glow, Views, np.ndarray] | None:
    """Line up new, an unlit photo next to a trusted one (anchor), and judge on,
    the lit photo between them. Returns new's warp, now trusted; the result;
    what it was judged from (on and new as the fitted photo sees them, and
    where both saw the board), so analyse can judge it again; and how the
    picture changed between the two unlit photos (board_change), which shows
    what this run's photos do when nothing is lit. None if they cannot be
    lined up with confidence.

    on is lined up twice: via new, and via the trusted anchor. The two must put
    the board in the same place, at its centre and at the LED if one lit.
    """
    pitch = photo_pitch(rect)
    new_warp = align_off(new, fitted, rect, track(anchor, anchor_warp, new, rect))
    if new_warp is None:
        return None
    via_new = align_on(on, new, new_warp, rect)
    via_anchor = align_on(on, anchor, anchor_warp, rect)
    if via_new is None or via_anchor is None:
        return None
    limit = MAX_DISAGREE_PITCHES * pitch
    if _gap(via_new, via_anchor, _corners(rect).mean(axis=0)) > limit:
        return None
    seen = seen_by(via_new, on, fitted) & seen_by(new_warp, new, fitted)
    views = warp_onto(on, via_new, fitted), warp_onto(new, new_warp, fitted), seen
    glow = find_glow(pin, views[0], views[1], rect, seen)
    if glow.photo_xy is not None and _gap(via_new, via_anchor, glow.photo_xy) > limit:
        return None  # agree at the centre but not here: the board turned
    rest = board_change(views[1], warp_onto(anchor, anchor_warp, fitted), pitch)
    both = seen_by(new_warp, new, fitted) & seen_by(anchor_warp, anchor, fitted)
    rest[_shrink(both, rest) == 0] = 0  # where either saw nothing, say nothing
    return new_warp, glow, views, rest


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

    The board is fitted once, on the first unlit frame that fits. Every other
    photo is lined up with that one on the board itself, so the board or camera
    may drift during the run.

    The holes repeat, so an alignment can lock one hole over and pass every
    other check, which would put an LED a column off. So alignments form a chain
    of trust. The fitted photo is trusted. Working outward from it, in both
    directions, each unlit photo is followed from the nearest trusted one
    (track), lined up with the fitted photo (align_off), and trusted only if the
    lit photo between them lines up the same way via both (_step). Otherwise
    that pin is set aside, and the next one is checked against the last trusted
    photo instead, so one slip cannot carry into later pins.

    The photos that hold no glow show how much this run's pictures differ when
    nothing happens: each pin that showed no core, and the two unlit photos
    either side of every lit one. Each lit pin is then judged again against
    them (find_glow's quiet), so a run with noisy photos asks more of a glow
    than a still one does. And a pin with no core whose board changed unevenly,
    but no more than the run's other such pins, is listed as calm (see Session).

    If the unlit board is washed out in the fitted photo, nothing is judged.
    """
    photographed = [p for p in pins if f"pin{p}_on" in frames and f"pin{p}_off" in frames]
    # Unlit photos in time order; the lit photo of photographed[k] sits between
    # unlit[k] and unlit[k + 1].
    unlit = [frames.get("base")] + [frames[f"pin{p}_off"] for p in photographed]
    rect, fitted, start = None, None, 0
    for i, image in enumerate(unlit):
        if image is not None:
            candidate = rectify(image)
            if candidate is not None and candidate.ok:
                rect, fitted, start = candidate, image, i
                break
    if rect is None or fitted is None:
        return Session(rect=None, shorted=frozenset(shorted))
    if too_bright(fitted, rect):
        return Session(rect, shorted=frozenset(shorted), too_bright=True)

    pitch = photo_pitch(rect)
    glows: dict[int, Glow] = {}
    moved: set[int] = set()
    lit: dict[int, Views] = {}
    quiet: list[np.ndarray] = []  # how the picture changed where nothing glowed
    coreless: dict[int, tuple[float, float]] = {}  # how unevenly it changed
    for direction in (1, -1):
        anchor: np.ndarray = fitted
        anchor_warp = np.eye(3)
        i = start + direction
        while 0 <= i < len(unlit):
            pin = photographed[i - 1 if direction == 1 else i]
            new, on = unlit[i], frames[f"pin{pin}_on"]
            done = None
            if new is not None:
                done = _step(pin, on, anchor, anchor_warp, new, fitted, rect)
            if new is None or done is None:
                moved.add(pin)
            else:
                anchor_warp, glows[pin], views, rest = done
                anchor = new
                quiet.append(rest)
                if glows[pin].status == "lit":
                    lit[pin] = views
                elif not glows[pin].core_px:
                    change = board_change(views[0], views[1], pitch)
                    board = board_mask(fitted.shape, rect) & views[2]
                    coreless[pin] = _uneven(change, _shrink(board, change) > 0)
                    quiet.append(change)
            i += direction
    calm: set[int] = set()
    if len(coreless) >= MIN_CALM:
        mean, peak = (float(np.median(v)) for v in zip(*coreless.values(), strict=True))
        calm = {
            pin
            for pin, (m, k) in coreless.items()
            if glows[pin].status == "unclear"
            and m < CALM_OVER_TYPICAL * mean
            and k < CALM_OVER_TYPICAL * peak
        }
    for pin, (on, off, seen) in lit.items():
        glows[pin] = find_glow(pin, on, off, rect, seen, quiet)
    return Session(
        rect, glows, frozenset(moved), frozenset(shorted), calm=frozenset(calm)
    )


@dataclass(frozen=True)
class Attempt:
    """One run_sequence: its frames (named as analyse expects), the pins it
    drove, and the pins that were tied to ground."""

    frames: dict[str, np.ndarray]
    pins: tuple[int, ...]
    shorted: tuple[int, ...] = ()


def retried(first: Session, again: Session) -> Session:
    """first, with the pins it set aside judged from a second run of just those.

    The second run is fitted and lined up on its own, so only its judgements
    carry over, and only if its fit is graded at least as well as the first
    run's: the verdict grades the first run's fit alone, and "works" needs a top
    grade. A glow's photo position is moved into the first run's photo, so it
    can be drawn there. A pin set aside again stays set aside.
    """
    if first.rect is None or first.too_bright:
        return first
    if again.rect is None or again.too_bright:
        return first
    if again.rect.confidence < first.rect.confidence:
        return first
    glows, moved, shorted = dict(first.glows), set(first.moved), set(first.shorted)
    for pin in first.moved:
        if pin in again.shorted:
            shorted.add(pin)
            moved.discard(pin)
        elif pin in again.glows:
            glows[pin] = _into(again.glows[pin], again.rect, first.rect)
            moved.discard(pin)
    calm = first.calm | (again.calm & first.moved)
    return Session(first.rect, glows, frozenset(moved), frozenset(shorted), calm=calm)


def _into(glow: Glow, source: Rectification, target: Rectification) -> Glow:
    """The glow as seen in target's photo rather than source's."""
    if glow.photo_xy is None:
        return glow
    to_target = np.linalg.inv(target.photo_to_canonical) @ source.photo_to_canonical
    point = np.array([[glow.photo_xy]], np.float64)
    x, y = cv2.perspectiveTransform(point, to_target)[0, 0]
    return replace(glow, photo_xy=(float(x), float(y)))


def analyse_attempts(attempts: Sequence[Attempt]) -> Session:
    """Judge a run and its retries: each retry fills in pins set aside before."""
    session = Session(rect=None)
    for i, attempt in enumerate(attempts):
        judged = analyse(attempt.frames, attempt.pins, attempt.shorted)
        session = judged if i == 0 else retried(session, judged)
    return session


def blink_and_watch(
    board: PinDriver,
    pins: Sequence[int],
    capture: Callable[[], np.ndarray],
    settle_s: float = 0.6,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = lambda _message: None,
    retries: int = MAX_RETRIES,
    retrying: Callable[[tuple[int, ...]], None] = lambda _pins: None,
) -> tuple[list[Attempt], Session]:
    """Blink every pin, judge the photos, and check set-aside pins again.

    A pin is set aside when its photos cannot be lined up, usually because the
    board moved while it was lit. Checking it again at once, on its own, costs a
    couple of seconds and usually gets a clean answer, where otherwise the
    student would have to run the whole check again. No retry is tried when the
    board was not found at all: the framing is the problem, and the same
    framing would fail again. say(message) reports progress to the person
    holding the board; retrying(pins) is called as a re-check starts. Safety is
    run_sequence's: all pins are released between runs, and while the photos
    are judged.

    Returns every run made, so it can be saved and judged again later
    (analyse_attempts), and the combined result.
    """
    frames, shorted = run_sequence(board, pins, capture, settle_s, sleep)
    attempts = [Attempt(frames, tuple(pins), tuple(shorted))]
    say("Working it out...")
    session = analyse(frames, pins, shorted)
    for _ in range(retries):
        if session.rect is None or not session.moved:
            break
        again = tuple(sorted(session.moved))
        listed = ", ".join(str(p) for p in again)
        say(f"Pin{'s' if len(again) > 1 else ''} {listed} again: hold still...")
        retrying(again)
        frames, shorted = run_sequence(board, again, capture, settle_s, sleep)
        attempts.append(Attempt(frames, again, tuple(shorted)))
        say("Working it out...")
        session = retried(session, analyse(frames, again, shorted))
    return attempts, session
