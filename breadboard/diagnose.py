"""Which leg to move: the checker's fixes for a described build, trusted only as
far as blinking confirms the description.

Stage B (ARCHITECTURE.md section 3[6]) needs every leg in a hole: a circuit file
of the build. In the finished system a trained model reads that from the photo.
Until it exists, a person describes the build in the same kind of file
(examples/basicboard_as_seen.json), so everything after the model runs for
real, and the model can later take the person's place unchanged.

A description is inference either way. People misread boards, and so will the
model. So before any fix is shown, the description is checked against
measurement: blink and watch (breadboard/blink.py) has already switched each pin
on and seen what lit. The description predicts where light can come from. Pin 4
drives the blue LED at columns 31-32, so if pin 4 lights anything, it is blue
and there; a pin with no LED lights nothing. If any light disagrees, the
description is wrong somewhere, and no fix is shown, because a fix computed from
a wrong description would send the student to move a leg that is fine.

Darkness contradicts nothing. An LED can be placed right and still stay dark,
for reasons no picture shows: it is the wrong way round, a leg is not pushed in,
or it is broken. So the answer works in two steps. First, is everything in the
right place? The checker (check.py) compares the description with the lab and
names the smallest fix, down to the hole. Only when there is nothing to move,
for an LED placed right that stayed dark, it suggests the hidden causes one at
a time, most likely first (HIDDEN_CAUSES), and the next check confirms or rules
each out. "Placed right" is only as good as the description, so it is said as
"looks placed right", and the later causes include checking each leg's hole. remember()
carries what was suggested from one check to the next. When the LED lights, the
answer says it is fixed.

Which way round an LED faces cannot be seen once it is seated, so a description
may leave it open (netlist.UNKNOWN), and a model's always will. Blinking settles
it: an LED that lights from its pin faces forwards, whatever the description
says (settle_directions).

"Works" still needs blinking to confirm every LED the lab expects (verify.py),
so a description cannot talk its way to a clean bill of health.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace

from breadboard.blink import Glow, Session
from breadboard.board import BoardError, is_rail, parse_hole
from breadboard.check import Finding, check
from breadboard.graph import build as build_graph
from breadboard.graph import wires_joined
from breadboard.netlist import Netlist
from breadboard.verify import (
    _WARM,
    Verdict,
    _a,
    _cap,
    _led,
    _same_colour,
    _summary,
    expected_leds,
    verify,
)

MISMATCH_SUMMARY = "The parts entered don't match what the board did."
_SUPPLIES = frozenset({"GND", "5V", "3V", "VIN", "RAW"})

# What to try when an LED placed right stays dark, most likely first: an LED put
# in backwards, then a leg not pushed in, then a broken LED. After that, a person
# has to look.
HIDDEN_CAUSES = ("turn", "reseat", "swap", "ask")

_TRY = {
    "turn": "It is most likely the wrong way round. Its longer leg should be in "
    "{long}, on pin {pin}'s side. If it is not, turn the LED around, then check "
    "again.",
    "reseat": "A leg may not be pushed in. Take the LED out and push it back in "
    "firmly, longer leg in {long}. Check its resistor and wires are firmly in the "
    "right holes too, then check again.",
    "swap": "The LED may be broken. Put in a new one, longer leg in {long}, then "
    "check again.",
    "ask": "I can't find what is wrong. Check every leg and wire on its path is in "
    "the right hole, then ask your teacher to look at it with you.",
}

# What most likely fixed an LED that lights again, by what was suggested.
_FIXED_BY = {
    ("hidden_fault", 0): "It was most likely the wrong way round.",
    ("hidden_fault", 1): "A leg was most likely not pushed in.",
    ("hidden_fault", 2): "The old LED was most likely broken.",
    ("reversed_polarity", 0): "It was most likely the wrong way round.",
}


@dataclass(frozen=True)
class Claim:
    """What a described build says switching one pin on will do.

    colour is the LED's colour when it should light, None when the pin should
    stay dark. columns are that LED's leg columns; component its id.
    """

    pin: int
    colour: str | None = None
    component: str | None = None
    columns: tuple[int, ...] = ()
    holes: tuple[str, ...] = ()

    @property
    def lights(self) -> bool:
        return self.component is not None


@dataclass(frozen=True)
class Suggested:
    """Where one pin stood at a check, for the next check to follow on from.

    kind is the main finding about the pin's LED; step, for a hidden fault,
    which of HIDDEN_CAUSES was suggested; holes where the LED was, so an LED
    moved since starts again from the first cause.
    """

    pin: int
    kind: str
    step: int = 0
    holes: frozenset[str] = frozenset()


def claims(build: Netlist, pins: Sequence[int]) -> dict[int, Claim]:
    """What the build says each pin will do when switched on alone.

    A pin with one LED wired forwards to ground lights it, if nothing unseen is
    wrong. A pin stays dark when no LED can be reached from it at all, or only
    ones facing away from it or never reaching ground. Anything murkier gets no
    claim, rather than a guess: two LEDs on one pin, a button in the path, an
    LED whose way round is not known, or an LED reachable some other way (a
    resistor shared between two LEDs, two pins joined by a wire).
    """
    expected, untestable = expected_leds(build)
    unsure = {u.pin for u in untestable if u.pin is not None and not u.dark}
    faces_away: dict[int, set[str]] = {}
    for u in untestable:
        if u.dark and u.pin is not None:
            faces_away.setdefault(u.pin, set()).add(u.component)
    reachable = _reachable_leds(build)
    out: dict[int, Claim] = {}
    for pin in pins:
        if pin in unsure:
            continue
        want = expected.get(pin)
        near = reachable(pin)
        if want is None:
            if not near - faces_away.get(pin, set()):
                out[pin] = Claim(pin)
            continue
        if near - {want.component}:
            continue
        holes = tuple(p.hole for p in build[want.component].pins.values())
        out[pin] = Claim(pin, want.colour, want.component, _columns(holes), holes)
    return out


def _reachable_leds(build: Netlist):
    """A function giving, for a pin, every LED any path from it touches before
    reaching a supply (ground or power), whichever way the LED faces."""
    graph = wires_joined(build_graph(build))
    stop = {n for n, names in graph.labels.items() if names & _SUPPLIES}
    links: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for e in graph.edges:
        leds = frozenset(i.component_id for i in e.items if i.type == "led")
        links.setdefault(e.a, []).append((e.b, leds))
        links.setdefault(e.b, []).append((e.a, leds))
    for part in graph.parts:  # a sensor may pass current between any of its legs
        nodes = [node for _, node in part.pins]
        for a in nodes:
            links.setdefault(a, []).extend((b, frozenset()) for b in nodes if b != a)

    def reach(pin: int) -> set[str]:
        start = [n for n, names in graph.labels.items() if f"D{pin}" in names]
        seen, todo, found = set(start), list(start), set()
        while todo:
            node = todo.pop()
            for other, leds in links.get(node, ()):
                found |= leds
                if other not in seen and other not in stop:
                    seen.add(other)
                    todo.append(other)
        return found

    return reach


def _columns(holes: Sequence[str]) -> tuple[int, ...]:
    cols = set()
    for hole in holes:
        try:
            where, index = parse_hole(hole)
        except BoardError:
            continue
        if not is_rail(where):
            cols.add(index)
    return tuple(sorted(cols))


def settle_directions(
    build: Netlist,
    session: Session,
    pins: Sequence[int],
    previous: Mapping[int, Suggested] | None = None,
) -> tuple[Netlist, tuple[str, ...]]:
    """The build with each LED's way round settled as far as blinking can.

    An LED that lit from its pin, in its colour and where the build puts it,
    faces forwards: its longer leg is on the pin's side. That holds whatever the
    build says, so a description never has to get it right. An LED the build
    leaves open that stayed dark is taken to face forwards, the way the lab
    wants it. If it is really backwards, that is the first hidden cause tried.
    So is an LED described as backwards that the last check said to turn around,
    and that is still dark: once turned, the description is out of date. Any
    other LED stays as described; the checker matches an open one either way.

    Returns the settled build, and the LEDs the build left open that blinking
    settled.
    """
    previous = previous or {}
    settled, told = build, []
    for led in build.of_type("led"):
        found = _lighting_way(settled, led.id, pins)
        if found is None:
            continue
        way, claim = found
        glow = session.glows.get(claim.pin)
        lit_here = glow is not None and glow.lit and _contradiction(claim, glow) is None
        holes = frozenset(h.lower() for h in claim.holes)
        doubted = not led.direction_known or _turned(previous.get(claim.pin), holes)
        if lit_here or doubted:
            settled = way
            if lit_here and not led.direction_known:
                told.append(led.id)
    return settled, tuple(told)


def _lighting_way(build: Netlist, cid: str, pins: Sequence[int]):
    """The build with this LED facing whichever way lets a pin light it, and
    that pin's claim. None if neither way does."""
    for flip in (False, True):
        trial = deepcopy(build)
        led = trial[cid]
        led.attrs = {k: v for k, v in led.attrs.items() if k != "direction"}
        if flip:
            led.pins = {"anode": led.pins["cathode"], "cathode": led.pins["anode"]}
        for claim in claims(trial, pins).values():
            if claim.component == cid:
                return trial, claim
    return None


def _turned(before: Suggested | None, holes: frozenset[str]) -> bool:
    """Whether the last check said to turn this LED around, where it still is."""
    if before is None or before.holes != holes:
        return False
    return before.kind == "reversed_polarity" or (
        before.kind == "hidden_fault" and HIDDEN_CAUSES[before.step] == "turn"
    )


def disagreements(build: Netlist, session: Session, pins: Sequence[int]) -> list[Finding]:
    """Every pin where what lit contradicts what the build says.

    Only light can contradict a description. A pin that stayed dark contradicts
    nothing: an LED placed right can stay dark for reasons no picture shows
    (hidden_causes). Nor does a pin set aside, tied to ground, or unclear
    (verify reports those), or a colour the camera cannot tell apart (red,
    orange, yellow; or "unknown"). A glow is placed to within a column of the
    LED's legs, since the LED's body stands above the board.
    """
    out = []
    for pin, claim in sorted(claims(build, pins).items()):
        glow = session.glows.get(pin)
        if glow is None or not glow.lit:
            continue
        said = _contradiction(claim, glow)
        if said is not None:
            out.append(_mismatch(claim, said))
    return out


def _contradiction(claim: Claim, glow: Glow) -> str | None:
    """What a lit pin showed that the build does not say, if anything."""
    pin = claim.pin
    seen = f"{_a(glow.colour)} LED lit"
    if glow.column is not None:
        seen += f" at column {glow.column}"
    if not claim.lights:
        return f"they say nothing lights from pin {pin}, but {seen}."
    if not _colour_fits(claim.colour, glow.colour):
        return f"they say pin {pin} lights {_led(claim.colour)}, but {seen}."
    if (
        glow.column is not None
        and claim.columns
        and not (min(claim.columns) - 1 <= glow.column <= max(claim.columns) + 1)
    ):
        ends = sorted({min(claim.columns), max(claim.columns)})
        where = "-".join(str(c) for c in ends)
        return (
            f"they put {_led(claim.colour)} on pin {pin} at column {where}, "
            f"but the light came from column {glow.column}."
        )
    return None


def _colour_fits(want: str | None, seen: str | None) -> bool:
    if _same_colour(want, seen) or seen in (None, "unknown"):
        return True
    return want in _WARM and seen in _WARM


def _mismatch(claim: Claim, what: str) -> Finding:
    return Finding(
        "entry_mismatch",
        f"The parts entered don't match the board: {what}",
        severity="uncertain",
        components=(claim.component,) if claim.component else (),
        holes=claim.holes,
        suggestion="Fix the entered parts to match the board, or the board to "
        "match them, then check again.",
        detail={"pin": claim.pin},
    )


def hidden_causes(
    build: Netlist,
    lab: Netlist,
    session: Session,
    pins: Sequence[int],
    previous: Mapping[int, Suggested] | None = None,
) -> list[Finding]:
    """What to try next for each LED placed to light that clearly stayed dark.

    Only LEDs the lab expects, on the pin and in the colour it expects them.
    Each gets the hidden cause after the one the last check suggested for it
    (previous), or the first, if there was none or the LED has moved since.
    Call it only when the checker finds nothing to move: a part in the wrong
    place comes first, and may be why the LED is dark.
    """
    previous = previous or {}
    expected, _ = expected_leds(lab)
    said = claims(build, pins)
    out = []
    for pin, want in sorted(expected.items()):
        glow, claim = session.glows.get(pin), said.get(pin)
        if glow is None or glow.status != "dark" or claim is None or not claim.lights:
            continue
        if not _same_colour(want.colour, claim.colour) or claim.component is None:
            continue
        legs = build[claim.component].pins
        long, short = legs["anode"].hole.lower(), legs["cathode"].hole.lower()
        step = _next_step(previous.get(pin), frozenset((long, short)))
        led = _led(claim.colour)
        out.append(
            Finding(
                "hidden_fault",
                f"{_cap(led)} looks placed right, but it did not light."
                if step == 0
                else f"{_cap(led)} still did not light.",
                components=(claim.component,),
                holes=(long, short),
                suggestion=_TRY[HIDDEN_CAUSES[step]].format(long=long, pin=pin),
                detail={"pin": pin, "step": step},
            )
        )
    return out


def _next_step(before: Suggested | None, holes: frozenset[str]) -> int:
    if before is None or before.holes != holes:
        return 0
    if before.kind == "hidden_fault":
        return min(before.step + 1, len(HIDDEN_CAUSES) - 1)
    if before.kind == "reversed_polarity":
        return HIDDEN_CAUSES.index("turn") + 1  # turning it round was tried
    return 0


def remember(
    verdict: Verdict,
    lab: Netlist,
    session: Session,
    previous: Mapping[int, Suggested] | None = None,
) -> dict[int, Suggested]:
    """What the next check needs to know about this one.

    Each pin the lab expects to light that blinking saw fail (dark, tied to
    ground, or an error about it), with the main thing said about it. A pin
    that worked is forgotten. A pin that could not be judged, or was only in
    doubt, keeps what it had, so one blurred photo does not restart the list of
    causes. So does a pin whose failure was only measured this time, with no
    new suggestion (a check where the parts entered did not match): the
    student has not been told anything new to try.
    """
    previous = previous or {}
    expected, _ = expected_leds(lab)
    said: dict[int, Finding] = {}
    for f in sorted(verdict.findings, key=lambda f: f.severity != "error"):
        pin = f.detail.get("pin")
        if isinstance(pin, int):
            said.setdefault(pin, f)
    out = {}
    for pin in sorted(expected):
        if pin in verdict.ok_pins:
            continue
        glow = session.glows.get(pin)
        judged = session.rect is not None and (
            pin in session.shorted or (glow is not None and glow.status != "unclear")
        )
        if not judged:
            if pin in previous:
                out[pin] = previous[pin]
            continue
        f = said.get(pin)
        if f is None or f.severity != "error":
            dark = glow is not None and glow.status == "dark"
            failed = pin in session.shorted or dark
            if not failed:  # lit, but in doubt: not a failure
                if pin in previous:
                    out[pin] = previous[pin]
                continue
            kind = "did_not_work"  # its own finding gave way to a fix, or none was made
        else:
            kind = f.kind
        if kind in _MEASURED_ONLY and pin in previous:
            out[pin] = previous[pin]
        elif f is not None and f.severity == "error":
            holes = frozenset(h.lower() for h in f.holes)
            out[pin] = Suggested(pin, kind, f.detail.get("step", 0), holes)
        else:
            out[pin] = Suggested(pin, kind)
    return out


# Failures blinking measured, which suggest nothing new to try.
_MEASURED_ONLY = frozenset(
    {"did_not_work", "led_does_not_light", "wrong_led_on_pin", "pin_tied_to_ground"}
)


def diagnose(
    build: Netlist,
    lab: Netlist,
    session: Session,
    pins: Sequence[int],
    previous: Mapping[int, Suggested] | None = None,
) -> Verdict:
    """The verdict for a blink run of a described build.

    previous is what the last check left (remember), so the answer can follow
    on from it: the next hidden cause to try, or that something is now fixed.

    If blinking contradicts the description, the answer says so, alongside what
    blinking found against the lab, and shows no fix. If nothing contradicts it,
    the checker's findings for the build (with the holes to change) replace
    blinking's findings about which LED lit, which they explain more exactly.
    An LED placed right but dark gets its next hidden cause. Every other finding
    blinking made is kept. A fix is stated as an error only when blinking
    clearly saw the pin of the LED it concerns; otherwise it is offered as not
    yet confirmed. Works only when the checker finds nothing and blinking
    confirms every LED the lab expects.
    """
    previous = previous or {}
    measured = verify(lab, session, pins)
    if session.rect is None:
        return measured
    expected, _ = expected_leds(lab)
    notes = [
        f"Fixed since the last check: {_led(expected[pin].colour)} lights now. "
        + _FIXED_BY.get((previous[pin].kind, previous[pin].step), "")
        for pin in sorted(previous)
        if pin in measured.ok_pins and pin in expected
    ]
    notes = [n.strip() for n in notes]
    build, told = settle_directions(build, session, pins, previous)
    contradictions = disagreements(build, session, pins)
    if contradictions:
        findings = (*contradictions, *measured.findings)
        return Verdict(
            False, MISMATCH_SUMMARY, findings, measured.ok_pins, notes=tuple(notes)
        )
    if told:
        leds = _leds([build[cid].attrs.get("color") for cid in told])
        verb = "is" if len(told) == 1 else "are"
        notes.append(
            f"Blinking showed which way round {leds} {verb}. The parts entered did "
            "not say."
        )

    fixes = [_confirmed_or_not(f, build, session, pins) for f in check(build, lab)]
    hidden = [] if fixes else hidden_causes(build, lab, session, pins, previous)
    if not fixes and not hidden:
        return replace(measured, notes=tuple(notes))
    dark = {f.detail["pin"] for f in hidden}
    kept = [
        f
        for f in measured.findings
        if not (f.kind == "led_does_not_light" and f.detail.get("pin") in dark)
    ]
    if fixes:
        explained = _explained_pins(build, lab, pins, fixes)
        kept = [
            f
            for f in kept
            if not (f.kind in _EXPLAINED_BY_FIXES and f.detail.get("pin") in explained)
        ]
    findings = [*fixes, *hidden, *kept]
    order = {"error": 0, "warning": 1, "uncertain": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    # "In the right place" only when nothing is to be moved (no fixes, since
    # hidden causes need none) and nothing else is wrong; doubts may remain.
    if hidden and all(
        f.kind == "hidden_fault" or f.severity == "uncertain" for f in findings
    ):
        leds = _leds([build[f.components[0]].attrs.get("color") for f in hidden])
        summary = f"Everything looks in the right place, but {leds} did not light."
    else:
        summary = _summary(findings)
    return Verdict(False, summary, tuple(findings), measured.ok_pins, notes=tuple(notes))


# Blinking's findings about which LED lit where. When a confirmed description
# names the fix, these say the same thing less exactly.
_EXPLAINED_BY_FIXES = frozenset(
    {"led_does_not_light", "wrong_led_on_pin", "unexpected_led"}
)


def _explained_pins(
    build: Netlist, lab: Netlist, pins: Sequence[int], fixes: Sequence[Finding]
) -> set[int]:
    """Pins whose trouble the fixes account for: the description does not place
    the lab's LED to light from them, or a fix names the LED it does place. A
    pin whose LED is placed right, and named by no fix, is not explained by a
    fix elsewhere, so what blinking found there is still said."""
    expected, _ = expected_leds(lab)
    named = {c for f in fixes for c in f.components}
    said = claims(build, pins)
    out = set(pins) - set(said)  # no claim: nothing says the LED is placed right
    for pin, claim in said.items():
        want = expected.get(pin)
        placed = (
            want is not None and claim.lights and _same_colour(want.colour, claim.colour)
        )
        if not placed or claim.component in named:
            out.add(pin)
    return out


def _leds(colours: Sequence[str | None]) -> str:
    """["blue"] -> "the blue LED"; ["white", "blue", "red"] -> "the white, blue
    and red LEDs"."""
    if len(colours) == 1:
        return _led(colours[0])
    named = [c for c in colours if c]
    if len(named) < len(colours):
        return f"{len(colours)} LEDs"
    return f"the {', '.join(named[:-1])} and {named[-1]} LEDs"


def _confirmed_or_not(
    fix: Finding, build: Netlist, session: Session, pins: Sequence[int]
) -> Finding:
    """The fix as is if blinking clearly saw the pins it concerns, else offered
    as not yet confirmed. Its pins are those of the LEDs it names; a fix naming
    no LED needs every pin the description makes a claim about to be clear. A
    fix concerning one pin says which (detail "pin")."""
    if fix.severity != "error":
        return fix
    expected, untestable = expected_leds(build)
    pin_of = {e.component: pin for pin, e in expected.items()}
    pin_of.update({u.component: u.pin for u in untestable if u.pin is not None})
    concerned = [pin_of[c] for c in fix.components if c in pin_of]
    if len(set(concerned)) == 1:
        fix = replace(fix, detail={**fix.detail, "pin": concerned[0]})
    if not concerned:
        concerned = list(claims(build, pins))
    clear = {p for p, g in session.glows.items() if g.status in ("lit", "dark")}
    if all(p in clear for p in concerned):
        return fix
    said = fix.message[0].lower() + fix.message[1:]
    return replace(fix, severity="uncertain", message=f"Not yet confirmed: {said}")
