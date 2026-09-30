"""Which leg to move: the checker's fixes for a described build, trusted only as
far as blinking confirms the description.

Stage B (ARCHITECTURE.md section 3[6]) needs every leg in a hole: a circuit file
of the build. In the finished system a trained model reads that from the photo.
Until it exists, a person describes the build in the same kind of file
(examples/basicboard_as_built.json), so everything after the model runs for
real, and the model can later take the person's place unchanged.

A description is inference either way. People misread boards, and so will the
model. So before any fix is shown, the description is checked against
measurement: blink and watch (breadboard/blink.py) has already switched each pin
on and seen what lit. The description predicts the same thing. Pin 4 drives the
blue LED at columns 31-32, so it should light blue there; a backwards LED, or a
pin with no LED, should stay dark. If any pin disagrees, the description is
wrong somewhere, and no fix is shown, because a fix computed from a wrong
description would send the student to move a leg that is fine.

When every pin agrees, the checker (check.py) compares the description with the
lab and names the smallest fix, down to the hole. "Works" still needs blinking to
confirm the lab's LEDs too (verify.py), so a description cannot talk its way to a
clean bill of health.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from breadboard.blink import Session
from breadboard.board import BoardError, is_rail, parse_hole
from breadboard.check import Finding, check
from breadboard.graph import build as build_graph
from breadboard.graph import wires_joined
from breadboard.netlist import Netlist
from breadboard.verify import (
    _WARM,
    Verdict,
    _a,
    _led,
    _same_colour,
    _summary,
    expected_leds,
    verify,
)

MISMATCH_SUMMARY = "The parts entered don't match what the board did."
_SUPPLIES = frozenset({"GND", "5V", "3V", "VIN", "RAW"})


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


def claims(build: Netlist, pins: Sequence[int]) -> dict[int, Claim]:
    """What the build says each pin will do when switched on alone.

    A pin with one LED wired forwards to ground lights it. A pin stays dark when
    no LED can be reached from it at all, or only ones facing away from it or
    never reaching ground. Anything murkier gets no claim, rather than a guess:
    two LEDs on one pin, a button in the path, or an LED reachable some other way
    (a resistor shared between two LEDs, two pins joined by a wire).
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


def disagreements(build: Netlist, session: Session, pins: Sequence[int]) -> list[Finding]:
    """Every pin where what lit contradicts what the build says.

    Only clear results count: a pin set aside, tied to ground, or unclear
    contradicts nothing (verify reports those). A colour the camera cannot tell
    apart (red, orange, yellow; or "unknown") is not a contradiction either.
    A glow is placed to within a column of the LED's legs, since the LED's body
    stands above the board.
    """
    out = []
    for pin, claim in sorted(claims(build, pins).items()):
        glow = session.glows.get(pin)
        if glow is None or glow.status == "unclear":
            continue
        if glow.status == "dark":
            if claim.lights:
                out.append(
                    _mismatch(
                        claim,
                        f"they say pin {pin} lights {_led(claim.colour)}, "
                        "but nothing lit.",
                    )
                )
            continue
        seen = f"{_a(glow.colour)} LED lit"
        if glow.column is not None:
            seen += f" at column {glow.column}"
        if not claim.lights:
            said = f"they say nothing lights from pin {pin}, but {seen}."
            out.append(_mismatch(claim, said))
            continue
        if not _colour_fits(claim.colour, glow.colour):
            out.append(
                _mismatch(
                    claim, f"they say pin {pin} lights {_led(claim.colour)}, but {seen}."
                )
            )
        elif (
            glow.column is not None
            and claim.columns
            and not (min(claim.columns) - 1 <= glow.column <= max(claim.columns) + 1)
        ):
            ends = sorted({min(claim.columns), max(claim.columns)})
            where = "-".join(str(c) for c in ends)
            out.append(
                _mismatch(
                    claim,
                    f"they put {_led(claim.colour)} on pin {pin} at column {where}, "
                    f"but the light came from column {glow.column}.",
                )
            )
    return out


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


def diagnose(
    build: Netlist, lab: Netlist, session: Session, pins: Sequence[int]
) -> Verdict:
    """The verdict for a blink run of a described build.

    If blinking contradicts the description, the answer says so, alongside what
    blinking found against the lab, and shows no fix. If nothing contradicts it,
    the checker's findings for the build (with the holes to change) replace
    blinking's findings about which LED lit, which they explain more exactly.
    Every other finding blinking made is kept. A fix is stated as an error only
    when blinking clearly saw the pin of the LED it concerns; otherwise it is
    offered as not yet confirmed. Works only when the checker finds nothing and
    blinking confirms every LED the lab expects.
    """
    measured = verify(lab, session, pins)
    if session.rect is None:
        return measured
    contradictions = disagreements(build, session, pins)
    if contradictions:
        return Verdict(False, MISMATCH_SUMMARY, (*contradictions, *measured.findings))
    fixes = [_confirmed_or_not(f, build, session, pins) for f in check(build, lab)]
    if not fixes:
        return measured
    kept = [f for f in measured.findings if f.kind not in _EXPLAINED_BY_FIXES]
    findings = [*fixes, *kept]
    order = {"error": 0, "warning": 1, "uncertain": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    return Verdict(False, _summary(findings), tuple(findings), measured.ok_pins)


# Blinking's findings about which LED lit where. When a confirmed description
# names the fix, these say the same thing less exactly.
_EXPLAINED_BY_FIXES = frozenset(
    {"led_does_not_light", "wrong_led_on_pin", "unexpected_led"}
)


def _confirmed_or_not(
    fix: Finding, build: Netlist, session: Session, pins: Sequence[int]
) -> Finding:
    """The fix as is if blinking clearly saw the pins it concerns, else offered
    as not yet confirmed. Its pins are those of the LEDs it names; a fix naming
    no LED needs every pin the description makes a claim about to be clear."""
    if fix.severity != "error":
        return fix
    expected, untestable = expected_leds(build)
    pin_of = {e.component: pin for pin, e in expected.items()}
    pin_of.update({u.component: u.pin for u in untestable if u.pin is not None})
    concerned = [pin_of[c] for c in fix.components if c in pin_of]
    if not concerned:
        concerned = list(claims(build, pins))
    clear = {p for p, g in session.glows.items() if g.status in ("lit", "dark")}
    if all(p in clear for p in concerned):
        return fix
    said = fix.message[0].lower() + fix.message[1:]
    return replace(fix, severity="uncertain", message=f"Not yet confirmed: {said}")
