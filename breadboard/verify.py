"""Does it work? What each pin lights, checked against what the lab expects.

This is stage A of verification (ARCHITECTURE.md section 3[6]), done by
measurement. The lab's circuit file says, for example, that pin 2 drives a
resistor and a red LED to ground, so switching pin 2 on should light the red
LED. Blink and watch (breadboard/blink.py) switches each pin on and records what
lit. Comparing the two answers "does it work?" without knowing where any leg is.

Why this and not the full checker (check.py). The checker needs a circuit file
with every leg in a hole. Blink and watch cannot see legs, resistors or wires,
so building that file would mean inventing parts nobody saw, which is exactly
what this project must not do. Where a leg must move (stage B, the minimum
edit) needs a description of every leg: diagnose.py takes one written by hand,
standing in for the model that will read it from the photo.

Findings use the checker's Finding type, so callers treat both the same.
"Everything works" is harder to earn than any single error (CLAUDE.md). It needs
every LED in the lab testable, and each one lit in the right colour from the
right pin. Nothing may be doubtful on any pin, and the board fit must be top
grade. An LED this cannot test is reported, never assumed fine.

During a run only the pin under test drives; every other pin is disconnected
(blink.run_sequence). So an LED lights only if its current really reaches ground,
not because some other pin happened to be held low. If the current reaches ground
through another LED, both glow, which reads as light in two places.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from breadboard.blink import OUTPUT_PINS, Glow, Session
from breadboard.check import Finding
from breadboard.graph import CircuitGraph, build
from breadboard.netlist import Netlist

# Colours the camera cannot reliably tell apart, so a mismatch between them is
# not an error. glow_colour's orange and yellow cut-offs are unvalidated.
_WARM = frozenset({"red", "orange", "yellow"})

# What may sit in series with a testable LED. Anything else, such as a button or
# a sensor, could block current during the run for reasons that are not faults.
_PASSIVE = frozenset({"led", "resistor", "wire"})

CAVEAT = (
    "Blinking cannot see resistors, and an LED with no resistor still lights, "
    "so check each LED has its resistor."
)


@dataclass(frozen=True)
class Expected:
    """One LED the lab says a pin should light. colour None means any colour."""

    pin: int
    component: str
    colour: str | None


@dataclass(frozen=True)
class Untestable:
    """An LED in the lab that blinking cannot test, and why.

    pin is the digital pin its chain starts from, when it has one. dark is True
    when, switched on alone, that pin cannot light it: it faces away from the
    pin, or its path never reaches ground.
    """

    component: str
    colour: str | None
    reason: str
    pin: int | None = None
    dark: bool = False


@dataclass(frozen=True)
class Verdict:
    """The answer for the student.

    works is True only when everything was confirmed. ok_pins are the pins whose
    expected LED lit correctly. findings explain everything else, errors before
    doubts. summary is one line to show first. caveat is what a "works" verdict
    cannot vouch for. notes are things worth saying that are not problems, such
    as what was fixed since the last check (diagnose.py).
    """

    works: bool
    summary: str
    findings: tuple[Finding, ...] = ()
    ok_pins: tuple[int, ...] = ()
    caveat: str | None = None
    notes: tuple[str, ...] = ()


def expected_leds(lab: Netlist) -> tuple[dict[int, Expected], list[Untestable]]:
    """What the lab expects each digital pin to light when switched on.

    Reads the lab's series chains (graph.py). Strips joined by wires count as one
    place, so an LED returning to a ground rail that a wire joins to GND counts
    as going to GND. An LED is expected to light from a pin when its chain runs
    from exactly one digital output pin to ground. It must hold only resistors
    and wires besides the LED, the LED must face forward (anode toward the pin),
    and no other LED may share the pin. Every other LED comes back as
    Untestable, with the reason.
    """
    full = build(lab)
    place = _wire_joined(full)
    names: dict[str, set[str]] = {}
    for node in full.node_set():
        names.setdefault(place(node), set()).update(full.label_of(node))

    graph = full.collapsed()
    per_pin: dict[int, list[Expected]] = {}
    untestable: list[Untestable] = []
    for edge in graph.edges:
        leds = [item for item in edge.items if item.type == "led"]
        if not leds:
            continue
        ends = {end: names.get(place(end), set()) for end in (edge.a, edge.b)}
        pin_end = next((e for e, n in ends.items() if _digital(n) is not None), None)
        pin = _digital(ends[pin_end]) if pin_end is not None else None
        other = edge.b if pin_end == edge.a else edge.a

        def fail(reason: str, leds=leds, pin=pin, dark: bool = False) -> None:
            untestable.extend(
                Untestable(led.component_id, _colour(led.attrs), reason, pin, dark)
                for led in leds
            )

        if len(leds) != 1:
            fail("it is in a chain with another LED")
        elif any(item.type not in _PASSIVE for item in edge.items):
            fail("something other than a resistor or wire is in its path")
        elif pin is None or pin_end is None:
            fail("it is not wired from a digital pin")
        elif "GND" not in ends[other]:
            fail("it does not return to ground", dark=True)
        elif leds[0].polarity == 0:
            fail("which way round it is was not seen")
        elif (leds[0].polarity == 1) != (pin_end == edge.a):
            fail("it faces away from its pin, so it cannot light from it", dark=True)
        elif pin not in OUTPUT_PINS:
            fail(f"the board's program cannot switch pin {pin}")
        else:
            led = leds[0]
            per_pin.setdefault(pin, []).append(
                Expected(pin, led.component_id, _colour(led.attrs))
            )

    expected: dict[int, Expected] = {}
    for pin, wants in per_pin.items():
        if len(wants) == 1:
            expected[pin] = wants[0]
        else:
            untestable += [
                Untestable(
                    w.component, w.colour, f"it shares pin {pin} with another LED", pin
                )
                for w in wants
            ]
    return expected, untestable


def _wire_joined(graph: CircuitGraph):
    """A function mapping each node to one representative of its wire-joined set."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in graph.edges:
        if all(item.type == "wire" for item in edge.items):
            parent[find(edge.a)] = find(edge.b)
    return find


def _digital(names: set[str] | frozenset[str]) -> int | None:
    """The digital pin number among a place's MCU pin names, if exactly one."""
    pins = [int(n[1:]) for n in names if n.startswith("D") and n[1:].isdigit()]
    return pins[0] if len(pins) == 1 else None


def _colour(attrs: tuple[tuple[str, object], ...]) -> str | None:
    value = dict(attrs).get("color")
    return None if value is None else str(value)


def verify(lab: Netlist, session: Session, pins_run: Sequence[int]) -> Verdict:
    """Compare a blink run with what the lab expects."""
    expected, untestable = expected_leds(lab)
    cannot = [
        Finding(
            "cannot_check",
            f"I can't check {_led(u.colour)} by blinking it, so check that one yourself.",
            severity="uncertain",
            components=(u.component,),
            detail={"reason": u.reason},
        )
        for u in untestable
    ]
    if not expected:
        return Verdict(
            False, "This lab has no LEDs that blinking can check.", tuple(cannot)
        )
    if session.rect is None:
        return Verdict(
            False,
            "I could not find the whole board in the picture.",
            (
                Finding(
                    "board_not_found",
                    "I could not find the whole board in the picture.",
                    severity="uncertain",
                    suggestion="Hold the camera so the whole board fills the view, "
                    "in good light, and check again.",
                ),
            ),
        )
    if session.too_bright:
        return Verdict(
            False,
            "The picture is too bright to check the LEDs.",
            (
                Finding(
                    "too_bright",
                    "The board is so bright in the picture that I could not tell "
                    "whether an LED lit.",
                    severity="uncertain",
                    suggestion="Move the board out of direct light, or turn the "
                    "lamp away, and check again.",
                ),
            ),
        )

    lit = {p: g for p, g in session.glows.items() if g.status == "lit"}
    findings: list[Finding] = list(cannot)
    ok: list[int] = []
    explained: set[int] = set()  # lit pins accounted for by another pin's finding

    for pin, want in sorted(expected.items()):
        if pin not in pins_run:
            findings.append(_not_checked(pin, "it was not part of this run"))
            continue
        problem = _judge(pin, want, session, lit, expected)
        if problem is None:
            ok.append(pin)
            continue
        finding, elsewhere = problem
        findings.append(finding)
        if elsewhere is not None:
            explained.add(elsewhere)

    # Pins the lab does not use must be quiet too: anything happening there is
    # either an extra part or something we could not see clearly.
    for pin in sorted(set(pins_run) - set(expected)):
        glow = session.glows.get(pin)
        if pin in session.shorted:
            findings.append(_tied_to_ground(pin, (), severity="warning"))
        elif pin in session.moved:
            findings.append(_not_checked(pin, "its photos could not be lined up"))
        elif glow is not None and glow.status == "unclear" and pin not in session.calm:
            # A calm pin showed no LED, only the room's light changing as it did
            # on every quiet pin. Nothing should be on this pin, so that will do.
            findings.append(_not_checked(pin, glow.note or "the picture was unclear"))
        elif glow is not None and glow.status == "lit" and pin not in explained:
            findings.append(
                Finding(
                    "unexpected_led",
                    f"Pin {pin} lit {_a(glow.colour)} LED that this lab does not use.",
                    severity="warning",
                    suggestion="If that LED should not be there, take it out. If it "
                    "should, check which pin it is wired to.",
                    detail={"pin": pin, "glow": pin},
                )
            )

    if session.rect.confidence < 1.0 and not findings:
        findings.append(
            Finding(
                "imperfect_view",
                "The board was only partly clear in the picture.",
                severity="uncertain",
                suggestion="Check again with the whole board sharp and in view.",
            )
        )

    order = {"error": 0, "warning": 1, "uncertain": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    works = not findings
    return Verdict(
        works, _summary(findings), tuple(findings), tuple(ok), CAVEAT if works else None
    )


def _summary(findings: list[Finding]) -> str:
    if not findings:
        return (
            "Every LED lights from the right pin, so your wiring works. "
            "If something is still wrong, look at your program."
        )
    errors = sum(f.severity == "error" for f in findings)
    warnings = sum(f.severity == "warning" for f in findings)
    if errors:
        return f"Found {errors} problem{'s' if errors > 1 else ''} with the wiring."
    if warnings:
        return f"Found {warnings} thing{'s' if warnings > 1 else ''} to look at."
    return "Not sure yet. Check again."


def _judge(
    pin: int,
    want: Expected,
    session: Session,
    lit: dict[int, Glow],
    expected: dict[int, Expected],
) -> tuple[Finding, int | None] | None:
    """None if the pin did what the lab expects; else a finding, and the other
    lit pin that finding accounts for, if any."""
    if pin in session.shorted:
        return _tied_to_ground(pin, (want.component,)), None
    if pin in session.moved:
        return _not_checked(pin, "its photos could not be lined up"), None
    glow = session.glows.get(pin)
    if glow is None:
        return _not_checked(pin, "there were no pictures of it"), None
    if glow.status == "unclear":
        return _not_checked(pin, glow.note or "the picture was unclear"), None

    colour = want.colour
    led = _led(colour)
    elsewhere = _lit_elsewhere(pin, colour, lit, expected)
    moved_to = f" {_cap(led)} lit from pin {elsewhere} instead." if elsewhere else ""

    if glow.status == "dark":
        return (
            Finding(
                "led_does_not_light",
                f"{_cap(led)} should light when pin {pin} turns on, "
                f"but nothing lit.{moved_to}",
                components=(want.component,),
                suggestion=(
                    f"Connect {led} to pin {pin}, not pin {elsewhere}."
                    if elsewhere
                    else f"Check {led}: its long leg goes toward pin {pin}'s side, "
                    "and each leg shares a strip of five holes with the part next "
                    "to it. Check its resistor and its wire to ground too."
                ),
                detail={"pin": pin, "glow": elsewhere},
            ),
            elsewhere,
        )

    seen = glow.colour
    if _same_colour(colour, seen):
        return None
    if seen is None or seen == "unknown" or {seen, colour} <= _WARM:
        return (
            _not_checked(pin, f"its LED's colour was hard to tell ({seen} or {colour})"),
            None,
        )
    return (
        Finding(
            "wrong_led_on_pin",
            f"Pin {pin} should light {led}, but it lit {_a(seen)} one.{moved_to}",
            components=(want.component,),
            suggestion=(
                f"Connect {led} to pin {pin}, not pin {elsewhere}."
                if elsewhere
                else f"Use {_a(colour)} LED here, or check which LED is wired to "
                f"pin {pin}."
            ),
            detail={"pin": pin, "glow": pin, "seen": seen},
        ),
        elsewhere,
    )


def _same_colour(want: str | None, seen: str | None) -> bool:
    return want is None or want == seen


def _lit_elsewhere(
    pin: int, colour: str | None, lit: dict[int, Glow], expected: dict[int, Expected]
) -> int | None:
    """The one other pin that lit an LED of this colour, if exactly one did.

    A pin that lit exactly the LED the lab wants there is doing its own job, so
    it is never offered as where this pin's LED went.
    """
    if colour is None:
        return None
    others = [
        q
        for q, g in lit.items()
        if q != pin
        and g.colour == colour
        and not (q in expected and _same_colour(expected[q].colour, g.colour))
    ]
    return others[0] if len(others) == 1 else None


def _tied_to_ground(
    pin: int, components: tuple[str, ...], severity: str = "error"
) -> Finding:
    return Finding(
        "pin_tied_to_ground",
        f"Pin {pin} could not switch on: something is pulling it down to ground.",
        severity=severity,
        components=components,
        suggestion=f"Look for a wire from pin {pin} straight to a - rail or GND, or an "
        "LED with no resistor, and fix it.",
        detail={"pin": pin},
    )


def _not_checked(pin: int, why: str) -> Finding:
    return Finding(
        "not_checked",
        f"I could not check pin {pin}: {why}.",
        severity="uncertain",
        suggestion="Hold the board and the camera as steady as you can, in steady "
        "light, and check again.",
        detail={"pin": pin},
    )


def _led(colour: str | None) -> str:
    return f"the {colour} LED" if colour else "the LED"


def _cap(text: str) -> str:
    """Capitalise the first letter only: "the red LED" -> "The red LED"."""
    return text[:1].upper() + text[1:]


def _a(colour: str | None) -> str:
    if not colour or colour == "unknown":
        return "an"
    return f"an {colour}" if colour[0] in "aeiou" else f"a {colour}"
