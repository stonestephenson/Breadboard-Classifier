"""Diagnosis: what is wrong with this circuit, and what should the student do.

Deliberately deterministic. Connectivity reasoning has to be exact, and we can
compute it exactly, so no language model decides what is broken -- one may later
rephrase what this module concludes, but never replace it.

The order of work encodes the project's central caution. Before reporting any
error we ask whether a *misreading* explains the difference: if swapping one
low-confidence endpoint for the alternative perception offered would make the
circuit correct, we report doubt and ask for a better look instead of accusing a
student of a mistake they did not make. For a 13-year-old, a confident wrong
answer carrying our authority is the worst thing this system can do.

Findings carry a `scope`: "lab" for parts the student was asked to place,
"baseline" for the factory-wired parts of the BasicBoard. Both are reported --
a knocked-loose factory LED is a real problem -- but the caller can phrase them
differently.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from breadboard.board import (
    N_COLS,
    ROWS,
    BoardError,
    hole_position,
    is_rail,
    node_of,
    parse_hole,
    same_rail_different_segment,
)
from breadboard.graph import build, equivalent
from breadboard.netlist import Component, Netlist, Pin

POWER_PINS = {"5V", "3V", "VIN", "RAW"}
GROUND_PINS = {"GND"}

# Most actionable first. A specific "move this here" beats "something is extra",
# which beats a bare statement that the circuits differ.
KIND_RANK = {
    "uncertain_reading": 0,
    "wrong_connection": 1,
    "reversed_polarity": 2,
    "shorted_component": 3,
    "short_circuit": 4,
    "led_without_resistor": 5,
    "dead_rail_segment": 6,
    "extra_component": 7,
    "missing_component": 8,
    "circuit_differs": 9,
}


@dataclass
class Finding:
    kind: str
    message: str
    severity: str = "error"  # error | uncertain | warning
    components: tuple[str, ...] = ()
    holes: tuple[str, ...] = ()
    suggestion: str | None = None
    scope: str = "lab"  # lab | baseline
    detail: dict = field(default_factory=dict)

    def rank(self) -> int:
        return KIND_RANK.get(self.kind, 99)


def check(student: Netlist, reference: Netlist | None = None) -> list[Finding]:
    """Diagnose a student's circuit, optionally against the lab's intended one.

    Without a reference only the rules that need no lab are applied -- shorts,
    an LED with no resistor, the dead half of a split rail. With one, the
    student's circuit is compared to it and the smallest repair is reported.
    """
    findings = _sanity_rules(student)

    if reference is not None:
        if equivalent(student, reference):
            return _ranked(findings)
        uncertain = _explained_by_misreading(student, reference)
        if uncertain is not None:
            # Suppress everything else: if we may have misread the board, we
            # have no business reporting what is on it.
            return [uncertain]
        findings.extend(_diff(student, reference))

    return _ranked(findings)


def _ranked(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (f.rank(), f.components, f.kind))


# --------------------------------------------------------------------------
# Rules that need no reference circuit
# --------------------------------------------------------------------------


def _sanity_rules(n: Netlist) -> list[Finding]:
    out: list[Finding] = []
    out += _shorted_components(n)
    out += _power_to_ground_shorts(n)
    out += _leds_without_resistors(n)
    out += _dead_rail_segments(n)
    return out


def _shorted_components(n: Netlist) -> list[Finding]:
    out = []
    for c in n:
        if not c.is_two_terminal:
            continue
        p, q = c.ordered_pins()
        if p.node == q.node:
            out.append(
                Finding(
                    kind="shorted_component",
                    message=(
                        f"Both legs of {c.id} are in the same row, so "
                        f"electricity skips straight past it."
                    ),
                    components=(c.id,),
                    holes=(p.hole, q.hole),
                    scope=_scope(c),
                    suggestion="Move one leg to a different column.",
                )
            )
    return out


def _power_to_ground_shorts(n: Netlist) -> list[Finding]:
    power, ground = _supply_nodes(n)
    for edge in build(n).collapsed().edges:
        # A collapsed chain of nothing but wires joining supply to ground is a
        # dead short: full current, no component to limit it.
        if all(i.type == "wire" for i in edge.items) and (
            (edge.a in power and edge.b in ground)
            or (edge.b in power and edge.a in ground)
        ):
            ids = tuple(i.component_id for i in edge.items if i.component_id)
            return [
                Finding(
                    kind="short_circuit",
                    message=(
                        "Power is wired directly to ground with nothing in "
                        "between. Disconnect this before plugging in again."
                    ),
                    components=ids,
                    severity="error",
                    suggestion="Remove the wire joining the power and ground rails.",
                )
            ]
    return []


def _leds_without_resistors(n: Netlist) -> list[Finding]:
    power, ground = _supply_nodes(n)
    driven = power | _driveable_nodes(n)
    out = []
    for edge in build(n).collapsed().edges:
        types = [i.type for i in edge.items]
        if "led" not in types or "resistor" in types:
            continue
        ends = {edge.a, edge.b}
        if ends & driven and ends & ground:
            led_ids = tuple(i.component_id for i in edge.items if i.type == "led")
            out.append(
                Finding(
                    kind="led_without_resistor",
                    message=(
                        "This LED has no resistor with it, so too much current "
                        "will flow through it."
                    ),
                    components=led_ids,
                    suggestion="Add a 330 ohm resistor in line with the LED.",
                )
            )
    return out


def _dead_rail_segments(n: Netlist) -> list[Finding]:
    """A rail half that nothing powers, while its other half is powered.

    Only on a board whose power rails are two separate runs (board.RAILS_SPLIT;
    the kit's own board's are not). There the paint runs the whole length, so a
    connection into the wrong half looks perfectly correct and simply does
    nothing -- one of the few errors that is genuinely invisible.
    """
    power, ground = _supply_nodes(n)
    supplied = power | ground
    used = n.nodes_used()
    out = []
    for node in sorted(used):
        if not is_rail(node) or node in supplied:
            continue
        sibling = next(
            (s for s in supplied if same_rail_different_segment(node, s)), None
        )
        if sibling is None:
            continue
        holes = tuple(p.hole for c in n for p in c.pins.values() if p.node == node)
        out.append(
            Finding(
                kind="dead_rail_segment",
                message=(
                    "This part of the power strip is not connected to power. "
                    "The strip is split in the middle, so the two halves are "
                    "separate even though the line looks continuous."
                ),
                holes=holes,
                suggestion="Move it to the same half of the strip as the power wire, "
                "or add a wire bridging the two halves.",
            )
        )
    return out


def _supply_nodes(n: Netlist) -> tuple[set[str], set[str]]:
    power = {
        p.node
        for c in n.of_type("mcu")
        for name, p in c.pins.items()
        if name in POWER_PINS
    }
    ground = {
        p.node
        for c in n.of_type("mcu")
        for name, p in c.pins.items()
        if name in GROUND_PINS
    }
    if not (power or ground):
        return set(), set()
    # Anything wired straight to a supply pin is part of that supply.
    for _ in range(4):  # a few passes is plenty for these circuit sizes
        for c in n:
            if c.type != "wire":
                continue
            a, b = (p.node for p in c.ordered_pins())
            for group in (power, ground):
                if a in group or b in group:
                    group.update({a, b})
    return power, ground


def _driveable_nodes(n: Netlist) -> set[str]:
    mcu = n.mcu()
    if mcu is None:
        return set()
    return {p.node for name, p in mcu.pins.items() if name.startswith(("D", "A"))}


# --------------------------------------------------------------------------
# Comparison against the lab's intended circuit
# --------------------------------------------------------------------------


def _explained_by_misreading(student: Netlist, reference: Netlist) -> Finding | None:
    """Would swapping one uncertain endpoint for its alternative fix this?

    Only single substitutions are tried. Two independent misreadings in one
    photo is far less likely than one, and trying every combination would let
    the checker explain away real errors.
    """
    for comp in student:
        for pin_name, pin in comp.pins.items():
            if pin.confidence >= 1.0 or not pin.alternatives:
                continue
            for alt in pin.alternatives:
                trial = deepcopy(student)
                trial[comp.id].pins[pin_name] = Pin(alt)
                if equivalent(trial, reference):
                    return Finding(
                        kind="uncertain_reading",
                        severity="uncertain",
                        message=(
                            "I could not see this part of the board clearly "
                            "enough to be sure. Try another photo with this "
                            "area in better view."
                        ),
                        components=(comp.id,),
                        holes=(pin.hole,),
                        scope=_scope(comp),
                        detail={
                            "read_as": pin.hole,
                            "might_be": alt,
                            "confidence": pin.confidence,
                        },
                    )
    return None


def _diff(student: Netlist, reference: Netlist) -> list[Finding]:
    inventory = _inventory_diff(student, reference)
    if inventory:
        return inventory

    # Wires are joins, not parts, so a build may use more or fewer than the lab's
    # drawing. But when the circuits differ, a missing or extra wire is the
    # likelier story than a correctly placed part being in the wrong hole.
    missing = _missing_wires(student, reference)
    if missing:
        return missing
    extra = _extra_wire(student, reference)
    if extra is not None:
        return [extra]

    repair = _single_pin_repair(student, reference)
    if repair is not None:
        return [repair]

    polarity = _polarity_repair(student, reference)
    if polarity is not None:
        return [polarity]

    swap = _pin_swap_repair(student, reference)
    if swap is not None:
        return [swap]

    return [
        Finding(
            kind="circuit_differs",
            message=(
                "This circuit is not wired the way the lab expects, and I "
                "could not work out a single change that would fix it."
            ),
            suggestion="Compare your board against the lab diagram step by step.",
        )
    ]


def _signature(c: Component) -> tuple:
    from breadboard.graph import SIGNIFICANT_ATTRS

    keys = SIGNIFICANT_ATTRS.get(c.type, ())
    return (c.type, tuple(sorted((k, c.attrs[k]) for k in keys if k in c.attrs)))


def _inventory_diff(student: Netlist, reference: Netlist) -> list[Finding]:
    """Parts present in one circuit and not the other, ignoring where they are.

    Covers sensors as well as two-terminal parts. The Metro Mini is exempt, since
    it is the fixed anchor rather than something a student adds, and so are
    wires: they are joins, not parts, and a build may need more or fewer of them
    than the lab's own drawing (graph.equivalent compares them as joins).
    """
    s = [c for c in student if c.type not in ("mcu", "wire")]
    r = [c for c in reference if c.type not in ("mcu", "wire")]
    pool = list(r)
    extra = []
    for c in s:
        match = next((x for x in pool if _signature(x) == _signature(c)), None)
        if match is None:
            extra.append(c)
        else:
            pool.remove(match)

    out = []
    for c in extra:
        out.append(
            Finding(
                kind="extra_component",
                message=f"There is an extra {_describe(c).removeprefix('the ')} on "
                "the board that the lab does not use.",
                components=(c.id,),
                holes=_holes(c),
                scope=_scope(c),
                suggestion="Take it off the board.",
            )
        )
    for c in pool:
        out.append(
            Finding(
                kind="missing_component",
                message=f"The lab needs a {_describe(c).removeprefix('the ')} that "
                "is not on the board.",
                components=(c.id,),
                scope=_scope(c),
                suggestion=f"Add the {_describe(c)}.",
            )
        )
    return out


def _missing_wires(student: Netlist, reference: Netlist) -> list[Finding]:
    """The lab's wires beyond as many as the build has, when it has fewer."""
    have = len(student.of_type("wire"))
    need = reference.of_type("wire")
    return [
        Finding(
            kind="missing_component",
            message="The lab needs a wire that is not on the board.",
            components=(c.id,),
            scope=_scope(c),
            suggestion="Add the wire.",
        )
        for c in need[have:]
    ]


def _extra_wire(student: Netlist, reference: Netlist) -> Finding | None:
    """A wire whose removal alone makes the circuit right."""
    for wire in student.of_type("wire"):
        without = deepcopy(student)
        without.components = [c for c in without.components if c.id != wire.id]
        if equivalent(without, reference):
            return Finding(
                kind="extra_component",
                message=f"There is an extra {_describe(wire).removeprefix('the ')} on "
                "the board that the lab does not use.",
                components=(wire.id,),
                holes=_holes(wire),
                scope=_scope(wire),
                suggestion="Take it off the board.",
            )
    return None


def _single_pin_repair(student: Netlist, reference: Netlist) -> Finding | None:
    """Find the endpoint that, moved, makes the circuit correct.

    Several different single moves often fix the same break, and they are not
    equally useful to a student. With a wire ending in a33 and an LED's leg in
    a32, moving either one closes the gap. Nothing says which of them the
    student put in the wrong hole: placement is free, so the lab has no places
    to compare with (where its own file puts things means nothing, and is never
    looked at).

    So every working repair is collected and then ranked. A repair that leaves
    a wire joining nothing comes last: when a wire's end has strayed, carrying
    the part's own leg all the way to where the wire should have gone also makes
    the circuits match, but it strands the wire and stretches the part across
    the board. Then the shortest move, since one hole off is the commonest slip;
    then a wire's end before a part's leg, since a wire is the easier thing to
    move and moving it leaves the part as it sits. Returning the first repair
    found instead would blame whichever component happened to come first in
    the list.
    """
    targets = _candidate_holes(student)

    repairs = []
    for comp in student:
        if comp.type == "mcu":
            continue
        for pin_name, pin in comp.pins.items():
            if not pin.placed:
                continue  # a leg on a net is in no hole, so it cannot be moved
            for hole in targets:
                if hole == pin.hole:
                    continue
                trial = deepcopy(student)
                trial[comp.id].pins[pin_name] = Pin(hole)
                if equivalent(trial, reference):
                    repairs.append((comp, pin_name, pin, hole, _stray_wires(trial)))

    if not repairs:
        return None

    def rank(r):
        comp, pin_name, pin, hole, stray = r
        return (
            stray,
            _hole_distance(pin.hole, hole),
            comp.type != "wire",
            comp.id,
            pin_name,
        )

    comp, pin_name, pin, hole, _ = min(repairs, key=rank)
    nice = _nearest_hole_in_same_row(pin.hole, hole)
    return Finding(
        kind="wrong_connection",
        message=(
            f"One leg of {_describe(comp)} is in the wrong place, so this "
            f"part of the circuit is not joined up."
        ),
        components=(comp.id,),
        holes=(pin.hole,),
        scope=_scope(comp),
        suggestion=f"Move it from {pin.hole} to {nice}.",
        detail={"leg": pin_name, "from": pin.hole, "to": nice},
    )


def _stray_wires(n: Netlist) -> int:
    """How many wires join nothing: no part's leg is in either end's strip, nor
    in any strip wired to them."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    wires = n.of_type("wire")
    for wire in wires:
        a, b = (p.node for p in wire.ordered_pins())
        parent[find(a)] = find(b)
    held = {find(p.node) for c in n if c.type != "wire" for p in c.pins.values()}
    return sum(find(wire.ordered_pins()[0].node) not in held for wire in wires)


def _hole_distance(a: str, b: str) -> float:
    try:
        (ax, ay), (bx, by) = hole_position(a), hole_position(b)
    except BoardError:
        return float("inf")
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _polarity_repair(student: Netlist, reference: Netlist) -> Finding | None:
    for comp in student:
        if not (comp.polarised and comp.is_two_terminal):
            continue
        trial = deepcopy(student)
        names = list(comp.pins)
        a, b = comp.pins[names[0]], comp.pins[names[1]]
        trial[comp.id].pins = {names[0]: b, names[1]: a}
        if equivalent(trial, reference):
            described = _describe(comp)
            return Finding(
                kind="reversed_polarity",
                # Not .capitalize(): that would lower-case "LED" in the rest of
                # the phrase.
                message=(
                    f"{described[0].upper()}{described[1:]} is in the wrong "
                    f"way round. LEDs only let electricity through one way."
                ),
                components=(comp.id,),
                holes=_holes(comp),
                scope=_scope(comp),
                suggestion="Turn it around so its longer leg is on the side "
                "the electricity comes from.",
            )
    return None


# How the worksheets name a sensor's legs. Students have seen these on the data
# sheet; they have never seen "vcc".
PIN_LABELS = {"vcc": "+", "gnd": "-", "out": "OUT", "trig": "trigger", "echo": "echo"}


def _pin_swap_repair(student: Netlist, reference: Netlist) -> Finding | None:
    """Two legs of a multi-terminal part exchanged.

    Needs its own rule because swapping takes *two* moves, so the single-move
    search can never find it. It is also the error that matters most on a
    sensor: reversing + and - can destroy the part rather than merely stop it
    working.
    """
    for comp in student:
        if comp.type == "mcu" or comp.is_two_terminal:
            continue
        names = sorted(comp.pins)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                trial = deepcopy(student)
                pins = trial[comp.id].pins
                pins[a], pins[b] = pins[b], pins[a]
                if not equivalent(trial, reference):
                    continue
                la, lb = PIN_LABELS.get(a, a), PIN_LABELS.get(b, b)
                return Finding(
                    kind="reversed_polarity",
                    message=(
                        f"The {la} and {lb} legs of {_describe(comp)} are "
                        f"swapped, so it is wired the wrong way round."
                    ),
                    components=(comp.id,),
                    holes=(comp.pins[a].hole, comp.pins[b].hole),
                    scope=_scope(comp),
                    suggestion=(
                        f"Swap those two wires: the {la} leg should go where "
                        f"the {lb} leg is now, and the other way round."
                    ),
                    detail={"swapped": [a, b]},
                )
    return None


def _candidate_holes(student: Netlist) -> list[str]:
    """Holes worth trying as a repair target.

    Every hole the student's circuit already uses, plus their immediate
    neighbours -- because the error we most expect is an endpoint one column
    off. A repair has to join the leg to something on the board, so nowhere
    else can help. The lab's own file adds nothing: it has no places.
    """
    holes = {p.hole for c in student for p in c.pins.values() if p.placed}
    for hole in list(holes):
        where, index = parse_hole(hole)  # every hole in a netlist is already valid
        if where in ROWS:
            for delta in (-1, 1):
                if 1 <= index + delta <= N_COLS:
                    holes.add(f"{where}{index + delta}")
    return sorted(holes)


def _nearest_hole_in_same_row(origin: str, target: str) -> str:
    """Express a repair in the student's own row where we can.

    "Move it from a21 to a20" is easier to follow than a jump to some other row
    that happens to be electrically identical.
    """
    try:
        row, _ = parse_hole(origin)
        _, col = parse_hole(target)
    except BoardError:
        return target
    if row in ROWS and node_of(f"{row}{col}") == node_of(target):
        return f"{row}{col}"
    return target


def _describe(c: Component) -> str:
    if c.type == "led":
        colour = c.attrs.get("color")
        return f"the {colour} LED" if colour else "the LED"
    if c.type == "resistor":
        ohms = c.attrs.get("ohms")
        return f"the {ohms} ohm resistor" if ohms else "the resistor"
    if c.type == "wire":
        # Colour cannot be trusted to infer a wire's *role* -- the curriculum
        # team confirmed students do not follow the conventions, which is why
        # graph.py ignores it for equivalence. But it is still the best way to
        # point at which physical wire to move when three of them are in play.
        colour = c.attrs.get("color")
        return f"the {colour} wire" if colour else "the wire"
    if c.type in ("sensor3", "sensor4"):
        kind = c.attrs.get("kind")
        return f"the {kind} sensor" if kind else "the sensor"
    return f"the {c.type}"


def _holes(c: Component) -> tuple[str, ...]:
    return tuple(p.hole for p in c.pins.values())


def _scope(c: Component) -> str:
    return "baseline" if c.origin == "factory" else "lab"
