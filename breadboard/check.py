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

from copy import copy, deepcopy
from dataclasses import dataclass, field

from breadboard.board import (
    N_COLS,
    PITCH_MM,
    ROWS,
    BoardError,
    hole_position,
    holes_of,
    is_rail,
    node_of,
    parse_hole,
    same_rail_different_segment,
)
from breadboard.graph import (
    CircuitGraph,
    Edge,
    Item,
    Part,
    build,
    equivalent,
    find_isomorphism,
    wires_joined,
)
from breadboard.netlist import Component, Netlist, Pin

POWER_PINS = {"5V", "3V", "VIN", "RAW"}
GROUND_PINS = {"GND"}

# Most actionable first. A specific "move this here" beats "something is extra",
# which beats a bare statement that the circuits differ.
KIND_RANK = {
    "uncertain_reading": 0,
    "wrong_connection": 1,
    "swapped_connections": 1,
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

    return _ranked(_said_once(findings))


def _said_once(findings: list[Finding]) -> list[Finding]:
    """Drop a finding that another says better.

    An LED with no resistor, and a resistor the lab needs that is not on the
    board, are one problem. Say it once, as the LED's: that names the place."""
    bare = sum(f.kind == "led_without_resistor" for f in findings)
    # Likewise a part with both legs in one strip, when a fix says where its leg
    # should go: the fix is the same thing, said better.
    moved = {c for f in findings if f.kind == "wrong_connection" for c in f.components}
    out = []
    for f in findings:
        if bare and f.kind == "missing_component" and f.detail.get("type") == "resistor":
            bare -= 1
            continue
        if f.kind == "shorted_component" and moved & set(f.components):
            continue
        out.append(f)
    return out


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
            first = n[led_ids[0]]
            out.append(
                Finding(
                    kind="led_without_resistor",
                    message=(
                        f"{_cap(_describe(first))} has no resistor with it, so too "
                        "much current will flow through it."
                    ),
                    components=led_ids,
                    holes=_holes(first),
                    scope=_scope(first),
                    suggestion="Add a 330 ohm resistor in line with it.",
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


# How many single changes the search will make in a row to reach the lab's
# circuit, and how many of them are shown at once. Three is about what a student
# can act on before checking again.
MAX_STEPS = 6
MAX_FIXES = 3


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
    return _repairs(student, reference)


def _repairs(student: Netlist, reference: Netlist) -> list[Finding]:
    """The changes that turn the student's circuit into the lab's.

    One change at a time. Each step takes the single change that puts most of
    the lab's circuit right (_best_change), makes it, and looks again, until
    the circuits match. One mistake takes one step, and gets the same answer a
    search for a single fix would give. Two or three separate mistakes take
    two or three: no one change fixes the whole circuit, but each puts one of
    the lab's links right, and that is a real repair whichever comes first.

    Fixes are passed on only when the whole run of them ends at the lab's
    circuit. A run that stalls short of it says nothing, however far it got: a
    step can raise the score and still be the wrong thing to do. At most
    MAX_FIXES are shown, the first ones made, with a note that more follow.
    """
    lab = wires_joined(build(reference)).collapsed()
    facts = _lab_facts(lab)
    working, steps, done = student, [], False
    while not done and len(steps) < MAX_STEPS:
        step = _best_change(working, lab, facts)
        if step is None:
            break
        finding, working, done = step
        steps.append(finding)
    if not done:
        # No run of changes was found that ends at the lab's circuit. The steps
        # taken so far each raised the score, but a step can raise it and still
        # be wrong (it may need undoing later), so none is passed on.
        return [
            Finding(
                kind="circuit_differs",
                message=(
                    "This circuit is not wired the way the lab expects, and I "
                    "could not work out what to change."
                ),
                suggestion="Compare your board against the lab diagram step by step.",
            )
        ]
    found = _parts_merged(_swaps_merged(steps), student, working)
    if len(found) <= MAX_FIXES:
        return found
    return [
        *found[:MAX_FIXES],
        # A caution, not another problem counted: it names nothing to fix.
        Finding(
            kind="circuit_differs",
            severity="warning",
            message="There is more to fix after these.",
            suggestion="Fix these, then check again.",
        ),
    ]


# One thing the lab's circuit has between named points: ("link", labels at one
# end, labels at the other, the link) or ("leg", the part, its leg, the labels
# of the point it sits on).
_Fact = tuple


def _lab_facts(lab: CircuitGraph) -> list[_Fact]:
    """What a student's circuit can be scored against, one item at a time.

    Each link of the lab's whose two ends carry Metro Mini pins, and each leg of
    a multi-leg part that sits on a named point. A link through an unnamed
    junction is left out: which of the student's strips is "the" junction is
    not given, so it cannot be judged on its own. A lab made only of those
    still gets a single fix, when one makes the whole circuit match.
    """
    facts: list[_Fact] = []
    for edge in lab.edges:
        a, b = lab.label_of(edge.a), lab.label_of(edge.b)
        if a and b:
            facts.append(("link", a, b, edge))
    for part in lab.parts:
        for leg, node in part.pins:
            if lab.label_of(node):
                facts.append(("leg", part, leg, lab.label_of(node)))
    return facts


def _facts_right(joined: CircuitGraph, g: CircuitGraph, facts: list[_Fact]) -> list[int]:
    """How much of each lab fact a circuit has, in points.

    A link that is whole scores two per part plus two, or one less if only an
    LED's way round differs. A broken one scores for its pieces still in place
    (_broken). A sensor's leg on the right named point scores two. joined is the
    circuit with wires made into joins; g is that with series runs merged.

    A part counts for one link only. Those of a whole link are that link's, so
    the resistor of a working LED is not taken for the one a broken LED lacks.
    """
    whole: dict[int, int] = {}
    claimed: set[str] = set()
    for i, f in enumerate(facts):
        if f[0] == "link":
            hit = _whole(g, f[1], f[2], f[3])
            if hit is not None:
                whole[i] = hit[0]
                claimed |= hit[1]
    # Built once here, not per fact: this runs for every trial circuit.
    links: _Links = {}
    for index, edge in enumerate(joined.edges):
        if len(edge.items) == 1 and edge.items[0].component_id not in claimed:
            links.setdefault(edge.a, []).append((index, edge.b, edge.items[0]))
            links.setdefault(edge.b, []).append((index, edge.a, edge.items[0].reversed()))
    named: dict[frozenset, list[str]] = {}
    for node, labels in joined.labels.items():
        if labels:
            named.setdefault(labels, []).append(node)
    out, used = [], frozenset()
    for i, f in enumerate(facts):
        if f[0] == "leg":
            out.append(2 * any(_leg_right(g, sp, f[1], f[2], f[3]) for sp in g.parts))
        elif i in whole:
            out.append(whole[i])
        else:
            points, used = _broken(links, named, f[1], f[2], f[3].items, used)
            out.append(points)
    return out


# For each point, the parts leading away from it: (the part's number, the point
# at its other end, the part as met going that way).
_Links = dict[str, list[tuple[int, str, Item]]]


def _whole(
    g: CircuitGraph, a: frozenset, b: frozenset, ref: Edge
) -> tuple[int, set[str]] | None:
    """A lab link the circuit has whole: its points, and the parts that make it."""
    size, found = len(ref.items), None
    for edge in g.edges:
        for e in (edge, edge.reversed()):
            if g.label_of(e.a) == a and g.label_of(e.b) == b:
                parts = {i.component_id for i in e.items}
                if e.matches(ref):
                    return 2 * size + 2, parts
                if found is None and Edge(e.a, e.b, _either_way(e.items)).matches(ref):
                    found = (2 * size + 1, parts)  # whole, but for an LED's way round
    return found


def _broken(
    links: _Links,
    named: dict[frozenset, list[str]],
    a: frozenset,
    b: frozenset,
    wanted: tuple[Item, ...],
    used: frozenset[int],
) -> tuple[int, frozenset[int]]:
    """The points for a lab link that is not whole, and the parts counted.

    Two for each of its parts that hangs, one after another, off the right
    named point at either end; one if that part is an LED the wrong way round.
    So a broken chain with its pieces in place scores for each piece, and
    mending either of two breaks in it counts as a step, as does turning the
    LED, before or after. Parts already counted for another link are left out.
    """
    best, taken = 0, used
    for start in named.get(a, ()):
        for points, rest, near in _runs(links, start, wanted, used, False):
            for end in named.get(b, ()) or [None]:
                far = (
                    _runs(links, end, rest, near, True)
                    if end is not None
                    else [(0, rest, near)]
                )
                for more, _, both in far:
                    if points + more > best:
                        best, taken = points + more, both
    return best, taken


def _either_way(items: tuple[Item, ...]) -> tuple[Item, ...]:
    """The same parts with every LED's way round left open."""
    return tuple(Item(i.type, i.attrs, 0, i.component_id) for i in items)


def _runs(
    links: _Links,
    node: str,
    wanted: tuple[Item, ...],
    used: frozenset[int],
    backwards: bool,
    passed: frozenset[str] = frozenset(),
):
    """Every run of the wanted parts, each used once, leading away from this
    point: (its points, the wanted parts left over, the parts used). The empty
    run is one of them. backwards is for a run from the link's far end, where a
    part the right way round is met facing the other way. A run never comes back
    to a point it has passed, so a part with both legs on one point, or two
    parts side by side, earn nothing: they lead nowhere."""
    yield 0, wanted, used
    passed = passed | {node}
    for index, other, item in links.get(node, ()):
        if index in used or other in passed:
            continue
        for i, ref in enumerate(wanted):
            if _either_way((item,))[0].compatible_with(ref):
                facing = -ref.polarity if backwards else ref.polarity
                points = 1 if item.polarity and facing and item.polarity != facing else 2
                rest = wanted[:i] + wanted[i + 1 :]
                for more, left, taken in _runs(
                    links, other, rest, used | {index}, backwards, passed
                ):
                    yield points + more, left, taken
                break


def _leg_right(
    g: CircuitGraph, mine: Part, ref: Part, leg: str, labels: frozenset
) -> bool:
    if mine.type != ref.type:
        return False
    have = dict(mine.attrs)
    if not all(have.get(k, v) == v for k, v in ref.attrs):
        return False
    return g.label_of(dict(mine.pins).get(leg, "")) == labels


def _best_change(
    n: Netlist, lab: CircuitGraph, facts: list[_Fact]
) -> tuple[Finding, Netlist, bool] | None:
    """The single change that does most good: its finding, the circuit after
    it, and whether that is now the lab's circuit. None if no change helps.

    A change that makes the circuits match outright always wins. Otherwise the
    one that puts most of the lab's circuit right, and it must put more right
    than it puts wrong. Ties go by kind (_changes), then by each kind's own
    order. Every working change is weighed before one is chosen: taking the
    first found would blame whichever component happened to come first.
    """
    joined = wires_joined(build(n))
    before = _facts_right(joined, joined.collapsed(), facts)
    best = None
    for kind, trial, tie, describe in _changes(n):
        joined = wires_joined(build(trial))
        g = joined.collapsed()
        done = find_isomorphism(g, lab) is not None
        after = _facts_right(joined, g, facts)
        gain = sum(after) - sum(before)
        if not done and gain <= 0:
            continue
        key = (not done, -gain, kind, *tie())
        if best is None or key < best[0]:
            best = (key, trial, describe, done, after)
    if best is None:
        return None
    _, trial, describe, done, after = best
    finding = describe()
    finding.detail["pins"] = _pins_put_right(facts, before, after)
    if finding.kind == "wrong_connection":
        # Carry on from the hole the student is sent to, not the one tried, so
        # a later fix is not sent to the same hole.
        comp = n[finding.components[0]]
        moved = {**comp.pins, finding.detail["leg"]: Pin(finding.detail["to"])}
        trial = _with_legs(n, comp, moved)
    return finding, trial, done


def _pins_put_right(facts: list[_Fact], before: list[int], after: list[int]) -> list[str]:
    """The Metro Mini pins (D4, A0) whose links a change put right."""
    names: set[str] = set()
    for fact, was, now in zip(facts, before, after, strict=True):
        if now > was:
            labels = fact[1] | fact[2] if fact[0] == "link" else fact[3]
            names |= {x for x in labels if x[:1] in "DA" and x[1:].isdigit()}
    return sorted(names, key=lambda x: (x[0], int(x[1:])))


def _changes(n: Netlist):
    """Every single change worth trying.

    Yields (kind, the circuit after it, a function giving its place among
    others of its kind, a function giving its Finding). Kinds, most likely
    story first: 0 take out a wire, 1 move one leg, 2 turn a two-leg part
    around, 3 swap two legs of a part with more.

    Moves are ordered among themselves like this. A move that leaves a wire
    joining nothing comes last: when a wire's end has strayed, carrying the
    part's own leg all the way to where the wire should have gone also makes
    the circuits match, but it strands the wire and stretches the part across
    the board. Then the shortest move, since one hole off is the commonest
    slip; then a wire's end before a part's leg, since a wire is the easier
    thing to move and moving it leaves the part as it sits. Nothing says which
    leg the student put in the wrong hole: placement is free, so the lab has no
    places to compare with.
    """
    for index, wire in enumerate(n.of_type("wire")):
        trial = copy(n)
        trial.components = [c for c in n.components if c is not wire]
        yield (
            0,
            trial,
            lambda index=index: (index,),
            lambda wire=wire: _extra_wire_finding(wire),
        )

    targets = _candidate_holes(n)
    for comp in n:
        if comp.type == "mcu":
            continue
        for leg, pin in comp.pins.items():
            if not pin.placed:
                continue  # a leg on a net is in no hole, so it cannot be moved
            for hole in targets:
                if hole == pin.hole or not _can_reach(comp, leg, hole):
                    continue
                trial = _with_legs(n, comp, {**comp.pins, leg: Pin(hole)})
                yield (
                    1,
                    trial,
                    lambda trial=trial, comp=comp, leg=leg, pin=pin, hole=hole: (
                        _stray_wires(trial),
                        _hole_distance(pin.hole, hole),
                        comp.type != "wire",
                        comp.id,
                        leg,
                    ),
                    lambda comp=comp, leg=leg, pin=pin, hole=hole: _move_finding(
                        n, comp, leg, pin, hole
                    ),
                )

    for index, comp in enumerate(n.components):
        if comp.polarised and comp.is_two_terminal:
            first, second = list(comp.pins)
            turned = {first: comp.pins[second], second: comp.pins[first]}
            trial = _with_legs(n, comp, turned)
            yield (
                2,
                trial,
                lambda index=index: (index,),
                lambda comp=comp: _turned_finding(comp),
            )

    # Swapping two legs of a sensor takes two moves, so the moves above cannot
    # reach it in one. It is also the error that matters most on a sensor:
    # reversing + and - can destroy the part rather than merely stop it working.
    for index, comp in enumerate(n.components):
        if comp.type == "mcu" or comp.is_two_terminal:
            continue
        names = sorted(comp.pins)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                swapped = {**comp.pins, a: comp.pins[b], b: comp.pins[a]}
                trial = _with_legs(n, comp, swapped)
                yield (
                    3,
                    trial,
                    lambda index=index, a=a, b=b: (index, a, b),
                    lambda comp=comp, a=a, b=b: _legs_swapped_finding(comp, a, b),
                )


# How far apart a part's two legs can sit, in hole pitches. An LED's legs are
# short and stiff. A resistor's are long, and reach well across the centre gap.
REACH = {"led": 4.0, "resistor": 20.0}
# A leg may always move this much further from its partner than it is now. The
# example circuits are drawn with parts spanning more than a real one could.
SLACK = 2.5


def _can_reach(comp: Component, leg: str, hole: str) -> bool:
    """Whether a student could push this leg into that hole, the part's other
    legs staying where they are. A fix nobody can make is not a fix: an LED
    cannot be stretched across the board to meet a wire that strayed, and a
    sensor's legs are one rigid row, so none moves alone. A wire reaches
    anywhere."""
    if comp.type == "wire":
        return True
    if not comp.is_two_terminal:
        return False
    other = next(p for name, p in comp.pins.items() if name != leg)
    if not other.placed:
        return True
    if node_of(hole) == other.node:
        return False  # both legs in one strip: the part would be skipped past
    now = _hole_distance(comp.pins[leg].hole, other.hole)
    limit = max(REACH.get(comp.type, 4.0) * PITCH_MM, now + SLACK * PITCH_MM)
    return _hole_distance(hole, other.hole) <= limit


def _with_legs(n: Netlist, comp: Component, pins: dict[str, Pin]) -> Netlist:
    """The circuit with one part's legs changed. Every other part is shared with
    n, not copied: thousands of these are made, and nothing changes them."""
    changed = copy(comp)
    changed.pins = pins
    trial = copy(n)
    trial.components = [changed if c is comp else c for c in n.components]
    return trial


def _extra_wire_finding(wire: Component) -> Finding:
    return Finding(
        kind="extra_component",
        message=f"There is an extra {_describe(wire).removeprefix('the ')} on "
        "the board that the lab does not use.",
        components=(wire.id,),
        holes=_holes(wire),
        scope=_scope(wire),
        suggestion="Take it off the board.",
        detail={"type": wire.type},
    )


def _move_finding(n: Netlist, comp: Component, leg: str, pin: Pin, hole: str) -> Finding:
    to = _free_hole(n, pin.hole, hole)
    return Finding(
        kind="wrong_connection",
        message=(
            f"One leg of {_describe(comp)} is in the wrong place, so this "
            f"part of the circuit is not joined up."
        ),
        components=(comp.id,),
        holes=(pin.hole,),
        scope=_scope(comp),
        suggestion=f"Move it from {pin.hole} to {to}.",
        detail={"leg": leg, "from": pin.hole, "to": to, "type": comp.type},
    )


def _turned_finding(comp: Component) -> Finding:
    return Finding(
        kind="reversed_polarity",
        message=(
            f"{_cap(_describe(comp))} is in the wrong way round. LEDs only let "
            "electricity through one way."
        ),
        components=(comp.id,),
        holes=_holes(comp),
        scope=_scope(comp),
        suggestion="Turn it around so its longer leg is on the side "
        "the electricity comes from.",
    )


def _legs_swapped_finding(comp: Component, a: str, b: str) -> Finding:
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


def _swaps_merged(steps: list[Finding]) -> list[Finding]:
    """Two moves that send each leg to where the other was are one fix: swap
    them. Said as one finding."""
    out: list[Finding] = []
    used: set[int] = set()
    for i, f in enumerate(steps):
        if i in used:
            continue
        j = next(
            (
                j
                for j in range(i + 1, len(steps))
                if j not in used and _exchange(f, steps[j])
            ),
            None,
        )
        if j is None:
            out.append(f)
            continue
        used.add(j)
        g = steps[j]
        wires = f.detail["type"] == g.detail["type"] == "wire"
        what = "ends" if wires else "legs"
        here, there = sorted(
            (f.detail["from"], g.detail["from"]), key=lambda h: (parse_hole(h)[1], h)
        )
        pins = sorted(
            {*f.detail["pins"], *g.detail["pins"]}, key=lambda x: (x[0], int(x[1:]))
        )
        out.append(
            Finding(
                kind="swapped_connections",
                # Said as what to do, which is true whichever pair of things
                # the student mixed up (two wires, or the two parts they go to).
                message=f"Two {'wires' if wires else 'legs'} need to swap places.",
                components=(*f.components, *g.components),
                holes=(here, there),
                scope=f.scope,
                suggestion=f"Swap the {what} in {here} and {there}.",
                detail={"pins": pins, "swapped": [here, there]},
            )
        )
    return out


def _parts_merged(
    found: list[Finding], student: Netlist, fixed: Netlist
) -> list[Finding]:
    """Two fixes to one part (a leg moved and the part turned, or both legs
    moved) are one thing to do: put it in these holes. Said once, with where
    each leg ends up."""
    out: list[Finding] = []
    for f in found:
        again = [
            g
            for g in found
            if g.components == f.components
            and len(f.components) == 1
            and g.kind in ("wrong_connection", "reversed_polarity")
        ]
        comp = student[f.components[0]] if len(again) > 1 else None
        if comp is None or comp.type == "wire" or not comp.is_two_terminal:
            out.append(f)
            continue
        if f is not again[0]:
            continue  # said with the first
        legs = {name: pin.hole for name, pin in fixed[comp.id].pins.items()}
        first, second = legs.values()
        pins = sorted(
            {p for g in again for p in g.detail.get("pins", ())}, key=_pin_order
        )
        out.append(
            Finding(
                kind="wrong_connection",
                message=f"{_cap(_describe(comp))} is in the wrong holes.",
                components=(comp.id,),
                holes=_holes(comp),
                scope=_scope(comp),
                suggestion=f"Put its longer leg in {legs['anode']} and its shorter leg "
                f"in {legs['cathode']}."
                if comp.type == "led"
                else f"Put its legs in {first} and {second}.",
                detail={"legs": legs, "pins": pins, "type": comp.type},
            )
        )
    return out


def _pin_order(name: str) -> tuple[str, int]:
    return name[0], int(name[1:])


def _exchange(f: Finding, g: Finding) -> bool:
    if not (f.kind == g.kind == "wrong_connection"):
        return False
    if "from" not in f.detail or "from" not in g.detail:
        return False
    try:
        return node_of(f.detail["from"]) == node_of(g.detail["to"]) and node_of(
            g.detail["from"]
        ) == node_of(f.detail["to"])
    except BoardError:
        return False


def _free_hole(n: Netlist, origin: str, target: str) -> str:
    """The hole to send a leg to: one in the target's strip with no leg in it.

    The leg's own row if that hole is free, which reads most easily ("from a33
    to a32"); otherwise the nearest free one. The target itself if the whole
    strip is taken.
    """
    try:
        row, _ = parse_hole(origin)
        strip = holes_of(node_of(target))
    except BoardError:
        return target
    taken = {p.hole.strip().lower() for c in n for p in c.pins.values() if p.placed}
    taken.discard(origin.strip().lower())
    free = [h for h in strip if h not in taken]
    if not free:
        return target
    return min(
        free, key=lambda h: (parse_hole(h)[0] != row, _hole_distance(origin, h), h)
    )


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
                detail={"type": c.type},
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
                suggestion=f"Add {_describe(c)}.",
                detail={"type": c.type},
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
    # Rounded: the same one-hole distance comes out a hair different from column
    # to column, and that hair must not decide which fix is named.
    return round(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, 3)


# How the worksheets name a sensor's legs. Students have seen these on the data
# sheet; they have never seen "vcc".
PIN_LABELS = {"vcc": "+", "gnd": "-", "out": "OUT", "trig": "trigger", "echo": "echo"}


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


def _cap(text: str) -> str:
    """Capitalise the first letter only: "the red LED" -> "The red LED". Not
    .capitalize(), which would lower-case "LED"."""
    return text[:1].upper() + text[1:]


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
