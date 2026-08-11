"""Circuit topology, and deciding when two circuits are the same one.

The problem this solves: the curriculum cannot tell students which holes to use,
so two builds of the same lab may share no positions at all. Worse, several
genuinely different wirings are equally correct -- a resistor works on either
side of an LED. Comparing netlists directly would flag working circuits as
broken, which is the failure this system can least afford.

So comparison happens on an abstraction, in two steps:

1. **Anonymise placement.** Breadboard strips become unlabelled vertices; only
   the Metro Mini's pins carry names. Matching then means finding a bijection
   between anonymous strips while holding D2 to D2. Placement freedom costs
   nothing, and the named pins keep the search tightly anchored -- this is not
   open graph isomorphism.

2. **Collapse series chains.** A run of two-terminal parts through nodes that
   nothing else touches is electrically one thing, and its internal order does
   not matter. Collapsing it to a single edge carrying an unordered bag of parts
   makes resistor-then-LED equal LED-then-resistor, while *direction* is kept so
   a reversed LED still fails.

Attributes are compared only where the reference specifies them: an author who
omits an LED colour means "any colour". Wire colour is never significant --
the curriculum team confirmed colours are conventions students do not follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

from breadboard.netlist import Component, Netlist

# Attributes that participate in equivalence, per component type. Anything not
# listed here is descriptive only and never makes two circuits differ.
SIGNIFICANT_ATTRS: dict[str, tuple[str, ...]] = {
    "led": ("color",),
    "resistor": ("ohms",),
    "wire": (),
    "button": (),
}


@dataclass(frozen=True)
class Item:
    """One component inside a (possibly collapsed) chain.

    `polarity` is +1 when the part's first pin faces the edge's `a` end, -1 when
    it faces `b`, and 0 for parts with no orientation. Reversing an edge negates
    it, which is exactly how a backwards LED stays detectable after collapsing.
    """

    type: str
    attrs: tuple[tuple[str, object], ...]
    polarity: int
    component_id: str = ""

    def reversed(self) -> Item:
        return Item(self.type, self.attrs, -self.polarity, self.component_id)

    def sort_key(self) -> tuple:
        return (self.type, tuple(sorted(map(str, self.attrs))), self.polarity)

    def compatible_with(self, ref: Item) -> bool:
        """Could this student part be the reference part?

        Attributes are checked only when the reference states them, so an author
        who does not care about colour need not write one.
        """
        if self.type != ref.type or self.polarity != ref.polarity:
            return False
        mine = dict(self.attrs)
        for key, want in ref.attrs:
            if key in mine and mine[key] != want:
                return False
        return True


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    items: tuple[Item, ...]

    def reversed(self) -> Edge:
        return Edge(self.b, self.a, tuple(i.reversed() for i in self.items))

    def matches(self, ref: Edge) -> bool:
        """True if these carry the same bag of parts, in any order.

        Chains hold a handful of parts at most, so trying every arrangement is
        cheaper and clearer than building a bipartite matching.
        """
        if len(self.items) != len(ref.items):
            return False
        for perm in permutations(self.items):
            if all(s.compatible_with(r) for s, r in zip(perm, ref.items)):
                return True
        return False


@dataclass
class CircuitGraph:
    labels: dict[str, frozenset[str]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def label_of(self, node: str) -> frozenset[str]:
        return self.labels.get(node, frozenset())

    def node_set(self) -> set[str]:
        out = set(self.labels)
        for e in self.edges:
            out.update((e.a, e.b))
        return out

    def degree(self, node: str) -> int:
        # Self-loops touch the node twice, which correctly keeps a shorted-out
        # component from looking like a pass-through.
        return sum((e.a == node) + (e.b == node) for e in self.edges)

    def collapsed(self) -> CircuitGraph:
        """Merge series runs into single edges. See module docstring, step 2."""
        edges = list(self.edges)
        while True:
            for node in sorted(self.node_set() - set(self._anchored())):
                touching = [e for e in edges if node in (e.a, e.b)]
                if len(touching) != 2 or any(e.a == e.b for e in touching):
                    continue
                e1, e2 = touching
                # Orient both so they run *into* and then *out of* `node`.
                first = e1 if e1.b == node else e1.reversed()
                second = e2 if e2.a == node else e2.reversed()
                if first.a == second.b and len(edges) > 2:
                    continue  # would close a loop onto itself; leave it alone
                edges = [e for e in edges if e not in (e1, e2)]
                edges.append(Edge(first.a, second.b, first.items + second.items))
                break
            else:
                break
        return CircuitGraph(labels=dict(self.labels), edges=edges)

    def _anchored(self) -> set[str]:
        return {n for n, lab in self.labels.items() if lab}


def build(netlist: Netlist) -> CircuitGraph:
    """Turn a netlist into a graph: MCU pins label nodes, parts become edges."""
    labels: dict[str, frozenset[str]] = {}
    for comp in netlist.of_type("mcu"):
        for pin_name, pin in comp.pins.items():
            labels[pin.node] = labels.get(pin.node, frozenset()) | {pin_name}

    edges = []
    for comp in netlist:
        if not comp.is_two_terminal:
            continue
        p, q = comp.ordered_pins()
        edges.append(Edge(p.node, q.node, (_item(comp),)))
    return CircuitGraph(labels=labels, edges=edges)


def _item(comp: Component) -> Item:
    keys = SIGNIFICANT_ATTRS.get(comp.type, ())
    attrs = tuple((k, comp.attrs[k]) for k in keys if k in comp.attrs)
    return Item(comp.type, attrs, 1 if comp.polarised else 0, comp.id)


def find_isomorphism(student: CircuitGraph,
                     reference: CircuitGraph) -> dict[str, str] | None:
    """Map student nodes onto reference nodes, or None if no mapping exists.

    Nodes carrying MCU pins are forced to their namesakes; the rest are searched
    by backtracking. Circuits are tiny once collapsed, so plain backtracking is
    ample and far easier to reason about than a canonical-labelling scheme.
    """
    s_nodes, r_nodes = student.node_set(), reference.node_set()
    if len(s_nodes) != len(r_nodes) or len(student.edges) != len(reference.edges):
        return None

    mapping: dict[str, str] = {}
    used: set[str] = set()

    # Anchor every labelled node first. A mismatch here is fatal immediately.
    r_by_label = {lab: n for n, lab in reference.labels.items() if lab}
    for node, lab in student.labels.items():
        if not lab:
            continue
        target = r_by_label.get(lab)
        if target is None:
            return None
        mapping[node] = target
        used.add(target)

    free = sorted(s_nodes - set(mapping))
    candidates = sorted(r_nodes - used)
    return _search(student, reference, mapping, free, candidates)


def _search(student: CircuitGraph, reference: CircuitGraph,
            mapping: dict[str, str], free: list[str],
            candidates: list[str]) -> dict[str, str] | None:
    if not free:
        return dict(mapping) if _edges_agree(student, reference, mapping) else None

    node, rest = free[0], free[1:]
    for cand in candidates:
        if student.degree(node) != reference.degree(cand):
            continue
        mapping[node] = cand
        if _partial_ok(student, reference, mapping):
            got = _search(student, reference, mapping, rest,
                          [c for c in candidates if c != cand])
            if got is not None:
                return got
        del mapping[node]
    return None


def _mapped_edges(g: CircuitGraph, mapping: dict[str, str]) -> list[Edge] | None:
    out = []
    for e in g.edges:
        if e.a not in mapping or e.b not in mapping:
            return None
        out.append(Edge(mapping[e.a], mapping[e.b], e.items))
    return out


def _consume(edge: Edge, pool: list[Edge]) -> bool:
    """Remove one reference edge that this mapped student edge satisfies."""
    for i, ref in enumerate(pool):
        for oriented in (edge, edge.reversed()):
            if (oriented.a, oriented.b) == (ref.a, ref.b) and oriented.matches(ref):
                pool.pop(i)
                return True
    return False


def _edges_agree(student: CircuitGraph, reference: CircuitGraph,
                 mapping: dict[str, str]) -> bool:
    mapped = _mapped_edges(student, mapping)
    if mapped is None:
        return False
    pool = list(reference.edges)
    return all(_consume(e, pool) for e in mapped) and not pool


def _partial_ok(student: CircuitGraph, reference: CircuitGraph,
                mapping: dict[str, str]) -> bool:
    """Cheap prune: every fully-mapped edge so far must have a partner."""
    pool = list(reference.edges)
    for e in student.edges:
        if (e.a in mapping and e.b in mapping
                and not _consume(Edge(mapping[e.a], mapping[e.b], e.items), pool)):
            return False
    return True


def equivalent(student: Netlist, reference: Netlist) -> bool:
    """True when both netlists describe the same working circuit."""
    return find_isomorphism(build(student).collapsed(),
                            build(reference).collapsed()) is not None
