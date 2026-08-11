"""The netlist: what a circuit *is*, independent of how we learned about it.

This is the contract between the two halves of the system. The electrical probe
(`tools/probe.py`) and the camera pipeline both produce a Netlist; the checker
(`breadboard/check.py`) consumes one. Neither side needs to know about the other.

Four things travel together on every pin:

- **hole** — where it physically is, so a repair can say "one column left".
- **node** — derived from the hole; the topology we actually verify against.
- **confidence** — how sure the perception stage is.
- **alternatives** — near-miss holes it might be instead.

The last two exist so the checker can distinguish *the student made a mistake*
from *I misread the photo*. Without them a blurry frame becomes a false
accusation, which for a 13-year-old is the worst failure this system has.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from breadboard.board import BoardError, node_of

BOARD = "WB-102"

ORIGINS = ("factory", "student")

# Metro Mini V2 (ATmega328P) pin names, as printed on the board. D0/D1 are the
# USB serial link. See docs/FIRMWARE_PROTOCOL.md.
MCU_PINS = tuple(
    [f"D{i}" for i in range(14)]
    + [f"A{i}" for i in range(6)]
    + ["5V", "3V", "GND", "VIN", "RST", "RAW"]
)

# type -> (pin names, is polarised). Polarised parts have an orientation the
# checker must preserve; unpolarised ones may be inserted either way round.
COMPONENT_TYPES: dict[str, tuple[tuple[str, ...], bool]] = {
    "resistor": (("1", "2"), False),
    "wire": (("1", "2"), False),
    "led": (("anode", "cathode"), True),
    "button": (("1", "2"), False),
    "sensor3": (("vcc", "out", "gnd"), True),
    "sensor4": (("vcc", "trig", "echo", "gnd"), True),
    "mcu": (MCU_PINS, True),
}

# Two-terminal parts form the edges of the circuit graph. Everything else is a
# labelled vertex.
TWO_TERMINAL = {t for t, (pins, _) in COMPONENT_TYPES.items() if len(pins) == 2}


class NetlistError(ValueError):
    """A netlist that could not exist on a real board."""


@dataclass(frozen=True)
class Pin:
    """One leg of a component, in one hole."""

    hole: str
    confidence: float = 1.0
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            node_of(self.hole)
            for alt in self.alternatives:
                node_of(alt)
        except BoardError as e:
            raise NetlistError(str(e)) from e
        if not 0.0 <= self.confidence <= 1.0:
            raise NetlistError(
                f"confidence must be between 0 and 1, got {self.confidence}")

    @property
    def node(self) -> str:
        return node_of(self.hole)

    @property
    def alternative_nodes(self) -> tuple[str, ...]:
        """Distinct nodes the alternatives would put this pin on."""
        here = self.node
        seen = {here}
        out = []
        for alt in self.alternatives:
            n = node_of(alt)
            if n not in seen:
                seen.add(n)
                out.append(n)
        return tuple(out)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"hole": self.hole}
        if self.confidence != 1.0:
            d["confidence"] = self.confidence
        if self.alternatives:
            d["alternatives"] = list(self.alternatives)
        return d

    @classmethod
    def from_json(cls, d: Any) -> Pin:
        if isinstance(d, str):  # bare hole string is a valid shorthand
            return cls(d)
        return cls(hole=d["hole"],
                   confidence=d.get("confidence", 1.0),
                   alternatives=tuple(d.get("alternatives", ())))


@dataclass
class Component:
    """A part on the board, with each of its legs placed in a hole."""

    id: str
    type: str
    pins: dict[str, Pin]
    attrs: dict[str, Any] = field(default_factory=dict)
    origin: str = "student"

    def __post_init__(self) -> None:
        if self.type not in COMPONENT_TYPES:
            raise NetlistError(
                f"unknown component type {self.type!r}; "
                f"expected one of {sorted(COMPONENT_TYPES)}")
        if self.origin not in ORIGINS:
            raise NetlistError(
                f"origin must be one of {ORIGINS}, got {self.origin!r}")

        allowed, _ = COMPONENT_TYPES[self.type]
        unknown = set(self.pins) - set(allowed)
        if unknown:
            raise NetlistError(
                f"{self.type} {self.id!r} has unexpected pins {sorted(unknown)}; "
                f"expected {list(allowed)}")
        if self.type != "mcu" and set(self.pins) != set(allowed):
            missing = sorted(set(allowed) - set(self.pins))
            raise NetlistError(
                f"{self.type} {self.id!r} is missing pins {missing}")

    @property
    def polarised(self) -> bool:
        return COMPONENT_TYPES[self.type][1]

    @property
    def is_two_terminal(self) -> bool:
        return self.type in TWO_TERMINAL

    def ordered_pins(self) -> tuple[Pin, Pin]:
        """The two legs of a two-terminal part, in canonical order.

        For polarised parts the order is meaningful (anode first); for the rest
        it is arbitrary but stable.
        """
        if not self.is_two_terminal:
            raise NetlistError(f"{self.id!r} is not a two-terminal component")
        names, _ = COMPONENT_TYPES[self.type]
        return self.pins[names[0]], self.pins[names[1]]

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type,
                             "pins": {k: v.to_json() for k, v in self.pins.items()}}
        if self.attrs:
            d["attrs"] = dict(sorted(self.attrs.items()))
        if self.origin != "student":
            d["origin"] = self.origin
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Component:
        return cls(id=d["id"], type=d["type"],
                   pins={k: Pin.from_json(v) for k, v in d["pins"].items()},
                   attrs=d.get("attrs", {}),
                   origin=d.get("origin", "student"))


@dataclass
class Netlist:
    """A whole circuit."""

    components: list[Component] = field(default_factory=list)
    name: str = ""
    board: str = BOARD

    def __post_init__(self) -> None:
        if self.board != BOARD:
            raise NetlistError(f"unsupported board {self.board!r}; expected {BOARD}")
        seen: set[str] = set()
        for c in self.components:
            if c.id in seen:
                raise NetlistError(f"duplicate component id {c.id!r}")
            seen.add(c.id)

    def __getitem__(self, cid: str) -> Component:
        for c in self.components:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def __iter__(self):
        return iter(self.components)

    def nodes_used(self) -> set[str]:
        return {p.node for c in self.components for p in c.pins.values()}

    def of_type(self, *types: str) -> list[Component]:
        return [c for c in self.components if c.type in types]

    def mcu(self) -> Component | None:
        found = self.of_type("mcu")
        return found[0] if found else None

    def to_json(self) -> str:
        return json.dumps(
            {"board": self.board, "name": self.name,
             "components": [c.to_json() for c in self.components]},
            indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, d: Any) -> Netlist:
        if isinstance(d, str):
            d = json.loads(d)
        return cls(board=d.get("board", BOARD),
                   name=d.get("name", ""),
                   components=[Component.from_json(c) for c in d["components"]])
