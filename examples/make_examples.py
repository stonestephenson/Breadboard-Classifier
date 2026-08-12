"""Regenerate the example circuits in this directory.

    ./venv/bin/python examples/make_examples.py

The baseline is the BasicBoard's *measured* topology -- pin 2 drives the red
LED, 3 white, 4 green, 5 blue, each through a 330 ohm resistor to ground --
recovered by driving pins over USB on 2026-08-11 (docs/FIRMWARE_PROTOCOL.md).

Column positions are representative, not surveyed. Placement is free, so nothing
in the system depends on them; they exist so the files look like a real board
and so repair suggestions have somewhere to point.
"""

from __future__ import annotations

import json
from pathlib import Path

from breadboard.netlist import Component, Netlist, Pin

HERE = Path(__file__).resolve().parent

LEDS = [("D2", "red", 12), ("D3", "white", 20), ("D4", "green", 28), ("D5", "blue", 36)]
MCU_COL = {"D2": 44, "D3": 46, "D4": 48, "D5": 50}
GND_COL = 58


def basicboard() -> Netlist:
    pins = {name: Pin(f"j{MCU_COL[name]}") for name, _, _ in LEDS}
    pins["GND"] = Pin(f"j{GND_COL}")
    parts: list[Component] = [
        Component(id="MCU", type="mcu", origin="factory", pins=pins)
    ]
    for name, colour, col in LEDS:
        parts += [
            Component(
                id=f"R_{colour}",
                type="resistor",
                origin="factory",
                attrs={"ohms": 330},
                pins={"1": Pin(f"j{MCU_COL[name]}"), "2": Pin(f"j{col}")},
            ),
            Component(
                id=f"LED_{colour}",
                type="led",
                origin="factory",
                attrs={"color": colour},
                pins={"anode": Pin(f"j{col}"), "cathode": Pin(f"j{GND_COL}")},
            ),
        ]
    return Netlist(name="basicboard", components=parts)


def led_moved() -> Netlist:
    """One LED leg a single column off -- the commonest real mistake."""
    n = basicboard()
    n.name = "basicboard (blue LED displaced)"
    n["LED_blue"].pins["anode"] = Pin("j37")
    return n


def led_reversed() -> Netlist:
    n = basicboard()
    n.name = "basicboard (green LED backwards)"
    c = n["LED_green"]
    c.pins = {"anode": c.pins["cathode"], "cathode": c.pins["anode"]}
    return n


def unsure_reading() -> Netlist:
    """Same displacement, but perception flagged it as a doubtful reading.

    The checker should decline to call this an error and ask for a better look.
    """
    n = basicboard()
    n.name = "basicboard (blue LED, uncertain reading)"
    n["LED_blue"].pins["anode"] = Pin("j37", confidence=0.45, alternatives=("j36",))
    return n


def missing_resistor() -> Netlist:
    """LED straight from a pin to ground. Caught with no lab file at all."""
    n = basicboard()
    n.name = "basicboard (red LED with no resistor)"
    n.components = [c for c in n.components if c.id != "R_red"]
    n["LED_red"].pins["anode"] = Pin(f"j{MCU_COL['D2']}")
    return n


def dead_rail() -> Netlist:
    """Power into one half of a split rail, the circuit into the other.

    Invisible on the board: the painted stripe runs the whole length, but the
    metal underneath does not.
    """
    n = basicboard()
    n.name = "basicboard (wired to the dead half of a rail)"
    n["MCU"].pins["5V"] = Pin("p1+:5")
    n.components.append(
        Component(
            id="W_power",
            type="wire",
            attrs={"color": "red"},
            pins={"1": Pin("p1+:40"), "2": Pin("j30")},
        )
    )
    return n


EXAMPLES = {
    "basicboard.json": basicboard,
    "basicboard_led_moved.json": led_moved,
    "basicboard_led_reversed.json": led_reversed,
    "basicboard_uncertain.json": unsure_reading,
    "basicboard_no_resistor.json": missing_resistor,
    "basicboard_dead_rail.json": dead_rail,
}


def main() -> None:
    for filename, factory in EXAMPLES.items():
        path = HERE / filename
        path.write_text(json.dumps(json.loads(factory().to_json()), indent=2) + "\n")
        print(f"wrote {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
