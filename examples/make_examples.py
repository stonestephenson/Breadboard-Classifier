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
# The same board as rewired by 2026-09-27, as blink and watch found it
# (tools/blink.py): pin 2 green, 3 blue, 4 white, 5 red.
REWIRED = [
    ("D2", "green", 12),
    ("D3", "blue", 20),
    ("D4", "white", 28),
    ("D5", "red", 36),
]
MCU_COL = {"D2": 44, "D3": 46, "D4": 48, "D5": 50, "D6": 52}
GND_COL = 58


def basicboard(leds=LEDS, name: str = "basicboard") -> Netlist:
    pins = {pin: Pin(f"j{MCU_COL[pin]}") for pin, _, _ in leds}
    pins["GND"] = Pin(f"j{GND_COL}")
    parts: list[Component] = [
        Component(id="MCU", type="mcu", origin="factory", pins=pins)
    ]
    for pin, colour, col in leds:
        parts += [
            Component(
                id=f"R_{colour}",
                type="resistor",
                origin="factory",
                attrs={"ohms": 330},
                pins={"1": Pin(f"j{MCU_COL[pin]}"), "2": Pin(f"j{col}")},
            ),
            Component(
                id=f"LED_{colour}",
                type="led",
                origin="factory",
                attrs={"color": colour},
                pins={"anode": Pin(f"j{col}"), "cathode": Pin(f"j{GND_COL}")},
            ),
        ]
    return Netlist(name=name, components=parts)


def basicboard_rewired() -> Netlist:
    return basicboard(REWIRED, "basicboard (rewired 2026-09-27)")


# The demo lab (2026-09-29): pin 3 white, 4 blue, 5 green, 6 red, each through a
# resistor to ground. It is what Stone's demo board below was built to do.
DEMO = [("D3", "white", 12), ("D4", "blue", 20), ("D5", "green", 28), ("D6", "red", 36)]


def basicboard_demo() -> Netlist:
    return basicboard(DEMO, "basicboard (demo lab 2026-09-29)")


# Stone's demo board as built on 2026-09-29, as Stone described it and blinking
# confirmed (pin 3 white at column 27, 4 blue at 32, 5 green at 38, 6 red at 44).
# The Metro Mini's top pins are at column 63: its digital pins in row c (13 at
# column 63 down to 3 at column 53), its power pins in row g (GND at column 59).
# Each LED: (pin, colour, short-leg column). Short leg in d<col>, long leg in
# d<col+1>; a resistor from e<col> across the centre gap to h<col>; a jumper from
# i<col> to the ground row of the rail beside row j; and a wire from a<col+1> to
# row a of its pin's column. One wire joins GND (h59) to that ground row.
AS_BUILT = [
    ("D3", "white", 27),
    ("D4", "blue", 31),
    ("D5", "green", 37),
    ("D6", "red", 43),
]
AS_BUILT_PIN_COL = {"D3": 53, "D4": 54, "D5": 55, "D6": 56}
AS_BUILT_GND_COL = 59


def rail_hole_above(col: int, rail: str) -> str:
    """The rail hole level with a terminal column. Rails come in groups of five
    starting at column 3, with a gap every sixth column."""
    group, within = divmod(col - 3, 6)
    if col < 3 or within == 5:
        raise ValueError(f"no rail hole at column {col}")
    return f"{rail}:{group * 5 + within + 1}"


def basicboard_as_built() -> Netlist:
    """The demo board as built, standing in for what a vision model will read."""
    mcu = {pin: Pin(f"c{col}") for pin, col in AS_BUILT_PIN_COL.items()}
    mcu["GND"] = Pin(f"g{AS_BUILT_GND_COL}")
    parts: list[Component] = [
        Component(id="MCU", type="mcu", origin="factory", pins=mcu),
        Component(
            id="W_ground",
            type="wire",
            pins={
                "1": Pin(f"h{AS_BUILT_GND_COL}"),
                "2": Pin(rail_hole_above(AS_BUILT_GND_COL, "p2-")),
            },
        ),
    ]
    for pin, colour, col in AS_BUILT:
        long = col + 1
        parts += [
            Component(
                id=f"W_{colour}",
                type="wire",
                pins={"1": Pin(f"a{AS_BUILT_PIN_COL[pin]}"), "2": Pin(f"a{long}")},
            ),
            Component(
                id=f"LED_{colour}",
                type="led",
                attrs={"color": colour},
                pins={"anode": Pin(f"d{long}"), "cathode": Pin(f"d{col}")},
            ),
            Component(
                id=f"R_{colour}",
                type="resistor",
                attrs={"ohms": 330},
                pins={"1": Pin(f"e{col}"), "2": Pin(f"h{col}")},
            ),
            Component(
                id=f"J_{colour}",
                type="wire",
                pins={"1": Pin(f"i{col}"), "2": Pin(rail_hole_above(col, "p2-"))},
            ),
        ]
    return Netlist(name="basicboard (demo board as built 2026-09-29)", components=parts)


def basicboard_as_seen() -> Netlist:
    """The demo board as a camera sees it: every leg in its hole, but not which
    way round each LED is, which a camera cannot see once it is seated. Each
    LED's legs are listed left to right (so "anode" means nothing here, and is
    the short leg on this board); blinking settles which is which."""
    build = basicboard_as_built()
    build.name = "basicboard (demo board as a camera sees it 2026-09-29)"
    for led in build.of_type("led"):
        legs = sorted(led.pins.values(), key=lambda p: int(p.hole[1:]))
        led.pins = {"anode": legs[0], "cathode": legs[1]}
        led.attrs["direction"] = "unknown"
    return build


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


# --------------------------------------------------------------------------
# Activity 3: Intro to Sensors
#
# The worksheet fixes the answer exactly: the light sensor's + goes to 3V (not
# 5V), - goes to GND, and OUT goes to A0, because the pre-loaded program reads
# A0. Three jumper wires, placed wherever the student likes.
#
# The Metro Mini straddles the centre channel, so its two pin rows land in the
# two halves of the board. Columns below are representative, not surveyed.
# --------------------------------------------------------------------------

MCU_LOWER = ["RST", "3V", "5V", "GND", "VIN", "A0", "A1", "A2", "A3", "A4", "A5"]
MCU_LOWER_COL0 = 30  # RST sits at f30, so 3V is f31, 5V f32, GND f33, A0 f35


def _mcu_lower(name: str, row: str = "f") -> str:
    return f"{row}{MCU_LOWER_COL0 + MCU_LOWER.index(name)}"


def activity3(power_from="3V", signal_from="A0", vcc_col=10, swap_power=False):
    """The Activity 3 answer, with knobs for building the wrong versions."""
    vcc, out, gnd = f"a{vcc_col}", "a12", "a14"
    # Only the *sensor* turns round; the wires stay in the holes they were in.
    # Swapping both would just mirror the circuit, which is still correct -- an
    # earlier version of this generator made exactly that mistake and produced
    # an example the checker rightly passed.
    s_vcc, s_gnd = (gnd, vcc) if swap_power else (vcc, gnd)
    pins = {name: Pin(_mcu_lower(name)) for name in ("3V", "5V", "GND", "A0", "A1")}
    return Netlist(
        name="activity3",
        components=[
            Component(id="MCU", type="mcu", origin="factory", pins=pins),
            Component(
                id="LIGHT",
                type="sensor3",
                attrs={"kind": "light"},
                pins={"vcc": Pin(s_vcc), "out": Pin(out), "gnd": Pin(s_gnd)},
            ),
            # Wires reach the Metro Mini's strips from the free rows below it.
            Component(
                id="W_power",
                type="wire",
                attrs={"color": "red"},
                pins={"1": Pin(_mcu_lower(power_from, "j")), "2": Pin(vcc)},
            ),
            Component(
                id="W_signal",
                type="wire",
                attrs={"color": "yellow"},
                pins={"1": Pin(_mcu_lower(signal_from, "j")), "2": Pin(out)},
            ),
            Component(
                id="W_ground",
                type="wire",
                attrs={"color": "black"},
                pins={"1": Pin(_mcu_lower("GND", "j")), "2": Pin(gnd)},
            ),
        ],
    )


def activity3_on_5v():
    """The + wire on 5V. The data sheet is explicit that the sensor needs 3V."""
    n = activity3(power_from="5V")
    n.name = "activity3 (sensor powered from 5V)"
    return n


def activity3_wrong_analog():
    """OUT on A1. The pre-loaded program only reads A0, so nothing arrives."""
    n = activity3(signal_from="A1")
    n.name = "activity3 (signal wire on A1)"
    return n


def activity3_sensor_backwards():
    """+ and - swapped. Worth catching: this can destroy the sensor."""
    n = activity3(swap_power=True)
    n.name = "activity3 (sensor + and - swapped)"
    return n


def activity3_wire_off_by_one():
    n = activity3()
    n.name = "activity3 (power wire one column off)"
    n["W_power"].pins["2"] = Pin("a11")
    return n


def activity3_uncertain():
    """The same displacement, but read doubtfully -- must not be an accusation."""
    n = activity3()
    n.name = "activity3 (power wire, uncertain reading)"
    n["W_power"].pins["2"] = Pin("a11", confidence=0.4, alternatives=("a10",))
    return n


EXAMPLES = {
    "activity3.json": activity3,
    "activity3_on_5v.json": activity3_on_5v,
    "activity3_wrong_analog.json": activity3_wrong_analog,
    "activity3_sensor_backwards.json": activity3_sensor_backwards,
    "activity3_wire_off_by_one.json": activity3_wire_off_by_one,
    "activity3_uncertain.json": activity3_uncertain,
    "basicboard.json": basicboard,
    "basicboard_rewired.json": basicboard_rewired,
    "basicboard_demo.json": basicboard_demo,
    "basicboard_as_built.json": basicboard_as_built,
    "basicboard_as_seen.json": basicboard_as_seen,
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
