"""End-to-end check against the real BasicBoard.

The topology here is not invented: it was measured on 2026-08-11 by driving each
Metro Mini pin in turn over USB and watching which LED lit (`tools/probe.py
--sweep`). Pin 2 lights the red LED, pin 3 white, pin 4 green, pin 5 blue, and
pins 6-12 light nothing. See docs/FIRMWARE_PROTOCOL.md.

Column positions are representative rather than surveyed -- placement is free
and the checker must not depend on it, which is itself one of the things these
tests pin down.
"""

from __future__ import annotations

from breadboard.check import check
from breadboard.netlist import Component, Netlist, Pin

# Measured: MCU pin -> LED colour.
LED_PINS = {"D2": "red", "D3": "white", "D4": "green", "D5": "blue"}


def basicboard(origin="factory", shift=0, reverse=None, move=None):
    """The factory-wired BasicBoard: four LEDs, each pin -> 330R -> LED -> GND.

    `shift` slides the whole build along the board, which must change nothing.
    `reverse` flips one LED. `move` displaces one LED's anode by a column.
    """
    def col(n):
        return n + shift

    mcu_pins = {name: Pin(f"j{col(40 + i * 2)}") for i, name in enumerate(LED_PINS)}
    mcu_pins["GND"] = Pin(f"j{col(56)}")
    parts = [Component(id="MCU", type="mcu", origin=origin, pins=mcu_pins)]

    for i, colour in enumerate(LED_PINS.values()):
        pin_col, mid_col = col(40 + i * 2), col(10 + i * 2)
        anode_col = mid_col + (1 if move == colour else 0)
        parts.append(Component(
            id=f"R_{colour}", type="resistor", origin=origin, attrs={"ohms": 330},
            pins={"1": Pin(f"j{pin_col}"), "2": Pin(f"j{mid_col}")}))
        anode, cathode = Pin(f"j{anode_col}"), Pin(f"j{col(56)}")
        if reverse == colour:
            anode, cathode = cathode, anode
        parts.append(Component(
            id=f"LED_{colour}", type="led", origin=origin, attrs={"color": colour},
            pins={"anode": anode, "cathode": cathode}))

    return Netlist(name="basicboard", components=parts)


class TestFactoryBoard:
    def test_the_board_as_shipped_is_clean(self):
        assert check(basicboard(), basicboard()) == []

    def test_no_sanity_rule_fires_on_a_correct_board(self):
        # Every LED has its resistor, nothing is shorted, no dead rail half.
        assert check(basicboard()) == []

    def test_the_same_build_slid_along_the_board_is_clean(self):
        # Placement is free; only topology is checked. This is the property the
        # whole architecture rests on.
        assert check(basicboard(shift=6), basicboard()) == []


class TestFaults:
    def test_a_displaced_led_is_found_and_named_by_colour(self):
        found = check(basicboard(move="blue"), basicboard())
        assert [f.kind for f in found] == ["wrong_connection"]
        assert found[0].components == ("LED_blue",)
        assert "blue LED" in found[0].message

    def test_the_repair_names_both_holes(self):
        f = check(basicboard(move="blue"), basicboard())[0]
        assert f.suggestion is not None
        assert f.detail["from"] != f.detail["to"]

    def test_a_reversed_led_is_reported_as_polarity(self):
        found = check(basicboard(reverse="green"), basicboard())
        assert [f.kind for f in found] == ["reversed_polarity"]
        assert found[0].components == ("LED_green",)

    def test_a_fault_in_factory_wiring_is_scoped_to_baseline(self):
        # Nothing in the lab asked the student to touch these, so the caller can
        # phrase it as "did you mean to move this?" rather than as an error.
        f = check(basicboard(move="red"), basicboard())[0]
        assert f.scope == "baseline"

    def test_only_the_faulty_led_is_reported(self):
        found = check(basicboard(move="white"), basicboard())
        assert len(found) == 1
        assert "white" in found[0].components[0]


def with_student_led(anode_col=30):
    """The factory board plus a student-added orange LED driven from D6.

    The added part must be *in* the circuit, not merely present. An LED dangling
    from a strip nothing else touches is electrically identical wherever that
    strip happens to be, so moving it changes nothing and the checker is right
    to stay quiet.
    """
    board = basicboard()
    board["MCU"].pins["D6"] = Pin("j34")
    board.components += [
        Component(id="R_student", type="resistor", attrs={"ohms": 330},
                  pins={"1": Pin("j34"), "2": Pin(f"j{anode_col}")}),
        Component(id="LED_student", type="led", attrs={"color": "orange"},
                  pins={"anode": Pin(f"j{anode_col}"), "cathode": Pin("j56")}),
    ]
    return board


class TestStudentAdditions:
    def test_a_correct_student_addition_is_clean(self):
        assert check(with_student_led(), with_student_led()) == []

    def test_a_displaced_student_part_is_scoped_to_the_lab(self):
        student = with_student_led()
        student["LED_student"].pins["anode"] = Pin("j31")
        f = check(student, with_student_led())[0]
        assert f.kind == "wrong_connection"
        assert f.scope == "lab"

    def test_a_dangling_part_is_not_a_position_error(self):
        # Both boards leave the LED hanging off a strip nothing else touches.
        # That is the same (non-)circuit either way, so there is nothing to fix.
        a, b = basicboard(), basicboard()
        for board, col in ((a, 30), (b, 31)):
            board.components.append(Component(
                id="LED_loose", type="led", attrs={"color": "orange"},
                pins={"anode": Pin(f"j{col}"), "cathode": Pin("j56")}))
        assert check(a, b) == []

    def test_a_wire_into_the_dead_half_of_a_rail_is_caught(self):
        board = basicboard()
        board["MCU"].pins["5V"] = Pin("p1+:5")
        board.components.append(Component(
            id="W_student", type="wire",
            pins={"1": Pin("p1+:40"), "2": Pin("j20")}))
        assert "dead_rail_segment" in [f.kind for f in check(board)]
