"""The checker: from "these circuits differ" to "move this wire one column left".

Three behaviours matter more than the rest:

- A correct build must never be flagged, however it was laid out.
- A near-miss *reading* must never become an accusation. If the circuit would be
  correct had we misread one endpoint, we say so and ask for a better look.
- A finding must point somewhere, with a repair a 13-year-old can carry out.
"""

from __future__ import annotations

import json
from pathlib import Path

from breadboard import board as board_module
from breadboard.check import check
from breadboard.netlist import Component, Netlist, Pin

RESISTOR_PINS = ("1", "2")
LED_PINS = ("anode", "cathode")


def mcu(**pins):
    return Component(
        id="MCU", type="mcu", origin="factory", pins={k: Pin(v) for k, v in pins.items()}
    )


def part(cid, ctype, a, b, origin="student", **attrs):
    names = LED_PINS if ctype == "led" else RESISTOR_PINS
    a_pin = a if isinstance(a, Pin) else Pin(a)
    b_pin = b if isinstance(b, Pin) else Pin(b)
    return Component(
        id=cid,
        type=ctype,
        attrs=attrs,
        origin=origin,
        pins={names[0]: a_pin, names[1]: b_pin},
    )


def blink(
    led_from: str | Pin = "a20",
    led_to: str | Pin = "a30",
    res_from: str | Pin = "a10",
    res_to: str | Pin = "a20",
    origin: str = "student",
    color: str = "red",
) -> Netlist:
    """D2 -> 330R -> red LED -> GND. The canonical one-LED lab."""
    return Netlist(
        name="blink",
        components=[
            mcu(D2="a10", GND="a30"),
            part("R1", "resistor", res_from, res_to, origin=origin, ohms=330),
            part("L1", "led", led_from, led_to, origin=origin, color=color),
        ],
    )


def kinds(findings):
    return [f.kind for f in findings]


class TestCorrectBuilds:
    def test_an_identical_build_is_clean(self):
        assert check(blink(), blink()) == []

    def test_the_same_circuit_built_elsewhere_is_clean(self):
        elsewhere = blink(res_from="a44", res_to="a50", led_from="a50", led_to="a58")
        elsewhere.components[0] = mcu(D2="a44", GND="a58")
        assert check(elsewhere, blink()) == []

    def test_components_reordered_in_series_is_clean(self):
        swapped = Netlist(
            name="blink",
            components=[
                mcu(D2="a10", GND="a30"),
                part("L1", "led", "a10", "a20", color="red"),
                part("R1", "resistor", "a20", "a30", ohms=330),
            ],
        )
        assert check(swapped, blink()) == []


class TestWrongConnection:
    def test_an_endpoint_one_column_off_is_located(self):
        student = blink(led_from="a21")  # resistor ends a20, LED starts a21
        found = check(student, blink())
        assert "wrong_connection" in kinds(found)

    def test_the_finding_names_the_part_and_where_it_is(self):
        found = check(blink(led_from="a21"), blink())
        f = next(f for f in found if f.kind == "wrong_connection")
        assert "L1" in f.components
        assert "a21" in f.holes

    def test_the_finding_suggests_where_it_should_go(self):
        found = check(blink(led_from="a21"), blink())
        f = next(f for f in found if f.kind == "wrong_connection")
        assert f.suggestion is not None
        assert "a20" in f.suggestion


class TestPolarity:
    def test_a_reversed_led_is_reported_as_polarity_not_wiring(self):
        student = blink()
        student["L1"].pins = {"anode": Pin("a30"), "cathode": Pin("a20")}
        assert "reversed_polarity" in kinds(check(student, blink()))

    def test_the_polarity_finding_names_the_led(self):
        student = blink()
        student["L1"].pins = {"anode": Pin("a30"), "cathode": Pin("a20")}
        f = next(f for f in check(student, blink()) if f.kind == "reversed_polarity")
        assert "L1" in f.components

    def test_an_led_whose_way_round_was_not_seen_is_never_called_backwards(self):
        student = blink()
        student["L1"].pins = {"anode": Pin("a30"), "cathode": Pin("a20")}
        student["L1"].attrs["direction"] = "unknown"
        assert check(student, blink()) == []

    def test_a_wire_off_by_one_is_found_whichever_way_the_led_was_listed(self):
        # The repair search must not depend on a direction nobody saw.
        for anode, cathode in (("a21", "a30"), ("a30", "a21")):
            student = blink(led_from=anode, led_to=cathode)
            student["L1"].attrs["direction"] = "unknown"
            assert kinds(check(student, blink())) == ["wrong_connection"]


class TestInventory:
    def test_a_missing_part_is_reported(self):
        student = blink()
        student.components = [c for c in student.components if c.id != "R1"]
        assert "missing_component" in kinds(check(student, blink()))

    def test_an_extra_part_is_reported(self):
        student = blink()
        student.components.append(part("L9", "led", "a40", "a45", color="green"))
        assert "extra_component" in kinds(check(student, blink()))


class TestUncertainty:
    def test_a_near_miss_reading_is_not_reported_as_an_error(self):
        # The camera read a21 but was unsure and offered a20. The circuit would
        # be correct at a20, so this is our doubt, not the student's mistake.
        student = blink(led_from=Pin("a21", confidence=0.4, alternatives=("a20",)))
        found = check(student, blink())
        assert kinds(found) == ["uncertain_reading"]
        assert "wrong_connection" not in kinds(found)

    def test_the_uncertain_finding_says_where_to_look_again(self):
        student = blink(led_from=Pin("a21", confidence=0.4, alternatives=("a20",)))
        f = check(student, blink())[0]
        assert "a21" in f.holes

    def test_a_confident_reading_that_is_wrong_is_still_an_error(self):
        # Same geometry, but perception was sure. Now it is a real finding.
        student = blink(led_from=Pin("a21", confidence=1.0))
        assert "wrong_connection" in kinds(check(student, blink()))

    def test_an_alternative_that_does_not_help_leaves_the_error(self):
        student = blink(led_from=Pin("a21", confidence=0.4, alternatives=("a22",)))
        assert "wrong_connection" in kinds(check(student, blink()))


class TestScopeTagging:
    def test_a_disturbed_factory_part_is_tagged_baseline(self):
        reference = blink(origin="factory")
        student = blink(origin="factory", led_from="a21")
        f = next(f for f in check(student, reference) if f.kind == "wrong_connection")
        assert f.scope == "baseline"

    def test_a_student_added_part_is_tagged_lab(self):
        f = next(
            f
            for f in check(blink(led_from="a21"), blink())
            if f.kind == "wrong_connection"
        )
        assert f.scope == "lab"


class TestSanityRulesWithoutAReference:
    def test_a_component_with_both_legs_in_one_strip_is_shorted(self):
        n = Netlist(
            components=[
                mcu(D2="a10", GND="a30"),
                part("L1", "led", "a20", "c20", color="red"),
            ]
        )
        assert "shorted_component" in kinds(check(n))

    def test_an_led_straight_across_power_has_no_resistor(self):
        n = Netlist(
            components=[
                mcu(D2="a10", GND="a30"),
                part("L1", "led", "a10", "a30", color="red"),
            ]
        )
        assert "led_without_resistor" in kinds(check(n))

    def test_a_resistor_in_series_clears_that_rule(self):
        assert "led_without_resistor" not in kinds(check(blink()))

    def test_power_wired_straight_to_ground_is_a_short(self):
        n = Netlist(
            components=[
                mcu(**{"5V": "a10", "GND": "a30"}),
                part("W1", "wire", "a10", "a30"),
            ]
        )
        assert "short_circuit" in kinds(check(n))

    def test_the_far_half_of_a_split_rail_is_flagged(self, monkeypatch):
        # On a board whose rails are two separate runs (the kit's own board's
        # are not; see board.RAILS_SPLIT). Power goes into one half,
        # the LED is wired to the other, and nothing works for no visible reason.
        monkeypatch.setattr(board_module, "RAILS_SPLIT", True)
        n = Netlist(
            components=[
                mcu(**{"5V": "p1+:5", "GND": "a30"}),
                part("R1", "resistor", "p1+:40", "a20", ohms=330),
                part("L1", "led", "a20", "a30", color="red"),
            ]
        )
        assert "dead_rail_segment" in kinds(check(n))

    def test_the_kits_continuous_rails_have_no_dead_half(self):
        n = Netlist(
            components=[
                mcu(**{"5V": "p1+:5", "GND": "a30"}),
                part("R1", "resistor", "p1+:40", "a20", ohms=330),
                part("L1", "led", "a20", "a30", color="red"),
            ]
        )
        assert "dead_rail_segment" not in kinds(check(n))

    def test_using_the_powered_half_of_a_rail_is_fine(self, monkeypatch):
        monkeypatch.setattr(board_module, "RAILS_SPLIT", True)
        n = Netlist(
            components=[
                mcu(**{"5V": "p1+:5", "GND": "a30"}),
                part("R1", "resistor", "p1+:20", "a20", ohms=330),
                part("L1", "led", "a20", "a30", color="red"),
            ]
        )
        assert "dead_rail_segment" not in kinds(check(n))


class TestFindingQuality:
    def test_every_finding_carries_a_message(self):
        student = blink()
        student.components = [c for c in student.components if c.id != "R1"]
        for f in check(student, blink()):
            assert f.message and f.message[0].isupper()

    def test_findings_are_ranked_most_actionable_first(self):
        student = blink(led_from="a21")
        student.components.append(part("L9", "led", "a40", "a45", color="green"))
        found = check(student, blink())
        assert found[0].kind in {"wrong_connection", "extra_component"}


# --- wires are joins, not parts ---------------------------------------------

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _example(name: str) -> Netlist:
    return Netlist.from_json(json.loads((EXAMPLES / f"{name}.json").read_text()))


class TestTheLabHasNoPlaces:
    """Placement is free, so where a lab's own file puts things means nothing."""

    def test_a_lab_written_as_a_circuit_gives_the_same_answers(self):
        as_circuit = Netlist(
            components=[
                mcu(D2="net:D2", GND="net:GND"),
                part("R1", "resistor", "net:D2", "net:mid", ohms=330),
                part("L1", "led", "net:mid", "net:GND", color="red"),
            ]
        )
        assert check(blink(), as_circuit) == []
        for student in (blink(led_from="a21"), blink(led_from="a30", led_to="a20")):
            told = [
                (f.kind, f.components, f.suggestion) for f in check(student, as_circuit)
            ]
            assert told == [
                (f.kind, f.components, f.suggestion) for f in check(student, blink())
            ]

    def test_where_the_labs_file_puts_things_does_not_steer_the_fix(self):
        # The same lab, laid out at other holes: nothing about the answer changes.
        student = blink(led_from="a21")
        elsewhere = Netlist(
            components=[
                mcu(D2="j40", GND="j50"),
                part("R1", "resistor", "j40", "j45", ohms=330),
                part("L1", "led", "j45", "j50", color="red"),
            ]
        )
        told = [(f.kind, f.components, f.suggestion) for f in check(student, elsewhere)]
        assert told == [
            (f.kind, f.components, f.suggestion) for f in check(student, blink())
        ]

    def test_a_wire_one_hole_off_is_moved_back_not_the_part_it_missed(self):
        # Both moves are one hole. A wire's end is the gentler thing to move.
        student = _example("basicboard_as_built")
        student["W_blue"].pins["2"] = Pin("a33")
        [found] = check(student, _example("basicboard_demo"))
        assert found.components == ("W_blue",)
        assert found.suggestion == "Move it from a33 to a32."

    def test_a_wire_end_far_from_home_is_moved_back_not_a_part_dragged_to_it(self):
        # Carrying the part's own leg to the pin's strip also makes the circuits
        # match, and is a shorter move, but it leaves the wire joining nothing
        # and asks the student to stretch a part across the board.
        lab = _example("activity3")
        student = _example("activity3")
        home = student["W_ground"].pins["1"].hole
        student["W_ground"].pins["1"] = Pin("j55")
        [found] = check(student, lab)
        assert found.components == ("W_ground",)
        assert found.suggestion == f"Move it from j55 to {home}."

        student = _example("basicboard_as_built")
        student["W_blue"].pins["1"] = Pin("a20")  # off pin 4's strip, far away
        [found] = check(student, _example("basicboard_demo"))
        assert found.components == ("W_blue",)
        assert found.suggestion == "Move it from a20 to a54."

    def test_a_build_with_no_places_that_differs_is_not_crashed_on(self):
        lab = _example("basicboard_demo")
        other = _example("basicboard_demo")
        white, blue = other["LED_white"], other["LED_blue"]
        white.pins["anode"], blue.pins["anode"] = blue.pins["anode"], white.pins["anode"]
        assert kinds(check(other, lab)) == ["circuit_differs"]

    def test_a_circuit_with_no_places_checks_clean_against_itself(self):
        lab = _example("basicboard_demo")
        assert check(lab, lab) == []


class TestWires:
    def test_a_build_using_jumpers_the_lab_drawing_does_not_is_the_same_circuit(self):
        # The demo board reaches every pin, and ground, through jumper wires;
        # the lab draws each resistor straight into the pin's strip.
        assert check(_example("basicboard_as_built"), _example("basicboard_demo")) == []

    def test_a_missing_jumper_is_reported_as_missing_not_as_a_part_moved(self):
        lab = _example("activity3")
        student = _example("activity3")
        student.components = [c for c in student.components if c.id != "W_signal"]
        found = check(student, lab)
        assert kinds(found) == ["missing_component"]
        assert "wire" in found[0].message

    def test_an_extra_jumper_is_taken_off_not_moved(self):
        lab = _example("activity3")
        student = _example("activity3")
        student.components.append(part("W_extra", "wire", "j36", "b12"))  # A1 to OUT
        found = check(student, lab)
        assert kinds(found) == ["extra_component"]
        assert found[0].components == ("W_extra",)
        assert found[0].suggestion == "Take it off the board."

    def test_a_wire_joining_two_pins_is_not_the_same_circuit(self):
        lab = _example("basicboard_demo")
        student = _example("basicboard_as_built")
        student.components.append(part("W_join", "wire", "b53", "b54"))  # pins 3 and 4
        assert check(student, lab) != []
