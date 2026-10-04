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

import pytest

from breadboard import board as board_module
from breadboard.check import check
from breadboard.graph import equivalent
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
        # Column 20's strip, in a hole the resistor's leg is not already in.
        found = check(blink(led_from="a21"), blink())
        f = next(f for f in found if f.kind == "wrong_connection")
        assert f.suggestion == "Move it from a21 to b20."


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


def _demo(**changes) -> Netlist:
    """The demo board as built, with some legs moved: W_blue={"2": "a33"}."""
    build = _example("basicboard_as_built")
    for part_id, legs in changes.items():
        for leg, hole in legs.items():
            build[part_id].pins[leg] = Pin(hole)
    return build


BLUE_BACKWARDS = {"LED_blue": {"anode": "d31", "cathode": "d32"}}


class TestSeveralMistakes:
    """More than one thing wrong: each is found, one change at a time, by taking
    the change that makes most of the lab's links right and looking again."""

    LAB = _example("basicboard_demo")

    def test_two_mistakes_on_two_leds_are_both_found(self):
        found = check(_demo(W_green={"2": "a39"}, **BLUE_BACKWARDS), self.LAB)
        assert kinds(found) == ["wrong_connection", "reversed_polarity"]
        assert found[0].components == ("W_green",)
        assert found[0].suggestion == "Move it from a39 to a38."
        assert found[1].components == ("LED_blue",)

    def test_each_fix_says_which_pins_link_it_puts_right(self):
        found = check(_demo(W_green={"2": "a39"}, **BLUE_BACKWARDS), self.LAB)
        assert [f.detail["pins"] for f in found] == [["D5"], ["D4"]]
        [alone] = check(_demo(W_blue={"2": "a33"}), self.LAB)
        assert alone.detail["pins"] == ["D4"]

    def test_two_mistakes_on_the_same_led_are_both_found(self):
        # The wire back in place makes the link right but for the LED's way
        # round, which counts as half right: enough to take the step.
        found = check(_demo(W_blue={"2": "a33"}, **BLUE_BACKWARDS), self.LAB)
        assert kinds(found) == ["wrong_connection", "reversed_polarity"]
        assert {f.components for f in found} == {("W_blue",), ("LED_blue",)}

    def test_three_mistakes_are_all_found(self):
        student = _demo(W_green={"2": "a39"}, J_red={"1": "i44"}, **BLUE_BACKWARDS)
        found = check(student, self.LAB)
        assert kinds(found) == [
            "wrong_connection",
            "wrong_connection",
            "reversed_polarity",
        ]

    def test_past_three_fixes_it_says_to_check_again(self):
        student = _demo(
            W_green={"2": "a39"},
            J_red={"1": "i44"},
            W_white={"2": "a29"},
            **BLUE_BACKWARDS,
        )
        found = check(student, self.LAB)
        assert len(found) == 4 and kinds(found)[-1] == "circuit_differs"
        assert found[-1].message == "There is more to fix after these."
        assert "check again" in (found[-1].suggestion or "")
        # Three problems are named. The last line is a caution, not a fourth.
        assert [f.severity for f in found] == ["error", "error", "error", "warning"]

    def test_two_wires_swapped_are_one_finding(self):
        found = check(_demo(W_white={"1": "a54"}, W_blue={"1": "a53"}), self.LAB)
        assert kinds(found) == ["swapped_connections"]
        assert set(found[0].components) == {"W_white", "W_blue"}
        assert set(found[0].holes) == {"a53", "a54"}
        assert found[0].suggestion == "Swap the ends in a53 and a54."
        assert found[0].detail["pins"] == ["D3", "D4"]

    def test_two_leds_in_each_others_places_are_fixed_by_swapping_their_wires(self):
        student = _demo()
        white, blue = student["LED_white"], student["LED_blue"]
        white.pins, blue.pins = blue.pins, white.pins
        found = check(student, self.LAB)
        assert kinds(found) == ["swapped_connections"]
        assert found[0].message == "Two wires need to swap places."
        assert found[0].suggestion == "Swap the ends in a53 and a54."

    def test_a_mistake_nothing_can_place_still_says_so(self):
        # Pins 3 and 4 joined by a wire: no single change the search knows.
        student = _demo()
        student.components.append(part("W_join", "wire", "b53", "b54"))
        student.components.append(part("W_also", "wire", "b55", "b56"))
        found = check(student, self.LAB)
        assert "circuit_differs" in kinds(found) or "extra_component" in kinds(found)


def _carried_out(student: Netlist, found) -> Netlist:
    """The student's circuit after doing what each finding says."""
    for f in found:
        if f.kind == "wrong_connection" and "legs" in f.detail:
            for leg, hole in f.detail["legs"].items():
                student[f.components[0]].pins[leg] = Pin(hole)
        elif f.kind == "wrong_connection":
            student[f.components[0]].pins[f.detail["leg"]] = Pin(f.detail["to"])
        elif f.kind == "reversed_polarity":
            led = student[f.components[0]]
            led.pins = {"anode": led.pins["cathode"], "cathode": led.pins["anode"]}
        elif f.kind == "swapped_connections":
            a, b = f.detail["swapped"]
            legs = [
                (c, name)
                for c in student
                if c.id in f.components
                for name, pin in c.pins.items()
                if pin.hole in (a, b)
            ]
            (c1, n1), (c2, n2) = legs
            c1.pins[n1], c2.pins[n2] = c2.pins[n2], c1.pins[n1]
        elif f.kind == "extra_component":
            student.components = [
                c for c in student.components if c.id != f.components[0]
            ]
    return student


def _slipped(student: Netlist, part_id: str) -> None:
    """One slip on one part: an LED turned round, or a wire's end one hole along
    (a W_ wire's LED end, a J_ jumper's resistor end)."""
    comp = student[part_id]
    if comp.type == "led":
        comp.pins = {"anode": comp.pins["cathode"], "cathode": comp.pins["anode"]}
        return
    leg = "2" if part_id.startswith("W_") else "1"
    hole = comp.pins[leg].hole
    comp.pins[leg] = Pin(f"{hole[0]}{int(hole[1:]) + 1}")


class TestTheFixesWork:
    """The point of a fix: carried out, it makes the circuit the lab's."""

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("W_white", "LED_blue"),  # two LEDs
            ("J_green", "W_red"),
            ("LED_white", "LED_red"),
            ("W_white", "W_blue"),
            ("J_white", "J_red"),
            ("W_blue", "J_blue"),  # both on one LED's path
            ("W_green", "LED_green"),
            ("J_red", "LED_red"),
        ],
    )
    def test_two_slips_are_found_and_the_fixes_put_it_right(self, first, second):
        lab = _example("basicboard_demo")
        student = _demo()
        _slipped(student, first)
        _slipped(student, second)
        found = check(student, lab)
        assert "circuit_differs" not in kinds(found)
        assert len(found) == 2
        assert equivalent(_carried_out(student, found), lab)

    @pytest.mark.parametrize(
        "changes",
        [
            # Three builds a cold review (2026-10-03) found the search giving a
            # false fix for, while it still passed on fixes from a search that
            # had stalled: both LED legs sent to one strip, a resistor put
            # beside its LED, a wire moved that was in the right place.
            {"W_white": {"2": "a29"}, "J_white": {"1": "b27"}},
            {"W_blue": {"1": "j54"}, "LED_blue": {"cathode": "g31"}},
            {"LED_green": {"anode": "d37"}, "R_green": {"1": "f37"}},
            {"R_blue": {"1": "e32", "2": "h32"}},  # a whole resistor one column along
            {"LED_blue": {"anode": "d33", "cathode": "d32"}},  # a whole LED likewise
            {"W_blue": {"1": "a55", "2": "a33"}},  # both ends of one wire
        ],
    )
    def test_fixes_are_named_only_when_they_end_at_the_labs_circuit(self, changes):
        lab = _example("basicboard_demo")
        found = check(_demo(**changes), lab)
        fixes = [f for f in found if f.kind != "circuit_differs"]
        if fixes:
            assert equivalent(_carried_out(_demo(**changes), fixes), lab)
            assert "shorted_component" not in kinds(found)
        else:
            assert [f.severity for f in found] == ["error"]

    def test_past_three_fixes_the_rest_come_at_the_next_check(self):
        lab = _example("basicboard_demo")
        student = _demo()
        for part_id in ("W_white", "J_green", "LED_red", "W_blue"):
            _slipped(student, part_id)
        for _ in range(2):
            found = check(student, lab)
            student = _carried_out(student, found)
        assert equivalent(student, lab)

    def test_a_part_one_hole_along_is_one_thing_to_do(self):
        lab = _example("basicboard_demo")
        [found] = check(_demo(LED_blue={"anode": "d33", "cathode": "d32"}), lab)
        assert found.message == "The blue LED is in the wrong holes."
        assert found.suggestion == "Put its longer leg in d32 and its shorter leg in d31."
        assert set(found.holes) == {"d33", "d32"}

    def test_two_fixes_into_one_strip_name_two_holes(self):
        lab = _example("basicboard_demo")
        found = check(_demo(J_white={"2": "p2+:30"}, J_blue={"2": "p2+:29"}), lab)
        assert len({f.detail["to"] for f in found}) == 2

    def test_three_slips_too(self):
        lab = _example("basicboard_demo")
        student = _demo()
        for part_id in ("W_white", "J_green", "LED_red"):
            _slipped(student, part_id)
        found = check(student, lab)
        assert len(found) == 3
        assert equivalent(_carried_out(student, found), lab)


class TestOnlyMovesAStudentCanMake:
    """A fix nobody can carry out is not a fix."""

    LAB = _example("basicboard_demo")

    def test_an_led_sitting_away_from_its_wire_is_met_not_stretched(self):
        # The whole LED is ten columns along. Its wire and its resistor's leg
        # go to it; the LED is not pulled apart to reach them, and it is not
        # called backwards.
        found = check(_demo(LED_blue={"anode": "d41", "cathode": "d40"}), self.LAB)
        assert [(f.components, f.suggestion) for f in found] == [
            (("R_blue",), "Move it from e31 to e40."),
            (("W_blue",), "Move it from a32 to a41."),
        ]

    def test_a_missing_wire_does_not_send_an_led_leg_across_the_board(self):
        # Joining the LED straight to pin 4's strip would match the lab, with
        # its legs 22 columns apart. Better to say no single change was found.
        student = _demo()
        student.components = [c for c in student.components if c.id != "W_blue"]
        assert kinds(check(student, self.LAB)) == ["circuit_differs"]

    def test_one_leg_of_a_sensor_is_never_moved_alone(self):
        # Its legs are one rigid row. The wires are what move.
        lab = _example("activity3")
        for wire in ("W_power", "W_ground", "W_signal"):
            student = _example("activity3")
            hole = student[wire].pins["2"].hole
            student[wire].pins["2"] = Pin(f"{hole[0]}{int(hole[1:]) + 1}")
            [found] = check(student, lab)
            assert found.components == (wire,), wire

    def test_two_changes_that_only_help_together_are_not_guessed_at(self):
        # Two wires both join pins 3 and 4. Taking out either changes nothing.
        student = _demo()
        student.components.append(part("W_join", "wire", "b53", "b54"))
        student.components.append(part("W_again", "wire", "e53", "e54"))
        assert kinds(check(student, self.LAB)) == ["circuit_differs"]


class TestTheHoleSuggested:
    def test_a_hole_with_a_leg_already_in_it_is_not_suggested(self):
        # d32 holds the LED's leg. The wire goes in a free hole of that strip.
        [found] = check(_demo(W_blue={"2": "d33"}), _example("basicboard_demo"))
        assert found.suggestion == "Move it from d33 to c32."
        assert found.detail["to"] == "c32"

    def test_the_legs_own_row_is_kept_when_that_hole_is_free(self):
        [found] = check(_demo(W_blue={"2": "a33"}), _example("basicboard_demo"))
        assert found.suggestion == "Move it from a33 to a32."


class TestNoResistor:
    def _student(self) -> Netlist:
        student = _example("basicboard_as_built")
        student.components = [c for c in student.components if c.id != "R_blue"]
        student.components.append(part("W_bypass", "wire", "e31", "h31"))
        return student

    def test_a_missing_resistor_is_said_once_and_names_its_led(self):
        [found] = check(self._student(), _example("basicboard_demo"))
        assert found.kind == "led_without_resistor"
        assert found.message.startswith("The blue LED has no resistor")
        assert set(found.holes) == {"d31", "d32"}

    def test_a_missing_part_is_asked_for_in_plain_words(self):
        student = _example("basicboard_as_built")
        student.components = [
            c for c in student.components if c.id not in ("R_blue", "LED_blue")
        ]
        said = " ".join(
            f.suggestion or "" for f in check(student, _example("basicboard_demo"))
        )
        assert "the the" not in said
        assert "Add the 330 ohm resistor." in said


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
