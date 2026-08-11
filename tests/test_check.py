"""The checker: from "these circuits differ" to "move this wire one column left".

Three behaviours matter more than the rest:

- A correct build must never be flagged, however it was laid out.
- A near-miss *reading* must never become an accusation. If the circuit would be
  correct had we misread one endpoint, we say so and ask for a better look.
- A finding must point somewhere, with a repair a 13-year-old can carry out.
"""

from __future__ import annotations

from breadboard.check import check
from breadboard.netlist import Component, Netlist, Pin

RESISTOR_PINS = ("1", "2")
LED_PINS = ("anode", "cathode")


def mcu(**pins):
    return Component(id="MCU", type="mcu", origin="factory",
                     pins={k: Pin(v) for k, v in pins.items()})


def part(cid, ctype, a, b, origin="student", **attrs):
    names = LED_PINS if ctype == "led" else RESISTOR_PINS
    a_pin = a if isinstance(a, Pin) else Pin(a)
    b_pin = b if isinstance(b, Pin) else Pin(b)
    return Component(id=cid, type=ctype, attrs=attrs, origin=origin,
                     pins={names[0]: a_pin, names[1]: b_pin})


def blink(led_from: str | Pin = "a20", led_to: str | Pin = "a30",
          res_from: str | Pin = "a10", res_to: str | Pin = "a20",
          origin: str = "student", color: str = "red") -> Netlist:
    """D2 -> 330R -> red LED -> GND. The canonical one-LED lab."""
    return Netlist(name="blink", components=[
        mcu(D2="a10", GND="a30"),
        part("R1", "resistor", res_from, res_to, origin=origin, ohms=330),
        part("L1", "led", led_from, led_to, origin=origin, color=color),
    ])


def kinds(findings):
    return [f.kind for f in findings]


class TestCorrectBuilds:
    def test_an_identical_build_is_clean(self):
        assert check(blink(), blink()) == []

    def test_the_same_circuit_built_elsewhere_is_clean(self):
        elsewhere = blink(res_from="a44", res_to="a50",
                          led_from="a50", led_to="a58")
        elsewhere.components[0] = mcu(D2="a44", GND="a58")
        assert check(elsewhere, blink()) == []

    def test_components_reordered_in_series_is_clean(self):
        swapped = Netlist(name="blink", components=[
            mcu(D2="a10", GND="a30"),
            part("L1", "led", "a10", "a20", color="red"),
            part("R1", "resistor", "a20", "a30", ohms=330),
        ])
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
        f = next(f for f in check(blink(led_from="a21"), blink())
                 if f.kind == "wrong_connection")
        assert f.scope == "lab"


class TestSanityRulesWithoutAReference:
    def test_a_component_with_both_legs_in_one_strip_is_shorted(self):
        n = Netlist(components=[mcu(D2="a10", GND="a30"),
                                part("L1", "led", "a20", "c20", color="red")])
        assert "shorted_component" in kinds(check(n))

    def test_an_led_straight_across_power_has_no_resistor(self):
        n = Netlist(components=[mcu(D2="a10", GND="a30"),
                                part("L1", "led", "a10", "a30", color="red")])
        assert "led_without_resistor" in kinds(check(n))

    def test_a_resistor_in_series_clears_that_rule(self):
        assert "led_without_resistor" not in kinds(check(blink()))

    def test_power_wired_straight_to_ground_is_a_short(self):
        n = Netlist(components=[mcu(**{"5V": "a10", "GND": "a30"}),
                                part("W1", "wire", "a10", "a30")])
        assert "short_circuit" in kinds(check(n))

    def test_the_far_half_of_a_split_rail_is_flagged(self):
        # The WB-102's rails are two separate runs. Power goes into one half,
        # the LED is wired to the other, and nothing works for no visible reason.
        n = Netlist(components=[
            mcu(**{"5V": "p1+:5", "GND": "a30"}),
            part("R1", "resistor", "p1+:40", "a20", ohms=330),
            part("L1", "led", "a20", "a30", color="red"),
        ])
        assert "dead_rail_segment" in kinds(check(n))

    def test_using_the_powered_half_of_a_rail_is_fine(self):
        n = Netlist(components=[
            mcu(**{"5V": "p1+:5", "GND": "a30"}),
            part("R1", "resistor", "p1+:20", "a20", ohms=330),
            part("L1", "led", "a20", "a30", color="red"),
        ])
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
