"""Multi-terminal parts in the circuit graph.

Activity 3 ("Intro to Sensors") is the first lab with real wiring and a fully
determined answer: the light sensor's + goes to 3V, - to GND, and OUT to A0,
via three jumper wires placed wherever the student likes. A three-pin part is
neither an edge nor an anchor, so the graph needs a third kind of thing.

Pin identity is what carries orientation here: VCC must land where VCC lands.
Swapping + and - is a real student error (and can destroy the sensor), so it
must not compare equal.
"""

from __future__ import annotations

from breadboard.check import check
from breadboard.graph import build, equivalent
from breadboard.netlist import Component, Netlist, Pin


def mcu(**pins):
    return Component(
        id="MCU",
        type="mcu",
        origin="factory",
        pins={k: Pin(v) for k, v in pins.items()},
    )


def sensor(vcc="a10", out="a12", gnd="a14", cid="LIGHT"):
    return Component(
        id=cid,
        type="sensor3",
        attrs={"kind": "light"},
        pins={"vcc": Pin(vcc), "out": Pin(out), "gnd": Pin(gnd)},
    )


def wire(cid, a, b):
    return Component(id=cid, type="wire", pins={"1": Pin(a), "2": Pin(b)})


def activity3(vcc="a10", out="a12", gnd="a14", power="j31", signal="j36", ground="j33"):
    """The Activity 3 answer: sensor + -> 3V, - -> GND, OUT -> A0."""
    return Netlist(
        name="activity3",
        components=[
            mcu(**{"3V": "f31", "5V": "f32", "GND": "f33", "A0": "f36", "A1": "f37"}),
            sensor(vcc=vcc, out=out, gnd=gnd),
            wire("W_power", power, vcc),
            wire("W_signal", signal, out),
            wire("W_ground", ground, gnd),
        ],
    )


class TestSensorInGraph:
    def test_a_sensor_appears_in_the_graph(self):
        g = build(activity3())
        assert len(g.parts) == 1
        assert g.parts[0].type == "sensor3"

    def test_its_pins_reach_the_nodes_it_sits_on(self):
        g = build(activity3())
        assert dict(g.parts[0].pins)["vcc"] == "T:ae:10"

    def test_the_nodes_it_touches_are_in_the_graph(self):
        assert "T:ae:10" in build(activity3()).node_set()

    def test_a_node_a_sensor_touches_is_never_collapsed(self):
        # A sensor pin is a real junction. Collapsing through it would merge the
        # power wire and the sensor into one meaningless chain.
        assert "T:ae:10" in build(activity3()).collapsed().node_set()


class TestEquivalence:
    def test_the_same_wiring_placed_elsewhere_matches(self):
        assert equivalent(
            activity3(vcc="a10", out="a12", gnd="a14"),
            activity3(vcc="c40", out="c44", gnd="c48"),
        )

    def test_swapping_plus_and_minus_does_not_match(self):
        swapped = activity3()
        s = swapped["LIGHT"]
        s.pins = {"vcc": s.pins["gnd"], "out": s.pins["out"], "gnd": s.pins["vcc"]}
        assert not equivalent(swapped, activity3())

    def test_power_on_5v_instead_of_3v_does_not_match(self):
        # The worksheet is explicit that the sensor needs 3V.
        assert not equivalent(activity3(power="j32"), activity3())

    def test_signal_on_the_wrong_analog_pin_does_not_match(self):
        # The pre-loaded program reads A0 specifically.
        assert not equivalent(activity3(signal="j37"), activity3())

    def test_a_missing_sensor_does_not_match(self):
        without = activity3()
        without.components = [c for c in without.components if c.id != "LIGHT"]
        assert not equivalent(without, activity3())

    def test_an_extra_sensor_does_not_match(self):
        extra = activity3()
        extra.components.append(sensor(vcc="a30", out="a32", gnd="a34", cid="LIGHT2"))
        assert not equivalent(extra, activity3())

    def test_a_wire_one_column_off_does_not_match(self):
        assert not equivalent(activity3(vcc="a10", power="j30"), activity3())

    def test_a_different_kind_of_sensor_does_not_match(self):
        other = activity3()
        other["LIGHT"].attrs["kind"] = "temperature"
        assert not equivalent(other, activity3())

    def test_a_kind_only_one_file_states_is_not_compared(self):
        unstated = activity3()
        del unstated["LIGHT"].attrs["kind"]
        assert equivalent(unstated, activity3())
        assert equivalent(activity3(), unstated)

    def test_a_four_leg_sensors_kind_is_compared_too(self):
        def lab(kind: str) -> Netlist:
            return Netlist(
                components=[
                    mcu(**{"5V": "f32", "GND": "f33", "D7": "f40", "D8": "f41"}),
                    Component(
                        id="RANGE",
                        type="sensor4",
                        attrs={"kind": kind},
                        pins={
                            "vcc": Pin("j32"),
                            "trig": Pin("j40"),
                            "echo": Pin("j41"),
                            "gnd": Pin("j33"),
                        },
                    ),
                ]
            )

        assert equivalent(lab("ultrasonic"), lab("ultrasonic"))
        assert not equivalent(lab("infrared"), lab("ultrasonic"))


class TestCheckerOnActivity3:
    def test_the_correct_build_is_clean(self):
        assert check(activity3(), activity3()) == []

    def test_a_displaced_wire_is_located(self):
        student = activity3()
        student["W_power"].pins["2"] = Pin("a11")
        found = check(student, activity3())
        assert [f.kind for f in found] == ["wrong_connection"]
        assert "a11" in found[0].holes

    def test_power_on_5v_instead_of_3v_is_located(self):
        found = check(activity3(power="j32"), activity3())
        assert [f.kind for f in found] == ["wrong_connection"]
        assert "j32" in found[0].holes
        assert found[0].suggestion and "j31" in found[0].suggestion

    def test_a_wire_is_identified_by_its_colour(self):
        # Three wires in this lab, so "the wire" would leave a student guessing.
        # Colour is useless for inferring a wire's role, but it is exactly right
        # for pointing at which one to move.
        student = activity3()
        student["W_power"].attrs["color"] = "red"
        student["W_power"].pins["2"] = Pin("a11")
        reference = activity3()
        reference["W_power"].attrs["color"] = "red"
        assert "red wire" in check(student, reference)[0].message

    def test_signal_on_the_wrong_analog_pin_is_located(self):
        found = check(activity3(signal="j37"), activity3())
        assert [f.kind for f in found] == ["wrong_connection"]
        assert "j37" in found[0].holes

    def test_a_sensor_inserted_backwards_is_reported_as_polarity(self):
        # Swapping + and - takes two moves, so the single-move repair cannot
        # find it. It needs its own rule -- and it matters, because reversed
        # power can destroy the sensor.
        student = activity3()
        s = student["LIGHT"]
        s.pins = {"vcc": s.pins["gnd"], "out": s.pins["out"], "gnd": s.pins["vcc"]}
        found = check(student, activity3())
        assert [f.kind for f in found] == ["reversed_polarity"]
        assert "LIGHT" in found[0].components

    def test_the_polarity_finding_identifies_which_two_legs(self):
        student = activity3()
        s = student["LIGHT"]
        s.pins = {"vcc": s.pins["gnd"], "out": s.pins["out"], "gnd": s.pins["vcc"]}
        f = check(student, activity3())[0]
        assert set(f.detail["swapped"]) == {"vcc", "gnd"}

    def test_the_polarity_message_uses_the_words_on_the_data_sheet(self):
        # The worksheet calls these the "+ PIN" and "- PIN". A student has never
        # seen "vcc", so the technical names stay in `detail` for us and out of
        # the sentence they read.
        student = activity3()
        s = student["LIGHT"]
        s.pins = {"vcc": s.pins["gnd"], "out": s.pins["out"], "gnd": s.pins["vcc"]}
        msg = check(student, activity3())[0].message
        assert "+" in msg and "-" in msg
        assert "vcc" not in msg.lower()

    def test_the_sensor_is_named_the_way_the_worksheet_names_it(self):
        student = activity3()
        s = student["LIGHT"]
        s.pins = {"vcc": s.pins["gnd"], "out": s.pins["out"], "gnd": s.pins["vcc"]}
        msg = check(student, activity3())[0].message
        assert "light sensor" in msg
        assert "sensor3" not in msg

    def test_a_missing_sensor_is_reported_as_missing(self):
        student = activity3()
        student.components = [c for c in student.components if c.id != "LIGHT"]
        assert "missing_component" in [f.kind for f in check(student, activity3())]

    def test_an_extra_sensor_is_reported_as_extra(self):
        student = activity3()
        student.components.append(sensor(vcc="a30", out="a32", gnd="a34", cid="L2"))
        assert "extra_component" in [f.kind for f in check(student, activity3())]

    def test_the_wrong_kind_of_sensor_is_reported_as_the_wrong_part(self):
        student = activity3()
        student["LIGHT"].attrs["kind"] = "temperature"
        found = check(student, activity3())
        assert [f.kind for f in found] == ["extra_component", "missing_component"]
        assert "temperature sensor" in found[0].message
        assert found[1].suggestion == "Add the light sensor."

    def test_a_build_that_leaves_the_kind_out_still_gets_its_wiring_fix(self):
        # Not "an extra sensor": the kind is compared only when both files say.
        student = activity3()
        del student["LIGHT"].attrs["kind"]
        student["W_power"].pins["2"] = Pin("a11")
        assert [f.kind for f in check(student, activity3())] == ["wrong_connection"]

    def test_an_uncertain_wire_reading_is_not_an_accusation(self):
        student = activity3()
        student["W_power"].pins["2"] = Pin("a11", confidence=0.4, alternatives=("a10",))
        assert [f.kind for f in check(student, activity3())] == ["uncertain_reading"]
