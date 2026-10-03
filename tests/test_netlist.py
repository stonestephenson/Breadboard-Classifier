"""The netlist: the contract between perception and reasoning.

Both the electrical probe and the camera pipeline emit one of these, and the
checker consumes it. It carries topology (what is connected), placement (so we
can say "move it one column left"), provenance (factory-wired or student-added),
and confidence (so the checker can tell a misreading from a mistake).
"""

from __future__ import annotations

import json

import pytest

from breadboard.netlist import (
    Component,
    Netlist,
    NetlistError,
    Pin,
)


def led(cid="LED1", anode="b24", cathode="b28", **kw):
    return Component(
        id=cid,
        type="led",
        attrs={"color": "red"},
        pins={"anode": Pin(anode), "cathode": Pin(cathode)},
        **kw,
    )


def resistor(cid="R1", a="d24", b="d20", **kw):
    return Component(
        id=cid,
        type="resistor",
        attrs={"ohms": 330},
        pins={"1": Pin(a), "2": Pin(b)},
        **kw,
    )


class TestPin:
    def test_node_is_derived_from_the_hole(self):
        assert Pin("b24").node == Pin("d24").node
        assert Pin("b24").node != Pin("g24").node

    def test_defaults_to_certain(self):
        assert Pin("b24").confidence == 1.0
        assert Pin("b24").alternatives == ()

    def test_carries_near_miss_alternatives(self):
        p = Pin("b24", confidence=0.55, alternatives=("b23", "b25"))
        assert p.confidence == 0.55
        assert p.alternatives == ("b23", "b25")

    def test_rejects_an_impossible_hole(self):
        with pytest.raises(NetlistError):
            Pin("z99")

    def test_a_leg_may_sit_on_a_net_instead_of_in_a_hole(self):
        # How a lab is written: which legs are joined, never where.
        a, b, other = Pin("net:white"), Pin("net:white"), Pin("net:GND")
        assert not a.placed and Pin("b24").placed
        assert a.node == b.node != other.node
        assert a.node != Pin("b24").node

    def test_a_net_is_written_as_a_net_not_as_a_hole(self):
        assert Pin("net:white").to_json() == {"net": "white"}
        assert Pin.from_json({"net": "white"}) == Pin("net:white")

    def test_a_place_that_is_not_text_is_rejected_not_crashed_on(self):
        # A hand-edited file can hold anything.
        for bad in (12, None):
            with pytest.raises(NetlistError):
                Pin(bad)  # pyright: ignore[reportArgumentType]
        with pytest.raises(NetlistError):
            Pin.from_json({"net": None})

    def test_spaces_round_a_nets_name_do_not_make_it_another_net(self):
        assert Pin("net: white ").node == Pin("net:white").node
        assert Pin.from_json({"net": " white"}).to_json() == {"net": "white"}

    def test_a_leg_on_a_net_has_no_reading_to_doubt(self):
        with pytest.raises(NetlistError):
            Pin("net:white", confidence=0.5)

    def test_a_net_needs_a_name_and_cannot_be_a_guess_at_a_hole(self):
        with pytest.raises(NetlistError):
            Pin("net:")
        with pytest.raises(NetlistError):
            Pin("b24", confidence=0.5, alternatives=("net:white",))

    def test_rejects_an_impossible_alternative(self):
        with pytest.raises(NetlistError):
            Pin("b24", alternatives=("nope",))

    def test_rejects_out_of_range_confidence(self):
        with pytest.raises(NetlistError):
            Pin("b24", confidence=1.5)


class TestComponent:
    def test_polarised_parts_are_flagged(self):
        assert led().polarised is True
        assert resistor().polarised is False

    def test_pin_names_must_match_the_component_type(self):
        with pytest.raises(NetlistError, match="pins"):
            Component(id="LED1", type="led", pins={"1": Pin("b24"), "2": Pin("b28")})

    def test_unknown_component_type_is_rejected(self):
        with pytest.raises(NetlistError, match="type"):
            Component(
                id="X1", type="flux_capacitor", pins={"1": Pin("b24"), "2": Pin("b28")}
            )

    def test_defaults_to_student_added(self):
        assert led().origin == "student"

    def test_can_be_marked_factory_wired(self):
        assert led(origin="factory").origin == "factory"

    def test_rejects_unknown_origin(self):
        with pytest.raises(NetlistError, match="origin"):
            led(origin="martian")

    def test_a_component_spanning_one_node_is_rejected(self):
        # Both legs in the same strip shorts the part out. It is a real student
        # error, but it is a *circuit* error for the checker to report, not a
        # malformed netlist -- so this must construct successfully.
        c = led(anode="b24", cathode="d24")
        assert c.pins["anode"].node == c.pins["cathode"].node

    def test_mcu_pins_are_named_by_the_metro_mini_silkscreen(self):
        mcu = Component(id="MCU", type="mcu", pins={"D2": Pin("a10"), "GND": Pin("a12")})
        assert set(mcu.pins) == {"D2", "GND"}

    def test_mcu_rejects_a_pin_the_metro_mini_does_not_have(self):
        with pytest.raises(NetlistError, match="D99"):
            Component(id="MCU", type="mcu", pins={"D99": Pin("a10")})

    def test_an_led_may_say_its_way_round_was_not_seen(self):
        seen = led()
        unseen = Component(
            id="LED2",
            type="led",
            attrs={"color": "red", "direction": "unknown"},
            pins={"anode": Pin("b24"), "cathode": Pin("b28")},
        )
        assert seen.direction_known and not unseen.direction_known

    @pytest.mark.parametrize(
        ("ctype", "direction"),
        [("led", "backwards"), ("led", "unkown"), ("resistor", "unknown")],
    )
    def test_a_direction_is_only_ever_unknown_and_only_on_an_led(self, ctype, direction):
        # A hand-edited file with a typo must not be read as a known direction.
        names = ("anode", "cathode") if ctype == "led" else ("1", "2")
        with pytest.raises(NetlistError, match="direction"):
            Component(
                id="X",
                type=ctype,
                attrs={"direction": direction},
                pins={names[0]: Pin("b24"), names[1]: Pin("b28")},
            )


class TestNets:
    def test_a_net_with_one_leg_on_it_is_a_spelling_mistake(self):
        # "whtie" joins nothing, and every correct build would then look wrong.
        with pytest.raises(NetlistError, match="whtie"):
            Netlist(
                components=[
                    Component(id="MCU", type="mcu", pins={"D2": Pin("net:D2")}),
                    Component(
                        id="R1",
                        type="resistor",
                        pins={"1": Pin("net:D2"), "2": Pin("net:whtie")},
                    ),
                ]
            )


class TestNetlist:
    def test_component_ids_must_be_unique(self):
        with pytest.raises(NetlistError, match="duplicate"):
            Netlist(components=[led("X"), resistor("X")])

    def test_lookup_by_id(self):
        n = Netlist(components=[led(), resistor()])
        assert n["LED1"].type == "led"
        with pytest.raises(KeyError):
            n["nope"]

    def test_reports_every_node_in_use(self):
        n = Netlist(components=[led(anode="b24", cathode="b28")])
        assert n.nodes_used() == {"T:ae:24", "T:ae:28"}

    def test_round_trips_through_json(self):
        original = Netlist(
            name="lab3",
            components=[
                led(origin="factory"),
                resistor(origin="factory"),
                Component(
                    id="W1",
                    type="wire",
                    attrs={"color": "black"},
                    pins={
                        "1": Pin("b28"),
                        "2": Pin("p1-:12", confidence=0.4, alternatives=("p1-:11",)),
                    },
                ),
            ],
        )
        restored = Netlist.from_json(json.loads(original.to_json()))
        assert restored == original

    def test_json_is_stable_across_round_trips(self):
        n = Netlist(components=[led(), resistor()])
        once = n.to_json()
        twice = Netlist.from_json(json.loads(once)).to_json()
        assert once == twice

    def test_rejects_an_unknown_board(self):
        with pytest.raises(NetlistError, match="board"):
            Netlist.from_json({"board": "WB-999", "components": []})

    def test_confidence_survives_the_round_trip(self):
        n = Netlist(
            components=[
                Component(
                    id="W1",
                    type="wire",
                    pins={
                        "1": Pin("b28"),
                        "2": Pin("b30", confidence=0.3, alternatives=("b29", "b31")),
                    },
                )
            ]
        )
        back = Netlist.from_json(json.loads(n.to_json()))
        assert back["W1"].pins["2"].confidence == 0.3
        assert back["W1"].pins["2"].alternatives == ("b29", "b31")
