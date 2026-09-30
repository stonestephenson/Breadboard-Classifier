"""Circuit topology, and what counts as "the same circuit".

Two builds of one lab can look nothing alike -- different columns, opposite
halves of the board, components in a different order along a series chain -- and
still be electrically identical. The graph layer is what makes those compare
equal, so the checker never tells a student their working circuit is broken.

The anchor is the Metro Mini: breadboard strips are anonymous and may map to any
other strip, but pin D2 is always pin D2.
"""

from __future__ import annotations

from breadboard.graph import build, equivalent, find_isomorphism
from breadboard.netlist import Component, Netlist, Pin


def mcu(**pins):
    return Component(
        id="MCU", type="mcu", pins={k: Pin(v) for k, v in pins.items()}, origin="factory"
    )


def two(cid, ctype, a, b, **attrs):
    names = {"led": ("anode", "cathode")}.get(ctype, ("1", "2"))
    return Component(
        id=cid, type=ctype, attrs=attrs, pins={names[0]: Pin(a), names[1]: Pin(b)}
    )


def blink_circuit(pin_col=10, mid_col=20, gnd_col=30, swap=False, reverse_led=False):
    """D2 -> resistor -> LED -> ground, built at caller-chosen columns.

    `swap` puts the LED before the resistor, which is electrically identical.
    `reverse_led` flips the LED, which is not.
    """
    a, b = f"a{mid_col}", f"a{gnd_col}"
    first, second = ("led", "resistor") if swap else ("resistor", "led")
    parts = []
    for cid, ctype, p, q in [("C1", first, f"a{pin_col}", a), ("C2", second, a, b)]:
        if ctype == "led" and reverse_led:
            p, q = q, p
        attrs = {"color": "red"} if ctype == "led" else {"ohms": 330}
        parts.append(two(cid, ctype, p, q, **attrs))
    return Netlist(components=[mcu(D2=f"a{pin_col}", GND=f"a{gnd_col}"), *parts])


class TestBuild:
    def test_mcu_pins_label_the_nodes_they_sit_on(self):
        g = build(blink_circuit())
        assert g.label_of("T:ae:10") == frozenset({"D2"})
        assert g.label_of("T:ae:30") == frozenset({"GND"})

    def test_strips_without_an_mcu_pin_are_anonymous(self):
        g = build(blink_circuit())
        assert g.label_of("T:ae:20") == frozenset()

    def test_two_terminal_parts_become_edges(self):
        g = build(blink_circuit())
        assert len(g.edges) == 2

    def test_holes_in_the_same_strip_are_one_node(self):
        # A resistor from a20 to c20 has both legs in the same strip: it is
        # shorted out. The graph must show that as a self-loop, not two nodes.
        n = Netlist(components=[two("R1", "resistor", "a20", "c20", ohms=330)])
        g = build(n)
        assert g.edges[0].a == g.edges[0].b


class TestSeriesCollapse:
    def test_a_chain_through_an_anonymous_node_becomes_one_edge(self):
        g = build(blink_circuit()).collapsed()
        assert len(g.edges) == 1
        assert {g.edges[0].a, g.edges[0].b} == {"T:ae:10", "T:ae:30"}

    def test_the_collapsed_edge_keeps_both_components(self):
        g = build(blink_circuit()).collapsed()
        assert sorted(i.type for i in g.edges[0].items) == ["led", "resistor"]

    def test_a_junction_node_is_not_collapsed(self):
        # Two LEDs sharing a node: that node has degree 3, so it is a real
        # junction and must survive.
        n = Netlist(
            components=[
                mcu(D2="a10", D3="a12", GND="a30"),
                two("R1", "resistor", "a10", "a20", ohms=330),
                two("L1", "led", "a20", "a30", color="red"),
                two("L2", "led", "a20", "a12", color="blue"),
            ]
        )
        assert "T:ae:20" in build(n).collapsed().node_set()

    def test_a_labelled_node_is_never_collapsed(self):
        # Even at degree 2, a node carrying an MCU pin is a fixed anchor.
        n = Netlist(
            components=[
                mcu(D2="a10", D3="a20", GND="a30"),
                two("R1", "resistor", "a10", "a20", ohms=330),
                two("L1", "led", "a20", "a30", color="red"),
            ]
        )
        assert "T:ae:20" in build(n).collapsed().node_set()


class TestEquivalence:
    def test_the_same_circuit_built_elsewhere_on_the_board_matches(self):
        assert equivalent(
            blink_circuit(pin_col=10, mid_col=20, gnd_col=30),
            blink_circuit(pin_col=10, mid_col=45, gnd_col=52),
        )

    def test_resistor_on_either_side_of_the_led_matches(self):
        # The case that killed plain graph isomorphism. Both work; both must pass.
        assert equivalent(blink_circuit(swap=False), blink_circuit(swap=True))

    def test_a_reversed_led_does_not_match(self):
        assert not equivalent(blink_circuit(), blink_circuit(reverse_led=True))

    def test_an_led_whose_way_round_was_not_seen_matches_either_way(self):
        # A camera cannot see which way an LED faces, so it cannot be wrong.
        for reverse in (False, True):
            unseen = blink_circuit(reverse_led=reverse)
            unseen["C2"].attrs["direction"] = "unknown"
            assert equivalent(unseen, blink_circuit())

    def test_an_led_whose_way_round_was_not_seen_still_needs_its_place(self):
        unseen = blink_circuit(gnd_col=31)
        unseen["C2"].attrs["direction"] = "unknown"
        unseen["C2"].pins["cathode"] = Pin("a30")  # not in the ground strip
        assert not equivalent(unseen, blink_circuit())

    def test_a_different_mcu_pin_does_not_match(self):
        other = blink_circuit()
        other.components[0] = mcu(D7="a10", GND="a30")
        assert not equivalent(blink_circuit(), other)

    def test_a_missing_component_does_not_match(self):
        short = blink_circuit()
        short.components.pop()
        assert not equivalent(blink_circuit(), short)

    def test_an_extra_component_does_not_match(self):
        extra = blink_circuit()
        extra.components.append(two("L9", "led", "a40", "a30", color="green"))
        assert not equivalent(blink_circuit(), extra)

    def test_a_different_led_colour_does_not_match(self):
        blue = blink_circuit()
        blue["C2"].attrs["color"] = "blue"
        assert not equivalent(blink_circuit(), blue)

    def test_an_unspecified_colour_in_the_reference_accepts_any(self):
        # Curriculum authors should not have to pin down attributes they do not
        # care about. Omitting one means "anything".
        loose = blink_circuit()
        loose["C2"].attrs.pop("color")
        assert equivalent(blink_circuit(), loose)

    def test_wire_colour_is_never_significant(self):
        # The curriculum team confirmed colours are conventions, not rules.
        a = Netlist(
            components=[
                mcu(D2="a10", GND="a30"),
                two("W1", "wire", "a10", "a30", color="red"),
            ]
        )
        b = Netlist(
            components=[
                mcu(D2="a10", GND="a30"),
                two("W1", "wire", "a10", "a30", color="green"),
            ]
        )
        assert equivalent(a, b)


class TestIsomorphism:
    def test_returns_the_node_mapping(self):
        m = find_isomorphism(
            build(blink_circuit(mid_col=20)).collapsed(),
            build(blink_circuit(mid_col=45)).collapsed(),
        )
        assert m is not None
        assert m["T:ae:10"] == "T:ae:10"

    def test_returns_none_when_there_is_no_match(self):
        assert (
            find_isomorphism(
                build(blink_circuit()).collapsed(),
                build(blink_circuit(reverse_led=True)).collapsed(),
            )
            is None
        )
