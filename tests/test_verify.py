"""Does it work? What each pin lit, checked against what the lab expects.

The behaviours that matter:

- The lab file is read as "pin N should light this LED", whatever the layout.
  That includes grounds reached through a rail. An LED it cannot test is said
  to be untestable, never silently dropped.
- Every way a pin can go wrong gets its own plain answer. It points at the fix
  when the evidence shows one, and never blames an LED that is doing its job.
- "Your wiring works" is the hardest answer to earn. Anything untestable,
  anything unclear on any pin, photos that could not be lined up, or an imperfect view of the
  board withholds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from breadboard.blink import Glow, Session
from breadboard.netlist import Component, Netlist, Pin
from breadboard.rectify import CANONICAL_SIZE, Rectification
from breadboard.verify import expected_leds, verify

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PINS = tuple(range(2, 11))


def _lab(name: str) -> Netlist:
    return Netlist.from_json(json.loads((EXAMPLES / f"{name}.json").read_text()))


BASICBOARD = _lab("basicboard")  # pin 2 red, 3 white, 4 green, 5 blue


# --- building small labs by hand -------------------------------------------


def _mcu(**pins: str) -> Component:
    return Component(id="MCU", type="mcu", pins={k: Pin(v) for k, v in pins.items()})


def _part(cid: str, kind: str, a: str, b: str, **attrs) -> Component:
    names = ("anode", "cathode") if kind == "led" else ("1", "2")
    return Component(
        id=cid, type=kind, attrs=attrs, pins={names[0]: Pin(a), names[1]: Pin(b)}
    )


def _colours(lab: Netlist) -> dict[int, str | None]:
    expected, _ = expected_leds(lab)
    return {p: e.colour for p, e in expected.items()}


def _untestable(lab: Netlist) -> dict[str, str]:
    _, untestable = expected_leds(lab)
    return {u.component: u.reason for u in untestable}


# --- blink results by hand --------------------------------------------------


def _fit(confidence: float = 1.0) -> Rectification:
    w, h = CANONICAL_SIZE
    return Rectification(np.eye(3), np.zeros((h, w, 3), np.uint8), confidence, True, 0.05)


def _session(lit: dict[int, str], confidence=1.0, **overrides) -> Session:
    """A run where each pin in `lit` lit an LED of that colour; the rest were dark."""
    glows = {p: Glow(p, "dark") for p in PINS}
    for pin, colour in lit.items():
        glows[pin] = Glow(pin, "lit", hole="c20", photo_xy=(0.0, 0.0), colour=colour)
    glows.update(overrides.pop("glows", {}))
    return Session(_fit(confidence), glows, **overrides)


AS_BUILT = {2: "red", 3: "white", 4: "green", 5: "blue"}


class TestExpectedLeds:
    def test_the_basicboard_lab_is_one_led_per_pin(self):
        assert _colours(BASICBOARD) == AS_BUILT
        assert _untestable(BASICBOARD) == {}

    def test_a_lab_without_leds_expects_nothing(self):
        assert expected_leds(_lab("activity3")) == ({}, [])

    @pytest.mark.parametrize("flip", [False, True], ids=["pin first", "ground first"])
    def test_direction_is_read_whichever_way_round_the_file_lists_parts(self, flip):
        r = ("j12", "j44") if flip else ("j44", "j12")
        lab = Netlist(
            [
                _mcu(D2="j44", GND="j58"),
                _part("R", "resistor", *r),
                _part("L", "led", "j12", "j58", color="red"),
            ]
        )
        assert _colours(lab) == {2: "red"}

    def test_an_led_facing_away_from_its_pin_is_untestable(self):
        assert "faces away" in _untestable(_lab("basicboard_led_reversed"))["LED_green"]

    def test_leds_returning_through_a_ground_rail_are_expected(self):
        # The usual layout: every LED to the - rail, one wire from the rail to GND.
        lab = Netlist(
            [
                _mcu(D2="j44", D3="j46", GND="j58"),
                _part("R2", "resistor", "j44", "j12"),
                _part("L2", "led", "j12", "p2-:5", color="red"),
                _part("R3", "resistor", "j46", "j20"),
                _part("L3", "led", "j20", "p2-:8", color="green"),
                _part("W", "wire", "j58", "p2-:12"),
            ]
        )
        assert _colours(lab) == {2: "red", 3: "green"}

    def test_two_leds_on_one_pin_are_both_untestable(self):
        lab = Netlist(
            [
                _mcu(D2="j44", GND="j58"),
                _part("R1", "resistor", "j44", "j12"),
                _part("L1", "led", "j12", "j58", color="red"),
                _part("R2", "resistor", "j44", "j20"),
                _part("L2", "led", "j20", "j58", color="green"),
            ]
        )
        assert _colours(lab) == {}
        assert set(_untestable(lab)) == {"L1", "L2"}

    def test_an_led_behind_a_button_is_untestable(self):
        # Nobody presses the button during a run, so "dark" would be a false alarm.
        lab = Netlist(
            [
                _mcu(D2="j44", GND="j58"),
                _part("B", "button", "j44", "j30"),
                _part("R", "resistor", "j30", "j12"),
                _part("L", "led", "j12", "j58", color="red"),
            ]
        )
        assert "other than a resistor" in _untestable(lab)["L"]


class TestVerify:
    def test_everything_right_works_with_the_resistor_caveat(self):
        verdict = verify(BASICBOARD, _session(AS_BUILT), PINS)
        assert verdict.works
        assert verdict.findings == ()
        assert verdict.ok_pins == (2, 3, 4, 5)
        assert verdict.caveat and "resistor" in verdict.caveat
        assert "program" in verdict.summary

    def test_a_dark_led_is_reported_with_what_to_check(self):
        lit = {**AS_BUILT}
        del lit[5]
        verdict = verify(BASICBOARD, _session(lit), PINS)
        assert not verdict.works
        (f,) = verdict.findings
        assert f.kind == "led_does_not_light"
        assert f.severity == "error"
        assert "The blue LED should light when pin 5 turns on" in f.message
        assert f.suggestion and "long leg" in f.suggestion

    def test_a_dark_led_that_lit_elsewhere_says_where(self):
        # The blue LED lit from pin 7, and pin 5 lit nothing.
        lit = {2: "red", 3: "white", 4: "green", 7: "blue"}
        verdict = verify(BASICBOARD, _session(lit), PINS)
        (f,) = verdict.findings  # pin 7 is explained, not reported twice
        assert f.kind == "led_does_not_light"
        assert "The blue LED lit from pin 7 instead" in f.message
        assert f.suggestion == "Connect the blue LED to pin 5, not pin 7."

    def test_two_leds_on_each_others_pins_are_both_named(self):
        lit = {2: "white", 3: "red", 4: "green", 5: "blue"}
        verdict = verify(BASICBOARD, _session(lit), PINS)
        kinds = [(f.kind, f.detail["pin"]) for f in verdict.findings]
        assert kinds == [("wrong_led_on_pin", 2), ("wrong_led_on_pin", 3)]
        assert (
            verdict.findings[0].suggestion == "Connect the red LED to pin 2, not pin 3."
        )

    def test_an_led_doing_its_own_job_is_never_blamed(self):
        # Two red LEDs in the lab; pin 3's works, pin 2's is dark. Telling the
        # student to move pin 3's LED would break a working part.
        lab = Netlist(
            [
                _mcu(D2="j44", D3="j46", GND="j58"),
                _part("R2", "resistor", "j44", "j12"),
                _part("L2", "led", "j12", "j58", color="red"),
                _part("R3", "resistor", "j46", "j20"),
                _part("L3", "led", "j20", "j58", color="red"),
            ]
        )
        verdict = verify(lab, _session({3: "red"}), PINS)
        (f,) = verdict.findings
        assert f.kind == "led_does_not_light"
        assert "pin 3" not in (f.suggestion or "")
        assert verdict.ok_pins == (3,)

    def test_a_wrong_colour_nowhere_else_suggests_the_led_itself(self):
        lit = {**AS_BUILT, 2: "green", 4: "green"}
        verdict = verify(BASICBOARD, _session(lit), PINS)
        (f,) = verdict.findings
        assert f.kind == "wrong_led_on_pin"
        assert f.suggestion and f.suggestion.startswith("Use a red LED here")

    def test_an_led_on_a_pin_the_lab_does_not_use_is_a_warning(self):
        verdict = verify(BASICBOARD, _session({**AS_BUILT, 8: "blue"}), PINS)
        (f,) = verdict.findings
        assert (f.kind, f.severity) == ("unexpected_led", "warning")
        assert f.suggestion
        assert not verdict.works
        assert verdict.summary == "Found 1 thing to look at."

    def test_warm_colours_the_camera_confuses_are_not_an_accusation(self):
        verdict = verify(BASICBOARD, _session({**AS_BUILT, 2: "orange"}), PINS)
        (f,) = verdict.findings
        assert (f.kind, f.severity) == ("not_checked", "uncertain")
        assert "Not sure" in verdict.summary

    @pytest.mark.parametrize(
        "overrides",
        [
            {"glows": {3: Glow(3, "unclear", "light appeared in two places")}},
            {"moved": frozenset({3})},
            {"glows": {8: Glow(8, "unclear", "the board brightened")}},
            {"moved": frozenset({9})},
        ],
        ids=["unclear", "moved", "unclear unused pin", "moved on unused pin"],
    )
    def test_doubt_about_any_pin_withholds_the_all_clear(self, overrides):
        verdict = verify(BASICBOARD, _session(AS_BUILT, **overrides), PINS)
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["not_checked"]
        assert verdict.caveat is None

    def test_a_pin_tied_to_ground_is_an_error(self):
        lit = {**AS_BUILT}
        del lit[3]
        verdict = verify(BASICBOARD, _session(lit, shorted=frozenset({3})), PINS)
        (f,) = verdict.findings
        assert (f.kind, f.severity) == ("pin_tied_to_ground", "error")

    def test_an_unused_pin_tied_to_ground_withholds_the_all_clear(self):
        verdict = verify(BASICBOARD, _session(AS_BUILT, shorted=frozenset({9})), PINS)
        assert not verdict.works
        (f,) = verdict.findings
        assert (f.kind, f.severity) == ("pin_tied_to_ground", "warning")

    def test_an_untestable_led_withholds_the_all_clear(self):
        # Pins 2-5 light correctly, but the file also has an LED blinking cannot
        # test. Saying "every LED works" would vouch for one nobody checked.
        lab = _lab("basicboard")
        lab.components.append(_part("L_extra", "led", "j60", "j62", color="yellow"))
        verdict = verify(lab, _session(AS_BUILT), PINS)
        assert not verdict.works
        (f,) = verdict.findings
        assert f.kind == "cannot_check"
        assert "yellow LED" in f.message

    def test_an_imperfect_view_withholds_the_all_clear(self):
        verdict = verify(BASICBOARD, _session(AS_BUILT, confidence=0.75), PINS)
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["imperfect_view"]

    def test_no_board_found_asks_for_a_better_picture(self):
        verdict = verify(BASICBOARD, Session(rect=None), PINS)
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["board_not_found"]

    def test_changing_light_on_pins_nothing_should_be_on_does_not_block_works(self):
        # The room's light kept changing: pins 6 and 7 showed no LED, but not
        # cleanly enough to be called dark, and no differently from each other
        # (calm). The lab puts nothing there, so that is quiet enough.
        note = "the board changed, but no LED core showed"
        unclear = {p: Glow(p, "unclear", note) for p in (6, 7)}
        run = _session(AS_BUILT, glows=unclear, calm=frozenset({6, 7}))
        assert verify(BASICBOARD, run, PINS).works

    def test_changing_light_on_a_pin_with_an_led_is_not_taken_for_a_dark_led(self):
        # The same on pin 3, where the lab has an LED: it is neither passed nor
        # said to be broken.
        lit = {p: c for p, c in AS_BUILT.items() if p != 3}
        unclear = {3: Glow(3, "unclear", "the board changed, but no LED core showed")}
        run = _session(lit, glows=unclear, calm=frozenset({3}))
        verdict = verify(BASICBOARD, run, PINS)
        assert not verdict.works
        kinds = [(f.kind, f.severity) for f in verdict.findings]
        assert kinds == [("not_checked", "uncertain")]

    def test_a_washed_out_board_asks_for_less_light_and_blames_nothing(self):
        # Nothing was judged, so no LED is said to be dark, and nothing works.
        verdict = verify(BASICBOARD, Session(_fit(), too_bright=True), PINS)
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["too_bright"]
        assert verdict.findings[0].severity == "uncertain"
        assert "light" in (verdict.findings[0].suggestion or "")

    def test_a_lab_without_leds_says_so(self):
        verdict = verify(_lab("activity3"), _session({}), PINS)
        assert not verdict.works
        assert "no LEDs" in verdict.summary


RUN = (
    Path(__file__).resolve().parent.parent
    / "data/cache/blink/basicboard-2026-09-27-overhead"
)


@pytest.mark.skipif(not RUN.exists(), reason="recorded blink run not present")
class TestRecordedRun:
    """The real board, rewired to pin 2 green, 3 blue, 4 white, 5 red."""

    @pytest.fixture(scope="class")
    @staticmethod
    def session():
        import cv2

        from breadboard.blink import analyse

        frames = {}
        for path in sorted(RUN.glob("*.png")):
            image = cv2.imread(str(path))
            assert image is not None
            frames[path.stem] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return analyse(frames, json.loads((RUN / "meta.json").read_text())["pins"])

    def test_it_works_against_the_lab_it_was_built_to(self, session):
        assert verify(_lab("basicboard_rewired"), session, PINS).works

    def test_against_the_original_lab_every_led_is_on_the_wrong_pin(self, session):
        verdict = verify(BASICBOARD, session, PINS)
        assert [f.kind for f in verdict.findings] == ["wrong_led_on_pin"] * 4
        assert (
            verdict.findings[0].suggestion == "Connect the red LED to pin 2, not pin 5."
        )
