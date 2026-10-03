"""Which leg to move: the checker on a described build, checked against blinking.

The behaviours that matter:

- A description is trusted only where blinking confirms it. Any pin that lit
  something the description does not predict means the description is wrong,
  and then no fix is shown at all.
- Darkness contradicts nothing: an LED placed right can stay dark for reasons no
  picture shows. It gets the hidden causes, one per check, most likely first,
  and the check after says whether it is fixed.
- Which way round an LED faces comes from blinking when it lights, whatever the
  description says. A description may leave it open, as a camera's must.
- Colours the camera cannot tell apart, and unclear pins, contradict nothing.
- When the description holds up, the answer names the fix down to the hole.
  "Works" still needs blinking to confirm every LED the lab expects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from breadboard.blink import Glow, Session
from breadboard.diagnose import (
    MISMATCH_SUMMARY,
    Suggested,
    claims,
    diagnose,
    disagreements,
    remember,
)
from breadboard.netlist import Component, Netlist, Pin
from breadboard.rectify import CANONICAL_SIZE, Rectification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from make_examples import basicboard_as_built, basicboard_as_seen, basicboard_demo

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PINS = tuple(range(2, 11))
LAB = basicboard_demo()  # pin 3 white, 4 blue, 5 green, 6 red
W, H = CANONICAL_SIZE
RECT = Rectification(
    photo_to_canonical=np.eye(3),
    canonical=np.zeros((H, W, 3), np.uint8),
    confidence=1.0,
    oriented=True,
    column_margin=0.05,
)
# Where each LED of the as-built board shows, by pin: (colour, glow column), as
# measured on the real board.
BOARD = {3: ("white", 27), 4: ("blue", 32), 5: ("green", 38), 6: ("red", 44)}


def _built(**changes) -> Netlist:
    """The as-built description, with some legs moved: LED_blue={"anode": "d31"}."""
    build = basicboard_as_built()
    for part, legs in changes.items():
        for leg, hole in legs.items():
            build[part].pins[leg] = Pin(hole)
    return build


FLIPPED_BLUE = {"LED_blue": {"anode": "d31", "cathode": "d32"}}


def _colours_swapped() -> Netlist:
    """The as-built description with the white and blue LEDs' colours swapped."""
    build = _built()
    build["LED_white"].attrs["color"], build["LED_blue"].attrs["color"] = "blue", "white"
    return build


def _kinds(verdict) -> list[str]:
    return [f.kind for f in verdict.findings]


def _hidden(verdict):
    [found] = [f for f in verdict.findings if f.kind == "hidden_fault"]
    return found


def _run(**pins) -> Session:
    """A blink run: every pin dark unless given as ("colour", column) or a
    status such as "unclear"."""
    glows = {}
    for pin in PINS:
        seen = pins.get(f"p{pin}", "dark")
        if isinstance(seen, tuple):
            colour, col = seen
            glows[pin] = Glow(
                pin, "lit", hole=f"c{col}", photo_xy=(0.0, 0.0), colour=colour
            )
        else:
            glows[pin] = Glow(pin, seen)
    return Session(RECT, glows)


AS_BUILT_RUN = {f"p{pin}": seen for pin, seen in BOARD.items()}


class TestClaims:
    def test_each_led_is_claimed_at_its_pin_colour_and_columns(self):
        said = claims(_built(), PINS)
        assert {p: (c.colour, c.columns) for p, c in said.items() if c.lights} == {
            3: ("white", (27, 28)),
            4: ("blue", (31, 32)),
            5: ("green", (37, 38)),
            6: ("red", (43, 44)),
        }

    def test_pins_with_no_led_are_claimed_dark(self):
        said = claims(_built(), PINS)
        assert all(not said[p].lights for p in (2, 7, 8, 9, 10))

    def test_a_backwards_led_is_claimed_dark(self):
        assert not claims(_built(**FLIPPED_BLUE), PINS)[4].lights

    def test_two_leds_on_one_pin_get_no_claim_rather_than_a_guess(self):
        build = _built(W_white={"1": "a54"})  # white's wire moved to pin 4
        assert 4 not in claims(build, PINS)


class TestDisagreements:
    def test_a_description_that_matches_the_board_has_none(self):
        assert disagreements(_built(), _run(**AS_BUILT_RUN), PINS) == []

    def test_an_led_that_stays_dark_contradicts_nothing(self):
        # Placed right, it can still be backwards, loose or broken. No picture
        # shows those, so the description is not wrong.
        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        assert disagreements(_built(), run, PINS) == []

    def test_light_where_the_description_says_none_can_come_disagrees(self):
        # Described backwards, so it cannot light from pin 4, but it did.
        [found] = disagreements(_built(**FLIPPED_BLUE), _run(**AS_BUILT_RUN), PINS)
        assert found.detail["pin"] == 4

    def test_a_different_colour_disagrees(self):
        run = _run(**{**AS_BUILT_RUN, "p4": ("white", 32)})
        assert [f.detail["pin"] for f in disagreements(_built(), run, PINS)] == [4]

    def test_colours_the_camera_cannot_tell_apart_do_not_disagree(self):
        run = _run(**{**AS_BUILT_RUN, "p6": ("orange", 44), "p4": ("unknown", 32)})
        assert disagreements(_built(), run, PINS) == []

    def test_light_more_than_a_column_from_the_legs_disagrees(self):
        near = _run(**{**AS_BUILT_RUN, "p4": ("blue", 33)})
        far = _run(**{**AS_BUILT_RUN, "p4": ("blue", 35)})
        assert disagreements(_built(), near, PINS) == []
        assert [f.detail["pin"] for f in disagreements(_built(), far, PINS)] == [4]

    def test_light_on_a_pin_described_as_empty_disagrees(self):
        run = _run(**{**AS_BUILT_RUN, "p7": ("red", 50)})
        [found] = disagreements(_built(), run, PINS)
        assert found.detail["pin"] == 7

    def test_an_unclear_pin_contradicts_nothing(self):
        run = _run(**{**AS_BUILT_RUN, "p4": "unclear"})
        assert disagreements(_built(), run, PINS) == []


class TestDiagnose:
    def test_a_matching_description_of_a_working_board_works(self):
        verdict = diagnose(_built(), LAB, _run(**AS_BUILT_RUN), PINS)
        assert verdict.works

    def test_a_confirmed_backwards_led_gets_the_fix_at_its_holes(self):
        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, run, PINS)
        assert not verdict.works
        kinds = [f.kind for f in verdict.findings]
        assert kinds == ["reversed_polarity"]  # not also "pin 4 did not light"
        assert set(verdict.findings[0].holes) == {"d31", "d32"}

    def test_a_description_the_board_contradicts_shows_no_fix(self):
        # Fixes computed from a wrong description would send the student to
        # move a leg that is fine.
        run = _run(**AS_BUILT_RUN)
        verdict = diagnose(_colours_swapped(), LAB, run, PINS)
        assert not verdict.works
        assert verdict.summary == MISMATCH_SUMMARY
        assert {f.kind for f in verdict.findings} == {"entry_mismatch"}

    def test_a_doubt_on_any_pin_withholds_works(self):
        run = _run(**{**AS_BUILT_RUN, "p8": "unclear"})
        verdict = diagnose(_built(), LAB, run, PINS)
        assert not verdict.works
        assert all(f.severity == "uncertain" for f in verdict.findings)

    def test_no_board_found_is_the_whole_answer(self):
        verdict = diagnose(_built(), LAB, Session(None), PINS)
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["board_not_found"]

    def test_a_description_contradicted_elsewhere_gets_no_hidden_causes(self):
        run = _run(**{**AS_BUILT_RUN, "p4": "dark", "p7": ("red", 50)})
        verdict = diagnose(_built(), LAB, run, PINS)
        assert verdict.summary == MISMATCH_SUMMARY
        assert "hidden_fault" not in _kinds(verdict)


class TestDirections:
    """Which way round an LED faces: from blinking when it lights."""

    def test_an_led_that_lights_faces_forwards_whatever_the_file_says(self):
        # Described backwards, but it lit from its pin: blinking wins, so there
        # is no "turn it around", and no mismatch either.
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, _run(**AS_BUILT_RUN), PINS)
        assert verdict.works, verdict.findings

    def test_leds_the_file_leaves_open_are_settled_by_blinking(self):
        verdict = diagnose(basicboard_as_seen(), LAB, _run(**AS_BUILT_RUN), PINS)
        assert verdict.works, verdict.findings
        assert any(
            "which way round the white, blue, green and red LEDs are" in n
            for n in verdict.notes
        )

    def test_an_open_led_that_stays_dark_is_never_called_backwards(self):
        # Nobody saw which way it faces, so the answer cannot say it is wrong.
        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        verdict = diagnose(basicboard_as_seen(), LAB, run, PINS)
        assert _kinds(verdict) == ["hidden_fault"]
        found = _hidden(verdict)
        assert found.holes == ("d32", "d31")  # the longer leg's hole first
        assert "longer leg should be in d32, on pin 4's side" in (found.suggestion or "")

    def test_a_wire_off_by_one_is_found_for_an_open_led(self):
        build = basicboard_as_seen()
        build["W_blue"].pins["2"] = Pin("a33")
        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        assert _kinds(diagnose(build, LAB, run, PINS)) == ["wrong_connection"]


class TestHiddenCauses:
    """Placed right, but dark: the next thing to try, one per check."""

    DARK = _run(**{**AS_BUILT_RUN, "p4": "dark"})

    def test_an_led_placed_right_that_stays_dark_gets_the_first_cause(self):
        verdict = diagnose(_built(), LAB, self.DARK, PINS)
        assert _kinds(verdict) == ["hidden_fault"]
        assert verdict.summary == (
            "Everything looks in the right place, but the blue LED did not light."
        )
        found = _hidden(verdict)
        assert found.severity == "error"
        assert found.detail == {"pin": 4, "step": 0}
        assert "wrong way round" in (found.suggestion or "")

    def test_each_check_moves_on_to_the_next_cause_then_stops(self):
        previous: dict[int, Suggested] = {}
        said = []
        for _ in range(5):
            verdict = diagnose(_built(), LAB, self.DARK, PINS, previous)
            found = _hidden(verdict)
            said.append((found.detail["step"], found.suggestion))
            previous = remember(verdict, LAB, self.DARK, previous)
        assert [step for step, _ in said] == [0, 1, 2, 3, 3]
        words = ["turn the LED around", "push it back in", "new one", "teacher"]
        for (_, suggestion), word in zip(said, words, strict=False):
            assert word in (suggestion or "")

    def test_an_led_moved_since_starts_again(self):
        elsewhere = {4: Suggested(4, "hidden_fault", 2, frozenset({"d40", "d41"}))}
        verdict = diagnose(_built(), LAB, self.DARK, PINS, elsewhere)
        assert _hidden(verdict).detail["step"] == 0

    def test_after_turning_a_backwards_led_as_told_the_next_cause_follows(self):
        # Described backwards and dark: turn it around. Turned, but still dark
        # (a leg is loose too), with the file not edited: the file is out of
        # date now, so the answer moves on instead of saying turn it again.
        first = diagnose(_built(**FLIPPED_BLUE), LAB, self.DARK, PINS)
        assert _kinds(first) == ["reversed_polarity"]
        previous = remember(first, LAB, self.DARK)
        again = diagnose(_built(**FLIPPED_BLUE), LAB, self.DARK, PINS, previous)
        assert _kinds(again) == ["hidden_fault"]
        assert _hidden(again).detail["step"] == 1

    def test_when_the_led_lights_again_the_answer_says_it_is_fixed(self):
        previous = remember(diagnose(_built(), LAB, self.DARK, PINS), LAB, self.DARK)
        verdict = diagnose(_built(), LAB, _run(**AS_BUILT_RUN), PINS, previous)
        assert verdict.works
        assert verdict.notes == (
            "Fixed since the last check: the blue LED lights now. It was most "
            "likely the wrong way round.",
        )
        assert remember(verdict, LAB, _run(**AS_BUILT_RUN), previous) == {}

    def test_any_fix_is_confirmed_when_the_led_lights(self):
        build = _built()
        build["W_blue"].pins["2"] = Pin("a33")
        previous = remember(diagnose(build, LAB, self.DARK, PINS), LAB, self.DARK)
        verdict = diagnose(_built(), LAB, _run(**AS_BUILT_RUN), PINS, previous)
        assert verdict.notes == ("Fixed since the last check: the blue LED lights now.",)

    def test_a_pin_that_could_not_be_judged_keeps_its_place(self):
        previous = remember(diagnose(_built(), LAB, self.DARK, PINS), LAB, self.DARK)
        blurred = _run(**{**AS_BUILT_RUN, "p4": "unclear"})
        unclear = diagnose(_built(), LAB, blurred, PINS, previous)
        previous = remember(unclear, LAB, blurred, previous)
        lost = Session(None)
        previous = remember(diagnose(_built(), LAB, lost, PINS), LAB, lost, previous)
        verdict = diagnose(_built(), LAB, self.DARK, PINS, previous)
        assert _hidden(verdict).detail["step"] == 1


class TestReviewFindings:
    """Cases a cold review found wrong (2026-09-29)."""

    def test_a_lab_blinking_cannot_test_never_reads_as_working(self):
        # Activity 3 has no LEDs. The answer must not say "your wiring works".
        lab = Netlist.from_json(json.loads((EXAMPLES / "activity3.json").read_text()))
        verdict = diagnose(lab, lab, _run(), PINS)
        assert not verdict.works
        assert "works" not in verdict.summary

    def test_a_fix_nothing_clearly_confirmed_is_not_stated_as_fact(self):
        # Every pin unclear: nothing contradicts the description, but nothing
        # confirms it either.
        run = _run(**{f"p{pin}": "unclear" for pin in PINS})
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, run, PINS)
        [fix] = [f for f in verdict.findings if f.kind == "reversed_polarity"]
        assert fix.severity == "uncertain"
        assert fix.message.startswith("Not yet confirmed")

    def test_a_pin_joined_by_wire_to_another_pins_led_gets_no_dark_claim(self):
        build = _built()
        build["MCU"].pins["D2"] = Pin("c52")
        build.components.append(
            Component(id="W_join", type="wire", pins={"1": Pin("a52"), "2": Pin("b53")})
        )
        said = claims(build, PINS)
        # Pin 2 can light the white LED through pin 3's strip, so claiming it
        # dark would wrongly contradict the board. (Joined, the two pins are one
        # point, and neither gets a claim; that is the safe side.)
        assert 2 not in said or said[2].lights

    def test_a_pin_tied_to_ground_is_still_reported_beside_a_fix(self):
        glows = {
            pin: Glow(pin, "lit", hole=f"c{col}", photo_xy=(0.0, 0.0), colour=colour)
            for pin, (colour, col) in BOARD.items()
            if pin not in (4, 6)
        }
        glows[4] = Glow(4, "dark")
        glows.update({pin: Glow(pin, "dark") for pin in (2, 7, 8, 9, 10)})
        run = Session(RECT, glows, shorted=frozenset({6}))
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, run, PINS)
        kinds = {f.kind for f in verdict.findings}
        assert {"reversed_polarity", "pin_tied_to_ground"} <= kinds


class TestHiddenCausesReview:
    """Cases a cold review of the hidden causes found wrong (2026-09-29)."""

    DARK = TestHiddenCauses.DARK

    def _two_dark_checks(self) -> dict[int, Suggested]:
        """What is remembered after "turn it around" and then "push it in"."""
        previous: dict[int, Suggested] = {}
        for _ in range(2):
            verdict = diagnose(_built(), LAB, self.DARK, PINS, previous)
            previous = remember(verdict, LAB, self.DARK, previous)
        assert previous[4].step == 1
        return previous

    def test_a_mismatch_does_not_make_working_leds_look_fixed_later(self):
        odd = _run(**{**AS_BUILT_RUN, "p7": ("red", 50)})
        previous = remember(diagnose(_built(), LAB, odd, PINS), LAB, odd)
        verdict = diagnose(_built(), LAB, _run(**AS_BUILT_RUN), PINS, previous)
        assert verdict.notes == ()

    def test_a_colour_in_doubt_is_not_later_called_fixed(self):
        # Red read as orange is a doubt, not a failure.
        doubt = _run(**{**AS_BUILT_RUN, "p6": ("orange", 44)})
        previous = remember(diagnose(_built(), LAB, doubt, PINS), LAB, doubt)
        verdict = diagnose(_built(), LAB, _run(**AS_BUILT_RUN), PINS, previous)
        assert verdict.notes == ()

    def test_a_typo_in_the_file_does_not_restart_the_causes(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import judge

        previous = self._two_dark_checks()
        broken = tmp_path / "as_built.json"
        broken.write_text("{")
        unreadable = judge(LAB, broken, self.DARK, PINS, previous)
        previous = remember(unreadable, LAB, self.DARK, previous)
        verdict = diagnose(_built(), LAB, self.DARK, PINS, previous)
        assert _hidden(verdict).detail["step"] == 2

    def test_a_mismatch_round_does_not_restart_the_causes(self):
        previous = self._two_dark_checks()
        odd = _run(**{**AS_BUILT_RUN, "p4": "dark", "p7": ("red", 50)})
        previous = remember(diagnose(_built(), LAB, odd, PINS), LAB, odd, previous)
        verdict = diagnose(_built(), LAB, self.DARK, PINS, previous)
        assert _hidden(verdict).detail["step"] == 2

    def test_no_hidden_cause_beside_a_fix_for_the_same_led(self):
        # Its short leg skips the resistor: "move this leg" and "placed right,
        # turn it around" would contradict each other. Placement comes first.
        verdict = diagnose(_built(LED_blue={"cathode": "h31"}), LAB, self.DARK, PINS)
        kinds = _kinds(verdict)
        assert "wrong_connection" in kinds
        assert "hidden_fault" not in kinds
        assert "led_does_not_light" not in kinds  # the fix says it better

    def test_a_possible_misplaced_leg_is_not_in_the_right_place(self):
        run = _run(**{**AS_BUILT_RUN, "p4": "dark", "p5": "unclear"})
        verdict = diagnose(_built(J_green={"1": "i36"}), LAB, run, PINS)
        assert "right place" not in verdict.summary
        assert "hidden_fault" not in _kinds(verdict)

    def test_a_dark_led_placed_right_is_still_said_beside_a_fix_elsewhere(self):
        # The green wire's fix does not explain why the blue LED is dark.
        green_off = _run(**{**AS_BUILT_RUN, "p4": "dark", "p5": "dark"})
        verdict = diagnose(_built(J_green={"1": "i36"}), LAB, green_off, PINS)
        pins_said = {f.detail.get("pin") for f in verdict.findings}
        assert 4 in pins_said

    def test_a_warning_elsewhere_is_not_everything_in_the_right_place(self):
        glows = dict(self.DARK.glows)
        del glows[7]
        run = Session(RECT, glows, shorted=frozenset({7}))
        verdict = diagnose(_built(), LAB, run, PINS)
        assert "hidden_fault" in _kinds(verdict)
        assert "right place" not in verdict.summary


class TestEditingTheFile:
    def test_a_file_with_a_typo_says_so_instead_of_crashing(self, tmp_path):
        # The file is edited by hand between checks, mid-demo.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import judge

        broken = tmp_path / "as_built.json"
        broken.write_text('{"board": "WB-102", "components": [')
        verdict = judge(LAB, broken, _run(**AS_BUILT_RUN), PINS)
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["entry_unreadable"]

    def test_a_file_of_the_wrong_shape_says_so_instead_of_crashing(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import judge

        odd = tmp_path / "as_built.json"
        odd.write_text('{"board": "WB-102", "components": [{"id": "X", "pins": []}]}')
        verdict = judge(LAB, odd, _run(**AS_BUILT_RUN), PINS)
        assert [f.kind for f in verdict.findings] == ["entry_unreadable"]

    def test_a_build_written_like_a_lab_is_refused(self, tmp_path):
        # Legs on nets say nothing about where anything is, so no fix could
        # name a hole. The lab's own file handed in as the build, by mistake.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import judge

        verdict = judge(
            LAB, EXAMPLES / "basicboard_demo.json", _run(**AS_BUILT_RUN), PINS
        )
        assert not verdict.works
        assert [f.kind for f in verdict.findings] == ["entry_unreadable"]

    def test_holes_typed_in_capitals_are_drawn_not_crashed_on(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import draw_verdict

        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        build = _built(LED_blue={"anode": "D31", "cathode": "D32"})
        verdict = diagnose(build, LAB, run, PINS)
        assert verdict.findings[0].holes
        picture = draw_verdict(np.zeros((H, W, 3), np.uint8), run, verdict, "note")
        assert picture.shape == (H, W, 3)

    def test_the_file_is_read_afresh_each_time(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import judge

        path = tmp_path / "as_built.json"
        path.write_text(_built().to_json())
        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        assert _kinds(judge(LAB, path, run, PINS)) == ["hidden_fault"]
        path.write_text(_built(**FLIPPED_BLUE).to_json())
        assert _kinds(judge(LAB, path, run, PINS)) == ["reversed_polarity"]


RUNS = Path(__file__).resolve().parent.parent / "data" / "cache" / "blink"


class TestRecordedRun:
    """The live demo of 2026-09-29, on the real board, held in a hand."""

    @staticmethod
    def _session(name: str) -> Session:
        run = RUNS / name
        if not run.exists():
            pytest.skip("recorded blink run not present")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from blink import load

        from breadboard.blink import analyse

        frames, pins, shorted = load(run)
        return analyse(frames, pins, shorted)

    def test_the_demo_board_as_described_works(self):
        verdict = diagnose(
            _built(), LAB, self._session("basicboard-2026-09-29-demo"), PINS
        )
        assert verdict.works, verdict.findings

    def test_a_flip_the_description_misses_gets_the_first_hidden_cause(self):
        # The blue LED turned around on the board, the file not edited: it is
        # placed right, so the first thing to try is turning it around.
        session = self._session("basicboard-2026-09-29-demo-flipped-unentered")
        verdict = diagnose(_built(), LAB, session, PINS)
        errors = [f for f in verdict.findings if f.severity == "error"]
        assert [(f.kind, f.detail["pin"]) for f in errors] == [("hidden_fault", 4)]
        assert errors[0].holes == ("d32", "d31")

    def test_colours_entered_wrong_are_caught_as_a_mismatch(self):
        session = self._session("basicboard-2026-09-29-demo")
        verdict = diagnose(_colours_swapped(), LAB, session, PINS)
        assert verdict.summary == MISMATCH_SUMMARY
        found = [f.detail["pin"] for f in verdict.findings if f.kind == "entry_mismatch"]
        assert found == [3, 4]
        assert not [f for f in verdict.findings if f.severity == "error"]

    def test_the_board_as_a_camera_sees_it_works_with_directions_from_blinking(self):
        session = self._session("basicboard-2026-09-29-demo")
        verdict = diagnose(basicboard_as_seen(), LAB, session, PINS)
        assert verdict.works, verdict.findings
        assert any("which way round" in n for n in verdict.notes)

    def test_the_board_works_on_a_laptop_camera_too(self):
        # 2026-10-02: first said "Not sure yet", the white LED's colour unknown.
        session = self._session("basicboard-2026-10-02-laptop")
        verdict = diagnose(basicboard_as_seen(), LAB, session, PINS)
        assert verdict.works, verdict.findings

    def test_a_flipped_led_a_camera_cannot_see_gets_turn_it_around(self):
        session = self._session("basicboard-2026-09-29-demo-flipped")
        verdict = diagnose(basicboard_as_seen(), LAB, session, PINS)
        errors = [f for f in verdict.findings if f.severity == "error"]
        assert [f.kind for f in errors] == ["hidden_fault"]
        assert "d32" in (errors[0].suggestion or "")

    def test_a_flip_the_description_records_gets_the_fix(self):
        # Then the file edited to match: the fix names the blue LED's holes.
        session = self._session("basicboard-2026-09-29-demo-flipped")
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, session, PINS)
        assert not verdict.works
        errors = [f for f in verdict.findings if f.severity == "error"]
        assert [f.kind for f in errors] == ["reversed_polarity"]
        assert set(errors[0].holes) == {"d31", "d32"}
