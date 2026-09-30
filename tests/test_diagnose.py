"""Which leg to move: the checker on a described build, checked against blinking.

The behaviours that matter:

- A description is trusted only where blinking confirms it. Any pin that lit
  something the description does not predict, or stayed dark when it should
  have lit, means the description is wrong, and then no fix is shown at all.
- A backwards LED, or a pin with no LED, is predicted to stay dark. That is what
  lets a description say "the blue LED is backwards" and be confirmed.
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
from breadboard.diagnose import MISMATCH_SUMMARY, claims, diagnose, disagreements
from breadboard.netlist import Component, Netlist, Pin
from breadboard.rectify import CANONICAL_SIZE, Rectification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from make_examples import basicboard_as_built, basicboard_demo

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

    def test_an_led_described_right_way_round_that_stays_dark_disagrees(self):
        # The blue LED was flipped on the board, but not in the description.
        run = _run(**{**AS_BUILT_RUN, "p4": "dark"})
        [found] = disagreements(_built(), run, PINS)
        assert found.detail["pin"] == 4
        assert "nothing lit" in found.message

    def test_an_led_described_backwards_that_lights_disagrees(self):
        # The description says the blue LED is backwards, but it lit: the
        # description is wrong, and "turn it around" would be a false accusation.
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
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, run, PINS)
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
        assert judge(LAB, path, run, PINS).summary == MISMATCH_SUMMARY
        path.write_text(_built(**FLIPPED_BLUE).to_json())
        kinds = [f.kind for f in judge(LAB, path, run, PINS).findings]
        assert kinds == ["reversed_polarity"]


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

    def test_a_flip_the_description_misses_is_caught_as_a_mismatch(self):
        # The blue LED turned around on the board, the file not yet edited.
        session = self._session("basicboard-2026-09-29-demo-flipped-unentered")
        verdict = diagnose(_built(), LAB, session, PINS)
        assert verdict.summary == MISMATCH_SUMMARY
        found = [f.detail["pin"] for f in verdict.findings if f.kind == "entry_mismatch"]
        assert found == [4]
        # What blinking itself found is still said; no fix is.
        assert "led_does_not_light" in [f.kind for f in verdict.findings]
        assert "reversed_polarity" not in [f.kind for f in verdict.findings]

    def test_a_flip_the_description_records_gets_the_fix(self):
        # Then the file edited to match: the fix names the blue LED's holes.
        session = self._session("basicboard-2026-09-29-demo-flipped")
        verdict = diagnose(_built(**FLIPPED_BLUE), LAB, session, PINS)
        assert not verdict.works
        errors = [f for f in verdict.findings if f.severity == "error"]
        assert [f.kind for f in errors] == ["reversed_polarity"]
        assert set(errors[0].holes) == {"d31", "d32"}
