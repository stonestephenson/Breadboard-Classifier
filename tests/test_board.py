"""WB-102 geometry: which holes share an electrical node.

This is the foundation everything else stands on. If hole-to-node mapping is
wrong, every circuit we extract is wrong in a way no later stage can detect.
"""

from __future__ import annotations

import pytest

from breadboard.board import (
    ALL_HOLES,
    RAILS,
    BoardError,
    hole_position,
    node_of,
    nodes,
    parse_hole,
)


class TestTerminalStrips:
    def test_rows_a_to_e_in_a_column_share_a_node(self):
        assert node_of("a24") == node_of("c24") == node_of("e24")

    def test_rows_f_to_j_in_a_column_share_a_node(self):
        assert node_of("f24") == node_of("h24") == node_of("j24")

    def test_the_two_halves_of_a_column_are_separate(self):
        # The centre channel splits every column. This is the single most
        # important electrical fact about a breadboard.
        assert node_of("e24") != node_of("f24")

    def test_adjacent_columns_are_separate(self):
        assert node_of("a24") != node_of("a25")

    def test_every_column_is_addressable(self):
        assert node_of("a1") != node_of("a63")
        for col in range(1, 64):
            assert node_of(f"a{col}")


class TestPowerRails:
    def test_holes_within_a_rail_segment_share_a_node(self):
        assert node_of("p1+:1") == node_of("p1+:13") == node_of("p1+:25")

    def test_rail_segments_are_electrically_split(self):
        # The WB-102's rails are two 25-hole runs, not one continuous strip.
        # Students wire power into one half and ground into the other and
        # cannot see why nothing works. We must model this.
        assert node_of("p1+:25") != node_of("p1+:26")

    def test_second_segment_is_internally_connected(self):
        assert node_of("p1+:26") == node_of("p1+:50")

    def test_the_four_rails_are_mutually_separate(self):
        first_segment_nodes = {node_of(f"{rail}:1") for rail in RAILS}
        assert len(first_segment_nodes) == len(RAILS)

    def test_rails_are_separate_from_terminal_strips(self):
        assert node_of("p1+:1") != node_of("a1")


class TestNodeInventory:
    def test_total_hole_count_matches_an_830_point_board(self):
        assert len(ALL_HOLES) == 830

    def test_hole_counts_split_correctly_between_terminal_and_rail(self):
        terminal = [h for h in ALL_HOLES if ":" not in h]
        rail = [h for h in ALL_HOLES if ":" in h]
        assert len(terminal) == 63 * 10 == 630
        assert len(rail) == 4 * 50 == 200

    def test_total_node_count(self):
        # 63 columns x 2 halves = 126 terminal nodes, plus 4 rails x 2
        # segments = 8. Every lead we localise resolves to one of these 134.
        assert len(nodes()) == 134
        assert len({node_of(h) for h in ALL_HOLES}) == 134

    def test_every_hole_maps_to_a_known_node(self):
        known = set(nodes())
        for hole in ALL_HOLES:
            assert node_of(hole) in known


class TestParsing:
    @pytest.mark.parametrize("hole,expected", [
        ("a1", ("a", 1)),
        ("j63", ("j", 63)),
        ("C24", ("c", 24)),  # accept upper case; students and authors will vary
    ])
    def test_terminal_holes_parse(self, hole, expected):
        assert parse_hole(hole) == expected

    @pytest.mark.parametrize("hole", [
        "k1",       # no such row
        "a0",       # columns are 1-based
        "a64",      # past the end of the board
        "a",        # no column
        "24",       # no row
        "p3+:1",    # no such rail
        "p1+:0",
        "p1+:51",
        "p1+",      # rail without a hole index
        "",
    ])
    def test_invalid_holes_are_rejected(self, hole):
        with pytest.raises(BoardError):
            parse_hole(hole)

    def test_node_of_rejects_invalid_holes_too(self):
        with pytest.raises(BoardError):
            node_of("z99")


class TestGeometry:
    def test_hole_position_is_in_millimetres_from_board_origin(self):
        x1, _ = hole_position("a1")
        x2, _ = hole_position("a2")
        assert x2 - x1 == pytest.approx(2.54, abs=1e-6)

    def test_rows_are_one_pitch_apart(self):
        _, y1 = hole_position("a10")
        _, y2 = hole_position("b10")
        assert abs(y2 - y1) == pytest.approx(2.54, abs=1e-6)

    def test_centre_channel_is_three_pitches(self):
        # 7.62mm = 0.3", the DIP package width. The rectifier relies on this
        # exact figure to index across the channel.
        _, y_e = hole_position("e10")
        _, y_f = hole_position("f10")
        assert abs(y_f - y_e) == pytest.approx(3 * 2.54, abs=1e-6)

    def test_positions_are_distinct_for_every_hole(self):
        assert len({hole_position(h) for h in ALL_HOLES}) == len(ALL_HOLES)
