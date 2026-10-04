"""WB-102 breadboard geometry: holes, electrical nodes, and physical positions.

Two facts about a breadboard drive this whole project:

1. **Holes in a column strip are electrically identical.** Rows a-e in column 24
   are one node; f-j are another. So a lead's *row* never matters — only its
   column and which half it is in. That collapses 830 physical holes into 130
   electrical nodes and relaxes the precision the camera must achieve by 5x
   along the axis that would otherwise hurt most.

2. **The power rails run the whole length** on the kit's real board (measured;
   see RAILS_SPLIT). Some boards split each rail into two separate runs, where
   students wire power into one half and ground into the other, see nothing
   work, and cannot tell why. RAILS_SPLIT models that too, which is what lets
   the checker name that error on such a board.

Hole addressing (also the format curriculum authors will write):

    terminal   "<row><column>"    rows a-j, columns 1-63     e.g. "c24", "j63"
    rail       "<rail>:<index>"   index 1-50                 e.g. "p1+:37"

Rail names follow the sibling generator repo so circuit configs stay portable:
p1+ and p1- are the two rails along one long edge, p2+ and p2- the other.
"""

from __future__ import annotations

import re

PITCH_MM = 2.54
CHANNEL_PITCHES = 3  # centre channel is 7.62mm = 0.3", the DIP package width

ROWS_UPPER = "abcde"
ROWS_LOWER = "fghij"
ROWS = ROWS_UPPER + ROWS_LOWER
N_COLS = 63

RAILS = ("p1+", "p1-", "p2+", "p2-")
RAIL_HOLES = 50
# Whether each rail is two separate runs of RAIL_SEGMENT_HOLES, as the generator's
# board_spec.json says. The kit's real board says not: on 2026-09-28 and again on
# 2026-09-29, LEDs grounded into the p2- rail at holes 11-25 lit through a single
# ground wire at hole 45-48, which a split between holes 25 and 26 would have
# made impossible. So the rails are modelled as continuous. The split model, and
# the dead_rail_segment rule it enables, stay for a board that turns out split.
RAILS_SPLIT = False
RAIL_SEGMENT_HOLES = 25  # the length of each run, when RAILS_SPLIT
RAIL_START_COL = 3  # rail hole 1 sits over terminal column 3
# Outermost terminal row (a or j) to the nearest rail row. Measured, not specified:
# the rectifier's hole template, fitted to real photos, puts it at 2.83 pitches, and
# its terminal holes agree with this module to 0.02 pitch. The generator's
# board_spec.json says 4.0 mm, which does not match the real board.
RAIL_OFFSET_MM = 7.2

_TERMINAL_RE = re.compile(r"^([a-j])([0-9]{1,2})$")
_RAIL_RE = re.compile(r"^(p[12][+-]):([0-9]{1,2})$")


class BoardError(ValueError):
    """A hole reference that this board does not have."""


def parse_hole(hole: str) -> tuple[str, int]:
    """Normalise a hole reference into (row_or_rail, index).

    Accepts either case; authors and students will not be consistent.
    """
    if not isinstance(hole, str):
        raise BoardError(f"hole must be a string, got {type(hole).__name__}")
    text = hole.strip().lower()

    m = _TERMINAL_RE.match(text)
    if m:
        row, col = m.group(1), int(m.group(2))
        if not 1 <= col <= N_COLS:
            raise BoardError(f"column {col} out of range 1-{N_COLS}: {hole!r}")
        return row, col

    m = _RAIL_RE.match(text)
    if m:
        rail, idx = m.group(1), int(m.group(2))
        if rail not in RAILS:
            raise BoardError(f"unknown rail {rail!r}: {hole!r}")
        if not 1 <= idx <= RAIL_HOLES:
            raise BoardError(f"rail hole {idx} out of range 1-{RAIL_HOLES}: {hole!r}")
        return rail, idx

    raise BoardError(f"not a valid WB-102 hole: {hole!r}")


def node_of(hole: str) -> str:
    """The electrical node a hole belongs to.

    Terminal nodes are named "T:<half>:<column>" where half is the row group
    that shares the strip. Rail nodes are "R:<rail>:<segment>"; with continuous
    rails (RAILS_SPLIT False) every rail is one segment.
    """
    where, index = parse_hole(hole)
    if where in ROWS:
        half = "ae" if where in ROWS_UPPER else "fj"
        return f"T:{half}:{index}"
    segment = 2 if RAILS_SPLIT and index > RAIL_SEGMENT_HOLES else 1
    return f"R:{where}:{segment}"


def nodes() -> list[str]:
    """Every electrical node on the board, in a stable order."""
    out = [f"T:{half}:{col}" for col in range(1, N_COLS + 1) for half in ("ae", "fj")]
    segments = (1, 2) if RAILS_SPLIT else (1,)
    out += [f"R:{rail}:{seg}" for rail in RAILS for seg in segments]
    return out


def _all_holes() -> list[str]:
    out = [f"{row}{col}" for col in range(1, N_COLS + 1) for row in ROWS]
    out += [f"{rail}:{i}" for rail in RAILS for i in range(1, RAIL_HOLES + 1)]
    return out


ALL_HOLES: tuple[str, ...] = tuple(_all_holes())


def holes_of(node: str) -> list[str]:
    """Every hole that belongs to a node, in board order."""
    return [hole for hole in ALL_HOLES if node_of(hole) == node]


def is_rail(node: str) -> bool:
    return node.startswith("R:")


def rail_of(node: str) -> str | None:
    """The rail a node belongs to, or None for terminal nodes."""
    return node.split(":")[1] if is_rail(node) else None


def same_rail_different_segment(a: str, b: str) -> bool:
    """True for two nodes on the same rail but on opposite sides of its split.

    This is the signature of a specific, common, invisible student error: the
    connection looks right because both holes are in the same painted stripe,
    but the board has no metal joining them.
    """
    if not (is_rail(a) and is_rail(b)):
        return False
    ra, sa = a.split(":")[1:]
    rb, sb = b.split(":")[1:]
    return ra == rb and sa != sb


def hole_position(hole: str) -> tuple[float, float]:
    """Physical position in millimetres from the board's top-left hole.

    Used to convert between the checker's world and the rectified image, and to
    express "move it one column left" as a real distance.
    """
    where, index = parse_hole(hole)
    if where in ROWS:
        x = (index - 1) * PITCH_MM
        row_i = ROWS.index(where)
        # Rows a-e are contiguous; f-j sit one channel width further down.
        y = row_i * PITCH_MM
        if where in ROWS_LOWER:
            y += (CHANNEL_PITCHES - 1) * PITCH_MM
        return x, y

    # Rail holes sit in groups of five with a gap between groups, and the rails
    # run outside the terminal area on both long edges. Each red (+) stripe sits
    # on the same side of its blue one, so + is the outer row along row a and the
    # inner row along row j.
    group, within = divmod(index - 1, 5)
    x = (RAIL_START_COL - 1 + group * 6 + within) * PITCH_MM
    if where.startswith("p1"):
        inner = -RAIL_OFFSET_MM
        y = inner - PITCH_MM if where.endswith("+") else inner
    else:
        _, row_j = hole_position(f"{ROWS[-1]}1")
        inner = row_j + RAIL_OFFSET_MM
        y = inner if where.endswith("+") else inner + PITCH_MM
    return x, y
