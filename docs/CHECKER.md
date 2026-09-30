# The circuit model and checker

The reasoning half of the system: given a circuit, decide whether it is right and — if not —
what one change would fix it. Deterministic and fully tested; no model, no camera, no hardware.

Four modules under `breadboard/`, each usable on its own:

| Module | Responsibility |
|--------|---------------|
| `board.py` | WB-102 geometry — which holes share an electrical node |
| `netlist.py` | The data format everything produces and consumes |
| `graph.py` | Topology, and deciding when two circuits are the same one |
| `check.py` | Diagnosis: findings with a repair a student can carry out |

Run the tests with `./venv/bin/python -m pytest tests/ -q` — 109 currently, ~0.6s.

## The two facts the board model encodes

**Holes in a column strip are electrically identical.** Rows a–e in column 24 are one node;
f–j are another. So a lead's *row* never matters, only its column and half. That turns 830
physical holes into **130 electrical nodes**, and relaxes by 5× the precision the camera has to
achieve along the axis that would otherwise hurt most.

**The power rails run the whole length.** The generator's board spec said each rail was two
separate 25-hole runs, joined by nothing but a continuous painted stripe. The kit's real board
disagrees: on 2026-09-28 and 2026-09-29, LEDs grounded into one half of a rail lit through a
ground wire in the other half. So `board.RAILS_SPLIT` is False. On a board with split rails
(set it True), a connection into the wrong half looks correct and simply does nothing, and
`check.py` names it (`dead_rail_segment`).

## Why comparison happens on an abstraction

The curriculum team confirmed lab instructions **cannot** specify hole positions. Two builds of
one lab may therefore share no positions at all. Worse, several genuinely different wirings are
equally correct — a resistor works on either side of an LED. Comparing netlists directly would
flag working circuits as broken, the failure this system can least afford.

So `graph.py` compares in two steps:

1. **Anonymise placement.** Breadboard strips become unlabelled vertices; only the Metro Mini's
   pins carry names. Matching means finding a bijection between anonymous strips while holding
   D2 to D2. Placement freedom then costs nothing, and the named pins keep the search tightly
   anchored — this is *not* open graph isomorphism, which is what makes plain backtracking
   sufficient.

2. **Collapse series chains.** A run of two-terminal parts through nodes nothing else touches is
   electrically one thing whose internal order is irrelevant. Collapsing it to a single edge
   carrying an unordered bag of parts makes resistor-then-LED equal LED-then-resistor, while
   *direction* is preserved so a reversed LED still fails.

Attributes are compared only where the reference states them: an author who omits an LED colour
means "any colour". Wire colour is never significant — the curriculum team confirmed colours are
conventions students do not reliably follow.

## Uncertainty comes before accusation

Every pin carries a confidence and a list of near-miss alternatives. Before reporting any error,
`check.py` asks whether a *misreading* explains the difference: if swapping one low-confidence
endpoint for an alternative would make the circuit correct, it emits a single
`uncertain_reading` finding and suppresses everything else.

Only single substitutions are tried. Two independent misreadings in one photo is far less likely
than one, and trying every combination would let the checker explain away real errors.

This is the machinery behind abstention. A confident wrong answer carrying our authority is the
worst thing this system can do to a 13-year-old.

## Findings

Each carries a kind, a message written for a young student, the components and holes involved, a
concrete suggestion, and a `scope`.

Needing no reference circuit:

| Kind | Meaning |
|------|---------|
| `shorted_component` | Both legs in one strip, so current skips past the part |
| `short_circuit` | Power wired straight to ground through wires alone |
| `led_without_resistor` | An LED across a supply with nothing to limit current |
| `dead_rail_segment` | Wired into the unpowered half of a split rail (split-rail boards only) |

Needing the lab's intended circuit:

| Kind | Meaning |
|------|---------|
| `uncertain_reading` | A near-miss reading would make this correct; ask for a better view |
| `wrong_connection` | One leg is in the wrong place; names where it is and where it goes |
| `reversed_polarity` | A polarised part is in backwards |
| `extra_component` | A part the lab does not use |
| `missing_component` | A part the lab needs that is absent |
| `circuit_differs` | Differs, and no single change explains it |

**`scope`** is `"lab"` for parts the student was asked to place and `"baseline"` for the
factory-wired BasicBoard. Both are reported — a knocked-loose factory LED is a real problem — but
the caller can phrase them differently ("did you mean to move this?" versus a lab error).

### Choosing between equally valid repairs

Several single moves often fix the same break. With a resistor ending at a20 and an LED starting
at a21, moving either closes the gap — but the resistor is exactly where the lab puts it, so the
LED is what actually moved. Every working repair is collected and then ranked, preferring to move
a leg that sits somewhere the lab never mentions, then the shortest move. Returning the first
repair found instead blames whichever component happened to come first in the list.

## Validated against real hardware

`tests/test_basicboard.py` builds the BasicBoard's measured topology — pin 2 → red, 3 → white,
4 → green, 5 → blue, each through 330 Ω to ground, recovered by driving pins over USB on
2026-08-11 (see `docs/FIRMWARE_PROTOCOL.md`). It confirms the board as shipped is clean, stays
clean when slid along the board, and that a displaced or reversed LED is found and named by
colour.

## What is not built yet

- **No lab reference circuits.** Waiting on the curriculum PDFs (`docs/OPEN_QUESTIONS.md` Q4).
  Authoring 7 of these is a bounded content task; an authoring tool would help.
- **Multi-terminal parts (`sensor3`, `sensor4`) exist in the netlist but not in the graph.** Only
  two-terminal parts become edges today. Sensors need to become labelled vertices before any lab
  using them can be checked.
- **No natural-language rendering.** Findings carry a `message`, but turning a set of findings
  into a paragraph for a student is a separate layer.
- **Nothing produces a Netlist yet from a photo.** The probe produces pin connectivity; wiring
  that into a Netlist is the next integration step. For LED labs, `breadboard/verify.py` already
  answers "does it work?" without one: it checks what each pin lit (blink and watch) against
  what the lab file expects, and returns findings in this module's `Finding` type.
