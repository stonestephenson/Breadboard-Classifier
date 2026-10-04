# The circuit model and checker

The reasoning half of the system: given a circuit, decide whether it is right and — if not —
which changes would fix it. Deterministic and fully tested; no model, no camera, no hardware.

Four modules under `breadboard/`, each usable on its own (the camera side, `rectify.py`,
`blink.py`, `verify.py` and `diagnose.py`, is described in their docstrings and
`ARCHITECTURE.md`):

| Module | Responsibility |
|--------|---------------|
| `board.py` | WB-102 geometry — which holes share an electrical node |
| `netlist.py` | The data format everything produces and consumes |
| `graph.py` | Topology, and deciding when two circuits are the same one |
| `check.py` | Diagnosis: findings with a repair a student can carry out |

Run the tests with `./venv/bin/python -m pytest tests/ -q` (about two minutes with the local
photos and recordings; see "Test data" in `CLAUDE.md`).

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
| `swapped_connections` | Two legs, usually two wire ends, need to swap places |
| `reversed_polarity` | A polarised part is in backwards |
| `extra_component` | A part the lab does not use |
| `missing_component` | A part the lab needs that is absent |
| `circuit_differs` | Differs, and the search found no run of changes that ends at the lab's circuit; or, as a caution after three fixes, more follow |

### Several mistakes at once

No single change fixes a circuit with two mistakes in it, so the search does not ask for one. It
scores the student's circuit by how much of the lab's it has. For each lab link between two named
pins: two points for each of its parts that hangs, one after another, off the right pin at either
end (one if that part is an LED the wrong way round), and two more when the link is whole. For a
sensor: two points for each leg on the right named point. Then it takes the single change that
raises the score most, makes it, and looks again, until the circuits match. A change that makes
them match outright always wins, so one mistake gets exactly the answer a search for a single fix
would give. A broken chain scores for each piece still in place, so two breaks in one LED's path
are mended one at a time, as is a wire off plus an LED backwards.

Fixes are passed on only when the whole run of them ends at the lab's circuit. A run that stalls
short of it says nothing, however far it got, because a step can raise the score and still be the
wrong thing to do. (A cold review found exactly that in a third of stalled runs.) At most three
fixes are shown, the first ones made, with a note that more follow.

Each fix records which pins' links it put right (`detail["pins"]`), so a caller with
measurements can check each fix against its own pins (`diagnose.py`). Two moves that send each
leg to where the other was are said once, as a swap. Two fixes to one part are said once too:
"The blue LED is in the wrong holes. Put its longer leg in d32 and its shorter leg in d31."

A part counts for one link only. The parts of a link that is whole are that link's, so a working
LED's resistor is not taken for the one a broken LED lacks.

Only moves a student could make are tried. A wire's end can go anywhere. An LED's leg stays
within four holes of its other leg, and a resistor's within twenty; a leg may always move a
little further than the part spans now, since the example circuits are drawn wider than real
parts. A sensor's legs are one rigid row, so none moves alone. Without this the search would
stretch an LED across the board to meet a wire that strayed.

Limits: missing or extra parts are reported first and stop the search. A missing wire is only
reported when the lab's own file has more wires than the build; otherwise nothing here adds a
part, so a wire the student left out is not suggested. Two changes that only help together give no first step:
both ends of one wire in the wrong place, or two wires that both join the same two pins. A lab link through a junction that
carries no Metro Mini pin cannot be scored on its own, so such a lab gets a fix only when one
change makes the whole circuit match.

### Where a moved leg is sent

The suggestion names a hole with no leg in it, in the right strip: the leg's own row if that hole
is free ("from a33 to a32"), otherwise the nearest free one.

**`scope`** is `"lab"` for parts the student was asked to place and `"baseline"` for the
factory-wired BasicBoard. Both are reported — a knocked-loose factory LED is a real problem — but
the caller can phrase them differently ("did you mean to move this?" versus a lab error).

### Choosing between equally valid repairs

Several single moves often fix the same break. With a wire ending in a33 and an LED's leg in a32,
moving either closes the gap, and nothing says which one the student misplaced. Every working
repair is collected and then ranked. A repair that leaves a wire joining nothing comes last: when
a wire's end has strayed far, carrying the part's own leg to where the wire should have gone also
makes the circuits match, but it strands the wire and stretches the part across the board. Then
the shortest move, since one hole off is the commonest slip, then a wire's end before a part's
leg, since a wire is the easier thing to move. Returning the first repair found instead blames
whichever component happened to come first in the list.

### The lab has no places

Placement is free, so a lab can only say which legs are joined, never where. A lab file can be
written that way: each leg on a named net, `{"net": "white"}`, instead of in a hole
(`examples/basicboard_demo.json`). A correct build can also serve as the lab. Either way the
checker uses only the circuit. Where a lab's own file puts things is never looked at: not as
places to try moving a leg to, and not to decide which of two equal repairs to suggest. (It once
was, on the idea that the lab is a reference build. The holes in a lab file are made up.)

## Findings and data flow

One check runs through these modules, each handing the next a plain data structure:

| Step | Module | Hands on |
|------|--------|----------|
| Find the board, name its holes | `rectify.py` | a `Rectification` (photo pixel to hole and back) |
| Blink each pin, see what lit | `blink.py` | a `Session`: one `Glow` per pin, pins set aside, pins tied to ground |
| Does it work? | `verify.py` | a `Verdict` of findings, against the lab's `Netlist` |
| The build, as described | a circuit file | a `Netlist` (every leg in a hole) |
| Trust it? What to move? Hidden causes? | `diagnose.py`, using `check.py` and `graph.py` | a `Verdict` |
| Draw and print it | `tools/blink.py` (`draw_verdict`), `tools/live.py` | the picture |

Wires are made into joins before anything is compared (`graph.wires_joined`): two strips a wire
connects are one point.

Every stage speaks in `Finding`s (`check.py`). The kinds above come from the checker. The others:

| Kind | From | Meaning |
|------|------|---------|
| `led_does_not_light`, `wrong_led_on_pin`, `unexpected_led` | `verify.py` | What a pin lit differs from the lab |
| `pin_tied_to_ground` | `verify.py` | The pin read low when driven, and was released |
| `not_checked`, `cannot_check`, `board_not_found`, `imperfect_view` | `verify.py` | Doubt: unclear photos, an LED blinking cannot test, no board, a poor fit |
| `entry_mismatch` | `diagnose.py` | Light where the described build predicts none; no fix is shown |
| `hidden_fault` | `diagnose.py` | Placed right but dark: the next cause to try |
| `entry_unreadable` | `tools/blink.py` | The build file could not be read, or is written like a lab |

`Finding.detail` carries what later stages need. Its keys:

| Key | Meaning |
|-----|---------|
| `pin` | The Metro Mini digital pin the finding is about, as an int |
| `pins` | Names (`"D4"`, `"A0"`) of the pins whose links a checker fix puts right |
| `glow` | The pin whose lit LED to ring on the photo |
| `leg`, `from`, `to` | One leg moved: its name, its hole, the free hole to send it to |
| `legs` | A whole part re-placed: each leg's hole |
| `swapped` | The two holes (or two sensor legs) to exchange |
| `step` | Which hidden cause was suggested (`diagnose.HIDDEN_CAUSES`) |
| `type` | The part's type |
| `seen`, `reason`, `read_as`, `might_be`, `confidence` | What was observed, for messages |

Pins are ints in `blink.py`, `verify.py` and `diagnose.py`, and names like `"D4"` in a `Netlist`.
A circuit file's parts take `attrs`: `color` (LED), `ohms` (resistor), `kind` (sensor), and
`direction: "unknown"` (an LED whose way round was not seen). A leg is `{"hole": "d31"}` or, in
a lab, `{"net": "white"}`.

## Validated against real hardware

`tests/test_basicboard.py` builds the BasicBoard's measured topology — pin 2 → red, 3 → white,
4 → green, 5 → blue, each through 330 Ω to ground, recovered by driving pins over USB on
2026-08-11 (see `docs/FIRMWARE_PROTOCOL.md`). It confirms the board as shipped is clean, stays
clean when slid along the board, and that a displaced or reversed LED is found and named by
colour.

## What is not built yet

- **Lab circuits for the remaining activities.** The demo board, the BasicBoard as shipped and
  Activity 3 are written; the curriculum's two design-challenge activities are not. A lab file
  here must name each pin, so a lab that leaves the choice of pin to the student cannot be
  written yet.
- **No fix that adds a part.** A wire the student left out is not suggested (unless the lab's
  file itself has more wires than the build); the answer falls back to "could not work out what
  to change" and what blinking saw.
- **No natural-language rendering.** Findings carry a `message`, but turning a set of findings
  into a paragraph for a student is a separate layer.
- **Nothing produces a Netlist yet from a photo.** The probe produces pin connectivity; wiring
  that into a Netlist is the next integration step. For LED labs, `breadboard/verify.py` already
  answers "does it work?" without one: it checks what each pin lit (blink and watch) against
  what the lab file expects, and returns findings in this module's `Finding` type.
