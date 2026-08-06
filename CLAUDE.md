# CLAUDE.md — Breadboard Classifier

**Read this file first.** It orients you; the detail lives in `ARCHITECTURE.md` (the design and
why) and `docs/DECISIONS.md` (what we rejected and why — read before proposing anything that
sounds like a good idea, it may already be a dead end we walked down).

## What this project is

A middle/high-school student builds a circuit on a breadboard, photographs it through their
school Chromebook's browser, and the system tells them **what is wrong with their wiring and
how to fix it**, in language a 13-year-old can act on.

Part of Sonoma State's Learning by Making (LbyM) / STEMACES program. The delivery vehicle is
the existing LbyM web app (AWS Amplify + DynamoDB + API Gateway, Firebase/Gmail auth, runs in
Chrome on school-issued machines).

## Status

**Design phase. No implementation yet.** The architecture below is settled in shape; the
riskiest assumption (rectification on real cluttered photos) is not yet validated.

Immediate work, in order:

1. **Rectifier spike** — validate homography fitting on the 248 real photos. Go/no-go for the
   whole architecture. Not started.
2. **Netlist schema + checker** — buildable today against hand-written netlists, no CV
   required. Not started.
3. Everything else is gated on the open questions below.

## The one principle

Answers can come from three places:

1. **Prior knowledge** — what we know before the photo exists (board geometry, kit contents,
   the lab's intended circuit).
2. **Measurement** — what the hardware can physically tell us (the Arduino probing its own
   connections over USB).
3. **Inference** — what a model squeezes out of pixels.

Inference is the unreliable one and its errors *compound*: 20 connections at 95% each yields a
fully-correct reconstruction 36% of the time. **Every design decision should move work out of
category 3 and into 1 or 2.** If a proposal adds something the model must infer, it needs to
justify itself against that.

## Hard constraints (confirmed, do not relitigate)

From the curriculum team:

- Students build from a **known set of curriculum labs**. The intended circuit is known.
- Lab instructions **cannot** specify exact hole positions. Placement is free. *(This is the
  constraint that shapes everything — see ARCHITECTURE.md "Why placement freedom is expensive".)*
- Wire colors are **conventions only**. Red does not reliably mean power. Colour carries no
  semantics, though it still helps separate crossing wires visually.
- The hardware kit **is standardized**: one breadboard model (WB-102), one microcontroller
  (Adafruit Metro Mini V2).
- **Multiple different builds can be correct** for the same lab (e.g. resistor on either side
  of the LED). The checker must accept functional equivalence, not one canonical answer.

From Stone (project side):

- Guided capture, check-deposit style, is **approved**. Students can see the screen while
  aiming, so a live alignment overlay is available.
- **Video clip capture is allowed** — many frames, not one photo.
- **Nothing may be added to the kit.** No printed alignment mat, no stand, no clip.
  Rectification must work off the board's own hole grid.
- **Student code is out of scope.** We judge wiring only. If the wiring is correct, the system
  says so and points the student at their program.
- **Inference runs in the student's browser.** Server-side is the fallback if model quality
  demands it. Latency budget: 15 seconds.
- The 248 existing photos are ours to use without restriction.

## Consequences worth remembering

- **"Correct" is a higher bar than "incorrect."** Because code is out of scope, a false
  "your wiring is fine" sends a student hunting through their program for a bug that isn't
  there. When unsure, abstain — never issue a clean bill of health on low confidence.
- **No kit additions** means the rectifier has to survive a board partly covered by wires,
  components, and the student's own fingers. This is the project's main technical risk.
- **Guided capture + video** are the two things that make that risk survivable.

## Open questions (blocking)

Sent to the curriculum/engineering team, awaiting answers. See `docs/OPEN_QUESTIONS.md` for
full text and what each unblocks.

| # | Question | Blocks |
|---|----------|--------|
| 1 | WebSerial to the Metro Mini? Can a diagnostic routine live in the shipped sketch? | The entire measurement channel + auto-labelling |
| 2 | Chromebook only or phone? Which Chromebook model? | Training data domain, model size budget |
| 3 | Which board is built on? What is "blank breadboard w/picture"? Is the circuit wired to the Arduino or coin-cell standalone? | Whether the measurement channel reaches the circuit at all |
| 4 | How many labs? Can we see 2–3 complete ones? | Netlist schema, checker rules, error taxonomy |
| 5 | How do we obtain a complete student kit? | Component crop library, all hardware questions |
| 6 | *(deferred)* Student-invented circuits in scope? | Scope only — does not change what we build first |

## Related repo

`../breadboard_generator` — the previous attempt. **Its renderer is retired** (see
`docs/DECISIONS.md`), but three things carry forward and should be reused rather than rewritten:

- `generator/grid.py` — the (row, col) → pixel coordinate system and board geometry. This
  becomes our canonical rectified space.
- `config/board_spec.json` — WB-102 physical dimensions, single source of truth.
- `generator/mutations.py` — the six error types. This is the seed of our error taxonomy and a
  free test-case generator for the checker.

Also holds `data/real/` — 248 iPhone HEIC photos of real WB-102 boards. These become the
component crop library and the frozen evaluation set. Note they are *phone* photos; if
production input is a 720p webcam they are the wrong domain for training the shipped model.

## Conventions

- Python 3.10+, type hints on public signatures, JSON for all configs.
- All physical dimensions come from the board spec. No hardcoded pixel math outside the grid
  module — that rule earned its place in the previous repo and still applies.
- Seed all randomness; same seed + same config = identical output.
- Prefer deterministic, testable code over learned components wherever the choice exists. See
  "The one principle".
