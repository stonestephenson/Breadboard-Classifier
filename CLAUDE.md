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

**Reasoning half built and tested; rectification works on real photos; closing the loop.**

| Piece | State |
|-------|-------|
| Circuit model + checker (`breadboard/`) | **Done**, including 3- and 4-leg sensors. Finds up to three mistakes in one check, names only fixes that end at the lab's circuit, and tries only moves a student can make. A lab is written as a circuit with no places (`{"net": "white"}`). `docs/CHECKER.md` |
| Electrical probe (`tools/probe.py`) | **Working on real hardware.** `docs/FIRMWARE_PROTOCOL.md` |
| Rectifier (`breadboard/rectify.py`) | **Working on real photos**, and live on a webcam (`tools/live.py`). Built on the vendored breadboard-normalizer. `docs/SPIKE_RECTIFY.md` |
| Blink and watch (`breadboard/blink.py`, `tools/blink.py`) | **Working on the real board**, on a phone camera and one laptop camera. Each LED found by its pin, colour and column, no trained model. The bar for a glow is set by each check's own photos in which nothing lit, above a small fixed floor, so a bright room or changing light does not change the answer. A pin set aside because the board moved is blinked again. |
| Does it work? (`breadboard/verify.py`) | **Working.** Blink results checked against a lab file: findings plus a verdict drawn on the photo. Demo: `tools/live.py --camera 1 --lab examples/basicboard_demo.json`, press `c`. |
| Which leg to move? (`breadboard/diagnose.py`) | **Working, with the build entered by hand** in place of the vision model: `examples/basicboard_as_seen.json` (LED directions left open, as a camera must) or `basicboard_as_built.json`. Trusted only where blinking confirms it; then the checker's fix, down to the hole, is drawn on the photo. An LED placed right but dark gets its hidden causes, one per check, and "fixed" when it lights. Demo: add `--build examples/basicboard_as_seen.json`. |
| Per-node occupancy from photos | Not started; a trained model needs data |
| Lab reference circuits | The demo board, the BasicBoard as shipped and Activity 3 are written; the two design-challenge activities are not |

Gate: `./.claude/verify.sh` (lint, format, types, tests; two to three minutes; it needs `pyright`
on the PATH, which the dev extras do not install). Tests alone:
`./venv/bin/python -m pytest tests/ -q`. CI (`.github/workflows/ci.yml`) runs lint and tests
only. **Setup, every run mode and the example circuits are in `README.md`.**

## What works for which kind of lab

The full loop (blink, check the description, name the fix, hidden causes) works for labs whose
parts are LEDs driven from output pins 2–10 to ground. For an analog sensor lab (Activity 3) the
checker finds wiring mistakes from a described build, and `tools/probe.py --analog` measures
whether it works, but that measurement is not part of the live check. For a lab with the
ultrasonic sensor the checker can represent the sensor; nothing measures it yet, though the
shipped sketch can read a distance. An activity built off the Arduino cannot be covered at all.
Student code is out of scope everywhere.

## Next

Vision (reading the build from a photo) is set aside for now; the hand-written build file stands
in for it. The Chrome/WebSerial port waits until everything else is an MVP. In order:

1. **The light sensor in the live check.** After blinking, ask the student to cover the sensor
   and pass if the analog reading changes; say which analog pin the signal is really on
   (`tools/probe.py --analog` already watches all six).
2. **A lab where the student chooses the pins.** A lab file must name each pin today
   (`graph.find_isomorphism` anchors on pin names).
3. **The ultrasonic sensor.** Ask the sketch for a distance; it works or it does not.
4. **Show what the build file stands for.** Draw the entered parts on the photo, the ones
   blinking confirmed in green; then a demo script and a one-page explainer. The demo is for
   Stone's professor.
5. **More cameras and rooms.** Two cameras so far, a phone and one laptop's own, in one home. A
   Chromebook camera is untested. That laptop camera takes about a second to recover after a
   bright LED goes off, longer than `SETTLE_S`, so its unlit photos are up to 20 levels dark; the
   glow measure tolerates it, but a settle time that waits for the picture to stop changing would
   be better and is not built.

**Known gap, found 2026-10-05, not fixed.** With the board held up to the laptop camera, one
check in eight read the green LED's colour as "unknown" and so said "Not sure yet"
(`data/cache/blink/basicboard-2026-10-05-laptop-held-green-unknown`; it is not in `LEDS`, because
it would fail). `find_glow` still names the peak channel from how much each channel *rose* in the
ring (`spill`), and on a bright board that camera can leave no channel risen at all. The likely
fix is to name it from the signed change beside the LED minus the far board, which the glow
test already measures (`_strength`). Do this before anything else in `blink.py`.

The checker's known limits are in `docs/CHECKER.md` ("What is not built yet"): no fix that adds a
part (a missing wire), and no first step when two changes only help together.

## Test data, baselines and knobs

The tests on real photos and recorded blink runs need files under `data/cache/`, which is not in
git. Without them those tests skip (about 58) and the gate still goes green, so a fresh clone and
CI never exercise real data. Do not tune `blink.py` or `rectify.py` on a machine without them.

- `data/cache/sample/` holds 12 photos sampled from `../breadboard_generator/data/real/`
  (`docs/SPIKE_RECTIFY.md`), and `data/cache/kit/basicboard.jpg` one photo of the kit's board.
  `tests/test_rectify.py` pins corner pixels for them.
- `data/cache/blink/<name>/` holds recorded blink runs. Every live check saves one as
  `data/cache/blink/<time>/` (`tools/live.py`, `tools/blink.py`). To make one a regression test:
  check by eye what lit, rename the folder to `basicboard-<date>-<what>`, and add its expected
  pins, colours and columns to `LEDS` in `tests/test_blink.py` (and a case in `TestRecordedRun`
  in `tests/test_diagnose.py` if a build file goes with it).
- `data/cache/replay_pages/` holds the source of two published pages that replay the checker's
  recorded trials (`record.py` and two templates). They are not part of the product. Re-record
  after changing `check.py`.
- `data/cache/blink_diag/` holds scratch scripts that measured the ranges quoted beside the
  constants in `blink.py`. `final_measure.py` prints what the code reads on every recorded run
  (glow, quiet samples, ring colours); `stress_fakes.py` feeds it faked glints and `stress.py
  bright` brightened runs; `mutate.py` breaks each rule in turn and checks a test fails. Not
  part of the product. Run them after changing a constant there.

Every tunable is a module constant with its measured range in the comment beside it: the
thresholds at the top of `breadboard/blink.py`, `MIN_COLUMN_MARGIN` in `rectify.py`, and
`MAX_STEPS`, `MAX_FIXES`, `REACH`, `SLACK` in `check.py`. The alignment limits in `blink.py` were
measured at the 0.6 s settle time (`SETTLE_S` in `tools/blink.py`) and go with it. The only
runtime knob is the `BREADBOARD_MODEL_DIR` environment variable (where the corner model is kept).

## Third-party material (local only)

The repo root holds material that is not ours and is gitignored: the curriculum PDFs (an active
study runs on these activities, and we were asked not to share them or put them on any public
service), the LbyM web app and Arduino sketch (reference only), and the kit contents list. This
repo is public. Read that material locally to author lab circuits; never commit it, quote it, or
put its content in a published page.

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

## Open questions

Sent to the curriculum/engineering team. Most are answered; `docs/OPEN_QUESTIONS.md` has the
answers in full and is the one place they are kept.

| # | Question | State |
|---|----------|-------|
| 1 | WebSerial to the Metro Mini? Can a diagnostic routine live in the shipped sketch? | Largely resolved: the shipped sketch's own commands are enough, with no firmware change. Chrome on school machines is not yet tried |
| 2 | Chromebook only or phone? Which Chromebook model? | Soft answer; the model is still unknown |
| 3 | Which board is built on? Is the circuit wired to the Arduino? | Answered: the pre-built BasicBoard, always wired to the Arduino. The blank breadboard is out of scope |
| 4 | How many labs? Can we see complete ones? | Mostly answered: about seven; the files for five are held locally |
| 5 | How do we obtain a complete student kit? | Answered |
| 6 | *(deferred)* Student-invented circuits in scope? | Not sent |

## Related repos

`github.com/hartleyblakey/breadboard-normalizer` (MIT) — **our rectifier**, vendored in
`breadboard/_vendor/` with a few small changes. Read the docstring of
`breadboard/_vendor/breadboard_normalizer/__init__.py` before editing it. Keep upstream's style (it is excluded from ruff and pyright), and record any change there.

`../breadboard_generator` — the previous attempt. **Its renderer is retired** (see
`docs/DECISIONS.md`). Three things carried forward as ideas; nothing here imports them, and
board geometry now lives in `breadboard/board.py`:

- `generator/grid.py` — the (row, col) → pixel coordinate system and board geometry. Canonical
  space is now the rectifier's 1024 × 340 frame instead (`breadboard/rectify.py`), and
  `board.hole_position` is tested to agree with it.
- `config/board_spec.json` — WB-102 physical dimensions. Its rail offset (4.0 mm) does not
  match the real board. The rectifier's template, fitted to real photos, gives 7.2 mm (see
  `RAIL_OFFSET_MM` in `breadboard/board.py`).
- `generator/mutations.py` — the six error types. This is the seed of our error taxonomy and a
  free test-case generator for the checker.

Also holds `data/real/` — 248 iPhone HEIC photos of real WB-102 boards. These become the
component crop library and the frozen evaluation set. Note they are *phone* photos; if
production input is a 720p webcam they are the wrong domain for training the shipped model.

## Conventions

- Python 3.10+, type hints on public signatures, JSON for all configs.
- All physical dimensions live in `breadboard/board.py`, and the canonical image's geometry in
  `breadboard/rectify.py`. No hardcoded pixel math anywhere else.
- No new `.md` files: notes go in docstrings or the existing docs.
- Before a commit, one cold reviewer (a fresh-context subagent) reads the diff. Commits are
  local; push only when Stone asks.
- Seed all randomness; same seed + same config = identical output.
- Prefer deterministic, testable code over learned components wherever the choice exists. See
  "The one principle".
