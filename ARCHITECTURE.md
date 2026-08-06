# Architecture

Read `CLAUDE.md` first. This document is the design and the reasoning behind it. Anything
marked **UNVALIDATED** is a belief, not a result.

---

## 1. The problem, stated precisely

**Input:** a short video clip of a WB-102 breadboard, captured in-browser under a guided
alignment overlay — from either a Chromebook webcam or a phone (§3[1]) — plus the identity of
the lab the student is attempting.

**Output:** either "your wiring matches the lab" (high confidence only), or a specific,
actionable repair — "the black wire in row 14 needs to move one column left" — or an honest
abstention that asks for a better view or a connectivity-tester measurement.

**Not in scope:** the student's program, circuits the student invented (deferred), anything
requiring changes to the physical kit.

## 2. Why this is hard, and where the difficulty actually lives

The naive framing — "classify the photo as correct or incorrect" — is the wrong problem, and it
is the framing the previous repo was built around. Two reasons it fails:

**A classifier cannot tell a student what to fix.** "Incorrect" is not actionable. The product
requires localisation and a repair instruction, which means reconstructing the circuit, not
scoring the image.

**Errors compound.** A lab circuit has on the order of 20 connections. At 95% per-connection
accuracy the whole reconstruction is right 36% of the time. Reliability is a product, not an
average, so the design has to minimise the *number of independent inferences* rather than
maximise the accuracy of any one of them.

### Why placement freedom is expensive

The curriculum cannot specify exact hole positions. This is the single most consequential
constraint, and it is worth understanding precisely why.

A breadboard's entire purpose is that the same circuit can be built anywhere on it. Electrical
meaning comes from *which holes share a metal strip*, never from where those strips sit. So two
builds of an identical circuit can be visually unrecognisable as the same thing — different
columns, opposite halves, mirrored, different wire lengths taking different paths.

Consequence: **no comparison based on appearance can work.** We cannot diff against a reference
photo, cannot check "is there a wire at a10", cannot template-match a correct build. We have to
climb from appearance up to *meaning* — the netlist — before any comparison is possible. That
climb is the entire computer-vision problem, and placement freedom forces us to make it.

### The compensating gift

The same fact that makes the board flexible also relaxes our hardest requirement. Holes a–e in
a column are electrically one node; f–j likewise. So we never need to know whether a lead is in
`a14` or `c14` — they mean the same thing.

| | count |
|---|---|
| Physical tie-points | 830 |
| Terminal nodes (63 columns × 2 halves) | 126 |
| Rail segment nodes (4 rails × 2 segments) | 8 |
| **Electrical nodes we must resolve** | **134** |

We need **column** and **which half**, not which hole. That is a 5× relaxation along the axis
that would otherwise hurt most, and it is what makes the hard path affordable.

## 3. Pipeline

```
[0] Lab context ──┐
                  ↓
[1] Capture → [2] Rectify → [3] Perceive ──┐
                                            ├→ [5] Fuse → [6] Verify → [7] Explain
[4] Measure (Arduino self-test) ────────────┘                       └→ [8] Abstain
```

### [0] Lab context — free prior

The lab's intended circuit as a netlist, plus the known kit vocabulary. Costs nothing, and it
constrains stage 3: when pixels are ambiguous, the expected circuit says what we are probably
looking at. This is the core advantage of *verification* over *discovery* and the reason the
deferred free-design case is so much harder.

**The inventory is known; the layout is not.** Students build on the **pre-assembled
BasicBoard** — a WB-102 shipped populated with 4 LEDs, 4 resistors, 4 black wires and the Metro
Mini. The separate blank breadboard is out of scope entirely.

What this does and does not buy us:

- ✅ **Known component inventory.** We know what is on the board — counts and types — before we
  look. That constrains perception meaningfully and is a genuine prior.
- ❌ **NOT known positions.** A student can move anything, and *moving a pre-positioned part is
  itself a plausible error we must catch*. The factory layout is a **prior for ranking
  hypotheses, never a constraint to verify against.** Nothing downstream may assume a component
  is where it shipped.
- ⚠️ **Anchoring must not depend on it.** Rectification anchors on features physically molded or
  printed into the board — the 3-pitch centre channel and the asymmetric red/blue rail stripe
  order — which cannot move. The Metro Mini is corroboration only: it is physically stubborn
  (24+ pins, no lab asks students to reseat it), but "stubborn" is not "guaranteed".

### [1] Capture — guided, multi-frame, two device paths

**Both paths are supported** (decided 2026-08-06). Every student can use the machine they
already have; students with a phone get the better result.

| | Chromebook webcam | Phone |
|---|---|---|
| Availability | universal | most students |
| Optical quality | ~720p, soft, wide-angle | 3–4× linear resolution |
| Serial link to the Metro Mini | ✅ same device | ❌ WebSerial is desktop-only |
| Session complexity | single device | needs pairing (QR / short code) |

Neither path dominates, which is why we build both: the Chromebook path is optically worse but
architecturally simpler (camera and USB on one machine); the phone path is optically better but
splits the two channels across devices and needs a pairing step, with the phone running vision
locally and uploading only the extracted netlist.

**Rectifying first is what makes two capture paths affordable.** Both cameras produce the *same
canonical view* — differing in effective resolution and sharpness, not geometry — so the
perception model sees one domain, not two. Size canonical space for the worst case (≈24 px per
pitch, which 720p supplies when guided capture makes the board fill the frame) and phone
capture simply oversamples it.

Cost: evaluation data is needed from both. Any laptop webcam is an adequate development proxy
for a Chromebook; we do not need a specific district model.

In both paths: live alignment overlay, capture refused until the board is framed, and a short
clip rather than a single frame. Natural hand movement means each frame sees the board slightly
differently, so a wire occluding a hole in one frame will not occlude it in the next.
Multi-frame fusion is the primary defence against occlusion and costs the student nothing.

### [2] Rectify — the foundation

The breadboard is effectively a giant fiducial marker: 830 holes on an exact 2.54 mm lattice,
plus rail stripes and printed labels, on a board whose dimensions we know exactly (165.1 ×
54.0 mm). Better than a QR code — larger and more regular.

Detect hole centres, fit a homography from the observed points to the known ideal lattice with
RANSAC, and warp. Every photo becomes a canonical top-down image in which node (half, column)
always lands on the same pixel.

Three properties matter:

- **Deterministic.** The geometry is solved, not learned. No training data required for this
  stage.
- **Self-checking.** Fit residuals tell us whether it worked. A bad photo gets *rejected*
  rather than silently guessed at — this is what makes abstention possible downstream.
- **Invertible.** A homography inverts, so anything found in canonical space can be drawn back
  onto the student's own photo. We circle the mistake in their picture.

**PARTIALLY VALIDATED (2026-08-06)** — see `docs/SPIKE_RECTIFY.md`. On a 12-photo sample of the
real data, 8 fitted successfully with median reprojection residual of **0.02–0.06 pitch**. We
need half-pitch accuracy to assign a lead to the right column, so the margin is roughly an order
of magnitude. Geometry is not the limiting factor; detection recall and occlusion are.

Still open: anchoring the lattice to the board spec (a bare grid is ambiguous up to an integer
origin and a 180° flip), and 4/12 failures on images with heavy background speckle. Neither is
expected to survive contact with guided multi-frame capture — we need one good frame per clip,
not one good photo per attempt.

### [3] Perceive — in canonical space

All perception happens *after* rectification, where perspective, scale and rotation are already
gone. This is a much smaller problem than perceiving in raw photo space.

1. **Occupancy** — is each of the 134 nodes occupied?
2. **Classification** — what is each occupying object? Vocabulary is small (see §5).
3. **Association** — which two nodes does each object bridge?

Association is the genuinely hard remainder. Helpful structure:

- Two-terminal components (resistors, LEDs) have a visible rigid body spanning their two holes.
  Detecting the body gives the pairing directly.
- Wires come in **fixed lengths** (12", 7.5 cm, 1", 2.5 cm pre-stripped), which bounds how far
  apart the two ends of any given wire can be.
- Colour separates crossing wires even though it carries no semantics.
- The Metro Mini and sensor modules are **rigid objects with known footprints** — find the
  outline once and every pin position follows from a rigid transform. This is the easy case.

### [4] Measure — the Arduino probes its own circuit

**Confirmed reachable (2026-08-06); protocol access pending.** The student's circuit is
*always* electrically connected to the Arduino — there is no standalone battery case to blind
us. A bidirectional command/response protocol over USB already exists and is in daily classroom
use: the web app sends commands, the sketch executes them, data comes back.

Constraint: **we may not flash our own sketch.** The shipped sketch is the integration point.

**RESOLVED (2026-08-06) — see `docs/FIRMWARE_PROTOCOL.md`.** The shipped sketch exposes
arbitrary memory read (`0xFE`) and write (`0xFD`). On an AVR the GPIO registers are
memory-mapped, so this is already complete control over pin direction, pull-ups, drive state and
reads. **A full connectivity probe is implementable today with no firmware change and no
external dependency.**

The probe: put every pin in INPUT_PULLUP, then drive one pin LOW at a time and read the port
registers. Any pin reading LOW shares an electrical node with the driven pin. ~65 transactions,
well under a second. `analogRead`'s 10 bits additionally separate a direct connection from one
through a 330 Ω resistor from a floating node.

Two capabilities this unlocks beyond plain connectivity:

- **LED polarity becomes measurable.** A diode conducts one way only, so an asymmetric response
  to reversed drive reveals orientation — retiring a perception problem §5 flags as known-hard.
- **Stimulus–response capture.** We already record a video clip; if the Arduino toggles pins
  during it, *whichever LED blinks in the video is the LED on that pin*. Component-to-node
  association by observation rather than inference.

Remaining dependency is no longer firmware but the **web app**: whether our code may send raw
serial commands through it.

This is *measurement*, not inference — it converts the least reliable part of the pipeline into
a hardware fact. The two channels fail in opposite directions:

> **Vision knows where things are but not what is connected.**
> **Electricity knows what is connected but not where things are.**

Both halves are needed for a useful instruction: "this branch is open" comes from electricity,
"your jumper is one column left of where it should be" comes from vision.

Secondary payoff, possibly larger: it is an **automatic labelling machine**. Build a circuit,
let the board report its own connectivity, photograph it — a perfectly labelled training
example with zero annotation effort. This is the problem CycleGAN was introduced to solve,
solved better and for free.

Bounds: only nodes reachable from Arduino pins are testable. Coin-cell-powered circuits and
isolated branches are invisible (see open question 3). It reports *that* something is
disconnected, rarely *which object* is at fault.

### [5] Fuse

Combine the vision-derived netlist with measured connectivity. Measurement wins where
available. **Disagreement between channels triggers abstention rather than a guess** — two
independent channels disagreeing is exactly the signal that we do not understand the image.

### [6] Verify — functional equivalence, then minimum edit

Multiple different builds are correct for the same lab, so comparing the student's graph
against one reference graph is wrong: it would flag working circuits as broken. Two stages:

**Stage A — does it work?** Collapse series chains into unordered sets and check *properties*,
not structure: "the LED lies on a path from a driven pin to ground, with a resistor somewhere
on that path, oriented anode-toward-positive." This accepts every functionally valid
arrangement automatically, including ones nobody enumerated, and it handles placement variation
for free since properties do not care about columns.

**Stage B — if it fails, what is the smallest fix?** Compute the minimum edit against whichever
valid arrangement is closest to what the student actually built. The edit *is* the diagnosis:
"move this one connection from here to there" rather than "wrong".

These graphs are tiny — 10–30 typed nodes — so exact computation is instant. This stage is
**deterministic code, fully unit-testable, with no learned components**, and `mutations.py`
from the generator repo supplies test cases for free.

### [7] Explain

Template the repair into friendly language and inverse-warp its location onto the student's own
photo. **No LLM in v1**: the checker's output is structured, so the text can be deterministic —
which for young students is arguably better (consistent phrasing, no possibility of a
hallucinated fix). An LLM later, if it earns its place, operates on the netlist as text and
never on the image.

### [8] Abstain

Low confidence → request a better view, or walk the student through a connectivity-tester
measurement. Every student already has a tester in their kit, so "let's measure it together" is
already in the curriculum's vocabulary, and it teaches debugging rather than dispensing
answers.

Abstention is asymmetric: the bar for declaring wiring **correct** is higher than the bar for
reporting an error, because code is out of scope and a false all-clear sends a student hunting
through their program for a bug that is not there.

## 4. Data strategy

The previous repo tried to make synthetic renders photorealistic with CycleGAN. That failed and
is retired — see `docs/DECISIONS.md`. The replacement:

**Composite real crops onto real boards, in rectified space.** Photograph each kit component
once from many angles to build a crop library, then paste real pixels into real rectified board
photos at programmatically chosen holes. Photorealistic by construction — every pixel came from
a camera — and perfectly labelled, because we chose the placement.

This reuses the generator's placement logic (`grid.py`, the circuit JSON schema,
`mutations.py`) and discards only the renderer.

Sources, in order of value:

1. Auto-labelled real builds via the measurement channel (§4), if open question 1 lands well.
2. Composited images from the crop library.
3. The 248 existing photos: crop library source, geometry development, frozen eval set.
4. Purely synthetic renders: optional pretraining only, where cartoon quality is acceptable.

**Domain caveat:** the 248 photos are from a phone. If production input is a 720p Chromebook
webcam, they are the wrong distribution for training the shipped model — roughly 3× the linear
resolution and 10× the pixels. A second collection pass through the actual capture device will
be needed. Gated on open question 2.

## 5. Component vocabulary

From the STEMACES kit contents list. Roughly eight classes — small, which is good.

| Item | Notes for perception |
|------|---------------------|
| Resistor, 330 Ω | **Single value only** — no colour-band decoding needed. Large saving. |
| LED (orange, white, variety) | Polarity matters and is **visually subtle**; the flat edge is often hidden once seated. Known-hard. |
| Jumper wire — 12", 7.5 cm, 1" | Fixed lengths bound endpoint association. Short ones are rigid and span a known column count. |
| Pre-stripped red wire, 2.5 cm | As above. |
| Light sensor | Mount type unknown — see below. |
| Temperature sensor | Mount type unknown. |
| Ultrasonic sensor | Design-challenge set; may be out of scope. |
| Metro Mini V2 | Rigid, known footprint. Straddles the centre channel, so all pins land in board holes. Easy case. |
| CR2032 + case | Implies some circuits are standalone, unreachable by the measurement channel. |
| Connectivity tester | A tool, not a circuit element. Enables the abstention fallback. |

The 3-pin and 4-pin **connectors** in the kit suggest at least some sensors attach by cable
rather than plugging into the board. Those are different problems: a board-mounted sensor
occupies known holes and needs orientation checking; a cable-attached one puts only its cable
pins in the board and may sit out of frame entirely. Resolved by getting a kit in hand (open
question 5), not by email.

## 6. Deployment

Inference runs **in the student's browser** (ONNX Runtime Web / WebGPU, WASM fallback).
Rationale: no image ever leaves the device, which for under-13 students on district accounts is
the difference between a hard legal conversation and none; zero marginal cost; scales to any
number of schools for free. Constraint: the model must be small and fast enough for managed
school hardware. Budget is 15 seconds, which is generous and permits multi-frame processing.

Server-side inference is the fallback if in-browser quality proves inadequate. That decision is
deliberately deferred, not foreclosed — keep the model boundary clean so it can move.

## 7. Known risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| ~~Rectification fails on cluttered real photos~~ | **Downgraded** 2026-08-06 | Spiked: 8/12 fit at 0.02–0.06 pitch residual. See `docs/SPIKE_RECTIFY.md` |
| Occlusion hides connections | High | Multi-frame capture, measurement channel, abstention |
| Wire endpoint association | High | Fixed lengths bound search; colour separates crossings; measurement confirms |
| LED polarity not visible once seated | Medium | May be electrically detectable; otherwise a UX nudge |
| Measurement channel unavailable (Q1/Q3 land badly) | High | Photo-only fallback; everything above still works, less reliably |
| Training data domain mismatch (phone vs webcam) | Medium | Second collection pass on the real device |
| Classroom reality: hands, clutter, neighbouring boards | Medium | Guided capture constrains framing |
| Answer keys do not exist yet as netlists | Medium | Build an authoring tool; needs a curriculum owner |
