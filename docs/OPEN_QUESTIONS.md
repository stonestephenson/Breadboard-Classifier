# Open questions

Sent to the SSU curriculum / engineering team on 2026-08-06. Update this file with answers as
they arrive, and note what each answer unblocks or changes.

Routing: Q1 → the programmer who wrote the LbyM technical brief. Q5 → Hannah
(Hellmanh@sonoma.edu), named as the kit contact in `STEMACES-Kit-Contents-List.docx`. Q2–Q4 →
curriculum side.

---

## Q1 — WebSerial and the diagnostic sketch  ⬛ AWAITING

> Can the web app talk to the Metro Mini over WebSerial today? And for running connection
> diagnostics, would it be possible either for us to load our own diagnostic sketch — or,
> probably simpler, for a diagnostic routine to be added to the sketch you already ship, that
> responds to a command over the serial connection?

**Unblocks:** the entire measurement channel (ARCHITECTURE.md §3[4]) *and* automatic labelling
of training data.

**Why it is the highest-leverage question:** connectivity is precisely what vision is worst at
and what electricity is best at. A yes converts the least reliable stage of the pipeline into a
hardware fact, and gives us labelled training data with zero annotation effort.

**If no:** everything still works, photo-only, with materially lower reliability and a much
more expensive data-collection story.

---

## Q2 — Camera and device  ⬛ AWAITING

> What type of camera will the students use to capture the breadboard — only the Chromebook, or
> are they allowed to use a phone? And which Chromebook model do they use?

**Unblocks:** training-data domain, model size budget, achievable precision.

**Why the model number matters:** from it we can look up camera resolution, processor, and
WebGPU support ourselves — one question, several answers.

**Stakes.** With the board filling a 720p frame we get ≈7.7 px/mm, about 20 px between adjacent
holes — workable for the lattice, marginal for fine detail like an LED's flat edge. At 1080p it
is ≈30 px per pitch and comfortable. Phone capture is ≈3× linear resolution again.

**Consequence either way:** the 248 existing photos are phone-sourced. If production is a
webcam, they remain valid for the crop library, geometry work and evaluation, but a second
collection pass through the real device is needed for training.

---

## Q3 — Which board, and is it electrically reachable  ⬛ AWAITING

> The docx mentions two boards the student has: a pre-assembled BasicBoard (with the Arduino and
> sketch), and a separate blank breadboard. Which one gets built on? What exactly is the "blank
> breadboard w/picture"? Is the student's circuit electrically connected to the Arduino during
> the lab, or can it be standalone on coin-cell power?

**Unblocks:** what a "correct" build even looks like, and — critically — whether the
measurement channel from Q1 can reach the student's circuit at all.

**Why the last part matters most:** the kit contains CR2032 coin cells and holders. If circuits
are built standalone on the blank board with battery power and no connection to the Arduino,
WebSerial gives us nothing for those labs and we are photo-only regardless of how Q1 lands.

**Also watching:** "w/picture" is unexplained. If it turns out to be a printed placement
diagram, some positional prior may come back — which would partially resurrect the cheap paths
rejected in DECISIONS.md R2/R3.

---

## Q4 — The labs themselves  ⬛ AWAITING

> Roughly how many labs will use photo troubleshooting? Can we see two or three complete labs as
> students receive them, with the actual instructions, plus whatever the "correct" build looks
> like?

**Unblocks:** the netlist schema, the checker's property rules, and the error taxonomy. This is
on the critical path — the diagnostic engine cannot be finished or meaningfully tested without
real examples.

**What we are reading them for:**
- Complexity range: how many components and connections in a typical lab.
- How much the instructions constrain student choices (they cannot fix hole positions, but they
  may still pin down circuit *structure*).
- How much topological freedom exists — how many genuinely different arrangements are correct.
  This determines the property rules in verification stage A.

---

## Q5 — Obtaining a kit  ⬛ AWAITING

> How can we obtain a complete kit that a student has, containing all the parts that we should
> be expecting / training the model on?

**Unblocks:** the component crop library, rectification testing on real webcam frames, hands-on
testing of the measurement channel, and every remaining hardware question — including whether
the light and temperature sensors plug into the board or attach by cable, which changes whether
sensor orientation is a checkable error.

Highest-priority logistical ask. A kit on the desk answers a dozen questions we have not thought
to ask yet, with no email round-trips.

---

## Q6 — Student-invented circuits  ⬛ DEFERRED (not sent)

Two earlier answers point in different directions: the PM noted students would "most likely"
photograph circuits they designed themselves, while Laura's view was that by the free-design
stage they would not need photo/AI troubleshooting.

**Deferred by project decision** — build for the curriculum-lab case first.

**Why deferring is safe:** everything up through extracting the circuit from the photo is
identical either way. Only the final checking stage differs — comparison against a known
intended circuit, versus open-ended analysis of an arbitrary one. This is a scope-and-promises
question, not an architecture question.

**Raise before** anyone describes the system's capabilities to a school.

---

## Answered

### Placement and conventions — ANSWERED
- Students build from a known set of curriculum labs. Intended circuit is known. ✅
- Instructions **cannot** specify exact hole positions. ❌ (drove DECISIONS.md R2, R3)
- Wire colours are conventions only; red does not reliably mean power. ❌
- Hardware kit **is** standardized: one board model, one microcontroller. ✅

### Parts list — ANSWERED
`STEMACES-Kit-Contents-List.docx`, summarised in ARCHITECTURE.md §5. Headline: 330 Ω is the
only resistor value, so no colour-band decoding is needed.

### Capture UX — ANSWERED (project side)
Guided check-deposit-style capture approved; students can see the screen while aiming. Video
clip capture allowed. **No additions to the kit** (drove DECISIONS.md R6).

### Student code — ANSWERED (project side)
Out of scope. We judge wiring only; if wiring is correct we say so and point at the program.
Drove the asymmetric-confidence rule in CLAUDE.md.

### Deployment and scale — ANSWERED (project side)
In-browser inference, 15 s latency budget, server-side deferred as fallback (DECISIONS.md R7).
Concurrency numbers deliberately not pursued.

### Rights to the 248 photos — ANSWERED
Ours, unrestricted use.
