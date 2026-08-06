# Open questions

Sent to the SSU curriculum / engineering team on 2026-08-06. Update this file with answers as
they arrive, and note what each answer unblocks or changes.

Routing: Q1 → the programmer who wrote the LbyM technical brief. Q5 → Hannah
(Hellmanh@sonoma.edu), named as the kit contact in `STEMACES-Kit-Contents-List.docx`. Q2–Q4 →
curriculum side.

---

## Q1 — WebSerial and the diagnostic sketch  🟡 PARTIAL (2026-08-06)

**Answered so far:**
- A **bidirectional command/response protocol already exists**: the web app sends commands over
  a USB interface, the sketch executes them, and *data is returned*. The measurement channel's
  transport is therefore already built and in daily classroom use.
- **We may not flash our own sketch.** "Any sketch that overwrites arduino is no bueno." The
  shipped sketch is the integration point, not a replacement for it.
- Adding a diagnostic routine to the existing sketch is considered plausible.
- **They offered us the sketch source.** Accept immediately.
- Exact WebSerial capability forwarded to Troy.

**What to look for in the sketch source — this is the pivotal detail.** If the existing command
set already exposes *generic pin primitives* (set pin high/low, read digital/analog pin), then
the entire connectivity probe can be implemented **client-side in JavaScript using commands
they already ship**. No firmware change, no Troy dependency, no deployment risk, nothing to
negotiate. Given that the platform's whole purpose is reading sensors from student code, such
primitives very likely exist in some form.

Only if they do not do we need a new firmware command, which is a slower path through their
release process.

---

## Q1 (original text)  ⬛ SUPERSEDED BY ABOVE

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

## Q2 — Camera and device  🟡 SOFT ANSWER + NEW PROBLEM (2026-08-06)

**Answered:** "I imagine a phone." Chromebook models "vary by district, would be impossible to
know."

**Treat as provisional** — "I imagine" is a guess, not policy, and it now determines system
topology (below). Needs confirmation.

**If phone is right, two good consequences:**
- The 248 existing photos are **the correct domain** after all. No re-collection needed.
- Resolution stops being a concern; phones give ~3–4× the linear resolution of a webcam.

**If Chromebook, design for the worst case** — model is unknowable, so assume a low-end 720p
sensor and let guided capture (board fills the frame) normalise effective resolution.

### NEW PROBLEM: the capture device and the measurement device may be different machines

**WebSerial does not exist on mobile browsers.** It is desktop Chrome/Edge only. So if the
student photographs with a phone while the Metro Mini is plugged into a Chromebook, the two
channels live on two devices in two browser sessions:

| | Camera | USB / serial |
|---|---|---|
| Phone | ✅ | ❌ |
| Chromebook | ✅ (poor) | ✅ |

**Proposed resolution — and it preserves the privacy story:** the phone runs the vision model
locally and uploads **only the extracted netlist**, which is a few hundred bytes of
non-identifying text. The Chromebook session contributes the electrical measurement. Fusion
happens in the LbyM app. No image ever leaves the phone, and both channels are available.

Requires a session-pairing step (QR code or short code shown in the web app). Phones are also
*better* hosts for in-browser inference than school Chromebooks, so this helps model budget too.

**Open:** do students actually have phones in class (many districts restrict them), and how is
the photo expected to reach the web app?

---

## Q2 (original text)  ⬛ SUPERSEDED BY ABOVE

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

## Q3 — Which board, and is it electrically reachable  ✅ ANSWERED (2026-08-06)

**The best answers in the whole exchange.**

- **Students build on the pre-assembled BasicBoard.**
- **The blank breadboard can be ignored entirely.** It is for non-Arduino activities and is out
  of scope.
- **The student's circuit is electrically connected to the Arduino.** Always.

**Consequences, in order of importance:**

1. **The measurement channel is universal, not partial.** Every lab circuit is reachable from
   the Arduino's pins. The coin-cell standalone case that would have blinded us does not arise.
   Combined with the confirmed command/response protocol in Q1, electrical measurement moves
   from "promising mitigation" to *the likely backbone of the system*.

2. **The starting state is known exactly.** The BasicBoard ships pre-populated (4 LEDs, 4
   resistors, 4 black wires, Metro Mini). Students *modify and extend* a known baseline rather
   than building from nothing, so the delta we must perceive is far smaller than a full circuit.

3. **Part of the board is at factory-fixed positions.** Placement freedom applies to what the
   student adds, not to what was pre-assembled. This partially restores the positional prior
   that DECISIONS.md R2/R3 gave up.

4. **It solves the rectification anchoring problem** (`docs/SPIKE_RECTIFY.md`). The Metro Mini
   is always present, is a large high-contrast rigid object, and sits at a known position on the
   BasicBoard — so detecting it fixes the 180° flip and the index origin directly, without
   needing to read rail-stripe order or printed labels.

---

## Q3 (original text)  ⬛ SUPERSEDED BY ABOVE

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

## Q4 — The labs themselves  🟢 MOSTLY ANSWERED (2026-08-06)

**~7 labs** will use photo troubleshooting. Small and tractable: authoring 7 reference circuits
is a bounded content task, not an open-ended one.

**Curriculum available from Hannah** (hellmanh@sonoma.edu) — same contact as the kit, so one
email covers both.

Still to read from the actual labs: complexity range, how much the instructions constrain
structure, and how many genuinely different arrangements are correct per lab.

---

## Q4 (original text)  ⬛ SUPERSEDED BY ABOVE

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

## Q5 — Obtaining a kit  ✅ ANSWERED (2026-08-06)

Contact **Hannah Helman, hellmanh@sonoma.edu**. Action item, not a question. Request the kit and
the curriculum for the ~7 labs in the same message.

---

## Q5 (original text)  ⬛ SUPERSEDED BY ABOVE

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
