# Open questions

Sent to the SSU curriculum / engineering team on 2026-08-06. Each question is given as it was
asked, with its current answer. The reasoning behind each question and the first partial answers
were removed on 2026-10-05 and are in git history.

Routing: Q1 → the programmer who wrote the LbyM technical brief. Q5 → Hannah, named as
the kit contact in the STEMACES kit contents list (held locally, not in this repo). Q2–Q4 →
curriculum side.

---

## Q1 — WebSerial and the diagnostic sketch  🟢 LARGELY RESOLVED

> Can the web app talk to the Metro Mini over WebSerial today? And for running connection
> diagnostics, would it be possible either for us to load our own diagnostic sketch — or,
> probably simpler, for a diagnostic routine to be added to the sketch you already ship, that
> responds to a command over the serial connection?

**The web app already talks to the board.** It sends commands over USB, the sketch carries them
out, and data comes back: a command and response protocol in daily classroom use.

**The firmware side is solved, with no firmware change.** We may not flash our own sketch ("any
sketch that overwrites arduino is no bueno"), so the shipped sketch is the integration point. Its
source arrived on 2026-08-06 and exposes arbitrary memory read and write, which on the ATmega328P
means full memory-mapped GPIO control. The connectivity probe (`tools/probe.py`) and blink and
watch (`tools/blink.py`) are built on that and work on the real board, from Python over USB. Full
analysis in `docs/FIRMWARE_PROTOCOL.md`.

**Still open:**

- **The browser.** Nothing has been run from Chrome over WebSerial yet, on a school machine or
  any other. That port waits until the rest is an MVP (`CLAUDE.md`, "Next").
- **The web app.** Whether our code may send raw serial commands through it. This is the only
  remaining dependency for the measurement channel, and it is a web-app question, not a firmware
  one. The web app's source has arrived (held locally, for reference only).

**Worth raising with Troy:**

- A first-class diagnostic opcode as the production path: faster, safer, and stable across
  firmware revisions than poking memory. The team's first reply called adding a diagnostic
  routine to the existing sketch plausible.
- A bug report: `digitalRead` on pins 0–5 and 7 is unreachable, shadowed by earlier branches in
  the dispatch chain.

---

## Q2 — Camera and device  🟡 SOFT ANSWER + NEW PROBLEM (2026-08-06)

> What type of camera will the students use to capture the breadboard — only the Chromebook, or
> are they allowed to use a phone? And which Chromebook model do they use?

**Answered:** "I imagine a phone." Chromebook models "vary by district, would be impossible to
know."

**Treat as provisional** — "I imagine" is a guess, not policy, and it now determines system
topology (below). Needs confirmation. Meanwhile both capture paths are supported
(ARCHITECTURE.md §3[1]).

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

## Q3 — Which board, and is it electrically reachable  ✅ ANSWERED (2026-08-06)

> The docx mentions two boards the student has: a pre-assembled BasicBoard (with the Arduino and
> sketch), and a separate blank breadboard. Which one gets built on? What exactly is the "blank
> breadboard w/picture"? Is the student's circuit electrically connected to the Arduino during
> the lab, or can it be standalone on coin-cell power?

- **Students build on the pre-assembled BasicBoard.**
- **The blank breadboard can be ignored entirely.** It is for non-Arduino activities and is out
  of scope.
- **The student's circuit is electrically connected to the Arduino.** Always.

**Consequences, in order of importance:**

1. **The measurement channel is universal, not partial.** Every lab circuit is reachable from
   the Arduino's pins. The coin-cell standalone case that would have blinded us does not arise.
   Combined with the shipped sketch's own commands (Q1), electrical measurement moves from
   "promising mitigation" to *the likely backbone of the system*.

2. **The component inventory is known**, though not the layout. The BasicBoard ships
   pre-populated (4 LEDs, 4 resistors, 4 black wires, Metro Mini), so we know *what* is on the
   board before looking. Useful.

3. **Positions are NOT known.** Students can move pre-positioned parts, and moving one is itself
   a plausible error we must catch. The factory layout is a prior for ranking hypotheses, never
   a constraint to verify against. The positional prior given up in DECISIONS.md R2/R3 stays
   given up.

4. **Anchoring must use board-intrinsic features**, not the Metro Mini
   (`docs/SPIKE_RECTIFY.md`). The 3-pitch centre channel and the asymmetric red/blue rail stripe
   order are molded and printed into the board and cannot move. The Metro Mini is corroboration
   only.

---

## Q4 — The labs themselves  🟢 MOSTLY ANSWERED

> Roughly how many labs will use photo troubleshooting? Can we see two or three complete labs as
> students receive them, with the actual instructions, plus whatever the "correct" build looks
> like?

**About seven labs** will use photo troubleshooting, so authoring their reference circuits is a
bounded content task, not an open-ended one.

**The files for five activities have arrived** and have been read. They are held locally, not
in this repo, and are not to be quoted here (`CLAUDE.md`, "Third-party material"). What each
kind of lab needs, and how far the system covers it, is in `CLAUDE.md` ("What works for which
kind of lab"). Lab circuits are written for the BasicBoard as shipped and for Activity 3. The two
design-challenge activities leave the choice of pin to the student, which a lab file cannot say
yet (`docs/CHECKER.md`, "What is not built yet").

**Still open:** the rest of the "about seven" have not been seen.

---

## Q5 — Obtaining a kit  ✅ ANSWERED

> How can we obtain a complete kit that a student has, containing all the parts that we should
> be expecting / training the model on?

Through **Hannah**, the kit contact (address in the kit contents list). The kit's pre-built
BasicBoard, with its Metro Mini, is in hand: everything this repo measures on real hardware was
measured on it. We can get any other part of the kit, the sensors included, when it is needed.

One thing the kit was wanted for is still not settled: how the temperature sensor mounts. The
light sensor plugs into the board on three legs (ARCHITECTURE.md §5).

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
The STEMACES kit contents list (held locally, not in this repo), summarised in
ARCHITECTURE.md §5. Headline: 330 Ω is the only resistor value, so no colour-band
decoding is needed.

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
