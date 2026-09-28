# Decisions

Approaches considered and **rejected**, with the reasoning. Read this before proposing an
approach — several obvious-sounding ideas here are dead ends we have already walked down, and
one of them consumed a previous repo.

Format: what was proposed, why it was appealing, why it fails, what replaced it.

---

## R1. Photorealistic synthetic images via CycleGAN — REJECTED

**Proposed:** render synthetic breadboard images programmatically, then use CycleGAN to
translate them into photorealistic ones, so training data could be generated in unlimited
quantity with free labels. Implemented in `../breadboard_generator` (phases 7–8).

**Why it failed:**

1. **GANs destroy fine repeating geometry**, and the hole lattice *is* the geometry we depend
   on. Training snapshots (`breadboard_generator/epochstages/`) show wavy, melted hole rows and
   two panels of outright mode collapse.
2. **Silent label corruption.** Labels are defined *by* the lattice. A distortion of half a hole
   pitch produces an image that looks plausible while every annotation on it is wrong. There is
   no cheap way to detect this, which makes the failure mode worse than an obvious one.
3. **Wrong direction.** Making fake things look real is an open research problem. Making real
   things look canonical is routine engineering. The effort was pointed the expensive way.

**Replaced by:** rectification (real → canonical) plus compositing of real component crops onto
real board photos. See ARCHITECTURE.md §2 and §4.

**Salvaged:** `grid.py`, `board_spec.json`, the circuit JSON schema, and `mutations.py` all
carry forward. Only the renderer is retired.

---

## R2. Golden-photo diffing — REJECTED

**Proposed:** a teacher photographs the correct build once; student photos are rectified and
diffed against it region by region. No component recognition, no netlist, no LLM — shippable in
two or three weeks.

**Why it fails:** requires that everyone build the circuit in the same holes. The curriculum
team confirmed instructions **cannot** specify exact hole positions. Two builds of the same
circuit can be visually unrecognisable as the same thing.

This was the cheapest path to a useful product and its loss is the reason the project needs a
full perception pipeline. Survives only as an internal debugging tool.

---

## R3. Verifying specific expected holes — REJECTED

**Proposed:** since the target circuit is known, check only the holes it involves ("is there a
red wire endpoint at a10?") rather than reading the whole board.

**Why it fails:** same root cause as R2. Free placement means the expected holes are not known
in advance. Replaced by whole-board occupancy plus topological comparison.

---

## R4. LLM as the fault-finder — REJECTED

**Proposed:** feed the extracted netlist to an LLM and ask what is wrong and how to fix it.

**Why it fails:**

1. Connectivity reasoning must be **exact**, and LLMs are unreliable at it.
2. We can compute it exactly in code — shorts, missing series resistors, reversed polarity,
   off-by-one placement, the split power rail. All deterministic graph checks, all unit-testable.
3. For young students a confidently wrong repair instruction is the worst possible failure, and
   it carries our authority.

**Replaced by:** a deterministic checker (ARCHITECTURE.md §3 stage 6) with templated
explanations. An LLM may later phrase output or answer follow-up questions — operating on the
netlist as text, never on the image, and never deciding what is broken.

---

## R5. Graph isomorphism against a single reference netlist — REJECTED

**Proposed:** extract the student's netlist, compare it to the lab's reference netlist as
labelled graphs up to node relabelling. Handles placement freedom cleanly.

**Why it fails:** multiple structurally different builds are functionally correct for the same
lab — a resistor may sit on either side of the LED in a series circuit, and both work. Exact
graph matching would flag working circuits as broken, which is the failure mode we can least
afford.

**Replaced by:** two-stage verification — check functional properties first (accepts all valid
variants, including unanticipated ones), then compute minimum edit only on failure to produce
an actionable repair instruction.

---

## R6. Adding an alignment aid to the kit — REJECTED (external constraint)

**Proposed:** a printed paper mat with fiducial markers in the corners, or a cardboard stand
holding the board at a fixed angle. Costs pennies, makes rectification near-trivial and robust
even when the board itself is heavily occluded.

**Why rejected:** the project side ruled kit changes out of scope. Not a technical judgement.

**Consequence:** rectification must work off the board's own hole grid with wires, components
and fingers covering part of it. This is now the project's primary technical risk and the
reason the rectifier is prototyped before anything else is built.

---

## R7. Server-side GPU inference — DEFERRED, not rejected

**Proposed:** serve the model from the lab's GPU hardware, called by the LbyM web app.

**Why deferred:** production is AWS serverless with browser clients; a university GPU box in
that path means a public endpoint with uptime obligations to schools that nobody has agreed to
own. In-browser inference costs nothing per use, scales to any number of schools, and keeps
student images on the student's device.

**Revisit if** in-browser model quality proves inadequate. Keep the model boundary clean so it
can move. The GPUs remain the training hardware regardless.

---

## Standing decisions (chosen, with rationale)

| Decision | Rationale |
|----------|-----------|
| Rectify real photos into canonical grid space | Deterministic, self-checking, invertible. See ARCHITECTURE.md §3[2]. |
| Rectify coarse to fine, with Hartley Blakey's normalizer | A pretrained corner model gives the rough outline; hole snapping gives the accuracy. It fitted 13/13 photos where our fit-the-grid-from-nothing spike fitted 8/12 and could not tell which end was column 1. `docs/SPIKE_RECTIFY.md` |
| A rectification is usable only if orientation and column are confirmed | The normaliser's own grade passes upside-down boards when the stripes are unreadable, and fits one column off. Either would put every lead in the wrong named hole with full confidence. `docs/SPIKE_RECTIFY.md` |
| Run the corner model on onnxruntime, not DocAligner's package | DocAligner's support library pins an onnxruntime that does not exist for Python 3.14. The same ONNX file is what an in-browser build would run. Identical corners on 13 photos. |
| Perceive only in rectified space | Perspective, scale and rotation are already removed — a far smaller problem. |
| Verification over discovery | The target circuit is known; it constrains ambiguous pixels. Free-design is deferred (open question 6). |
| Deterministic checker, templated explanations | Exactness where exactness is required; no hallucinated fixes. |
| Abstain rather than guess | A wrong confident answer to a 13-year-old is worse than asking for a better photo. |
| Asymmetric confidence thresholds | Code is out of scope, so a false "wiring is correct" sends a student hunting in the wrong place. |
