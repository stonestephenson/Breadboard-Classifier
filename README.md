# Breadboard Classifier

A student builds a circuit on a breadboard, photographs it, and the system tells them what is
wired wrong and how to fix it — in language a 13-year-old can act on.

Part of Sonoma State's Learning by Making / STEMACES program.
**Start with [CLAUDE.md](CLAUDE.md)** for the project's state and constraints,
[ARCHITECTURE.md](ARCHITECTURE.md) for the design and why.

## Setup

```bash
python -m venv venv
./venv/bin/pip install -e ".[dev]"
./.claude/verify.sh          # lint, format, types, tests — should print PASS
```

## Try it

Six example circuits live in `examples/`, all built from the BasicBoard's **measured** topology
(pin 2 drives the red LED, 3 white, 4 green, 5 blue — recovered over USB, not assumed).

Look at a circuit:

```bash
./venv/bin/python -m breadboard show examples/basicboard.json
```

Check a correct build against the lab it belongs to:

```bash
./venv/bin/python -m breadboard check examples/basicboard.json --lab examples/basicboard.json
# This circuit matches 'basicboard'. Nothing to fix.
```

Now break something:

```bash
./venv/bin/python -m breadboard check examples/basicboard_led_moved.json \
    --lab examples/basicboard.json
# x wrong_connection   [baseline]
#   One leg of the blue LED is in the wrong place, so this part of the circuit is not joined up.
#   at j37
#   -> Move it from j37 to j36.
```

The others are worth running to see the behaviours that matter most:

| File | What it demonstrates |
|------|---------------------|
| `basicboard_led_reversed.json` | Reversed polarity reported as such, not as bad wiring |
| `basicboard_uncertain.json` | **The same displacement, but flagged as a doubtful reading — the checker declines to call it an error and asks for a better photo instead** |
| `basicboard_no_resistor.json` | Caught with **no lab file at all** (`check` without `--lab`) |
| `basicboard_dead_rail.json` | The unpowered half of a split power rail — invisible on the board |

Placement is free, so a correct circuit built anywhere passes. Try editing a hole number in
`examples/basicboard.json` — move the whole build ten columns along and it still reports clean.
Move only one leg and it does not.

Regenerate the examples after editing the generator: `./venv/bin/python examples/make_examples.py`

Machine-readable output for wiring into something else: `check ... --json`.
Exit code is 0 when clean, 1 when there are findings.

## Talk to the hardware

Plug a Metro Mini in over USB (a **data** cable, not charge-only):

```bash
./venv/bin/python tools/probe.py --list      # find the port
./venv/bin/python tools/probe.py             # measure pin connectivity
./venv/bin/python tools/probe.py --sweep     # light each LED in turn; watch the board
```

This runs against the **shipped** LbyM sketch with no firmware change and needs nothing from the
web app — see [docs/FIRMWARE_PROTOCOL.md](docs/FIRMWARE_PROTOCOL.md) for how, and for the safety
measures that stop it shorting an output driver.

## Look at photos

```bash
./venv/bin/python spikes/hole_detect.py data/cache/sample/*.jpg   # -> data/cache/holes_overlay.jpg
./venv/bin/python spikes/lattice_fit.py data/cache/sample/*.jpg   # -> data/cache/rectified.jpg
```

Exploratory, and **not finished** — see [docs/SPIKE_RECTIFY.md](docs/SPIKE_RECTIFY.md).

## What works, and what does not

| | State |
|---|---|
| Circuit model and checker | **Works.** 109 tests. [docs/CHECKER.md](docs/CHECKER.md) |
| Electrical probe | **Works on real hardware.** [docs/FIRMWARE_PROTOCOL.md](docs/FIRMWARE_PROTOCOL.md) |
| Rectifying a photo | Partly. Fits, but sometimes onto the wrong lattice. [docs/SPIKE_RECTIFY.md](docs/SPIKE_RECTIFY.md) |
| Photo → circuit | **Not started.** Needs training data. |
| Probe → circuit | **Not started.** See below. |
| Lab reference circuits | **Blocked** on curriculum PDFs. |

**The two halves do not join up yet.** You can measure a real board, and you can check a circuit,
but nothing yet turns either a photograph or a probe reading into a `Netlist` for the checker to
consume. Every example in `examples/` is hand-authored.

Connecting the probe is the smaller of the two gaps and is next. Note it can only ever be
partial: passive continuity finds *wires* reliably, but a path through an LED and a 330 Ω
resistor leaves the pin in the chip's undefined input band, so the reading there is arbitrary.
Identifying which LED sits on which pin needs the `--sweep` stimulus with something watching —
a person today, the camera eventually.

## Layout

```
breadboard/     the circuit model and checker — deterministic, fully tested
  board.py      WB-102 geometry: which holes share an electrical node
  netlist.py    the data format everything produces and consumes
  graph.py      topology; deciding when two circuits are the same one
  check.py      diagnosis: findings with a repair a student can carry out
  cli.py        python -m breadboard
tools/probe.py  measure a real board over USB
spikes/         exploratory photo work, not production
examples/       runnable circuits, generated by make_examples.py
tests/          109 tests; test_basicboard.py checks against measured hardware
docs/           design decisions, open questions, protocol notes, spike results
```
