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

Twelve example circuits live in `examples/`. The BasicBoard ones are built from its **measured**
topology (pin 2 drives the red LED, 3 white, 4 green, 5 blue — recovered over USB, not assumed);
the Activity 3 ones come from the curriculum worksheet.

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

**Activity 3 (Intro to Sensors)** is the first lab with real wiring, and its answer is fixed by
the worksheet: the light sensor's + to 3V, − to GND, OUT to A0, with three jumper wires placed
wherever the student likes.

```bash
./venv/bin/python -m breadboard check examples/activity3_on_5v.json --lab examples/activity3.json
# -> One leg of the red wire is in the wrong place. Move it from j32 to j31.
```

| File | The mistake |
|------|-------------|
| `activity3_on_5v.json` | Sensor powered from 5V; the data sheet says 3V |
| `activity3_wrong_analog.json` | Signal wire on A1, but the program only reads A0 |
| `activity3_sensor_backwards.json` | + and − swapped — can destroy the sensor |
| `activity3_wire_off_by_one.json` | A wire one column short of the sensor |
| `activity3_uncertain.json` | The same displacement, read doubtfully |

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
./venv/bin/python tools/probe.py --analog    # watch A0-A5 while you change the light
```

`--analog` is Activity 3's own acceptance test, measured directly. Continuity cannot see that
lab at all — a sensor is not a short, so the three jumper wires join nothing a scan detects — but
the worksheet's criterion is that "the brightness readings change when the light changes", and
that is measurable. It answers *whether the circuit works*; the checker answers *why it does not*.

This runs against the **shipped** LbyM sketch with no firmware change and needs nothing from the
web app — see [docs/FIRMWARE_PROTOCOL.md](docs/FIRMWARE_PROTOCOL.md) for how, and for the safety
measures that stop it shorting an output driver.

## Find the board in a photo

```bash
./venv/bin/python -m breadboard rectify photo.jpg -o out/
# hole grid    good (confidence 1.00)
# orientation  confirmed by the rail stripes
# column fit   margin +0.053 (needs 0.02)
# verdict      usable
#   a1   at photo pixel (657, 333)
#   ...
# wrote out/photo_holes.jpg        every hole marked on the original photo
# wrote out/photo_rectified.jpg    the top-down canonical view
```

From Python, `rectify(load_photo(path))` gives a `Rectification`. It answers "where is hole j37
in this photo?" (`to_photo`) and "which hole is under this pixel?" (`hole_at`). Its `ok` is
False when the fit cannot be trusted: the holes fit badly, the rail stripes are unreadable, or
the column is ambiguous. Then the answer is "take a better photo". The first run downloads the
83 MB corner model into `data/models/` and checks its hash.

The rectifier is **Hartley Blakey's**
[breadboard-normalizer](https://github.com/hartleyblakey/breadboard-normalizer), vendored in
`breadboard/_vendor/` (credits and local changes in its `__init__.py`). Its corner model is
[DocAligner](https://github.com/DocsaidLab/DocAligner) (Apache 2.0). Results and how it compares
with our earlier attempt: [docs/SPIKE_RECTIFY.md](docs/SPIKE_RECTIFY.md).

## What works, and what does not

| | State |
|---|---|
| Circuit model and checker | **Works.** [docs/CHECKER.md](docs/CHECKER.md) |
| Electrical probe | **Works on real hardware.** [docs/FIRMWARE_PROTOCOL.md](docs/FIRMWARE_PROTOCOL.md) |
| Rectifying a photo | **Works** on all 13 test photos; live webcam not yet tried. [docs/SPIKE_RECTIFY.md](docs/SPIKE_RECTIFY.md) |
| Photo → circuit | **Not started.** Needs training data. |
| Probe → circuit | **Not started.** See below. |
| Activity 3 reference circuit | **Done**, from the worksheet. |
| Activities 1, 4, 5 references | Not authored yet. |

**The two halves do not join up yet.** You can measure a real board, find the board in a photo,
and check a circuit. But nothing yet turns a photograph or a probe reading into a `Netlist` for
the checker to consume. Every example in `examples/` is hand-authored.

What electricity *can* do on its own is answer whether a circuit works: `--sweep` for the LED
labs, `--analog` for the sensor labs. What it cannot do is say *where* a wire should move, and
for Activity 3 it cannot see the wiring at all. That gap is the camera's job.

Note too that passive continuity finds *wires* reliably but not component paths: through an LED
and a 330 Ω resistor the pin sits in the chip's undefined input band, so the reading is arbitrary.
Identifying which LED is on which pin needs the `--sweep` stimulus with something watching — a
person today, the camera eventually.

## Layout

```
breadboard/     the circuit model and checker — deterministic, fully tested
  board.py      WB-102 geometry: which holes share an electrical node
  netlist.py    the data format everything produces and consumes
  graph.py      topology; deciding when two circuits are the same one
  check.py      diagnosis: findings with a repair a student can carry out
  rectify.py    find the board in a photo; map holes between photo and board
  cli.py        python -m breadboard
  _vendor/      Hartley Blakey's normalizer, kept in upstream style
tools/probe.py  measure a real board over USB
spikes/         our earlier rectifier spike, superseded; kept for the record
examples/       runnable circuits, generated by make_examples.py
tests/          test_basicboard.py checks against measured hardware; photo tests
                need the local sample photos in data/cache and skip without them
docs/           design decisions, open questions, protocol notes, spike results
```
