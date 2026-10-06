# Breadboard Classifier

A student builds a circuit on a breadboard, photographs it, and the system tells them what is
wired wrong and how to fix it — in language a 13-year-old can act on.

Part of Sonoma State's Learning by Making / STEMACES program.
**Start with [CLAUDE.md](CLAUDE.md)** for the project's state and constraints,
[ARCHITECTURE.md](ARCHITECTURE.md) for the design and why.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -e ".[dev]"
./.claude/verify.sh          # lint, format, types, tests — should print PASS
```

## Try it

The example circuits live in `examples/`. The BasicBoard ones are built from its **measured**
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
#   -> Move it from j37 to i36.
```

The others are worth running to see the behaviours that matter most:

| File | What it demonstrates |
|------|---------------------|
| `basicboard_led_reversed.json` | Reversed polarity reported as such, not as bad wiring |
| `basicboard_uncertain.json` | **The same displacement, but flagged as a doubtful reading — the checker declines to call it an error and asks for a better photo instead** |
| `basicboard_no_resistor.json` | Caught with **no lab file at all** (`check` without `--lab`) |
| `basicboard_dead_rail.json` | The unpowered half of a split power rail. The kit's own board turned out to have continuous rails (`board.RAILS_SPLIT`), so this only applies to a split-rail board |
| `basicboard_rewired.json` | Not a mistake: the BasicBoard as rewired on 2026-09-27 (pin 2 green, 3 blue, 4 white, 5 red) |
| `basicboard_demo.json` | The lab for the live demo board, rebuilt 2026-09-29: pin 3 white, 4 blue, 5 green, 6 red. Written as a pure circuit: each leg on a named net (`{"net": "white"}`), no holes, because a lab cannot say where things go |
| `basicboard_as_built.json` | That demo board as built, every leg in its hole, as a person described it |
| `basicboard_as_seen.json` | The same, but not which way round each LED is, which a camera cannot see: the file a vision model will one day produce |

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

**Blink and watch** puts a camera on `--sweep`, so nobody has to watch:

```bash
./venv/bin/python tools/blink.py --camera 1           # the camera that sees the board
# pin  3  white LED at column 27
# pin  4  blue LED at column 32
# pin  5  green LED at column 38
# pin  6  red LED at column 44
# pins 2, 7-10  nothing lit
./venv/bin/python tools/blink.py --replay data/cache/blink/<run>   # re-judge a saved run
./venv/bin/python tools/blink.py --camera 1 --lab examples/basicboard_demo.json
# Every LED lights from the right pin, so your wiring works. ...
```

It switches pins 2-10 on one at a time and finds each LED as the spot that turns white, inside
the board. A pin whose photos could not be lined up, because the board moved, is blinked again on
its own straight away, up to twice. It gives which pin drives which LED, its colour, and its column (the row is
approximate, because the LED stands above the board). When what it sees is not one clean LED,
such as a glow with no bright centre or light in two places, it says "unclear" rather than
guessing. The bar for a glow is set by each check's own photos in which nothing lit, above a
small fixed floor, so a brighter room, another camera, or light that keeps changing does not
change the answer. A board so bright that it is washed out in the picture is said to be too
bright, and nothing is judged. Frame the board large, in a lit room. The board may be held up to the camera in your
hands: the photos are lined up on the board itself before they are compared. Only
the pin under test is ever driven. Every other pin is disconnected, so pins wired together
cannot fight, and no pin can pose as a ground. A pin tied to ground is released at once. However
the run ends, the board is handed back as the program set it up.

Add `--build examples/basicboard_as_seen.json` (to `tools/blink.py` or `tools/live.py`) to say
which leg to move. That file describes the board as built, standing in for the vision model that
will one day read it from the photo; edit it to match the board and check again. It is trusted
only where blinking confirms it. If light comes from somewhere it does not predict, the answer
says the entered parts don't match the board, and shows no fix. Otherwise it goes in two steps.
First, is every part in the right place? If not, the fix down to the hole ("Move it from a33 to
a32"), drawn on the photo. Up to three fixes are named in one check, each confirmed by its own
pin, and two wires on each other's pins are said as one swap. Then, for an LED placed right that stayed dark, the causes no picture
shows, one per check, most likely first: it is the wrong way round, a leg is not pushed in, the
LED is broken, and last, ask your teacher. `tools/live.py` remembers between checks, and says
"Fixed since the last check" when the LED lights.

A camera cannot see which way round an LED is once it is in the board, so `basicboard_as_seen.json`
leaves it open (`"direction": "unknown"`): the checker accepts it either way round, and blinking
settles it, since an LED that lights from its pin must face forwards. `basicboard_as_built.json`
gives the directions, as a person can: an LED entered backwards that stays dark gets "The blue LED
is in the wrong way round… at d31, d32".

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

Live, on a webcam: `./venv/bin/python tools/live.py` (`--list` to find cameras; an iPhone via
Continuity Camera works well). It marks every hole on the moving picture, green when usable.

**The demo:** `tools/live.py --camera 1 --lab examples/basicboard_demo.json` with the board on
USB. Press `c`. Each LED blinks in turn on screen, then the answer is drawn on the picture: green
circles for LEDs on the right pin, red for the wrong pin, and a plain sentence for each problem
and its fix. Pull an LED leg and press `c` again to see a dark LED reported.

From Python, `rectify(load_photo(path))` gives a `Rectification`. It answers "where is hole j37
in this photo?" (`to_photo`) and "which hole is under this pixel?" (`hole_at`). Its `ok` is
False when the fit cannot be trusted: the holes fit badly, the rail stripes are unreadable, or
the column is ambiguous. Then the answer is "take a better photo". The first run downloads the
83 MB corner model into `data/models/` and checks its hash.

The rectifier is
[breadboard-normalizer](https://github.com/hartleyblakey/breadboard-normalizer) (MIT), vendored in
`breadboard/_vendor/` (credits and local changes in
`breadboard/_vendor/breadboard_normalizer/__init__.py`). Its corner model is
[DocAligner](https://github.com/DocsaidLab/DocAligner) (Apache 2.0). Results and how it compares
with our earlier attempt: [docs/SPIKE_RECTIFY.md](docs/SPIKE_RECTIFY.md).

## What works, and what does not

| | State |
|---|---|
| Circuit model and checker | **Works.** [docs/CHECKER.md](docs/CHECKER.md) |
| Electrical probe | **Works on real hardware.** [docs/FIRMWARE_PROTOCOL.md](docs/FIRMWARE_PROTOCOL.md) |
| Rectifying a photo | **Works** on all 13 test photos, and live on a webcam with the board held in a hand. [docs/SPIKE_RECTIFY.md](docs/SPIKE_RECTIFY.md) |
| Does it work? (blinking against the lab) | **Works** on the real board, for LED labs. |
| Which leg to move? | **Works**, from a build described by hand and checked by blinking. |
| Photo → circuit | **Not started.** Needs training data. A hand-written file stands in for it. |
| Probe → circuit | **Not started.** Blinking gives per-pin facts instead. |
| Activity 3 reference circuit | **Done**, from the worksheet. |
| The other activities' reference circuits | Not authored yet. |

**For LED labs the loop is closed, except for one step.** A check blinks each LED, compares what
lit with the lab, and, given a description of the build, names the leg to move. Only that
description is still written by hand; nothing yet turns a photograph into a `Netlist`. Every
example in `examples/` is hand-authored.

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
  blink.py      find each LED from a camera watching its pin switch on
  verify.py     does it work? what each pin lit, against the lab
  diagnose.py   which leg to move: a described build, trusted where blinking agrees
  cli.py        python -m breadboard
  _vendor/      the vendored breadboard-normalizer, kept in upstream style
tools/probe.py  measure a real board over USB
tools/live.py   the rectifier on a live webcam
tools/blink.py  light each LED in turn and find it on camera
spikes/         our earlier rectifier spike, superseded; kept for the record
examples/       runnable circuits, generated by make_examples.py
tests/          test_basicboard.py checks against measured hardware; photo tests
                need the local sample photos in data/cache and skip without them
docs/           design decisions, open questions, protocol notes, spike results
```
