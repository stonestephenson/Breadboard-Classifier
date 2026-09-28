# LbyM firmware protocol — what it gives us

**Source:** the LbyM driver sketch `lbymDriver2024-HC-SR04-ALT/` (received 2026-08-06, built
with Arduino IDE 1.8.19). LbyM's own code — held locally, deliberately not committed here.
**Target:** Adafruit Metro Mini V2 = ATmega328P @ 16 MHz. Serial at 115200 baud.
**Analysed:** 2026-08-06.

## Headline

**We can run a full connectivity probe today, with no firmware change and no dependency on
Troy.** The sketch exposes arbitrary memory read and write, and on an AVR the GPIO registers are
memory-mapped — so we already have complete control over pin direction, pull-ups, drive state,
and reads. This was the pivotal unknown in ARCHITECTURE.md §3[4]; it is resolved.

Ask for a first-class diagnostic opcode anyway, as the *production* path (see Caveats).

## Protocol shape

`loop()` reads one byte per iteration and hands it to `dispatch()`, an `else if` chain on the
opcode. Multi-byte commands block inside `ugetc()` until their operands arrive. There is a
`delay(1)` per iteration, so budget ~1 ms per command — a full probe is well under a second.

Returns are little-endian: `write16` = 2 bytes, `write32` = 4 bytes, `uputc` = 1 byte.

## Opcode map

| Opcode | Effect | Returns |
|--------|--------|---------|
| `0xFF` | ping | `0x30` |
| `0xFE` | **read memory** — operands: addr_lo, addr_hi | 1 byte at that address |
| `0xFD` | **write memory** — operands: addr_lo, addr_hi, count, then `count` bytes | — |
| `0xF1` | init SI1145 UV/light sensor (I2C) | `0x0000` on success |
| `0xF6` | `millis()` | u32 |
| `0xF7` | ultrasonic ping distance | u32 (see `EnNeg`) |
| `0xF8` / `0xF9` | cached velocity / acceleration | u32 |
| `0xC7` | ultrasonic `distNow()` | u32 |
| `0xE2`–`0xEC` | `digitalWrite(pin, HIGH)`, pins 2–12 | — |
| `0xEF` | `digitalWrite(LED_BUILTIN, HIGH)` | — |
| `0xD2`–`0xDC` | `digitalWrite(pin, LOW)`, pins 2–12 | — |
| `0xDF` | `digitalWrite(LED_BUILTIN, LOW)` | — |
| `0xC0`–`0xC5` | `analogRead(A0..A5)` | u16, 10-bit |
| `0xC6`, `0xC8`–`0xCC` | `digitalRead(6, 8, 9, 10, 11, 12)` | 1 byte |

### Dead opcodes — a real bug, worth knowing

The block of `digitalRead` handlers is marked "Depreciated from prior lines" and is **partly
unreachable**, because the `else if` chain matches earlier branches first:

| Intended | Shadowed by | Result |
|----------|-------------|--------|
| `0xC0`–`0xC5` → `digitalRead(0..5)` | `analogRead(0..5)` | **dead** |
| `0xC7` → `digitalRead(7)` | ultrasonic `distNow()` | **dead** |
| `0xC6`, `0xC8`–`0xCC` | nothing earlier | **live** |

So digital reads are only directly available on pins 6, 8, 9, 10, 11, 12. This does not block us
— memory reads of `PINB`/`PINC`/`PIND` give every pin at once — but it is worth reporting back
to Troy as a genuine defect.

Verified programmatically by walking the dispatch chain in order, not by eye.

## The escape hatch: memory-mapped GPIO

ATmega328P data-space addresses:

| Port | PIN (read) | DDR (direction) | PORT (drive / pull-up) | Arduino pins |
|------|-----------|-----------------|------------------------|--------------|
| B | `0x23` | `0x24` | `0x25` | D8–D13 (bits 0–5) |
| C | `0x26` | `0x27` | `0x28` | A0–A5 (bits 0–5) |
| D | `0x29` | `0x2A` | `0x2B` | D0–D7 (bits 0–7) |

Read `PIND`:  `FE 29 00` → one byte, the live state of D0–D7.
Write `DDRD`: `FD 2A 00 01 <value>`.

That is complete GPIO control: set any pin to input-with-pull-up or output, drive it either way,
and read all eight pins of a port in a single transaction.

## Connectivity probe

Standard pin-to-pin continuity scan:

1. Put all candidate pins in **INPUT_PULLUP** (`DDR` bit = 0, `PORT` bit = 1). Every pin now
   floats high.
2. For each probe pin *P*:
   - set *P* to **OUTPUT LOW** (`DDR` bit = 1, `PORT` bit = 0);
   - read `PINB`, `PINC`, `PIND` (3 transactions);
   - **any other pin reading LOW shares an electrical node with *P***;
   - restore *P* to INPUT_PULLUP.
3. The result is a full connectivity matrix over the Arduino's pins — *measured*, not inferred.

~13 probe pins × ~5 transactions ≈ 65 commands ≈ well under a second.

### analogRead makes it richer than continuity

`analogRead` returns 10 bits, so a node's voltage distinguishes **direct connection** from
**connection through a 330 Ω resistor** from **floating**. We can detect whether a resistor is
actually in a path, not merely whether two points are joined.

### LED polarity becomes measurable

ARCHITECTURE.md §5 flags LED polarity as visually subtle — the flat edge is usually hidden once
seated. Electrically it is easy: a diode conducts one way only, so driving a pair of nodes one
way and then the other gives an asymmetric response. **This retires a known-hard perception
problem.**

### Stimulus–response capture — links the image to the circuit directly

We are already capturing a *video clip*. If the Arduino toggles pins while the camera is
recording, then **whichever LED blinks in the video is the LED connected to that pin.** That
resolves component-to-node association by observation instead of inference, for every LED that
lights.

Cheap on the Chromebook path (camera and serial on one device). On the phone path it needs the
pairing channel to carry timing. See ARCHITECTURE.md §3[1].

**Built (2026-09-27): `tools/blink.py`, via `Board.drive_only` in `tools/probe.py`.** It uses
memory writes, not the `0xE0+pin` opcodes. The pin under test is set to OUTPUT HIGH and every
other pin to INPUT with no pull-up. So two pins wired together can never fight, and no pin
held LOW can act as a ground for someone else's LED. The switch releases the old output first,
then sets PORT, then DDR, so no pin is ever driven LOW on the way. The new pin is read back
the instant it drives. If it reads LOW, it is released before anything else is sent. Restoring
the sketch's state disconnects every pin first, then sets PORT, then DDR. Every command first
discards any stale reply, and a check starts with a resync (flush plus ping). So an
interrupted read cannot shift later replies by a byte. All of this is tested against an
emulated chip in `tests/test_probe_board.py`. Measured on the board: a
disconnected pin with nothing attached keeps the level it was last driven to, for seconds. So
pin readings in this state cannot prove a wire between two pins, and nothing claims to.

## Caveats and reserved pins

| Pin(s) | Why hands off |
|--------|---------------|
| **D0 / D1** | RX/TX for the USB serial link. Touching `DDRD` bits 0–1 kills our own connection. **Mask them in every write.** |
| **D10 / D11** | Ultrasonic trigger / echo (`ping.h`). |
| **A4 / A5** | I2C (SDA/SCL) for the SI1145 light sensor. |
| **D13** | `LED_BUILTIN`, also SCK. |

Other cautions:

- `setup()` leaves D2–D10 as **OUTPUT** and D11–D12 as INPUT_PULLUP. A student circuit that
  fights a driven output can source or sink real current — probe with pull-ups, drive only one
  pin at a time, and restore state afterwards.
- Writing arbitrary memory can corrupt RAM. Confine writes to the port registers above. A board
  reset recovers.
- This is an **undocumented internal**, not an API. It breaks if the sketch or MCU changes.

## Recommendation

- **Now:** build the probe on memory-mapped GPIO. Unblocks development immediately; needs
  nothing from anyone.
- **Production:** still ask Troy for a first-class diagnostic opcode — one command that runs the
  scan on-device and returns the matrix. Faster (no per-pin round trips), safer (bounded, no raw
  memory writes), and stable across firmware revisions.
- **Report the dead-opcode bug** — `digitalRead` on pins 0–5 and 7 is unreachable.
- **Open:** whether the web app will let our code send raw serial commands. This is the real
  remaining dependency, and it is a *web app* question, not a firmware one — so accept the offer
  to see the web app source.
