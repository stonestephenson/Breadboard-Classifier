"""Connectivity probe for the LbyM BasicBoard, over USB serial.

Measures which of the Metro Mini's pins are electrically joined by the student's
circuit. This is the "measure" stage of ARCHITECTURE.md section 3[4] — hardware
fact rather than inference — and it runs against the *shipped* LbyM sketch with
no firmware change, using the memory read/write opcodes it already exposes.
Protocol details and register addresses: docs/FIRMWARE_PROTOCOL.md.

Talks straight to the board over USB, so it needs nothing from the LbyM web app.

    ./venv/bin/python tools/probe.py --list
    ./venv/bin/python tools/probe.py                 # auto-detect port, scan
    ./venv/bin/python tools/probe.py --port /dev/cu.usbmodem1101
    ./venv/bin/python tools/probe.py --blink 4       # flash pin 4 (see the LED)

SAFETY. Driving a pin LOW while the student's circuit ties it to +5V would short
the output driver. Before scanning, every pin is read with pull-ups OFF; a pin
that still reads HIGH is externally driven and is never driven low. Only one pin
is driven at a time, and the original pin state is restored on exit.
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
import serial.tools.list_ports

# --- ATmega328P register addresses in the data address space -----------------
PINB, DDRB, PORTB = 0x23, 0x24, 0x25
PINC, DDRC, PORTC = 0x26, 0x27, 0x28
PIND, DDRD, PORTD = 0x29, 0x2A, 0x2B

# Arduino pin -> (PIN reg, DDR reg, PORT reg, bit). D0-D7 on port D, D8-D13 on
# port B, A0-A5 on port C.
PINS: dict[str, tuple[int, int, int, int]] = {}
for _n in range(8):
    PINS[f"D{_n}"] = (PIND, DDRD, PORTD, _n)
for _n in range(8, 14):
    PINS[f"D{_n}"] = (PINB, DDRB, PORTB, _n - 8)
for _n in range(6):
    PINS[f"A{_n}"] = (PINC, DDRC, PORTC, _n)

# D0/D1 are the USB serial link itself — touching them kills our connection.
RESERVED = {"D0", "D1"}
PROBE_PINS = [p for p in PINS if p not in RESERVED]

OP_PING = 0xFF
OP_READ_MEM = 0xFE
OP_WRITE_MEM = 0xFD
OP_ANALOG = {i: 0xC0 + i for i in range(6)}


class Board:
    def __init__(
        self,
        port: str,
        baud: int = 115200,
        verbose: bool = False,
        ser: serial.Serial | None = None,
    ):
        self.verbose = verbose
        # `ser` lets tests stand in an emulated board for the real port.
        self.ser = ser if ser is not None else serial.Serial(port, baud, timeout=1.0)
        # Opening the port toggles DTR, which resets the ATmega. Wait for the
        # sketch's "OK!" banner rather than guessing a delay.
        deadline = time.time() + 5.0
        banner = b""
        while time.time() < deadline:
            banner += self.ser.read(64)
            if b"OK!" in banner:
                break
        self.ser.reset_input_buffer()
        self.banner_seen = b"OK!" in banner
        self._drive: dict[int, int] | None = None  # direction regs, for drive_only
        self._keep_portd = 0  # PORTD bits of D0/D1, the serial link

    def close(self) -> None:
        self.ser.close()

    def resync(self) -> None:
        """Discard any late reply and check the board answers.

        An interrupted read (Ctrl-C, a signal) can leave a reply byte queued,
        which would shift every reply after it by one. Raises OSError if the
        board does not answer.
        """
        time.sleep(0.05)
        self.ser.reset_input_buffer()
        if not self.ping():
            raise OSError("the board is not answering")

    def _cmd(self, payload: bytes, want: int = 0) -> bytes:
        # The sketch only ever speaks when asked, so anything already waiting is
        # a stale reply to an interrupted command. Dropping it keeps this
        # command's reply aligned.
        self.ser.reset_input_buffer()
        self.ser.write(payload)
        self.ser.flush()
        out = self.ser.read(want) if want else b""
        if self.verbose:
            print(f"    -> {payload.hex(' ')}  <- {out.hex(' ') or '(none)'}")
        time.sleep(0.002)  # the sketch's loop() delays 1ms between commands
        return out

    def ping(self) -> bool:
        return self._cmd(bytes([OP_PING]), 1) == b"\x30"

    def read_mem(self, addr: int) -> int:
        r = self._cmd(bytes([OP_READ_MEM, addr & 0xFF, (addr >> 8) & 0xFF]), 1)
        if len(r) != 1:
            raise OSError(f"no reply reading 0x{addr:02x}")
        return r[0]

    def write_mem(self, addr: int, value: int) -> None:
        self._cmd(bytes([OP_WRITE_MEM, addr & 0xFF, (addr >> 8) & 0xFF, 1, value & 0xFF]))

    def analog_read(self, ch: int) -> int:
        r = self._cmd(bytes([OP_ANALOG[ch]]), 2)
        if len(r) != 2:
            raise OSError(f"no reply reading A{ch}")
        return r[0] | (r[1] << 8)

    # --- the sketch's own pin commands ---------------------------------------
    def digital_write(self, pin: int, high: bool) -> None:
        """digitalWrite through the sketch's opcodes: 0xE0+pin HIGH, 0xD0+pin LOW.

        Pins 2-12 only. This is the exact path the LbyM web app drives, and
        setup() leaves D2-D10 as outputs.
        """
        if not 2 <= pin <= 12:
            raise ValueError(f"the sketch only switches pins 2-12, not {pin}")
        self._cmd(bytes([(0xE0 if high else 0xD0) + pin]))

    def pin_level(self, pin: int) -> int:
        """The voltage a digital pin reads right now: 1 high, 0 low."""
        reg, _, _, bit = PINS[f"D{pin}"]
        return (self.read_mem(reg) >> bit) & 1

    def drive_only(self, pin: int | None) -> int | None:
        """Drive `pin` high and disconnect every other pin: input, no pull-up.

        With only one pin driving, no two pins can fight through a wire between
        them, and no other pin can act as a ground the way a pin held low
        would. D0/D1, the USB serial link, are left as they are. None
        disconnects every pin. Restore a snapshot afterwards to hand the board
        back to the sketch.

        Returns what `pin` reads the moment it starts driving: 1 high, 0 low,
        None if no pin. A 0 means the circuit is holding it at ground. The pin
        is then released straight away, before anything else is sent, so the
        short lasts two messages.

        No pin is ever driven low on the way. The outputs that are going away
        are released first, the one currently driven before any other. Then the
        output levels are set, which on an input only enables its pull-up. Only
        then does the new pin start driving, already high. Each register is
        verified afterwards.
        """
        if pin is not None and not 2 <= pin <= 13:
            raise ValueError(f"not a pin drive_only may drive: {pin}")
        if self._drive is None:
            self._drive = {r: self.read_mem(r) for r in (DDRB, DDRC, DDRD)}
            self._keep_portd = self.read_mem(PORTD) & 0x03
        state = self._drive
        ddr = {DDRB: 0, DDRC: 0, DDRD: state[DDRD] & 0x03}
        port = {PORTB: 0, PORTC: 0, PORTD: self._keep_portd}
        reg = bit = None
        if pin is not None:
            _, reg, port_reg, bit = PINS[f"D{pin}"]
            ddr[reg] |= 1 << bit
            port[port_reg] |= 1 << bit

        # Release: registers with outputs going away, the fullest first.
        for r in sorted(ddr, key=lambda r: -bin(state[r] & ~ddr[r]).count("1")):
            if state[r] & ~ddr[r]:
                state[r] &= ddr[r]
                self.write_mem(r, state[r])
        for r, v in port.items():
            self.write_mem(r, v)
        level = None
        if reg is not None and bit is not None:
            state[reg] = ddr[reg]
            self.write_mem(reg, ddr[reg])
            level = self.pin_level(pin) if pin is not None else None
            if level == 0:
                state[reg] &= ~(1 << bit)
                self.write_mem(reg, state[reg])
        for r, v in {**state, **port}.items():
            got = self.read_mem(r)
            if got != v:
                raise OSError(f"0x{r:02x} reads 0b{got:08b}, expected 0b{v:08b}")
        return level

    # --- register state ------------------------------------------------------
    # Registers are always written whole, from state computed in Python. An
    # earlier version did read-modify-write per pin (~350 round trips for one
    # scan); a single desynced byte then corrupted a direction register, turned
    # pins into outputs driving low, and made every pin look shorted to every
    # other. Whole-register writes with read-back verification remove both the
    # round-trip count and the failure mode.
    def write_regs(self, vals: dict[int, int], verify: bool = True) -> None:
        for reg, v in vals.items():
            self.write_mem(reg, v)
        if not verify:
            return
        for reg, v in vals.items():
            got = self.read_mem(reg)
            if got != v:
                raise OSError(
                    f"write to 0x{reg:02x} did not stick: "
                    f"wrote 0b{v:08b}, read back 0b{got:08b} "
                    f"(serial desync?)"
                )

    def read_all(self) -> dict[str, int]:
        """One read per port, then unpack — 3 transactions for every pin."""
        regs = {
            PINB: self.read_mem(PINB),
            PINC: self.read_mem(PINC),
            PIND: self.read_mem(PIND),
        }
        return {p: (regs[PINS[p][0]] >> PINS[p][3]) & 1 for p in PINS}

    def snapshot(self) -> dict[int, int]:
        return {r: self.read_mem(r) for r in (DDRB, PORTB, DDRC, PORTC, DDRD, PORTD)}

    def restore(self, snap: dict[int, int]) -> None:
        """Put the pin registers back as a snapshot had them.

        In three steps, so no two outputs ever fight on the way. First every pin
        is disconnected except D0/D1, then the output levels are set, then the
        directions. Write-only, so a confused reply stream cannot corrupt it.
        """
        self.write_mem(DDRB, 0)
        self.write_mem(DDRC, 0)
        self.write_mem(DDRD, snap[DDRD] & 0x03)
        for reg in (PORTB, PORTC, PORTD):
            self.write_mem(reg, snap[reg])
        for reg in (DDRB, DDRC, DDRD):
            self.write_mem(reg, snap[reg])
        self._drive = None


# Pull-up masks covering exactly the probe pins of each port.
PULLUP = {PORTB: 0x3F, PORTC: 0x3F, PORTD: 0xFC}  # D8-D13, A0-A5, D2-D7


def make_state(driver: str | None, keep_ddrd: int, keep_portd: int) -> dict[int, int]:
    """All probe pins INPUT_PULLUP, except `driver` which is OUTPUT LOW.

    D0/D1 bits are carried through from the board's own startup state — they are
    the serial link and must not be disturbed.
    """
    regs = {
        DDRB: 0x00,
        DDRC: 0x00,
        DDRD: keep_ddrd & 0x03,
        PORTB: PULLUP[PORTB],
        PORTC: PULLUP[PORTC],
        PORTD: PULLUP[PORTD] | (keep_portd & 0x03),
    }
    if driver:
        _, ddr, port, bit = PINS[driver]
        regs[ddr] |= 1 << bit
        regs[port] &= ~(1 << bit) & 0xFF
    return regs


def watch_analog(b: Board, seconds: float) -> dict[str, tuple[int, int]]:
    """Sample A0-A5 while the student changes the light, and report the spread.

    This is the electrical half of Activity 3. Pin-to-pin continuity cannot see
    that lab at all -- a sensor is not a short, so the three jumper wires join
    nothing the scan can detect. But the lab's own acceptance criterion is that
    "the brightness readings change when the light changes", and that is
    directly measurable. So this answers *whether the circuit works*, and the
    checker answers *why it does not*.
    """
    print(f"\nwatching A0-A5 for {seconds:.0f}s — cover the sensor, then shine a")
    print("light on it, so we can see which channel responds\n")

    lo = dict.fromkeys(OP_ANALOG, 1023)
    hi = dict.fromkeys(OP_ANALOG, 0)
    deadline = time.time() + seconds
    while time.time() < deadline:
        for ch in OP_ANALOG:
            v = b.analog_read(ch)
            lo[ch], hi[ch] = min(lo[ch], v), max(hi[ch], v)
        bars = "  ".join(f"A{ch}={b.analog_read(ch):4d}" for ch in sorted(OP_ANALOG))
        print(f"\r  {bars}", end="", flush=True)
    print("\n")
    return {f"A{ch}": (lo[ch], hi[ch]) for ch in sorted(OP_ANALOG)}


def describe_analog(ranges: dict[str, tuple[int, int]]) -> list[str]:
    """Turn min/max per channel into a verdict.

    Thresholds are deliberately loose. The point is to separate "this channel
    is doing something" from "this channel is pinned", not to measure lux.
    """
    out = []
    for name, (lo, hi) in ranges.items():
        spread = hi - lo
        if spread >= 30:
            verdict = f"responding (varied {lo}-{hi})"
        elif hi <= 20:
            verdict = f"stuck low ({hi}) — no power reaching it, or OUT tied to ground"
        elif lo >= 1000:
            verdict = (
                f"stuck high ({lo}) — nothing pulling it down; OUT may be unconnected"
            )
        else:
            verdict = f"steady at ~{(lo + hi) // 2}, not responding"
        out.append(f"  {name}  {verdict}")
    return out


def scan(
    b: Board, keep_ddrd: int, keep_portd: int, idle: dict[str, int]
) -> tuple[dict[str, set[str]], set[str]]:
    """Drive each pin low in turn; a pin that *changes* to low shares its node.

    `idle` is the resting state with every pin floating on its pull-up. The
    comparison against it is essential: pins the circuit already ties low read
    low no matter what we drive, so without subtracting the baseline they look
    connected to everything and union-find collapses the whole board into one
    node. Only a pin that was high at rest and went low because we drove the
    driver is genuinely joined to it.

    Returns (adjacency, conflicts). A conflict is a pin that stayed HIGH while
    we drove it LOW — something is forcing it up, so we release it immediately
    rather than fight a 5V rail through the output driver.

    Limitation: two pins that are *both* tied low at rest cannot be told apart
    by this method; they are reported together in the resting-low group instead.
    """
    joined: dict[str, set[str]] = {}
    conflicts: set[str] = set()

    for driver in PROBE_PINS:
        if idle[driver] == 0:
            continue  # already low; driving it teaches us nothing
        b.write_regs(make_state(driver, keep_ddrd, keep_portd))
        time.sleep(0.005)
        state = b.read_all()
        # Release the pin before doing anything else with the reading.
        b.write_regs(make_state(None, keep_ddrd, keep_portd), verify=False)

        if state[driver] == 1:
            conflicts.add(driver)
            continue
        partners = {
            p for p in PROBE_PINS if p != driver and idle[p] == 1 and state[p] == 0
        }
        if partners:
            joined[driver] = partners
    return joined, conflicts


def merge_nodes(joined: dict[str, set[str]]) -> list[set[str]]:
    """Collapse the pairwise results into electrical nodes (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, partners in joined.items():
        for c in partners:
            ra, rc = find(a), find(c)
            if ra != rc:
                parent[ra] = rc

    groups: dict[str, set[str]] = {}
    for x in parent:
        groups.setdefault(find(x), set()).add(x)
    return [g for g in groups.values() if len(g) > 1]


def autodetect() -> str | None:
    for p in serial.tools.list_ports.comports():
        blob = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(
            k in blob for k in ("usbmodem", "usbserial", "wchusb", "arduino", "metro")
        ):
            return p.device
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument(
        "--blink",
        type=int,
        metavar="PIN",
        help="flash a digital pin 5x — confirms we can drive the circuit",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="drive pins 2-12 HIGH one at a time; watch which LED lights",
    )
    ap.add_argument(
        "--hold",
        type=float,
        default=1.5,
        metavar="SEC",
        help="seconds to hold each pin during --sweep (default 1.5)",
    )
    ap.add_argument(
        "--analog",
        nargs="?",
        type=float,
        const=8.0,
        metavar="SEC",
        help="watch A0-A5 while you change the light (default 8s). This is "
        "Activity 3's own acceptance test, measured directly.",
    )
    ap.add_argument("--verbose", action="store_true", help="show every byte exchanged")
    args = ap.parse_args()

    if args.list:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("no serial ports found — is the board plugged in?")
        for p in ports:
            print(f"  {p.device:<28} {p.description}")
        return 0

    port = args.port or autodetect()
    if not port:
        print("No board found. Plug in the Metro Mini, or run --list and pass --port.")
        return 1

    print(f"port          {port}")
    b = Board(port, verbose=args.verbose)
    try:
        print(
            f"banner        {'OK! seen' if b.banner_seen else 'not seen (may be fine)'}"
        )
        pong = "OK (0x30)" if b.ping() else "NO REPLY — wrong sketch or port?"
        print(f"ping          {pong}")

        for reg, name in ((PIND, "PIND"), (PINB, "PINB"), (PINC, "PINC")):
            print(f"{name:<14}0b{b.read_mem(reg):08b}   (memory read works)")

        snap = b.snapshot()
        keep_ddrd, keep_portd = snap[DDRD] & 0x03, snap[PORTD] & 0x03

        if args.analog is not None:
            ranges = watch_analog(b, args.analog)
            print("\n".join(describe_analog(ranges)))
            responding = [n for n, (lo, hi) in ranges.items() if hi - lo >= 30]
            print()
            if "A0" in responding:
                print("A0 is responding to light, so the sensor circuit works.")
            elif responding:
                print(
                    f"{', '.join(responding)} responded but A0 did not. The lab's "
                    "program only reads A0, so the signal wire is probably on the "
                    "wrong analog pin."
                )
            else:
                print(
                    "Nothing responded. Either the light did not change enough, or "
                    "the sensor is not wired to the Arduino correctly — check the "
                    "board against the lab with `python -m breadboard check`."
                )
            return 0

        if args.sweep:
            # Uses the sketch's own digitalWrite opcodes rather than memory
            # pokes: setup() already leaves D2-D10 as outputs, and this is the
            # exact path the LbyM web app itself drives, so it doubles as a test
            # of the interface we will use in production.
            print(f"\ndriving each pin HIGH for {args.hold}s — watch the board\n")
            try:
                for pin in range(2, 13):
                    print(f"  pin {pin:>2} HIGH", flush=True)
                    b.digital_write(pin, True)
                    time.sleep(args.hold)
                    b.digital_write(pin, False)
                    time.sleep(0.2)
            finally:
                b.restore(snap)
            print("\nWhich pins lit an LED, and what colour?")
            return 0

        if args.blink is not None:
            pin = f"D{args.blink}"
            print(f"\nblinking {pin} five times — watch the board")
            try:
                for _ in range(5):
                    b.write_regs(make_state(pin, keep_ddrd, keep_portd), verify=False)
                    time.sleep(0.25)
                    b.write_regs(make_state(None, keep_ddrd, keep_portd), verify=False)
                    time.sleep(0.25)
            finally:
                b.restore(snap)
            return 0

        try:
            # Baseline: everything floating high on its internal pull-up. Any
            # pin reading low here is tied low by the circuit.
            b.write_regs(make_state(None, keep_ddrd, keep_portd))
            time.sleep(0.05)
            idle = b.read_all()
            low = sorted(p for p in PROBE_PINS if idle[p] == 0)
            print(
                f"\ntied LOW at rest (share the ground node): "
                f"{', '.join(low) or '(none)'}"
            )

            joined, conflicts = scan(b, keep_ddrd, keep_portd, idle)
            if conflicts:
                print(
                    f"forced HIGH by the circuit (drive aborted): "
                    f"{', '.join(sorted(conflicts))}"
                )
        finally:
            b.restore(snap)

        print("\npin-to-pin connections found by driving:")
        nodes = merge_nodes(joined)
        if not nodes:
            print("  (none — no two floating pins are joined to each other)")
        for g in nodes:
            print(f"  {' = '.join(sorted(g))}")

        analog = {f"A{i}": b.analog_read(i) for i in range(6)}
        print("\nanalog levels (0-1023, distinguishes direct / resistive / floating):")
        print("  " + "   ".join(f"{k}={v:4d}" for k, v in analog.items()))
    finally:
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
