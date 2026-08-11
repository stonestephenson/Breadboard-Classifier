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
for _n in range(0, 8):
    PINS[f"D{_n}"] = (PIND, DDRD, PORTD, _n)
for _n in range(8, 14):
    PINS[f"D{_n}"] = (PINB, DDRB, PORTB, _n - 8)
for _n in range(0, 6):
    PINS[f"A{_n}"] = (PINC, DDRC, PORTC, _n)

# D0/D1 are the USB serial link itself — touching them kills our connection.
RESERVED = {"D0", "D1"}
PROBE_PINS = [p for p in PINS if p not in RESERVED]

OP_PING = 0xFF
OP_READ_MEM = 0xFE
OP_WRITE_MEM = 0xFD
OP_ANALOG = {i: 0xC0 + i for i in range(6)}


class Board:
    def __init__(self, port: str, baud: int = 115200, verbose: bool = False):
        self.verbose = verbose
        self.ser = serial.Serial(port, baud, timeout=1.0)
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

    def close(self) -> None:
        self.ser.close()

    def _cmd(self, payload: bytes, want: int = 0) -> bytes:
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
            raise IOError(f"no reply reading 0x{addr:02x}")
        return r[0]

    def write_mem(self, addr: int, value: int) -> None:
        self._cmd(bytes([OP_WRITE_MEM, addr & 0xFF, (addr >> 8) & 0xFF, 1, value & 0xFF]))

    def analog_read(self, ch: int) -> int:
        r = self._cmd(bytes([OP_ANALOG[ch]]), 2)
        if len(r) != 2:
            raise IOError(f"no reply reading A{ch}")
        return r[0] | (r[1] << 8)

    # --- bit helpers ---------------------------------------------------------
    def _set_bit(self, reg: int, bit: int, high: bool) -> None:
        v = self.read_mem(reg)
        self.write_mem(reg, (v | (1 << bit)) if high else (v & ~(1 << bit)))

    def set_input(self, pin: str, pullup: bool) -> None:
        _, ddr, port, bit = PINS[pin]
        self._set_bit(ddr, bit, False)
        self._set_bit(port, bit, pullup)

    def set_output_low(self, pin: str) -> None:
        _, ddr, port, bit = PINS[pin]
        self._set_bit(port, bit, False)
        self._set_bit(ddr, bit, True)

    def read_all(self) -> dict[str, int]:
        """One read per port, then unpack — 3 transactions for every pin."""
        regs = {PINB: self.read_mem(PINB), PINC: self.read_mem(PINC),
                PIND: self.read_mem(PIND)}
        return {p: (regs[PINS[p][0]] >> PINS[p][3]) & 1 for p in PINS}

    def snapshot(self) -> dict[int, int]:
        return {r: self.read_mem(r) for r in (DDRB, PORTB, DDRC, PORTC, DDRD, PORTD)}

    def restore(self, snap: dict[int, int]) -> None:
        for reg, val in snap.items():
            self.write_mem(reg, val)


def find_hot_pins(b: Board) -> set[str]:
    """Pins held HIGH by the circuit itself. Never drive these low."""
    for p in PROBE_PINS:
        b.set_input(p, pullup=False)
    time.sleep(0.05)
    state = b.read_all()
    return {p for p in PROBE_PINS if state[p] == 1}


def scan(b: Board, hot: set[str]) -> dict[str, set[str]]:
    """Drive each safe pin low in turn; any pin reading low shares its node."""
    joined: dict[str, set[str]] = {}
    for p in PROBE_PINS:
        b.set_input(p, pullup=True)
    time.sleep(0.05)

    for driver in PROBE_PINS:
        if driver in hot:
            continue
        b.set_output_low(driver)
        time.sleep(0.01)
        state = b.read_all()
        b.set_input(driver, pullup=True)
        partners = {p for p in PROBE_PINS
                    if p != driver and p not in hot and state[p] == 0}
        if partners:
            joined[driver] = partners
    return joined


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
        if any(k in blob for k in ("usbmodem", "usbserial", "wchusb", "arduino", "metro")):
            return p.device
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--blink", type=int, metavar="PIN",
                    help="flash a digital pin 5x — confirms we can drive the circuit")
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
        print(f"banner        {'OK! seen' if b.banner_seen else 'not seen (may be fine)'}")
        print(f"ping          {'OK (0x30)' if b.ping() else 'NO REPLY — wrong sketch or port?'}")

        for reg, name in ((PIND, "PIND"), (PINB, "PINB"), (PINC, "PINC")):
            print(f"{name:<14}0b{b.read_mem(reg):08b}   (memory read works)")

        if args.blink is not None:
            pin = f"D{args.blink}"
            print(f"\nblinking {pin} five times — watch the board")
            snap = b.snapshot()
            for _ in range(5):
                b.set_output_low(pin)
                time.sleep(0.25)
                _, _, portreg, bit = PINS[pin]
                b._set_bit(portreg, bit, True)
                time.sleep(0.25)
            b.restore(snap)
            return 0

        snap = b.snapshot()
        try:
            hot = find_hot_pins(b)
            if hot:
                print(f"\nexternally driven HIGH (not driven low): {', '.join(sorted(hot))}")
            joined = scan(b, hot)
        finally:
            b.restore(snap)

        print("\nconnected pin groups:")
        nodes = merge_nodes(joined)
        if not nodes:
            print("  (none — no two pins are joined by the circuit)")
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
