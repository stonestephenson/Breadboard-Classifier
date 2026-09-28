"""The probe's pin driving, run against an emulated board.

tools/probe.py talks to the shipped LbyM sketch, which exposes raw reads and
writes of the chip's memory. The pin registers live in that memory. FakeAvr
answers those commands and models the pin registers, including a pin a circuit
holds at ground. So these tests run the real Board code and check what the pins
actually did, not just the order of calls.

The guarantees that matter, because a real student's board is on the other end:

- During a check at most one pin ever drives, and it drives high. No pin is
  ever driven low on the way, so no two pins can fight through a wire.
- A pin held at ground is released within two messages.
- The serial pins D0/D1 are never touched.
- Restoring hands the board back exactly, without two outputs fighting.
- A stale reply from an interrupted command cannot shift later readings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import probe
from probe import DDRB, DDRC, DDRD, PINS, PORTB, PORTC, PORTD, Board

# The shipped sketch's setup(), as read from the real board after a reset:
# D2-D10 and D13 outputs driven low, D11-D12 inputs with pull-ups.
SETUP = {DDRB: 0x27, PORTB: 0x18, DDRC: 0x00, PORTC: 0x00, DDRD: 0xFC, PORTD: 0x00}
PIN_TO_PORT = {pin_reg: (ddr, port) for pin_reg, ddr, port, _ in PINS.values()}
CHECKED = [p for p in PINS if p not in ("D0", "D1")]


class FakeAvr:
    """Answers the sketch's ping, memory read and memory write commands."""

    def __init__(self, regs=SETUP, grounded=()):
        self.reg = dict(regs)
        self.grounded = {f"D{p}" for p in grounded}
        self.inbox = bytearray(b"OK!")  # the banner the sketch prints on reset
        self.pending = bytearray()
        self.commands = 0
        self.states: list[dict[str, int]] = []  # driven level per output, per write

    # --- the parts of serial.Serial that Board uses --------------------------
    def write(self, data: bytes) -> None:
        self.pending += data
        while self._step():
            pass

    def read(self, n: int) -> bytes:
        out, self.inbox = bytes(self.inbox[:n]), self.inbox[n:]
        return out

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.inbox.clear()

    def close(self) -> None:
        pass

    # --- the sketch -----------------------------------------------------------
    def _step(self) -> bool:
        p = self.pending
        if not p:
            return False
        if p[0] == 0xFF:
            del p[:1]
            self.inbox.append(0x30)
        elif p[0] == 0xFE and len(p) >= 3:
            addr = p[1] | p[2] << 8
            del p[:3]
            self.inbox.append(self._read(addr))
        elif p[0] == 0xFD and len(p) >= 5:
            addr, value = p[1] | p[2] << 8, p[4]
            del p[:5]
            self.reg[addr] = value
            self.states.append(self.outputs())
        elif p[0] in (0xFE, 0xFD):
            return False  # wait for the rest of the command
        else:
            del p[:1]
        self.commands += 1
        return True

    def _read(self, addr: int) -> int:
        if addr not in PIN_TO_PORT:
            return self.reg.get(addr, 0)
        value = 0
        for name, (pin_reg, _, _, bit) in PINS.items():
            if pin_reg == addr and self.level(name):
                value |= 1 << bit
        return value

    def level(self, name: str) -> int:
        _, _, port, bit = PINS[name]
        high = self.reg[port] >> bit & 1  # driven level, or pull-up on an input
        return 0 if name in self.grounded else high

    def outputs(self) -> dict[str, int]:
        """Each output pin other than D0/D1, and the level it drives."""
        out = {}
        for name in CHECKED:
            _, ddr, port, bit = PINS[name]
            if self.reg[ddr] >> bit & 1:
                out[name] = self.reg[port] >> bit & 1
        return out


@pytest.fixture(autouse=True)
def _no_delays(monkeypatch):
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)


def _board(avr: FakeAvr) -> Board:
    return Board("emulated", ser=avr)  # type: ignore[arg-type]


class TestDriveOnly:
    def test_at_most_one_pin_ever_drives_and_it_drives_high(self):
        avr = FakeAvr()
        board = _board(avr)
        board.drive_only(None)
        avr.states.clear()
        for pin in range(2, 11):
            assert board.drive_only(pin) == 1
            board.drive_only(None)
        for state in avr.states:
            assert len(state) <= 1, state
            assert all(level == 1 for level in state.values()), state

    def test_leaving_the_sketchs_state_never_drives_a_pin_high_against_one_low(self):
        avr = FakeAvr()
        board = _board(avr)
        board.drive_only(4)
        for state in avr.states:
            assert not (1 in state.values() and 0 in state.values()), state

    def test_a_pin_held_at_ground_is_released_within_two_messages(self):
        avr = FakeAvr(grounded={4})
        board = _board(avr)
        board.drive_only(None)
        avr.states.clear()
        start = avr.commands
        assert board.drive_only(4) == 0
        # Find when D4 started and stopped driving, in commands.
        driving = [i for i, s in enumerate(avr.states) if "D4" in s]
        assert driving, "it never drove the pin at all"
        assert len(driving) == 1  # released by the very next register write
        assert avr.states[-1].get("D4") is None
        assert avr.commands - start < 40  # and verification still ran

    def test_the_serial_pins_are_never_touched(self):
        regs = {**SETUP, DDRD: 0xFE, PORTD: 0x03}
        avr = FakeAvr(regs)
        board = _board(avr)
        for pin in (2, 7, 9, None):
            board.drive_only(pin)
        assert avr.reg[DDRD] & 0x03 == 0x02
        assert avr.reg[PORTD] & 0x03 == 0x03

    @pytest.mark.parametrize("pin", [0, 1, 14])
    def test_refuses_the_serial_pins_and_pins_that_do_not_exist(self, pin):
        with pytest.raises(ValueError):
            _board(FakeAvr()).drive_only(pin)


class TestRestore:
    def test_hands_the_board_back_exactly_without_two_outputs_fighting(self):
        avr = FakeAvr()
        board = _board(avr)
        snapshot = board.snapshot()
        board.drive_only(5)
        avr.states.clear()
        board.restore(snapshot)
        assert {r: avr.reg[r] for r in SETUP} == SETUP
        for state in avr.states:
            assert not (1 in state.values() and 0 in state.values()), state


class TestStaleReplies:
    def test_a_late_reply_cannot_shift_the_next_reading(self):
        # Ctrl-C between sending a read and collecting its reply leaves the
        # reply waiting. Without care, every later reading is off by one.
        avr = FakeAvr()
        board = _board(avr)
        avr.inbox.append(0xAA)
        assert board.read_mem(DDRB) == SETUP[DDRB]
        assert board.read_mem(PORTB) == SETUP[PORTB]

    def test_resync_checks_the_board_answers(self):
        avr = FakeAvr()
        board = _board(avr)
        avr.inbox += b"\x01\x02"
        board.resync()
        assert board.read_mem(DDRC) == SETUP[DDRC]
