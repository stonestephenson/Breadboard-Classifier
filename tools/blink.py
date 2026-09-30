"""Blink and watch: light each LED in turn and let the camera find it.

    ./venv/bin/python tools/blink.py --camera 1       # run it on the board
    ./venv/bin/python tools/blink.py --replay DIR     # re-judge a saved run
    ... --lab examples/basicboard_demo.json           # and check it against a lab
    ... --build examples/basicboard_as_built.json     # and say which leg to move

Needs the board on USB, a lit room, and a camera that sees the whole board for
the ~20 s a run takes. The board can be held in a hand: the photos are lined up
on the board before they are compared. The laptop switches the output pins
(2-10) on one at a time, by writing the pin registers through the shipped
sketch's memory commands (tools/probe.py Board.drive_only). The camera
photographs the board with each pin on and then off, and breadboard/blink.py
works out what lit.

With --build, a circuit file describing the board as built stands in for the
vision model that will one day read it from the photo (breadboard/diagnose.py).
If blinking confirms the description, the answer names the fix down to the
hole; if not, it says the description does not match the board.

It prints one line per pin, and writes:
  data/cache/blink/<time>/   the frames, so the run can be re-judged (--replay)
  out/blink/<time>.jpg       the photo, each LED circled and labelled

Safety: only the pin under test ever drives; every other pin is disconnected. A
pin the circuit holds at ground is released within two messages and reported,
not photographed. However the run ends (finishing, an error, Ctrl-C, or the
terminal closing), the board is handed back as the sketch set it up. What this
cannot catch is in breadboard.blink.run_sequence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from live import first_frame, open_camera
from probe import Board, autodetect

from breadboard.blink import OUTPUT_PINS, Session, analyse, run_sequence
from breadboard.check import Finding
from breadboard.diagnose import diagnose
from breadboard.netlist import Netlist, NetlistError
from breadboard.verify import Verdict, verify

SETTLE_S = 0.6  # after switching, let the LED and the camera's exposure settle
SAMPLES = 6  # frames averaged per photo, to beat sensor noise
SESSIONS = Path("data/cache/blink")

LABEL_COLOURS = {  # RGB, for drawing
    "red": (255, 60, 60),
    "orange": (255, 150, 40),
    "yellow": (255, 230, 40),
    "green": (40, 230, 80),
    "blue": (60, 140, 255),
    "white": (255, 255, 255),
}


class Grabber(threading.Thread):
    """Reads the camera continuously so a sample is never a stale buffered frame."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._count = 0
        self.running = True

    def run(self) -> None:
        while self.running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._frame, self._count = frame, self._count + 1
            else:
                time.sleep(0.01)  # a hiccup: do not spin

    def sample(self, n: int = SAMPLES) -> np.ndarray:
        """The mean of the next n new frames, as RGB."""
        frames, seen = [], self._count
        deadline = time.monotonic() + 5.0
        while len(frames) < n:
            if time.monotonic() > deadline:
                raise OSError("the camera stopped sending frames")
            if self._count != seen and self._frame is not None:
                seen = self._count
                frames.append(self._frame.astype(np.float32))
            time.sleep(0.005)
        mean = np.mean(frames, axis=0).astype(np.uint8)
        return cv2.cvtColor(mean, cv2.COLOR_BGR2RGB)


def _exit_on_signal(signum: int, _frame: object) -> None:
    # Closing the terminal (SIGHUP) or `kill` (SIGTERM) would otherwise end the
    # process without running the finally blocks that switch pins off.
    raise SystemExit(128 + signum)


def record(camera: int, pins: tuple[int, ...]) -> Path:
    """Run the blink sequence on the real board and save every frame."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    cap = open_camera(camera)
    grabber: Grabber | None = None
    try:
        if not cap.isOpened() or first_frame(cap) is None:
            raise OSError(f"camera {camera} is not sending pictures")
        grabber = Grabber(cap)
        grabber.start()
        port = autodetect()
        if not port:
            raise OSError("no board found on USB")
        board = Board(port)
        try:
            board.resync()
            snapshot = board.snapshot()
            try:
                frames, shorted = run_sequence(board, pins, grabber.sample, SETTLE_S)
            finally:
                with contextlib.suppress(OSError):
                    board.resync()  # restore is write-only, so it runs either way
                board.restore(snapshot)
        finally:
            board.close()
    finally:
        # Stop the reader before releasing the camera: releasing it mid-read
        # crashes the process on macOS. If the reader is stuck, leave the camera
        # to the process exit rather than risk that.
        if grabber is not None:
            grabber.running = False
            grabber.join(timeout=2.0)
        if grabber is None or not grabber.is_alive():
            cap.release()

    return save_run(frames, pins, shorted, camera)


def save_run(
    frames: dict[str, np.ndarray],
    pins: tuple[int, ...],
    shorted: list[int],
    camera: int,
) -> Path:
    """Save a run's frames to data/cache/blink/<time>/, for --replay."""
    run = SESSIONS / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = run / f"{name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise OSError(f"could not write {path}")
    meta = {"camera": camera, "pins": list(pins), "shorted": shorted}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run


def load(run: Path) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    meta = json.loads((run / "meta.json").read_text())
    frames = {}
    for path in sorted(run.glob("*.png")):
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"cannot read {path}")
        frames[path.stem] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return frames, meta["pins"], meta.get("shorted", [])


def report(session: Session, pins: list[int]) -> list[str]:
    if session.rect is None:
        return ["Could not find the board in any frame. Nothing was judged."]
    lines, dark = [], []
    for pin in pins:
        glow = session.glows.get(pin)
        if pin in session.shorted:
            lines.append(f"pin {pin:>2}  did not go high: tied to ground? Switched off.")
        elif pin in session.moved:
            lines.append(f"pin {pin:>2}  not judged: its photos could not be lined up")
        elif glow is None:
            lines.append(f"pin {pin:>2}  not judged: no pictures")
        elif glow.status == "lit":
            lines.append(f"pin {pin:>2}  {glow.colour} LED at column {glow.column}")
        elif glow.status == "unclear":
            lines.append(f"pin {pin:>2}  unclear: {glow.note}. Check it by eye.")
        else:
            dark.append(pin)
    if dark:
        lines.append(f"pin{'s' if len(dark) > 1 else ''} {_ranges(dark)}  nothing lit")
    return lines


def _ranges(pins: list[int]) -> str:
    """[6, 7, 8, 10] -> "6-8, 10"."""
    runs: list[list[int]] = []
    for pin in sorted(pins):
        if runs and pin == runs[-1][-1] + 1:
            runs[-1].append(pin)
        else:
            runs.append([pin])
    return ", ".join(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs)


def draw(frames: dict[str, np.ndarray], session: Session) -> np.ndarray:
    """The base photo with each lit LED circled and numbered, and a legend."""
    out = frames["base"].copy()
    scale = out.shape[1] / 1920
    font, legend = cv2.FONT_HERSHEY_SIMPLEX, []

    def text(t: str, org: tuple[int, int], colour: tuple[int, int, int], size: float):
        for thickness, ink in ((6, (0, 0, 0)), (2, colour)):
            cv2.putText(
                out, t, org, font, size * scale, ink, max(1, round(thickness * scale))
            )

    for glow in sorted(session.glows.values(), key=lambda g: g.pin):
        if not glow.lit or glow.photo_xy is None:
            continue
        x, y = (round(v) for v in glow.photo_xy)
        colour = LABEL_COLOURS.get(glow.colour or "", (255, 255, 255))
        radius = round(28 * scale)
        cv2.circle(out, (x, y), radius, colour, max(2, round(4 * scale)))
        text(
            str(glow.pin),
            (x - round(10 * scale), y - radius - round(8 * scale)),
            colour,
            1.0,
        )
        legend.append(
            (f"pin {glow.pin}: {glow.colour} LED, column {glow.column}", colour)
        )
    for i, (line, colour) in enumerate(legend):
        text(line, (round(20 * scale), round((50 + 45 * i) * scale)), colour, 1.1)
    return out


VERDICT_COLOURS = {  # RGB
    "ok": (40, 210, 80),
    "error": (240, 60, 50),
    "warning": (255, 170, 0),
    "uncertain": (255, 170, 0),
}


def draw_verdict(
    photo: np.ndarray, session: Session, verdict: Verdict, note: str | None = None
) -> np.ndarray:
    """The photo with each lit LED circled by verdict, and the answer in a panel.

    Green: that pin lit the LED the lab wants. Red: an LED lit on the wrong pin.
    Amber: something doubtful or unexpected. A dark LED cannot be circled; nobody
    knows where it is, unless a finding names its holes (from a described build):
    those holes are ringed in the finding's colour. note, if given, is the last
    line of the panel.
    """
    out = photo.copy()
    scale = out.shape[1] / 1920
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(t: str, org: tuple[int, int], colour: tuple[int, int, int], size: float):
        for thickness, ink in ((6, (0, 0, 0)), (2, colour)):
            cv2.putText(
                out, t, org, font, size * scale, ink, max(1, round(thickness * scale))
            )

    lines: list[tuple[str, tuple[int, int, int], float]] = []
    head = (
        "ok"
        if verdict.works
        else (
            "error"
            if any(f.severity == "error" for f in verdict.findings)
            else "uncertain"
        )
    )
    lines.append((verdict.summary, VERDICT_COLOURS[head], 1.25))
    width = out.shape[1] - round(60 * scale)
    for f in verdict.findings:
        colour = VERDICT_COLOURS.get(f.severity, (255, 255, 255))
        for i, part in enumerate(_wrap(f.message, width, 0.9 * scale)):
            lines.append((("- " if i == 0 else "  ") + part, colour, 0.9))
        if f.suggestion:
            for part in _wrap("-> " + f.suggestion, width, 0.8 * scale):
                lines.append(("    " + part, (230, 230, 230), 0.8))
    if verdict.caveat:
        for part in _wrap(verdict.caveat, width, 0.8 * scale):
            lines.append((part, (200, 200, 200), 0.8))
    if note:
        for part in _wrap(note, width, 0.8 * scale):
            lines.append((part, (120, 200, 255), 0.8))

    step = round(42 * scale)
    panel_h = step * len(lines) + round(30 * scale)
    shade = out[:panel_h].astype(np.float32) * 0.35
    out[:panel_h] = shade.astype(np.uint8)
    for i, (line, colour, size) in enumerate(lines):
        text(line, (round(24 * scale), round(20 * scale) + step * (i + 1)), colour, size)
    flagged: dict[int, str] = {}
    for f in verdict.findings:
        glow_pin = f.detail.get("glow")
        if isinstance(glow_pin, int):
            flagged.setdefault(glow_pin, f.severity)
    if session.rect is not None:
        for f in verdict.findings:
            colour = VERDICT_COLOURS.get(f.severity, (255, 255, 255))
            spots = []
            for hole in f.holes:
                try:  # holes as typed by hand: "D31" is d31
                    spots.append(
                        tuple(round(v) for v in session.rect.to_photo(hole.lower()))
                    )
                except KeyError:
                    continue
            for x, y in spots:
                ring = max(2, round(4 * scale))
                cv2.circle(out, (x, y), round(14 * scale), colour, ring)
            if spots:
                x, y = min(spots, key=lambda p: p[1])
                label = ", ".join(f.holes)
                text(label, (x - round(20 * scale), y - round(24 * scale)), colour, 0.7)
    for glow in session.glows.values():
        if not glow.lit or glow.photo_xy is None:
            continue
        mark = "ok" if glow.pin in verdict.ok_pins else flagged.get(glow.pin, "warning")
        colour = VERDICT_COLOURS[mark]
        x, y = (round(v) for v in glow.photo_xy)
        radius = round(30 * scale)
        cv2.circle(out, (x, y), radius, colour, max(3, round(6 * scale)))
        text(
            str(glow.pin),
            (x - round(10 * scale), y - radius - round(10 * scale)),
            colour,
            1.1,
        )

    return out


def _wrap(line: str, width_px: int, font_scale: float) -> list[str]:
    """Split text into lines that fit width_px at this font scale."""
    words, rows, current = line.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        if w > width_px and current:
            rows.append(current)
            current = word
        else:
            current = trial
    return [*rows, current] if current else rows


def print_verdict(verdict: Verdict) -> None:
    print(verdict.summary)
    for f in verdict.findings:
        print(f"  [{f.severity}] {f.message}")
        if f.holes:
            print(f"      at {', '.join(f.holes)}")
        if f.suggestion:
            print(f"      -> {f.suggestion}")
    if verdict.caveat:
        print(f"  ({verdict.caveat})")


def load_lab(path: Path) -> Netlist:
    try:
        return Netlist.from_json(json.loads(path.read_text()))
    except (OSError, ValueError, NetlistError, KeyError) as e:
        raise SystemExit(f"cannot read the lab file {path}: {e}") from e


def entered_note(build: Path) -> str:
    """The on-screen reminder that a person, not a model, described the build."""
    return f"Parts entered by hand ({build.name}). A trained model will do this step."


def judge(
    lab: Netlist, build: Path | None, session: Session, pins: Sequence[int]
) -> Verdict:
    """The verdict: blinking against the lab, or, given a description of the
    build, the fixes it confirms. The description is read afresh each time, so
    it can be edited between checks. One that cannot be read is said to be so."""
    if build is None:
        return verify(lab, session, pins)
    try:
        described = Netlist.from_json(json.loads(build.read_text()))
    except (OSError, ValueError, NetlistError, KeyError, TypeError, AttributeError) as e:
        # A hand-edited file can be malformed in any shape; say so, don't crash.
        return Verdict(
            False,
            "Could not read the entered parts.",
            (
                Finding(
                    "entry_unreadable",
                    f"{build}: {e}",
                    severity="uncertain",
                    suggestion="Fix the file and check again.",
                ),
            ),
        )
    return diagnose(described, lab, session, pins)


def main() -> int:
    ap = argparse.ArgumentParser(description="Light each LED in turn; find it on camera.")
    ap.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    ap.add_argument("--replay", type=Path, help="re-judge a saved run instead")
    ap.add_argument("--lab", type=Path, help="check the result against this lab file")
    ap.add_argument(
        "--build", type=Path, help="circuit file of the board as built (needs --lab)"
    )
    ap.add_argument("--out", type=Path, default=Path("out/blink"))
    args = ap.parse_args()
    if args.build and not args.lab:
        ap.error("--build needs --lab")
    lab = load_lab(args.lab) if args.lab else None

    try:
        run = args.replay or record(args.camera, OUTPUT_PINS)
    except OSError as e:
        print(f"Could not run: {e}")
        return 1
    frames, pins, shorted = load(run)
    session = analyse(frames, pins, shorted)
    print("\n".join(report(session, pins)))

    verdict = judge(lab, args.build, session, pins) if lab is not None else None
    if verdict is not None:
        note = entered_note(args.build) if args.build else None
        print()
        print_verdict(verdict)
        if note:
            print(f"({note})")
        picture_rgb = draw_verdict(frames["base"], session, verdict, note)
    else:
        picture_rgb = draw(frames, session)

    args.out.mkdir(parents=True, exist_ok=True)
    picture = args.out / f"{run.name}.jpg"
    cv2.imwrite(str(picture), cv2.cvtColor(picture_rgb, cv2.COLOR_RGB2BGR))
    print(f"\nframes   {run}\npicture  {picture}")
    if verdict is not None:
        return 0 if verdict.works else 1
    return 0 if session.rect is not None else 1


if __name__ == "__main__":
    sys.exit(main())
