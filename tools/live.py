"""Live view: the rectifier running on a webcam, so you can watch it find the board.

    ./venv/bin/python tools/live.py              # the default camera
    ./venv/bin/python tools/live.py --camera 1   # another camera, e.g. an iPhone
    ./venv/bin/python tools/live.py --list       # which camera numbers work
    ./venv/bin/python tools/live.py --camera 1 --lab examples/basicboard_rewired.json

Point the camera so the whole board is in view. Every hole is drawn where the
rectifier thinks it is. Green means the fit is usable; orange means it is not,
with the reason across the top. The top-down view sits underneath.

Keys: q quits, s saves the current frame, its overlay and its top-down view to
out/live/.

With --lab and the board on USB, c checks the wiring. It blinks each LED in turn
while you watch (tools/blink.py), then shows what works and what to fix, drawn on
the picture. Any key returns to the live view; c checks again.

The rectifier takes about 0.2 s a frame, so it runs in a background thread on
the newest frame while the video keeps playing. If the board moves, the overlay
trails the video by a frame or two.

On a Mac, the first run asks for camera access for your terminal app. If you
declined, allow it in System Settings > Privacy & Security > Camera. An iPhone
near the Mac shows up as another camera (Continuity Camera). It gives a much
sharper image, and it is easier to hold above the board than a laptop.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from breadboard.rectify import (
    MIN_COLUMN_MARGIN,
    Rectification,
    draw_holes,
    rectify,
)

if TYPE_CHECKING:
    from probe import Board

    from breadboard.netlist import Netlist

GOOD = (40, 200, 40)
BAD = (255, 140, 0)
BUSY = (70, 150, 255)
VIEW_WIDTH = 1280  # on-screen width of each panel
# An iPhone via Continuity Camera takes about 5 s to send its first real frame.
WARMUP_SECONDS = 10.0
# Tolerate a camera hiccup this long before giving up.
STALL_SECONDS = 3.0
CORNER_HOLES = ("a1", "j1", "a63", "j63")


def verdict(rect: Rectification | None) -> tuple[bool, str]:
    """Whether the fit is usable, and what to tell the person holding the camera."""
    if rect is None:
        return False, "No board in view - get the whole board in the picture"
    if rect.confidence == 0.0:
        return False, "Holes don't line up with the grid - hold steady, fill the view"
    if not rect.oriented:
        return False, "Can't read the rail stripes - which end is column 1?"
    if rect.column_margin < MIN_COLUMN_MARGIN:
        return False, "Can't pin down the columns - get both ends of the board in view"
    grade = "" if rect.confidence == 1.0 else " (imperfect fit)"
    return True, f"Board found: all 830 holes located{grade}"


def annotate(frame: np.ndarray, rect: Rectification | None) -> np.ndarray:
    """The camera frame (RGB) with holes marked and the verdict across the top."""
    out = draw_holes(frame, rect, CORNER_HOLES) if rect is not None else frame.copy()
    ok, message = verdict(rect)
    return banner(out, message, GOOD if ok else BAD)


def banner(image: np.ndarray, message: str, colour: tuple[int, int, int]) -> np.ndarray:
    """The image with a coloured bar across the top carrying a message."""
    out = image.copy()
    w = out.shape[1]
    scale = w / 1600
    cv2.rectangle(out, (0, 0), (w, round(60 * scale)), colour, -1)
    cv2.putText(
        out,
        message,
        (round(16 * scale), round(42 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2 * scale,
        (0, 0, 0),
        max(1, round(2 * scale)),
    )
    return out


def screen(frame: np.ndarray, rect: Rectification | None) -> np.ndarray:
    """What goes in the window: the annotated frame above the top-down view."""
    top = _fit_width(annotate(frame, rect), VIEW_WIDTH)
    if rect is not None:
        bottom = _fit_width(rect.canonical, VIEW_WIDTH)
    else:
        bottom = np.zeros((VIEW_WIDTH * 340 // 1024, VIEW_WIDTH, 3), np.uint8)
    return np.vstack([top, bottom])


def _fit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


class Fitter(threading.Thread):
    """Rectifies the newest submitted frame in the background; older ones are dropped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._pending: np.ndarray | None = None
        self.latest: tuple[np.ndarray, Rectification | None] | None = None

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame
        self._ready.set()

    def run(self) -> None:
        while True:
            self._ready.wait()
            with self._lock:
                frame, self._pending = self._pending, None
                self._ready.clear()
            if frame is not None:
                self.latest = (frame, rectify(frame))


def open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def first_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """The first real frame, waiting out a slow start. None if none arrives."""
    deadline = time.monotonic() + WARMUP_SECONDS
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.any():
            return frame
        time.sleep(0.1)
    return None


def list_cameras() -> int:
    found = False
    for index in range(6):
        cap = open_camera(index)
        if not cap.isOpened():
            break  # cameras are numbered from 0 with no gaps
        start = time.monotonic()
        frame = first_frame(cap)
        cap.release()
        if frame is None:
            print(f"camera {index}: opened, but sent no picture")
            continue
        found = True
        waited = time.monotonic() - start
        slow = (
            f"  (took {waited:.0f} s to start: probably an iPhone)" if waited > 2 else ""
        )
        print(f"camera {index}: {frame.shape[1]} x {frame.shape[0]}{slow}")
    if not found:
        print("No camera could be opened. " + _PERMISSION_HINT)
    return 0 if found else 1


_PERMISSION_HINT = (
    "On a Mac, allow camera access for your terminal app in System Settings > "
    "Privacy & Security > Camera, then run again."
)


def save(frame: np.ndarray, rect: Rectification | None, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = out / time.strftime("%Y%m%d-%H%M%S")
    images = {"frame": frame, "holes": annotate(frame, rect)}
    if rect is not None:
        images["rectified"] = rect.canonical
    for name, image in images.items():
        cv2.imwrite(f"{stem}_{name}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"saved {stem}_*.jpg")


class _Watched:
    """Passes pin commands to the board, remembering which pin is on for display."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self.pin: int | None = None

    def digital_write(self, pin: int, high: bool) -> None:
        self.board.digital_write(pin, high)
        self.pin = pin if high else None

    def pin_level(self, pin: int) -> int:
        return self.board.pin_level(pin)


def check_now(
    cap: cv2.VideoCapture, board: Board, lab: Netlist, title: str
) -> np.ndarray:
    """Blink and watch through the live window. Returns the verdict picture, RGB."""
    # Imported here: blink.py imports this module's camera helpers.
    from blink import SAMPLES, SETTLE_S, draw_verdict, print_verdict

    from breadboard.blink import OUTPUT_PINS, analyse, run_sequence
    from breadboard.verify import verify

    watched = _Watched(board)

    def show(bgr: np.ndarray, message: str) -> None:
        rgb = _fit_width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), VIEW_WIDTH)
        cv2.imshow(title, cv2.cvtColor(banner(rgb, message, BUSY), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def read() -> np.ndarray | None:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            time.sleep(0.01)
            return None
        on = f"pin {watched.pin} on" if watched.pin else "all pins off"
        show(bgr, f"Checking the wiring: {on}")
        return bgr

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            read()

    def capture() -> np.ndarray:
        frames: list[np.ndarray] = []
        deadline = time.monotonic() + 5.0
        while len(frames) < SAMPLES:
            if time.monotonic() > deadline:
                raise OSError("the camera stopped sending frames")
            bgr = read()
            if bgr is not None:
                frames.append(bgr.astype(np.float32))
        mean = np.mean(frames, axis=0).astype(np.uint8)
        return cv2.cvtColor(mean, cv2.COLOR_BGR2RGB)

    snapshot = board.snapshot()
    try:
        frames, shorted = run_sequence(watched, OUTPUT_PINS, capture, SETTLE_S, pump)
    finally:
        board.restore(snapshot)
    show(cv2.cvtColor(frames["base"], cv2.COLOR_RGB2BGR), "Working it out...")
    session = analyse(frames, OUTPUT_PINS, shorted)
    result = verify(lab, session, OUTPUT_PINS)
    print_verdict(result)
    return draw_verdict(frames["base"], session, result)


def _connect() -> Board:
    """Open the board on USB. Raises OSError, with a plain reason, if it cannot."""
    from probe import Board, autodetect

    port = autodetect()
    if not port:
        raise OSError("no board found on USB. Plug it in and press c again")
    try:
        return Board(port)
    except OSError as e:
        raise OSError(
            f"{port} is busy or unavailable ({e}). Close anything else using the "
            "board, such as the LbyM web app, and press c again"
        ) from e


def _exit_on_signal(signum: int, _frame: object) -> None:
    # Closing the terminal or `kill` would otherwise skip the cleanup that
    # switches pins off.
    raise SystemExit(128 + signum)


def run(index: int, out: Path, lab: Netlist | None = None) -> int:
    cap = open_camera(index)
    if not cap.isOpened():
        print(f"Could not open camera {index}. {_PERMISSION_HINT}")
        return 1
    # The board is connected on the first c, not here: opening the port resets
    # the Arduino, which is pointless if nobody checks.
    board: Board | None = None
    if lab is not None:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _exit_on_signal)
    print("Loading the corner model (the first run downloads it, 83 MB)...")
    rectify(np.full((480, 640, 3), 255, np.uint8))  # load it before the window opens
    print(f"Waiting for camera {index} (an iPhone takes a few seconds)...")
    if first_frame(cap) is None:
        print(f"Camera {index} opened but sent no picture. Is it awake and nearby?")
        return 1
    fitter = Fitter()
    fitter.start()
    title = "breadboard live - q quits, s saves" + (
        ", c checks the wiring" if lab else ""
    )
    last_frame = time.monotonic()
    result: np.ndarray | None = None  # the verdict picture, while it is on screen
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                if time.monotonic() - last_frame > STALL_SECONDS:
                    print("The camera stopped sending frames.")
                    return 1
                time.sleep(0.02)
                continue
            last_frame = time.monotonic()
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if result is None:
                fitter.submit(frame)
                rect = fitter.latest[1] if fitter.latest else None
                view = screen(frame, rect)
            else:
                view = _fit_width(result, VIEW_WIDTH)
            cv2.imshow(title, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("c") and lab is not None:
                try:
                    if board is None:
                        board = _connect()
                    result = check_now(cap, board, lab, title)
                except (OSError, cv2.error) as e:
                    print(f"Could not check: {e}")
                    if board is not None and isinstance(e, OSError):
                        print(
                            "If an LED stayed on, unplug and replug the USB cable "
                            "to reset the board."
                        )
                    result = None
            elif key == ord("s"):
                if result is not None:
                    out.mkdir(parents=True, exist_ok=True)
                    path = out / time.strftime("%Y%m%d-%H%M%S_verdict.jpg")
                    cv2.imwrite(str(path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    print(f"saved {path}")
                elif fitter.latest:
                    save(*fitter.latest, out)
            elif key != 255 and result is not None:
                result = None
    finally:
        if board is not None:
            board.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch the rectifier find the board live.")
    ap.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    ap.add_argument("--list", action="store_true", help="list working camera numbers")
    ap.add_argument("--out", default="out/live", help="where s saves (default out/live)")
    ap.add_argument("--lab", type=Path, help="lab file; c then checks the wiring")
    args = ap.parse_args()
    if args.list:
        return list_cameras()
    lab = None
    if args.lab:
        from blink import load_lab

        lab = load_lab(args.lab)
    return run(args.camera, Path(args.out), lab)


if __name__ == "__main__":
    sys.exit(main())
