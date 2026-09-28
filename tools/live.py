"""Live view: the rectifier running on a webcam, so you can watch it find the board.

    ./venv/bin/python tools/live.py              # the default camera
    ./venv/bin/python tools/live.py --camera 1   # another camera, e.g. an iPhone
    ./venv/bin/python tools/live.py --list       # which camera numbers work

Point the camera so the whole board is in view. Every hole is drawn where the
rectifier thinks it is. Green means the fit is usable; orange means it is not,
with the reason across the top. The top-down view sits underneath.

Keys: q quits, s saves the current frame, its overlay and its top-down view to
out/live/.

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
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from breadboard.rectify import (
    MIN_COLUMN_MARGIN,
    Rectification,
    draw_holes,
    rectify,
)

GOOD = (40, 200, 40)
BAD = (255, 140, 0)
VIEW_WIDTH = 1280  # on-screen width of each panel
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
    w = out.shape[1]
    scale = w / 1600
    bar = round(60 * scale)
    cv2.rectangle(out, (0, 0), (w, bar), GOOD if ok else BAD, -1)
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


def list_cameras() -> int:
    found = False
    for index in range(6):
        cap = open_camera(index)
        ok, frame = cap.read()
        cap.release()
        if ok:
            found = True
            print(f"camera {index}: {frame.shape[1]} x {frame.shape[0]}")
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


def run(index: int, out: Path) -> int:
    cap = open_camera(index)
    if not cap.isOpened():
        print(f"Could not open camera {index}. {_PERMISSION_HINT}")
        return 1
    print("Loading the corner model (the first run downloads it, 83 MB)...")
    rectify(np.full((480, 640, 3), 255, np.uint8))  # load it before the window opens
    fitter = Fitter()
    fitter.start()
    title = "breadboard live - q quits, s saves"
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("The camera stopped sending frames.")
                return 1
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fitter.submit(frame)
            rect = fitter.latest[1] if fitter.latest else None
            cv2.imshow(title, cv2.cvtColor(screen(frame, rect), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s") and fitter.latest:
                save(*fitter.latest, out)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch the rectifier find the board live.")
    ap.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    ap.add_argument("--list", action="store_true", help="list working camera numbers")
    ap.add_argument("--out", default="out/live", help="where s saves (default out/live)")
    args = ap.parse_args()
    if args.list:
        return list_cameras()
    return run(args.camera, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
