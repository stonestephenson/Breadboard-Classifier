"""Command line entry point, so the checker can be used without writing Python.

python -m breadboard show  examples/basicboard.json
python -m breadboard check examples/basicboard_led_moved.json \
    --lab examples/basicboard.json
python -m breadboard check board.json          # sanity rules only, no lab
python -m breadboard rectify photo.jpg -o out/   # find the board in a photo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from breadboard.check import Finding, check
from breadboard.graph import build
from breadboard.netlist import Netlist, NetlistError

MARK = {"error": "x", "uncertain": "?", "warning": "!"}


def _load(path: str) -> Netlist:
    try:
        return Netlist.from_json(json.loads(Path(path).read_text()))
    except FileNotFoundError:
        sys.exit(f"no such file: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON: {e}")
    except (NetlistError, KeyError) as e:
        sys.exit(f"{path} is not a valid circuit: {e}")


def _render(f: Finding) -> str:
    lines = [
        f"{MARK.get(f.severity, '-')} {f.kind}"
        + (f"   [{f.scope}]" if f.scope != "lab" else "")
    ]
    lines.append(f"  {f.message}")
    if f.holes:
        lines.append(f"  at {', '.join(f.holes)}")
    if f.suggestion:
        lines.append(f"  -> {f.suggestion}")
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace) -> int:
    n = _load(args.netlist)
    g = build(n)
    collapsed = g.collapsed()
    print(f"name        {n.name or '(unnamed)'}")
    print(f"board       {n.board}")
    print(f"components  {len(n.components)}")
    print(f"nodes used  {len(n.nodes_used())}")
    print(
        f"topology    {len(g.edges)} connections, "
        f"{len(collapsed.edges)} after collapsing series chains"
    )
    print()
    for c in sorted(n.components, key=lambda c: (c.type, c.id)):
        where = " ".join(f"{k}={v.hole}" for k, v in sorted(c.pins.items()))
        attrs = " ".join(f"{k}={v}" for k, v in sorted(c.attrs.items()))
        flag = "" if c.origin == "student" else f" ({c.origin})"
        unsure = "" if all(p.confidence >= 1.0 for p in c.pins.values()) else "  ~unsure"
        print(f"  {c.id:<14}{c.type:<10}{where}  {attrs}{flag}{unsure}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    student = _load(args.netlist)
    reference = _load(args.lab) if args.lab else None
    findings = check(student, reference)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "kind": f.kind,
                        "severity": f.severity,
                        "message": f.message,
                        "components": list(f.components),
                        "holes": list(f.holes),
                        "suggestion": f.suggestion,
                        "scope": f.scope,
                        "detail": f.detail,
                    }
                    for f in findings
                ],
                indent=2,
            )
        )
        return 1 if findings else 0

    if not findings:
        if reference is None:
            print(
                "No problems found. (No lab given, so only general rules ran --"
                " pass --lab to compare against the intended circuit.)"
            )
        else:
            print(f"This circuit matches '{reference.name or args.lab}'. Nothing to fix.")
        return 0

    for f in findings:
        print(_render(f))
        print()
    if any(f.severity == "uncertain" for f in findings):
        print("Not reporting an error: a clearer photo of that area may resolve it.")
    return 1


def cmd_rectify(args: argparse.Namespace) -> int:
    # Imported here so the checker commands do not pay for loading the vision stack.
    import cv2

    from breadboard.rectify import MIN_COLUMN_MARGIN, draw_holes, load_photo, rectify

    try:
        photo = load_photo(args.photo)
    except (FileNotFoundError, OSError) as e:
        sys.exit(f"cannot read {args.photo}: {e}")
    rect = rectify(photo)
    if rect is None:
        print("No board found in this photo.")
        return 1
    grade = {1.0: "good", 0.75: "imperfect"}.get(rect.confidence, "rejected")
    print(f"hole grid    {grade} (confidence {rect.confidence:.2f})")
    print(
        "orientation  "
        + ("confirmed by the rail stripes" if rect.oriented else "NOT confirmed")
    )
    print(f"column fit   margin {rect.column_margin:+.3f} (needs {MIN_COLUMN_MARGIN})")
    print(
        "verdict      "
        + ("usable" if rect.ok else "NOT usable -- ask for a better photo")
    )
    for hole in ("a1", "j1", "a63", "j63"):
        x, y = rect.to_photo(hole)
        print(f"  {hole:<4} at photo pixel ({x:.0f}, {y:.0f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.photo).stem
    overlay, canonical = out / f"{stem}_holes.jpg", out / f"{stem}_rectified.jpg"
    cv2.imwrite(str(overlay), cv2.cvtColor(draw_holes(photo, rect), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(canonical), cv2.cvtColor(rect.canonical, cv2.COLOR_RGB2BGR))
    print(f"wrote {overlay}\nwrote {canonical}")
    return 0 if rect.ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m breadboard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="summarise a circuit file")
    s.add_argument("netlist")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("check", help="diagnose a circuit, optionally against a lab")
    c.add_argument("netlist")
    c.add_argument("--lab", help="the lab's intended circuit to compare against")
    c.add_argument("--json", action="store_true", help="machine-readable output")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("rectify", help="find the board in a photo and name its holes")
    r.add_argument("photo")
    r.add_argument("-o", "--out", default="out", help="directory for the images")
    r.set_defaults(func=cmd_rectify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
