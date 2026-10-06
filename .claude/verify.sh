#!/usr/bin/env bash
# Done-gate: everything here must pass before a change is finished.
# Takes about two and a half minutes, nearly all of it the tests. They exercise
# the rectifier too, on the photos in data/cache where those are present.
# Run: ./.claude/verify.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

VENV=./venv/bin
fail=0
run() {
  local name="$1"; shift
  printf '%-10s' "$name"
  if out=$("$@" 2>&1); then
    echo "ok"
  else
    echo "FAIL"
    echo "$out" | sed 's/^/  /'
    fail=1
  fi
}

run lint    "$VENV/ruff" check breadboard tests tools spikes examples
run format  "$VENV/ruff" format --check breadboard tests tools spikes examples
run types   pyright --outputjson
run tests   "$VENV/python" -m pytest tests/ -q

[ "$fail" -eq 0 ] && echo "PASS" || echo "FAILED"
exit "$fail"
