#!/usr/bin/env bash
# Done-gate: everything here must pass before a change is finished.
# Fast by design (~5s) so it can run on every edit. Run: ./.claude/verify.sh
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

# The spikes are exploratory, but they must at least still execute -- they are
# the only thing exercising the rectifier and it is easy to break silently.
if compgen -G "data/cache/sample/*.jpg" > /dev/null; then
  run rectify "$VENV/python" spikes/lattice_fit.py data/cache/sample/IMG_3220.jpg
fi

[ "$fail" -eq 0 ] && echo "PASS" || echo "FAILED"
exit "$fail"
