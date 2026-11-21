#!/bin/sh
# Run the vc4 backend tests.
#
# Two board-level quirks are handled here, not in the tests:
#   * vc4d (this build) does not promptly reclaim the arena region of a
#     client that exits, so a *second* GPU process may be refused with
#     "out of arena memory" even though the first one is gone;
#   * the board is shared, so another client can hold the whole arena or be
#     killed mid-launch (which wedges the V3D until the next reboot).
# Each test therefore gets a fresh daemon and is retried when (and only when)
# it failed for one of those two environmental reasons.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || { echo "no venv at $PWD/$PY"; exit 1; }

run() {
    name="$1"
    shift
    attempt=1
    while [ "$attempt" -le 5 ]; do
        sudo systemctl restart "${VC4D_UNIT:-vc4d}"
        sleep 1
        log="/tmp/torch_vc4_${name}.log"
        "$PY" "$@" >"$log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "=== $name: PASS"
            return 0
        fi
        if grep -qE "out of arena memory|out of arena/slots|arena exhausted|GPU timeout|daemon is dead|device I/O error|denied by daemon policy|context is dead" "$log"; then
            echo "=== $name: environmental failure (attempt $attempt), retrying"
            attempt=$((attempt + 1))
            sleep 3
            continue
        fi
        echo "=== $name: FAIL (see $log)"
        tail -25 "$log"
        return 1
    done
    echo "=== $name: FAIL after $((attempt - 1)) environmental retries"
    return 1
}

fails=0
# Device-free checks: registration, broadcast and reduction plans, the GELU
# derivative.  No vc4 context is opened, so this runs before the restarts.
cpu_log="${TMPDIR:-/tmp}/torch_vc4_cpu.log"
if "$PY" tests/test_cpu_registration.py >"$cpu_log" 2>&1; then
    echo "=== cpu: PASS"
else
    echo "=== cpu: FAIL (see $cpu_log)"
    tail -25 "$cpu_log"
    fails=$((fails + 1))
fi
run ops tests/test_ops.py || fails=$((fails + 1))
run complete tests/test_complete.py || fails=$((fails + 1))
run autograd tests/test_autograd.py || fails=$((fails + 1))
run strict tests/test_backend.py || fails=$((fails + 1))

if [ "$fails" -ne 0 ]; then
    echo "=== $fails suite(s) failed"
    exit 1
fi
echo "=== all suites passed"
