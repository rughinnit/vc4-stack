#!/bin/sh
# run_tests.sh - the whole verification suite, serialised and health-checked.
#
# Every test takes /tmp/vc4-gpu.lock itself (flock) and every GPU run is
# followed by a board state check.  A stuck QPU cannot be reset from the ARM:
# if STATE is not LIVE, stop and power-cycle the board.
#
#   REPS=5 ./run_tests.sh
#
# Unprivileged through the vc4d daemon:
#
#   REPS=5 VC4INFER_DAEMON=1 ./run_tests.sh
#
# With the flag the GPU tests run as the invoking user (no sudo): vc4d owns
# /dev/mem and the V3D registers, so only the daemon needs privilege.  The
# daemon takes /tmp/vc4-gpu.lock for each launch, which is why the tests do
# not hold it themselves in that mode (tests/util.h).
set -e
cd "$(dirname "$0")"
REPS=${REPS:-5}
STATUS=${STATUS:-/home/raspi/vc4cl/tools/read_v3d_status}

DAEMON=${VC4INFER_DAEMON:-0}
case "$DAEMON" in
    0|"")
        SUDO="sudo -n"; DMENV=""; BACKEND="direct root (/dev/mem)";;
    *)
        SUDO=""; DMENV="VC4INFER_DAEMON=$DAEMON"; BACKEND="vc4d (unprivileged)";;
esac
RUN="$SUDO env $DMENV VC4INFER_TMO_MS=20000"

clocks() {
    echo "--- clocks: $(vcgencmd measure_clock core) $(vcgencmd measure_clock v3d) $(vcgencmd get_throttled)"
}

board() {
    echo "--- board:"
    sudo -n "$STATUS" | grep -E "STATE|SRQCS|V3D_ERRORS"
    sudo -n "$STATUS" | grep -q "STATE: LIVE" || {
        echo "!!! board is not LIVE - power-cycle it before running again"
        exit 1
    }
}

echo "=== backend: $BACKEND"
clocks

make -j4

echo "=== maths (no GPU)"
make check-math

echo "=== test_dispatch (no GPU)"
./build/test_dispatch

# Evidence for the security claim: a real suite client, with a live socket and
# an arena mapping, has no /dev/mem and no capabilities.
if [ "$DAEMON" != 0 ]; then
    echo "=== unprivileged client evidence"
    ./build/test_matmul 1 >/tmp/vc4-client-evidence.log 2>&1 &
    pid=$!
    alive=0
    for i in $(seq 1 100); do
        if [ -e /proc/$pid/status ]; then alive=1; break; fi
    done
    sleep 0.5
    if [ "$alive" = 1 ] && [ -e /proc/$pid/status ]; then
        echo "  client pid=$pid"
        grep -E '^(Uid|Gid|CapEff|NoNewPrivs):' /proc/$pid/status | sed 's/^/  /'
        echo "  open fds:"
        ls -l /proc/$pid/fd | sed 's/^/    /'
        if ls -l /proc/$pid/fd | grep -q '/dev/mem'; then
            echo "  FAIL: client has /dev/mem open"
            kill $pid 2>/dev/null || true
            exit 1
        fi
        cap=$(awk '/^CapEff:/{print $2}' /proc/$pid/status)
        uid=$(awk '/^Uid:/{print $2}' /proc/$pid/status)
        [ "$cap" = "0000000000000000" ] || {
            echo "  FAIL: CapEff=$cap, expected 0"; kill $pid 2>/dev/null || true; exit 1; }
        [ "$uid" = "1000" ] || {
            echo "  FAIL: uid=$uid, expected 1000"; kill $pid 2>/dev/null || true; exit 1; }
        echo "  PASS: uid 1000, CapEff=0, no /dev/mem"
    else
        echo "  FAIL: evidence client exited before it could be inspected"
        exit 1
    fi
    wait $pid
    tail -2 /tmp/vc4-client-evidence.log | sed 's/^/  /'
fi

for t in test_ew test_matmul test_conv; do
    echo "=== $t  (reps=$REPS)"
    timeout -k 5 900 $RUN ./build/$t "$REPS"
    board
done

echo "=== test_train_ops (training primitives)"
timeout -k 5 300 $RUN ./build/test_train_ops
board

echo "=== test_grad (finite differences)"
timeout -k 5 600 $RUN ./build/test_grad 1
board

echo "=== test_convgrad (conv dX/dW, pool; finite differences)"
timeout -k 5 900 $RUN ./build/test_convgrad 1
board

for o in sgd adam; do
    echo "=== test_train --opt $o (300 steps)"
    timeout -k 5 600 $RUN ./build/test_train --opt $o --steps 300
    board
done

for o in sgd adam; do
    echo "=== test_convnet --opt $o (150 steps)"
    timeout -k 5 900 $RUN ./build/test_convnet --opt $o --steps 150
    board
done

clocks
echo "ALL TESTS PASSED"
