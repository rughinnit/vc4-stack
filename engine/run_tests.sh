#!/bin/sh
# run_tests.sh - the whole verification suite, serialised and health-checked.
#
# Every test takes /tmp/vc4-gpu.lock itself (flock) and every GPU run is
# followed by a board state check.  A stuck QPU cannot be reset from the ARM:
# if STATE is not LIVE, stop and power-cycle the board.
#
#   REPS=5 ./run_tests.sh
set -e
cd "$(dirname "$0")"
REPS=${REPS:-5}
STATUS=${STATUS:-/home/raspi/vc4cl/tools/read_v3d_status}

board() {
    echo "--- board:"
    sudo -n "$STATUS" | grep -E "STATE|SRQCS|V3D_ERRORS"
    sudo -n "$STATUS" | grep -q "STATE: LIVE" || {
        echo "!!! board is not LIVE - power-cycle it before running again"
        exit 1
    }
}

make -j4

echo "=== maths (no GPU)"
make check-math

echo "=== test_dispatch (no GPU)"
./build/test_dispatch

for t in test_ew test_matmul test_conv; do
    echo "=== $t  (reps=$REPS)"
    timeout -k 5 900 sudo -n env VC4INFER_TMO_MS=20000 ./build/$t "$REPS"
    board
done

echo "=== test_train_ops (training primitives)"
timeout -k 5 300 sudo -n env VC4INFER_TMO_MS=20000 ./build/test_train_ops
board

echo "=== test_grad (finite differences)"
timeout -k 5 600 sudo -n env VC4INFER_TMO_MS=20000 ./build/test_grad 1
board

echo "=== test_convgrad (conv dX/dW, pool; finite differences)"
timeout -k 5 900 sudo -n env VC4INFER_TMO_MS=20000 ./build/test_convgrad 1
board

for o in sgd adam; do
    echo "=== test_train --opt $o (300 steps)"
    timeout -k 5 600 sudo -n env VC4INFER_TMO_MS=20000 ./build/test_train --opt $o --steps 300
    board
done

for o in sgd adam; do
    echo "=== test_convnet --opt $o (150 steps)"
    timeout -k 5 900 sudo -n env VC4INFER_TMO_MS=20000 ./build/test_convnet --opt $o --steps 150
    board
done

echo "ALL TESTS PASSED"
