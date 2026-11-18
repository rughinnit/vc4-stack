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

make -j4

echo "=== maths (no GPU)"
make check-math

echo "=== test_dispatch (no GPU)"
./build/test_dispatch | tail -5

for t in test_ew test_matmul test_conv; do
    echo "=== $t  (reps=$REPS)"
    timeout -k 5 900 sudo -n env VC4INFER_TMO_MS=20000 ./build/$t "$REPS"
    echo "--- board:"
    sudo -n "$STATUS" | grep -E "STATE|SRQCS|V3D_ERRORS"
    sudo -n "$STATUS" | grep -q "STATE: LIVE" || {
        echo "!!! board is not LIVE - power-cycle it before running again"
        exit 1
    }
done
echo "ALL TESTS PASSED"
