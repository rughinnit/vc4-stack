#!/bin/sh
# The wheel ships a copy of the kernel generator; they must stay identical.
python3 "$(dirname "$0")/../tools/check_kgen_sync.py" || exit 1
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
    echo "--- clocks: $(vcgencmd measure_clock core) $(vcgencmd measure_clock v3d) $(vcgencmd get_throttled) $(vcgencmd measure_temp)"
}

board() {
    echo "--- board: $(vcgencmd measure_temp)"
    sudo -n "$STATUS" | grep -E "STATE|SRQCS|V3D_ERRORS"
    sudo -n "$STATUS" | grep -q "STATE: LIVE" || {
        echo "!!! board is not LIVE - power-cycle it before running again"
        exit 1
    }
}

# --- thermal -----------------------------------------------------------------
# The fan was removed on 2026-11-20: the board idles at ~56 C and a sustained
# QPU load adds ~15 C in two minutes, after which the firmware cuts the clock
# (docs/CLOCK-VARIANCE.md).  So every section starts from the same thermal
# state, and each section prints the temperature it ran at.  TEMP_START_LIMIT
# is the "cool enough to start" threshold; COOL_MAX_S bounds the wait.
TEMP_START_LIMIT=${TEMP_START_LIMIT:-66}
COOL_MAX_S=${COOL_MAX_S:-300}

temp_c() { vcgencmd measure_temp 2>/dev/null | sed "s/^temp=//; s/.C$//"; }

cool() {
    _t=$(temp_c); _n=0
    while [ -n "$_t" ] && [ "${_t%%.*}" -ge "$TEMP_START_LIMIT" ] && \
          [ "$_n" -lt "$COOL_MAX_S" ]; do
        sleep 2; _n=$((_n + 2)); _t=$(temp_c)
    done
    echo "--- cool: $(temp_c) C after ${_n}s (start limit ${TEMP_START_LIMIT} C)"
}

echo "=== backend: $BACKEND"
clocks

# Build against this tree's daemon client library and headers, not a copy.
: "${VC4D_DIR:=../vc4d}"
VC4D_DIR=$(cd "$VC4D_DIR" && pwd)
export VC4D_DIR

make -j4

echo "=== maths (no GPU)"
make check-math

echo "=== test_dispatch (no GPU)"
./build/test_dispatch

# --- daemon regressions -----------------------------------------------------
# G1/G9/G11 need a daemon they can provoke (timeout, departed client, lock
# contention) without disturbing the one everything else talks to, so they run
# against a private vc4d on its own socket.  --gpu-lock makes the two daemons
# (and any direct-mode client) take turns on the QPUs.
echo "=== test_daemon_recover (G1: a launch timeout must not kill the daemon)"
echo "=== test_daemon_arena   (G9: the arena comes back with the client)"
TMPD=${TMPDIR:-$HOME/scratch/tmp}
mkdir -p "$TMPD"
TSOCK=/run/vc4d-test-$$.sock
sudo -n rm -f "$TSOCK"
sudo -n "$VC4D_DIR/build/vc4d" --socket "$TSOCK" --arena-mb 16 \
    --group vc4 --drop-user nobody --gpu-lock --max-timeout-ms 5000 \
    </dev/null >"$TMPD/vc4d-test-$$.log" 2>&1 &
TPID=$!
trap 'sudo -n kill $TPID 2>/dev/null; sudo -n rm -f "$TSOCK"' EXIT
for i in $(seq 1 100); do [ -S "$TSOCK" ] && break; sleep 0.05; done
[ -S "$TSOCK" ] || { echo "FAIL: private vc4d did not start"; tail -5 "$TMPD/vc4d-test-$$.log"; exit 1; }
./build/test_daemon_recover "$TSOCK"
./build/test_daemon_arena "$TSOCK"
./build/test_daemon_reap "$TSOCK" 3
board
cool

echo "=== test_direct_lock (G11: direct mode and vc4d share the GPU lock)"
sudo -n ./build/test_direct_lock "$TSOCK"
board
cool

# G3/G4: the daemon side of both fixes, against the tree's own vc4d.
echo "=== test_prog_lru (G3: 250 programs, LRU eviction, re-run the first)"
"$VC4D_DIR/build/test_prog_lru" "$TSOCK"
echo "=== test_uni_mask (G4: a uniform word that looks like an address)"
"$VC4D_DIR/build/test_uni_mask" "$TSOCK"
echo "=== test_daemon_kernels (G3: 200 kernels through the engine)"
env VC4D_SOCKET="$TSOCK" VC4INFER_DAEMON=1 ./build/test_daemon_kernels 200
echo "=== test_daemon_uni (G4: Clip/LeakyRelu/Affine <= -2.0)"
env VC4D_SOCKET="$TSOCK" VC4INFER_DAEMON=1 ./build/test_daemon_uni
echo "=== test_daemon_values (G4: Pow/Clip/LeakyRelu through the runner)"
env VC4D_SOCKET="$TSOCK" VC4INFER_DAEMON=1 VC4INFER_LIB="$PWD/build/libvc4infer.so" \
    PYTHONPATH="$PYTHONPATH:." timeout -k 5 600 python3 tests/test_daemon_values.py
board
cool
sudo -n kill $TPID 2>/dev/null
trap - EXIT
sudo -n rm -f "$TSOCK"

# G2: needs its own private daemons (it restarts one on purpose) and CMA that
# it can hold and release, so it runs on its own sockets.  On a shared board it
# reports SKIP when another process is holding CMA - that is not a failure.
echo "=== test_daemon_g2 (G2: retry, loud degrade, restore without a reboot)"
./tests/test_daemon_g2.sh "${G2_MB:-16}"

echo "=== test_pool_sizing (G10: pool sized from the model, no 8 MiB floor)"
timeout -k 5 300 $RUN PYTHONPATH=python TMPDIR="$TMPD" \
    python3 tests/test_pool_sizing.py

# Evidence for the security claim: a real suite client, with a live socket and
# an arena mapping, has no /dev/mem and no capabilities.
if [ "$DAEMON" != 0 ]; then
    echo "=== unprivileged client evidence"
    # Long enough to still be running when inspected: one 256^3 matmul is
    # ~4 ms, so a short run exits before the evidence can be read (that made
    # the suite report a failure whenever the kernel cache was warm).
    ./build/test_matmul 300 >/tmp/vc4-client-evidence.log 2>&1 &
    pid=$!
    alive=0
    # Wait for the client to open its daemon socket, not for a fixed time.
    for i in $(seq 1 200); do
        if [ ! -e /proc/$pid/status ]; then break; fi
        alive=1
        if ls -l /proc/$pid/fd 2>/dev/null | grep -q 'socket:'; then break; fi
        sleep 0.02
    done
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

echo "=== test_gate (unverified kernels must refuse, not launch)"
$RUN ./build/test_gate

for t in test_ew test_matmul test_conv; do
    echo "=== $t  (reps=$REPS)"
    timeout -k 5 900 $RUN ./build/$t "$REPS"
    board
done

echo "=== test_nhwc (NHWC data plane: conv/depthwise/pool/concat/upsample)"
timeout -k 5 900 $RUN ./build/test_nhwc "$REPS"
board
cool

echo "=== test_onnxops (generic maxpool, forward softmax)"
timeout -k 5 300 $RUN ./build/test_onnxops

# Pooling / normalisation / softmax variants / losses through the ONNX runner.
# Python, so it uses the interpreter that has vc4infer importable (the tree).
echo "=== test_onnx_norm (17 ops through the runner)"
PYTHONPATH=$PYTHONPATH:. timeout -k 5 600 python3 tests/test_onnx_norm.py
board
cool

echo "=== test_onnx_gaps (variadic/conversion/gather family through the runner)"
PYTHONPATH=$PYTHONPATH:. timeout -k 5 900 python3 tests/test_onnx_gaps.py
board
cool

echo "=== test_onnx_geom (geometry/detection family through the runner)"
PYTHONPATH=$PYTHONPATH:. timeout -k 5 1200 python3 tests/test_onnx_geom.py
board
cool

echo "=== test_copyops (layout copy kernels, byte-exact)"
timeout -k 5 300 $RUN ./build/test_copyops
board
cool

echo "=== test_coherency (cacheable arena mapping maintenance)"
timeout -k 5 300 $RUN ./build/test_coherency
board
cool

echo "=== test_train_ops (training primitives)"
timeout -k 5 300 $RUN ./build/test_train_ops
board
cool

echo "=== test_grad (finite differences)"
timeout -k 5 600 $RUN ./build/test_grad 1
board
cool

echo "=== test_bwd_one (each backward op in its own process: hang bisect)"
for op in softmax_bwd softmax_ce_bwd batchnorm_bwd layernorm_bwd \
          avgpool_bwd adaptive_avgpool_bwd global_avgpool_bwd maxpooln_bwd \
          binop_bwd cat_bwd split_bwd pad_bwd transpose_bwd reduce_bwd \
          resize_nearest_bwd resize_bilinear_bwd gather_bwd scatter_bwd \
          bwop_run gate; do
    timeout -k 2 120 $RUN ./build/test_bwd_one $op || {
        echo "FAIL: test_bwd_one $op"; exit 1; }
done
board
cool

cool
echo "=== test_convgrad (conv dX/dW, pool; finite differences)"
timeout -k 5 900 $RUN ./build/test_convgrad 1
board
cool

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
