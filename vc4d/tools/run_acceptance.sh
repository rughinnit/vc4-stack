#!/bin/bash
# run_acceptance.sh - the whole vc4d acceptance run.
#
#   ./tools/run_acceptance.sh
#
# Must be run as the unprivileged user (raspi); it uses sudo only to set up the
# vc4 group, start the daemon, inspect root-only /proc entries, and run the
# direct-root baseline.  The board-wide GPU lock is taken by the daemon around
# every launch (bounded wait) and by bench_direct itself, so this script never
# holds /tmp/vc4-gpu.lock (that would deadlock against the engine's own tests).
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
LOG=$ROOT/acceptance.log
SOCK=/run/vc4d.sock
DEMO_PID=/tmp/vc4d-demo.pid

exec > >(tee -a "$LOG") 2>&1

say() { echo; echo "=== $*"; }
clock() {
    echo "  core=$(vcgencmd measure_clock core 2>/dev/null) v3d=$(vcgencmd measure_clock v3d 2>/dev/null) throttled=$(vcgencmd get_throttled 2>/dev/null)"
}

stop_daemon() {
    sudo -n systemctl stop vc4d 2>/dev/null || true
    sudo -n pkill -x vc4d 2>/dev/null || true
    for _ in $(seq 1 30); do
        [ -S "$SOCK" ] || return 0
        sleep 0.1
    done
    sudo -n rm -f "$SOCK"
}

wait_daemon() {
    for _ in $(seq 1 100); do
        [ -S "$SOCK" ] && [ -n "$(pgrep -x vc4d)" ] && return 0
        sleep 0.1
    done
    return 1
}

say "board state before"
clock
uname -a
id

say "group setup (vc4)"
sudo -n groupadd -f vc4
sudo -n usermod -aG vc4 raspi
VGID=$(getent group vc4 | cut -d: -f3)
echo "  vc4 gid=$VGID  session groups=$(id -G)"
if [ "${VC4D_IN_SG:-0}" != 1 ] && ! id -G | tr ' ' '\n' | grep -qx "$VGID"; then
    echo "  re-exec under 'sg vc4' so the new group membership is active"
    VC4D_IN_SG=1 exec sg vc4 -c "$0 $*"
fi

say "build"
make -j4 || exit 1
make bench || true

say "stop any old daemon"
stop_daemon

say "start vc4d (root, then it drops privileges itself)"
sudo -n env VC4D_GPU_LOCK=1 VC4D_VERBOSE=1 setsid ./build/vc4d \
    --socket "$SOCK" --arena-mb 64 --group vc4 --drop-user nobody \
    --max-timeout-ms 10000 --verbose > /tmp/vc4d.log 2>&1 &
wait_daemon || { echo "FAIL: daemon did not come up"; cat /tmp/vc4d.log; exit 1; }
DPID=$(pgrep -x vc4d | head -1)
echo "daemon pid=$DPID"
ls -l "$SOCK"
sed 's/^/  /' /tmp/vc4d.log

say "daemon privilege state (must be uid!=0, CapEff=0, no /dev/mem)"
sudo -n grep -E '^(Uid|Gid|CapEff|CapPrm|NoNewPrivs):' /proc/$DPID/status | sed 's/^/  /'
echo "  open fds:"
sudo -n ls -l /proc/$DPID/fd | sed 's/^/    /'
if sudo -n ls -l /proc/$DPID/fd | grep -q /dev/mem; then
    echo "  FAIL: daemon still holds /dev/mem"
    DAEMON_RAWMEM=1
else
    echo "  OK: daemon holds no /dev/mem"
    DAEMON_RAWMEM=0
fi
NR_DAEMON_FDS_BEFORE=$(sudo -n ls /proc/$DPID/fd | wc -l)

say "demo_gemm as user raspi (3 runs) - unprivileged"
rm -f "$DEMO_PID"
./build/demo_gemm --reps 3 --hold 12 --pidfile "$DEMO_PID" &
DEMOJOB=$!
for _ in $(seq 1 200); do [ -s "$DEMO_PID" ] && break; sleep 0.1; done
CPID=$(cat "$DEMO_PID" 2>/dev/null || true)
CLIENT_RAWMEM=1
CLIENT_CAPS=1
if [ -n "$CPID" ]; then
    say "client privilege proof (pid $CPID)"
    grep -E '^(Uid|Gid|CapEff|CapPrm|CapBnd):' /proc/$CPID/status | sed 's/^/  /'
    echo "  open fds:"
    ls -l /proc/$CPID/fd | sed 's/^/    /'
    if ls -l /proc/$CPID/fd | grep -q /dev/mem; then
        echo "  FAIL: client has /dev/mem open"
    else
        echo "  OK: client has no /dev/mem"
        CLIENT_RAWMEM=0
    fi
    case "$(grep '^CapEff:' /proc/$CPID/status | awk '{print $2}')" in
        0000000000000000) echo "  OK: client CapEff is empty (no CAP_SYS_RAWIO)"; CLIENT_CAPS=0;;
        *) echo "  FAIL: client has capabilities";;
    esac
fi
wait $DEMOJOB; DEMO_RC=$?
echo "demo_gemm exit=$DEMO_RC"
clock

say "direct root path (engine-fast2 libvc4infer) for comparison"
sudo -n ./build/bench_direct 3 256
BENCH_RC=$?
echo "bench_direct exit=$BENCH_RC"
clock

say "hostile inputs"
./build/test_hostile --fuzz 2000
HOSTILE_RC=$?
echo "test_hostile exit=$HOSTILE_RC"
clock

say "daemon after the hostile suite"
if sudo -n kill -0 "$DPID" 2>/dev/null; then
    echo "  OK: daemon still running (pid $DPID)"
    DAEMON_ALIVE=0
else
    echo "  FAIL: daemon died"
    DAEMON_ALIVE=1
fi
NR_DAEMON_FDS_AFTER=$(sudo -n ls /proc/$DPID/fd 2>/dev/null | wc -l)
echo "  daemon fds: before=$NR_DAEMON_FDS_BEFORE after=$NR_DAEMON_FDS_AFTER"
if [ "$NR_DAEMON_FDS_AFTER" -le $((NR_DAEMON_FDS_BEFORE + 1)) ]; then
    echo "  OK: no fd leak"
    FDLEAK=0
else
    echo "  FAIL: fd count grew"
    FDLEAK=1
fi
sed 's/^/  /' /tmp/vc4d.log | tail -15

say "strict mode smoke test (allowlist + no raw SRQ)"
stop_daemon
( cd blobs && sha256sum *.bin ) > /tmp/vc4d-allow.txt
sudo -n env VC4D_GPU_LOCK=1 setsid ./build/vc4d --socket "$SOCK" --arena-mb 48 \
    --group vc4 --drop-user nobody --strict --allowlist /tmp/vc4d-allow.txt \
    --verbose > /tmp/vc4d-strict.log 2>&1 &
wait_daemon || { echo "FAIL: strict daemon did not come up"; cat /tmp/vc4d-strict.log; }
DPID2=$(pgrep -x vc4d | head -1)
echo "strict daemon pid=$DPID2"
./build/demo_gemm --reps 1 --blobs blobs | tail -4
STRICT_RC=${PIPESTATUS[0]}
echo "strict demo exit=$STRICT_RC"
./build/test_hostile --expect-denied --blobs blobs
DENY_RC=$?
echo "allowlist negative test exit=$DENY_RC"
[ "$DENY_RC" = 0 ] && STRICT_RC=0 || STRICT_RC=1
if [ -n "$DPID2" ] && sudo -n kill -0 "$DPID2" 2>/dev/null; then
    echo "  OK: strict daemon alive"
else
    echo "  FAIL: strict daemon died"
fi
sed 's/^/  /' /tmp/vc4d-strict.log | tail -10

say "recovery (G1) and arena reclaim (G9) on a private socket"
# A daemon these can provoke without disturbing the acceptance daemon: the
# recovery test times a launch out on purpose and the arena test orphans a
# connection.  --gpu-lock keeps the two daemons taking turns.
TSOCK=/run/vc4d-test-$$.sock
sudo -n rm -f "$TSOCK"
sudo -n setsid ./build/vc4d --socket "$TSOCK" --arena-mb 16 --group vc4 \
    --drop-user nobody --gpu-lock --max-timeout-ms 5000 \
    > /tmp/vc4d-test.log 2>&1 &
for _ in $(seq 1 100); do [ -S "$TSOCK" ] && break; sleep 0.1; done
TPID=$(pgrep -f -- "--socket $TSOCK" | head -1)
RECOVER_RC=1
ARENA_RC=1
if [ -S "$TSOCK" ] && [ -x ../infer/build/test_daemon_recover ]; then
    ../infer/build/test_daemon_recover "$TSOCK"; RECOVER_RC=$?
    ../infer/build/test_daemon_arena "$TSOCK"; ARENA_RC=$?
    echo "  recovery/log:"; grep -E "TIMEOUT|reset|reclaim" /tmp/vc4d-test.log | tail -6 | sed 's/^/    /'
else
    echo "  SKIP: build ../infer (test_daemon_recover) first"
fi
[ -n "$TPID" ] && sudo -n kill "$TPID" 2>/dev/null
sudo -n rm -f "$TSOCK"
echo "test_daemon_recover=$RECOVER_RC test_daemon_arena=$ARENA_RC"

say "stop daemon"
stop_daemon
[ -S "$SOCK" ] && echo "  WARN: $SOCK still present" || echo "  OK: socket removed"

say "summary"
echo "demo_gemm=$DEMO_RC bench=$BENCH_RC hostile=$HOSTILE_RC strict=$STRICT_RC recover=$RECOVER_RC arena=$ARENA_RC"
echo "daemon_rawmem=$DAEMON_RAWMEM client_rawmem=$CLIENT_RAWMEM client_caps=$CLIENT_CAPS daemon_alive=$DAEMON_ALIVE fdleak=$FDLEAK"
clock
if [ "$DEMO_RC" = 0 ] && [ "$HOSTILE_RC" = 0 ] && [ "$STRICT_RC" = 0 ] &&
   [ "$BENCH_RC" = 0 ] && [ "$DAEMON_RAWMEM" = 0 ] && [ "$CLIENT_RAWMEM" = 0 ] &&
   [ "$CLIENT_CAPS" = 0 ] && [ "$DAEMON_ALIVE" = 0 ] && [ "$FDLEAK" = 0 ] &&
   [ "$RECOVER_RC" = 0 ] && [ "$ARENA_RC" = 0 ]; then
    echo "ACCEPTANCE PASSED"
    exit 0
fi
echo "ACCEPTANCE FAILED"
exit 1
