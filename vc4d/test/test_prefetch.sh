#!/bin/sh
# test_prefetch.sh - the same request script against a synchronous daemon and a
# prefetching one, and the two client transcripts must be identical.
#
# The two daemons are private (own socket, own arena, 24 MiB) and run one after
# the other: two arenas at once is how a window loses its CMA block.  The
# daemon's exit line carries what a client cannot see - how many datagrams were
# read early and how many requests were prepared under a launch - and this
# checks both modes for the value that mode must produce.
#
# usage: test/test_prefetch.sh [VC4D] [TEST_PREFETCH]
#   defaults are ../build/vc4d and ../build/test_prefetch
#   needs sudo -n (a private vc4d is root for /dev/mem) and no other GPU client.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
VC4D=${1:-$HERE/../build/vc4d}
TEST=${2:-$HERE/../build/test_prefetch}
MB=${VC4D_PF_ARENA_MB:-24}
WORK=${TMPDIR:-/tmp}
WORK=$WORK/vc4d-prefetch
fail=0
daemon_pid=""

cleanup() {
    if [ -n "$daemon_pid" ] && sudo -n kill -0 "$daemon_pid" 2>/dev/null; then
        sudo -n kill "$daemon_pid" 2>/dev/null
        for _ in $(seq 50); do
            sudo -n kill -0 "$daemon_pid" 2>/dev/null || break
            sleep 0.1
        done
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "$WORK" || exit 1

run_mode() {
    mode=$1
    flag=$2
    sock="/run/vc4d-pf-$mode.sock"
    log="$WORK/$mode.log"
    out="$WORK/$mode.out"
    pfile="$WORK/$mode.pid"
    # The daemon cannot unlink its own socket after it drops to nobody (/run is
    # root-owned), and a stale file makes the next bind land on a new inode
    # while the old name is still gone: two daemons, one path, clients reaching
    # whichever bound last.  Remove it as root and start from a clean pid file.
    sudo -n rm -f "$sock" "$pfile"
    rm -f "$log" "$out"

    sudo -n sh -c "echo \$\$ > '$pfile'; exec '$VC4D' --socket '$sock' \
        --arena-mb $MB --group vc4 --drop-user nobody \
        --max-timeout-ms 10000 --gpu-lock $flag" >"$log" 2>&1 &
    for _ in $(seq 100); do
        [ -S "$sock" ] && [ -s "$pfile" ] && break
        sleep 0.1
    done
    daemon_pid=$(cat "$pfile" 2>/dev/null)
    if [ -z "$daemon_pid" ] || ! sudo -n kill -0 "$daemon_pid" 2>/dev/null; then
        echo "$mode: daemon did not start (log follows)"
        tail -5 "$log"
        return 1
    fi
    if ! grep -q "listening on" "$log"; then
        echo "$mode: daemon is not listening (log follows)"
        tail -5 "$log"
        return 1
    fi

    "$TEST" "$sock" >"$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "$mode: test failed (rc=$rc):"
        sed 's/^/    /' "$out"
        fail=1
    fi

    sudo -n kill "$daemon_pid" 2>/dev/null
    for _ in $(seq 100); do
        sudo -n kill -0 "$daemon_pid" 2>/dev/null || break
        sleep 0.1
    done
    if sudo -n kill -0 "$daemon_pid" 2>/dev/null; then
        echo "$mode: daemon $daemon_pid survived SIGTERM"
        fail=1
    fi
    daemon_pid=""
    # the arena is released at exit; give the kernel a moment before the next one
    sleep 0.5
    return 0
}

run_mode sync "" || fail=1
run_mode async "--prefetch" || fail=1

echo
echo "--- client transcripts"
if diff -u "$WORK/sync.out" "$WORK/async.out"; then
    echo "identical: $(grep -c PASS "$WORK/sync.out") pass line(s)"
else
    echo "FAIL: the two modes did not produce the same replies"
    fail=1
fi

echo "--- daemon counters"
if grep -q "prepared launches disabled" "$WORK/sync.log"; then
    echo "sync:  prepared launches disabled"
else
    echo "FAIL: the sync daemon did not report itself disabled"
    fail=1
fi
if grep -q "prepared launches enabled" "$WORK/async.log"; then
    echo "async: prepared launches enabled"
else
    echo "FAIL: the async daemon did not report itself enabled"
    fail=1
fi
sym=$(grep -o "[0-9]* prepared under a launch" "$WORK/sync.log" | cut -d' ' -f1)
asy=$(grep -o "[0-9]* prepared under a launch" "$WORK/async.log" | cut -d' ' -f1)
read_s=$(grep -o "[0-9]* datagram(s) read early" "$WORK/sync.log" | cut -d' ' -f1)
read_a=$(grep -o "[0-9]* datagram(s) read early" "$WORK/async.log" | cut -d' ' -f1)
echo "sync:  ${read_s:-?} read early, ${sym:-?} prepared under a launch"
echo "async: ${read_a:-?} read early, ${asy:-?} prepared under a launch"
[ "${sym:-x}" = "0" ] || { echo "FAIL: the sync daemon prepared something early"; fail=1; }
[ "${asy:-0}" -gt 0 ] || { echo "FAIL: the async daemon prepared nothing early"; fail=1; }

if [ $fail -eq 0 ]; then
    echo "PASSED"
else
    echo "FAILED"
fi
exit $fail
