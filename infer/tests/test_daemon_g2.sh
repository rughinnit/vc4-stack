#!/bin/sh
# test_daemon_g2.sh - G2: a degraded arena must not stay degraded.
#
# The daemon's arena is one contiguous CMA block.  Until 1.2.7 a failed
# allocation went straight to half the size (then a quarter, an eighth) and the
# daemon kept the smaller arena for the rest of its life: a transient CMA
# shortage - another client exiting, the DRM driver holding buffers for a
# moment - turned into "a 320x320 model fails with pool exhausted until you
# reboot".  `vc4-query` showed the reduced arena.
#
# What this does, against private daemons on their own sockets (they never
# launch anything, so they cannot disturb the installed one):
#
#   A. retry: CMA is held while the daemon starts and released a moment later.
#      The daemon retries the full block with a short backoff (0/200/400 ms) and
#      must end up at the requested size without a restart.
#   B. degrade + restore: CMA is held for the daemon's whole startup, so it must
#      fall back and say so; then the pressure is released and - with no client
#      using the arena - the daemon exits 75 so a fresh daemon (systemd
#      Restart=always, emulated here) claims the full block.  No reboot.
#
# Each phase gets its own socket and log so a daemon that is shutting down can
# never be confused with the next one.
#
# usage: tests/test_daemon_g2.sh [WANT_MB]
#   WANT_MB default 32 (16 is the minimum that can degrade: half of it is still
#   ARENA_MIN).  Needs sudo -n (a private vc4d is root for /dev/mem).
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
VC4D_DIR=${VC4D_DIR:-$ROOT/../vc4d}
WANT_MB=${1:-32}
# Phase A asks for a block that can actually be satisfied once the pressure is
# gone.  On a board whose CMA is already fragmented (the DRM driver's buffers
# are scattered through it) a 32 MiB contiguous block may be impossible at any
# CmaFree value, which is exactly the bug's premise; the retry is then proven
# with the size the board can still supply.
RETRY_MB=${RETRY_MB:-16}
TMPD=${TMPDIR:-$HOME/scratch/tmp}
mkdir -p "$TMPD"

BASE=/run/vc4d-g2-$$
LOGD=$TMPD/g2-$$
PIDBASE=$TMPD/cmahold-$$
SUDO="sudo -n"
SOCK=""
LOG=""
DPID=""
HPID=""
fails=0
skipped=0
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
skip() { printf '  SKIP  %s\n' "$*"; skipped=$((skipped + 1)); }
say()  { printf '%s\n' "$*"; }

free_mb() { awk '/^CmaFree:/{print int($2/1024)}' /proc/meminfo; }
field_mb() { sed -n "s/.*$1=\([0-9]*\).*/\1/p"; }

stop_daemon() {
    [ -n "$SOCK" ] || return 0
    $SUDO pkill -f -- "--socket $SOCK" 2>/dev/null || true
    if [ -n "$DPID" ]; then
        wait "$DPID" 2>/dev/null
        DPID=""
    fi
    $SUDO rm -f "$SOCK"
}
release_cma() {
    for pf in "$PIDBASE".*; do
        [ -s "$pf" ] || continue
        $SUDO kill "$(cat "$pf")" 2>/dev/null || true
    done
    i=0
    while [ $i -lt 40 ]; do
        pgrep -f "cmahold --mb" >/dev/null 2>&1 || break
        i=$((i + 1))
        sleep 0.05
    done
    [ -n "$HPID" ] && wait "$HPID" 2>/dev/null
    HPID=""
    rm -f "$PIDBASE".*
}
cleanup() {
    stop_daemon
    release_cma
    $SUDO rm -f "$BASE"*.sock
}
trap cleanup EXIT INT TERM

# Hold CMA until CmaFree is at most $1 MiB (+ one 4 MiB chunk of granularity).
hold_until_free() {
    target=$1
    n=0
    while [ "$(free_mb)" -gt $((target + 4)) ] && [ $n -lt 3 ]; do
        need=$(( $(free_mb) - target ))
        n=$((n + 1))
        pf=$PIDBASE.$n
        rm -f "$pf"
        $SUDO "$CMA" --mb "$need" --chunk-mb 4 --pidfile "$pf" \
              >"$pf.log" 2>&1 &
        HPID=$!
        i=0
        while [ $i -lt 100 ] && [ ! -s "$pf" ]; do
            i=$((i + 1))
            sleep 0.05
        done
        [ -s "$pf" ] || break
    done
    [ "$(free_mb)" -le $((target + 4)) ]
}

# Start the daemon (without waiting) / wait until it answers.
launch_daemon() {
    mb=${1:-$WANT_MB}
    $SUDO rm -f "$SOCK"
    $SUDO "$VC4D" --socket "$SOCK" --arena-mb "$mb" --group vc4 \
          --drop-user nobody --max-timeout-ms 5000 >"$LOG" 2>&1 &
    DPID=$!
}
wait_ready() {
    i=0
    while [ $i -lt 200 ]; do
        "$PROBE" "$SOCK" >/dev/null 2>&1 && return 0
        i=$((i + 1))
        sleep 0.05
    done
    return 1
}
wait_gone() {
    i=0
    while [ $i -lt $1 ]; do
        "$PROBE" "$SOCK" >/dev/null 2>&1 || return 0
        i=$((i + 1))
        sleep 0.5
    done
    return 1
}
start_daemon() { launch_daemon && wait_ready; }

# A shared board: a concurrent run can hold CMA for minutes, and then "the
# daemon restored" is simply not testable.  Wait, and report a skip rather than
# blaming the daemon.
wait_for_cma() {
    n=0
    while [ "$(free_mb)" -lt "$1" ] && [ $n -lt "$2" ]; do
        n=$((n + 2))
        sleep 2
    done
    [ "$(free_mb)" -ge "$1" ]
}

# A previous failed attempt at this size leaves a backoff stamp in /run that
# would (correctly) delay the restore; the test starts from a clean slate.
$SUDO rm -f /run/vc4d/arena-restore-*.state /tmp/arena-restore-*.state

VC4D=$VC4D_DIR/build/vc4d
PROBE=$VC4D_DIR/build/arena_size
CMA=$VC4D_DIR/build/cmahold
for f in "$VC4D" "$PROBE" "$CMA"; do
    [ -x "$f" ] || { echo "missing $f (make -C vc4d && make -C infer)" >&2; exit 1; }
done

say "== G2: arena restore (want ${WANT_MB} MiB, retry ${RETRY_MB} MiB, CmaFree $(free_mb) MiB; logs in $TMPD/g2-$$-*.log)"
if ! wait_for_cma $((WANT_MB + 8)) 120; then
    skip "whole test skipped: CmaFree is $(free_mb) MiB and $((WANT_MB + 8)) MiB is needed; another process is holding CMA (shared board)"
    printf 'G2 SKIP (%d)\n' "$skipped"
    exit 0
fi
# Leave room for the fallback block: want/2 must still be obtainable, and
# ARENA_MIN (8 MiB) must fit with a margin, or the daemon refuses to start at
# all and there is nothing to observe (that is a skip, not a failure).
leave=$((WANT_MB / 2 + 4))

# ---------------------------------------------------------------- A. retry --
SOCK=$BASE-a.sock
LOG=$LOGD-a.log
say "-- A. retry: asking for ${RETRY_MB} MiB while CMA is held down to ${leave} MiB free"
if ! hold_until_free "$leave"; then
    skip "phase A skipped: could not hold CMA that low ($(free_mb) MiB free)"
else
    say "   held; CmaFree $(free_mb) MiB"
    launch_daemon "$RETRY_MB"
    sleep 0.1                       # the first attempt is in flight
    release_cma
    say "   released while the daemon was starting; CmaFree $(free_mb) MiB"
    if wait_ready; then
        out=$("$PROBE" "$SOCK" "$RETRY_MB" 2>&1)
        rc=$?
        if [ "$rc" = 0 ]; then
            pass "retry: the same daemon reached the full ${RETRY_MB} MiB ($out)"
        else
            fail "retry: expected ${RETRY_MB} MiB ($out)"
        fi
    else
        fail "retry: the daemon stopped during its retries (see $LOG)"
    fi
    grep -E "arena attempt|obtained on attempt|ARENA DEGRADED" "$LOG" \
        | sed 's/^/   log: /' || true
    stop_daemon
fi

# ------------------------------------------------------ B. degrade + restore
SOCK=$BASE-b.sock
LOG=$LOGD-b.log
say "-- B. degrade: holding CMA down to ${leave} MiB free across startup"
hold_until_free "$leave" || say "   (could only reach $(free_mb) MiB free)"
say "   CmaFree $(free_mb) MiB"
launch_daemon
if ! wait_ready; then
    if [ "$(free_mb)" -lt 14 ]; then
        skip "phase B skipped: the daemon could not get even its minimum block with $(free_mb) MiB of CMA free"
    else
        fail "private vc4d did not start (B)"
    fi
else
    out=$("$PROBE" "$SOCK" 2>&1)
    say "   $out"
    deg=$(printf '%s\n' "$out" | field_mb degraded)
    got=$(printf '%s\n' "$out" | field_mb arena)
    if [ "${deg:-0}" = 1 ] && [ -n "$got" ] && [ "$got" -lt "$WANT_MB" ]; then
        pass "degraded to ${got} MiB of ${WANT_MB} MiB while CMA was held"
    else
        # CMA is allowed to migrate movable pages, so "CmaFree < request" is not
        # a guarantee that the request must fail.  Nothing to observe here.
        skip "phase B skipped: the ${WANT_MB} MiB block was available despite $(free_mb) MiB of CmaFree (CMA can migrate pages)"
    fi
    if [ "${deg:-0}" = 1 ]; then
        if grep -q "ARENA DEGRADED" "$LOG"; then
            pass "the daemon says so, with the sizes and what to do:"
            sed -n '/ARENA DEGRADED/,+6p' "$LOG" | sed 's/^/   log: /'
        else
            fail "no ARENA DEGRADED message in $LOG"
        fi

        say "-- B. restore: releasing the CMA; the idle daemon must exit 75"
        release_cma
        say "   CmaFree $(free_mb) MiB"
        alive=1
        i=0
        while [ $i -lt 120 ]; do
            if ! kill -0 "$DPID" 2>/dev/null; then alive=0; break; fi
            i=$((i + 1))
            sleep 0.5
        done
        rc=0
        if [ "$alive" = 0 ]; then
            wait "$DPID" 2>/dev/null || rc=$?
        fi
        DPID=""
        if [ "$rc" = 75 ] && [ "$alive" = 0 ]; then
            pass "the idle degraded daemon exited 75 (what systemd restarts)"
        elif wait_gone 2 && grep -q "exit: restarting" "$LOG"; then
            pass "the idle degraded daemon exited for a restart"
        elif [ "$(free_mb)" -lt $((WANT_MB + 4)) ]; then
            skip "the daemon is still waiting: CmaFree $(free_mb) MiB < $((WANT_MB + 4)) MiB"
        else
            fail "the degraded daemon did not exit while idle (CmaFree $(free_mb) MiB)"
        fi
        if grep -q "exit: restarting" "$LOG"; then
            pass "the log says why:"
            grep -E "arena was .* requested|exit: restarting|restart reason" "$LOG" \
                | sed 's/^/   log: /'
        else
            fail "no deliberate restart in the log (see $LOG)"
        fi
    fi
    stop_daemon
fi

# systemd's Restart=always, done by hand for a private daemon
SOCK=$BASE-c.sock
LOG=$LOGD-c.log
if start_daemon; then
    out=$("$PROBE" "$SOCK" "$WANT_MB" 2>&1)
    rc=$?
    if [ "$rc" = 0 ]; then
        pass "after the restart the arena is the full ${WANT_MB} MiB ($out)"
    elif [ "$(free_mb)" -lt $((WANT_MB + 4)) ]; then
        skip "after the restart CmaFree is $(free_mb) MiB: the full block is not available ($out)"
    else
        fail "after the restart the arena is still reduced ($out)"
    fi
    grep -E "^vc4d: arena " "$LOG" | tail -2 | sed 's/^/   log: /' || true
else
    fail "the replacement daemon did not start (see $LOG)"
fi
stop_daemon

if [ "$fails" = 0 ]; then
    printf 'G2 PASS (%d skipped)\n' "$skipped"
    exit 0
fi
printf 'G2 FAIL (%d)\n' "$fails"
exit 1
