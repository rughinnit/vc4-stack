#!/bin/sh
# thermal-guard.sh - run a command while watching the SoC temperature.
#
#   tools/thermal-guard.sh [--max C] [--interval S] -- command [args...]
#
# The fan was removed when the board was moved (2026-11-20).  Without it the SoC
# idles at ~56 C and a sustained QPU load climbs past 70 C in under two minutes;
# the firmware's soft limit is 80 C, and once it throttles, every measurement in
# the run is taken at a different clock.  This wrapper kills the command at the
# limit and prints the peak, so a benchmark either completes in the thermal
# envelope or fails loudly instead of producing quiet nonsense.
#
# Exit codes: the command's own, or 99 when the thermal limit stopped it.

max=72
interval=2
while [ $# -gt 0 ]; do
	case $1 in
	--max) shift; max=${1:-72} ;;
	--interval) shift; interval=${1:-2} ;;
	--) shift; break ;;
	*) break ;;
	esac
	shift
done
[ $# -gt 0 ] || { echo "usage: $0 [--max C] [--interval S] -- command ..." >&2; exit 2; }

temp() { vcgencmd measure_temp 2>/dev/null | sed "s/^temp=//; s/.C$//"; }
whole() { echo "${1%%.*}"; }

start=$(temp)
echo "thermal-guard: start ${start} C, limit ${max} C" >&2
"$@" &
pid=$!
peak=$start
hot=0
while kill -0 "$pid" 2>/dev/null; do
	t=$(temp)
	[ -n "$t" ] || { sleep "$interval"; continue; }
	[ "$(whole "$t")" -gt "$(whole "$peak")" ] && peak=$t
	if [ "$(whole "$t")" -gt "$max" ]; then
		hot=1
		kill -TERM "$pid" 2>/dev/null
		sleep 1
		kill -KILL "$pid" 2>/dev/null
		break
	fi
	sleep "$interval"
done
wait "$pid" 2>/dev/null
rc=$?
end=$(temp)
thr=$(vcgencmd get_throttled | cut -d= -f2)
clk=$(vcgencmd measure_clock core | cut -d= -f2)
echo "thermal-guard: peak ${peak} C, end ${end} C, core ${clk} Hz, throttled=${thr}" >&2
if [ "$hot" = 1 ]; then
	echo "thermal-guard: STOPPED at ${max} C - the run is not a valid measurement" >&2
	exit 99
fi
exit "$rc"
