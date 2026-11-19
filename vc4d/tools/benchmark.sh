#!/bin/bash
# benchmark.sh - daemon path vs direct root path, same board state, R reps each.
# The daemon must already be running (started by run_acceptance.sh or manually).
set -u
cd "$(dirname "$0")/.."
R=${1:-5}
echo "core=$(vcgencmd measure_clock core) v3d=$(vcgencmd measure_clock v3d) throttled=$(vcgencmd get_throttled)"
echo "--- via vc4d (unprivileged)"
./build/demo_gemm --reps "$R" --size 256
echo "--- direct root path (libvc4infer)"
sudo -n ./build/bench_direct "$R" 256
echo "core=$(vcgencmd measure_clock core) v3d=$(vcgencmd measure_clock v3d) throttled=$(vcgencmd get_throttled)"
