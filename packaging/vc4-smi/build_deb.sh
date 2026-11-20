#!/bin/sh
# Build vc4-smi, the nvidia-smi analogue.
#
#   packaging/vc4-smi/build_deb.sh [VERSION]
#
# No engine dependency: it parses `vc4top --once` and the read-only vc4mon
# register window, so it works on a box where the daemon is down (which is
# exactly when you want a status tool).
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-1.0.0}
out=$root/../dist
stage=$root/build/vc4-smi-deb
build=$root/build/vc4-smi

mkdir -p "$build"
gcc -O2 -g -Wall -Wextra -std=gnu11 -o "$build/vc4-smi" "$here/vc4-smi.c"

rm -rf "$stage"
install -d "$stage/usr/bin" "$stage/usr/share/man/man1" "$stage/usr/share/doc/vc4-smi"
install -m 0755 "$build/vc4-smi" "$stage/usr/bin/vc4-smi"
strip --strip-unneeded "$stage/usr/bin/vc4-smi"

gzip -9n -c "$here/vc4-smi.1" > "$stage/usr/share/man/man1/vc4-smi.1.gz"
chmod 0644 "$stage/usr/share/man/man1/vc4-smi.1.gz"
install -m 0644 "$here/README.md" "$stage/usr/share/doc/vc4-smi/README.md"

PKG=vc4-smi
VER=$ver
ARCH=arm64
DEPENDS="libc6 (>= 2.38), vc4top (>= 1.0.5)"
SECTION=utils
SHORT="One-shot VideoCore IV status summary (the nvidia-smi analogue)"
LONG="Prints state, QPU utilisation, stock core/V3D clocks, temperature,
throttle bits, CMA memory, QPU state and the clients attached to vc4d, in one
command.  Thin wrapper over 'vc4top --once' plus the read-only
/sys/kernel/vc4mon register window; --json for monitoring.
 .
 It exits non-zero when the GPU is unhealthy (active throttle bits,
undervoltage, a scheduler queue error or ERRSTAT error bits).  It changes no
clock, governor or voltage."

. "$here/../common.sh"
deb_finish "$stage" "$out"
