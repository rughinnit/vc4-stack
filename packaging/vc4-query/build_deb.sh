#!/bin/sh
# Build vc4-query, the deviceQuery analogue.
#
#   packaging/vc4-query/build_deb.sh [VERSION]
#
# Links against the engine's shared library (not the static archive): the
# package therefore depends on the engine, and the binary stays small.  The
# vc4d wire header is used only at build time for the HELLO handshake.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-1.0.0}
out=$root/dist
stage=$root/build/vc4-query-deb
build=$root/build/vc4-query

mkdir -p "$build"
cp "$root/vc4d/proto/vc4d_proto.h" "$build/vc4d_proto.h"

# The engine must be built first (make -C infer); fail loudly rather than link
# a stale object.
[ -f "$root/infer/build/libvc4infer.so" ] || {
	echo "missing infer/build/libvc4infer.so -- run make -C infer first" >&2
	exit 1
}

gcc -O2 -g -Wall -Wextra -std=gnu11 \
	-I"$root/infer/include" -I"$build" \
	-o "$build/vc4-query" "$here/vc4-query.c" \
	-L"$root/infer/build" -l:libvc4infer.so -lm

rm -rf "$stage"
install -d "$stage/usr/bin" "$stage/usr/share/man/man1" "$stage/usr/share/doc/vc4-query"
install -m 0755 "$build/vc4-query" "$stage/usr/bin/vc4-query"
strip --strip-unneeded "$stage/usr/bin/vc4-query"

gzip -9n -c "$here/vc4-query.1" > "$stage/usr/share/man/man1/vc4-query.1.gz"
chmod 0644 "$stage/usr/share/man/man1/vc4-query.1.gz"

install -m 0644 "$here/README.md" "$stage/usr/share/doc/vc4-query/README.md"

PKG=vc4-query
VER=$ver
ARCH=arm64
DEPENDS="libc6 (>= 2.38), vc4infer (>= 1.2.1) | vc4-toolkit-fast"
SECTION=utils
SHORT="VideoCore IV device query (the deviceQuery analogue)"
LONG="One command that reports the engine and daemon versions, the kernel, the
V3D revision and QPU count, the stock core/V3D clocks, die temperature, the
throttle register, CMA total/free, the daemon arena size, the op registry
counts, whether the installed engine carries the training/backward kernels,
the system kernel cache, and the installed vc4 package versions.
 .
 It reads /sys/kernel/vc4mon (vc4top's read-only module) and vcgencmd; it
 never opens /dev/mem.  Use --json for machine-readable output."
EXTRA="Recommends: vc4d
Suggests: vc4top-dkms"

. "$here/../common.sh"
deb_finish "$stage" "$out"
