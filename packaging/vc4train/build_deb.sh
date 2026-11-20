#!/bin/sh
# Build vc4train: the engine's own toy trainer, installed as a command.
#
#   packaging/vc4train/build_deb.sh [VERSION]
#
# training/tests/test_train.c is the MLP (64-32-10) full-batch trainer: every
# multiply-accumulate, activation, the fused softmax+CE gradient and the whole
# optimizer step are QPU kernels.  It prints the loss per step and fails if the
# loss does not drop 5x, which is what docs/INSTALL.md tells a new user to run
# to verify that the installed engine can train.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-1.0.0}
out=$root/../dist
stage=$root/build/vc4train-deb
build=$root/build/vc4train

# The static archive keeps the binary self-contained apart from libc/libm and
# is exactly what the test suite links; build it first.
[ -f "$root/infer/build/libvc4infer.so" ] || {
	echo "missing infer/build/libvc4infer.so -- run make -C infer first" >&2
	exit 1
}

# Link the shared engine: the demo then exercises the installed library (and,
# on vc4-toolkit-fast, the -O3 rebuild) rather than a private copy of it.
mkdir -p "$build"
gcc -O2 -g -Wall -Wextra -std=gnu11 \
	-I"$root/infer/include" -I"$root/infer/tests" \
	-o "$build/vc4train" "$root/training/tests/test_train.c" \
	-L"$root/infer/build" -l:libvc4infer.so -lm

rm -rf "$stage"
install -d "$stage/usr/bin" "$stage/usr/share/man/man1" "$stage/usr/share/doc/vc4train"
install -m 0755 "$build/vc4train" "$stage/usr/bin/vc4train"
strip --strip-unneeded "$stage/usr/bin/vc4train"

gzip -9n -c "$here/vc4train.1" > "$stage/usr/share/man/man1/vc4train.1.gz"
chmod 0644 "$stage/usr/share/man/man1/vc4train.1.gz"
install -m 0644 "$here/README.md" "$stage/usr/share/doc/vc4train/README.md"

PKG=vc4train
VER=$ver
ARCH=arm64
DEPENDS="libc6 (>= 2.38), vc4infer (>= 1.2.1) | vc4-toolkit-fast"
SECTION=utils
SHORT="Toy QPU trainer demo (MLP 64-32-10) that proves the engine can train"
LONG="Installs the engine's own training demo: a two-layer MLP trained full
batch with SGD or Adam, where every multiply-accumulate, activation, the fused
softmax+cross-entropy gradient and the optimizer step run as QPU kernels.  It
prints the loss every step and exits non-zero if the loss does not drop 5x.
 .
 This is a demonstration that the backward kernels in libvc4infer work on the
 QPUs, not a training framework.  Run it with:
 .
   vc4train --opt adam --steps 300
 .
 It uses the vc4d socket when VC4INFER_DAEMON=1 (set by /etc/profile.d/vc4.sh)
 or /dev/mem when run as root without the daemon."
EXTRA="Recommends: vc4d"

. "$here/../common.sh"
deb_finish "$stage" "$out"
