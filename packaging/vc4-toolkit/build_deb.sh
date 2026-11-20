#!/bin/sh
# Build vc4-toolkit, the meta-package that installs the whole stack.
#
#   packaging/vc4-toolkit/build_deb.sh [VERSION]
#
# `apt install vc4-toolkit` = daemon + engine + -dev headers + device query +
# status monitor + the training demo + vc4top and its kernel module.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-1.2.2}
out=$root/dist
stage=$root/build/vc4-toolkit-deb

rm -rf "$stage"
install -d "$stage/usr/share/doc/vc4-toolkit"
install -m 0644 "$here/README.md" "$stage/usr/share/doc/vc4-toolkit/README.md"

PKG=vc4-toolkit
VER=$ver
ARCH=arm64
DEPENDS="vc4d (>= 1.2.1), vc4infer (>= 1.2.2), vc4infer-dev (>= 1.2.2), vc4-query (>= 1.0.0), vc4-smi (>= 1.0.0), vc4train (>= 1.0.0), vc4top (>= 1.0.5), vc4top-dkms (>= 1.0.5)"
SECTION=metapackages
SHORT="VideoCore IV toolkit: daemon, engine, headers, query, monitor, trainer"
LONG="The single install command for the whole vc4-stack:
 .
   vc4d          privileged helper (socket in /run, vc4 group)
   vc4infer      engine, Python API and CLI (inference)
   vc4infer-dev  header, pkg-config and CMake package
   vc4-query     the deviceQuery analogue
   vc4-smi       the nvidia-smi analogue
   vc4train      the engine's own training demo
   vc4top        realtime QPU telemetry
   vc4top-dkms   the read-only V3D register/counter module
 .
 Everything runs on the QPUs at stock clocks (core 400 MHz / V3D 300 MHz);
 nothing in this stack changes a clock or a voltage.
 .
 Install vc4-toolkit-fast instead for the same stack rebuilt from source with
 -O3 -march=native and a prewarmed kernel cache."

. "$here/../common.sh"
deb_finish "$stage" "$out"
