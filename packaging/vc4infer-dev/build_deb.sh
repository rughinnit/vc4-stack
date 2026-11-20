#!/bin/sh
# Build vc4infer-dev: the header, the linker symlink, pkg-config and CMake
# package files.
#
#   packaging/vc4infer-dev/build_deb.sh [VERSION]
#
# The header lives here, not in vc4infer: vc4infer 1.2.2 stopped shipping it so
# a runtime-only install does not carry a compiler interface it cannot use.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-1.2.2}
out=$root/../dist
stage=$root/build/vc4infer-dev-deb
triplet=aarch64-linux-gnu

[ -f "$root/infer/include/vc4infer.h" ] || {
	echo "missing infer/include/vc4infer.h" >&2
	exit 1
}

rm -rf "$stage"
install -d "$stage/usr/include" \
	"$stage/usr/lib/$triplet/pkgconfig" \
	"$stage/usr/lib/$triplet/cmake/vc4infer" \
	"$stage/usr/share/doc/vc4infer-dev"

install -m 0644 "$root/infer/include/vc4infer.h" "$stage/usr/include/vc4infer.h"

# the soname link: the runtime package ships libvc4infer.so.1
ln -s libvc4infer.so.1 "$stage/usr/lib/$triplet/libvc4infer.so"

sed "s/@VERSION@/$ver/" "$here/vc4infer.pc.in" \
	> "$stage/usr/lib/$triplet/pkgconfig/vc4infer.pc"
sed "s/@VERSION@/$ver/" "$here/vc4inferConfig.cmake.in" \
	> "$stage/usr/lib/$triplet/cmake/vc4infer/vc4inferConfig.cmake"

cat > "$stage/usr/lib/$triplet/cmake/vc4infer/vc4inferConfigVersion.cmake" <<EOF
set(PACKAGE_VERSION "$ver")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
EOF
chmod 0644 "$stage/usr/lib/$triplet/cmake/vc4infer/"*.cmake \
	"$stage/usr/lib/$triplet/pkgconfig/vc4infer.pc"

cat > "$stage/usr/share/doc/vc4infer-dev/README.md" <<'EOF'
# vc4infer-dev

Headers and build-system files for the VideoCore IV engine.

```c
#include <vc4infer.h>

vc4_ctx *ctx;
if (vc4_init(&ctx, NULL, 0) != VC4_OK) { ... }
```

- `pkg-config --cflags --libs vc4infer`
- CMake: `find_package(vc4infer 1.2 REQUIRED)` then link `vc4infer::vc4infer`

The engine runs unprivileged through `/run/vc4d.sock` when
`VC4INFER_DAEMON=1` is set (the vc4-toolkit `/etc/profile.d/vc4.sh` does that);
otherwise it needs root and `/dev/mem`.  Every float operation runs on the
QPUs; unsupported ops return `VC4_ERR_UNSUPPORTED` and there is no CPU
fallback.
EOF

PKG=vc4infer-dev
VER=$ver
ARCH=arm64
DEPENDS="vc4infer (>= 1.2.2) | vc4-toolkit-fast (>= 1.2.2), libc6 (>= 2.38)"
SECTION=libdevel
SHORT="Headers and build files for libvc4infer (VideoCore IV engine)"
LONG="Installs /usr/include/vc4infer.h, the libvc4infer.so linker symlink, a
pkg-config file and a CMake package so C clients can build against the engine:
 .
   pkg-config --cflags --libs vc4infer
   find_package(vc4infer 1.2 REQUIRED)
 .
 The runtime library itself comes from vc4infer (or from vc4-toolkit-fast,
 which ships an -O3 rebuild under the same soname)."

. "$here/../common.sh"
deb_finish "$stage" "$out"
