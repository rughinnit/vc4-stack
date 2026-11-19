#!/bin/sh
# Build the installable vc4infer package.
#
#   infer/packaging/vc4infer/build_deb.sh [VERSION]
#
# Produces dist/vc4infer_<version>_arm64.deb.  The tree is assembled by hand
# (dpkg-deb, no debhelper) because the payload is small and fully known:
# the Python package (CLI + API + executor + binding), the engine .so + header,
# the kernel JIT and the selftest models.
#
# The engine library must be built first (make -C infer), with KGEN_DEFAULT
# pointing at the installed generator so the shipped .so carries no build-tree
# path.  The script fails if
# infer/build/libvc4infer.so is missing rather than shipping a stale one.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-$(python3 - "$root" <<'EOF'
import re, sys
s = open(sys.argv[1] + "/python/vc4infer/__init__.py").read()
print(re.search(r'__version__ = "([^"]+)"', s).group(1))
EOF
)}
arch=arm64
triplet=aarch64-linux-gnu
stage=$root/build/vc4infer-deb
out=$root/../dist
so=$root/build/libvc4infer.so
pkg=$stage/usr/lib/python3/dist-packages

[ -f "$so" ] || {
	echo "missing $so -- build the engine first: make -C infer" >&2
	exit 1
}
[ -f "$root/cli/vc4infer" ] || { echo "missing infer/cli/vc4infer" >&2; exit 1; }
[ -f "$root/python/vc4infer/cli.py" ] || { echo "missing python/vc4infer package" >&2; exit 1; }
[ -d "$here/data/models" ] || { echo "missing selftest data" >&2; exit 1; }

rm -rf "$stage"
install -d "$stage/DEBIAN" \
	"$stage/usr/bin" \
	"$pkg/vc4infer/kgen" \
	"$pkg/vc4infer/contrib" \
	"$stage/usr/lib/$triplet" \
	"$stage/usr/lib/vc4infer/kgen" \
	"$stage/usr/include" \
	"$stage/usr/share/vc4infer/models" \
	"$stage/usr/share/doc/vc4infer" \
	"$stage/usr/share/man/man1" \

# ---- entry point ------------------------------------------------------------
install -m 0755 "$root/cli/vc4infer" "$stage/usr/bin/vc4infer"

# ---- the Python package (same code as the wheel) ----------------------------
install -m 0644 "$root/python/vc4infer"/*.py "$pkg/vc4infer/"
install -m 0644 "$root/python/vc4infer/kgen"/*.py "$pkg/vc4infer/kgen/"
install -m 0644 "$root/python/vc4infer/contrib"/*.py "$pkg/vc4infer/contrib/"

# ---- engine library + header ------------------------------------------------
install -m 0755 "$so" "$stage/usr/lib/$triplet/libvc4infer.so.1"
install -m 0644 "$root/include/vc4infer.h" "$stage/usr/include/vc4infer.h"

# ---- kernel generator for C clients (the Python package has its own copy) ---
for f in kgen.py qenc.py gen.py vpmwords.py; do
	install -m 0644 "$root/tools/$f" "$stage/usr/lib/vc4infer/kgen/$f"
done

# ---- examples ---------------------------------------------------------------
install -d "$stage/usr/share/doc/vc4infer/examples/models"
install -m 0644 "$root/examples"/*.py "$stage/usr/share/doc/vc4infer/examples/"
install -m 0644 "$root/examples/models"/* \
	"$stage/usr/share/doc/vc4infer/examples/models/"

# ---- selftest models --------------------------------------------------------
install -m 0644 "$here"/data/models/* "$stage/usr/share/vc4infer/models/"

# ---- docs + man page --------------------------------------------------------
install -m 0644 "$root/../docs/VC4INFER.md" "$stage/usr/share/doc/vc4infer/README.md"
install -m 0644 "$root/../docs/PYTHON-API.md" \
	"$stage/usr/share/doc/vc4infer/PYTHON-API.md"
install -m 0644 "$here/copyright" "$stage/usr/share/doc/vc4infer/copyright"
if [ -f "$here/changelog.Debian.gz" ]; then
	install -m 0644 "$here/changelog.Debian.gz" \
		"$stage/usr/share/doc/vc4infer/changelog.Debian.gz"
else
	printf 'vc4infer (%s) stable; urgency=medium\n\n  * Initial release: run/bench/nodes/selftest, Python API.\n\n -- rughinnit <rughinnit@users.noreply.github.com>  %s\n' \
		"$ver" "$(date -R)" | gzip -9n > \
		"$stage/usr/share/doc/vc4infer/changelog.Debian.gz"
	chmod 0644 "$stage/usr/share/doc/vc4infer/changelog.Debian.gz"
fi
gzip -9n -c "$here/vc4infer.1" > "$stage/usr/share/man/man1/vc4infer.1.gz"
chmod 0644 "$stage/usr/share/man/man1/vc4infer.1.gz"

strip --strip-unneeded "$stage/usr/lib/$triplet/libvc4infer.so.1"

# no byte-compiled python: the package must work with any >= 3.11 interpreter
find "$stage" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

# ---- control ----------------------------------------------------------------
size=$(du -sk "$stage/usr" | cut -f1)
cat > "$stage/DEBIAN/control" <<EOF
Package: vc4infer
Version: $ver
Architecture: $arch
Maintainer: rughinnit <rughinnit@users.noreply.github.com>
Installed-Size: $size
Depends: python3 (>= 3.11), python3-numpy, python3-onnx (>= 1.14), python3-pil, libc6 (>= 2.38)
Suggests: vc4d
Section: misc
Priority: optional
Homepage: https://github.com/rughinnit/vc4-stack
Description: ONNX inference on the Raspberry Pi VideoCore IV QPUs
 vc4infer executes fixed-shape fp32 ONNX graphs on the VideoCore IV (V3D) QPUs
 of the Raspberry Pi 1/2/3, Zero and CM1/CM3.  It ships the libvc4infer engine
 library and headers, a CLI (run, bench, nodes, selftest), the QPU kernel
 generator, a set of reference models, and the Python package system-wide
 (vc4infer.Engine, vc4infer.Device, vc4infer.postprocess, vc4infer.yolo).
 .
 Every float operation runs on the QPUs; the ARM only moves bytes and does
 integer shape bookkeeping.  An operator or attribute the engine cannot
 dispatch raises an error naming the node and its ONNX op type -- there is no
 CPU fallback.  Detection heads (YOLOv8/YOLO11) are decoded on the host with
 class-aware NMS.
 .
 Unprivileged use goes through the vc4d helper daemon (/run/vc4d.sock); the
 client never opens /dev/mem.
EOF
chmod 0644 "$stage/DEBIAN/control"

( cd "$stage" && find . -type f ! -path './DEBIAN/*' -printf '%P\0' \
	| xargs -0 md5sum > DEBIAN/md5sums )
chmod 0644 "$stage/DEBIAN/md5sums"

install -d "$out"
rm -f "$out/vc4infer_${ver}_${arch}.deb"
dpkg-deb --root-owner-group -Zxz --build "$stage" \
	"$out/vc4infer_${ver}_${arch}.deb" >/dev/null

echo "built $out/vc4infer_${ver}_${arch}.deb ($(du -h "$out/vc4infer_${ver}_${arch}.deb" | cut -f1))"
dpkg-deb -I "$out/vc4infer_${ver}_${arch}.deb" | sed -n '1,14p'
