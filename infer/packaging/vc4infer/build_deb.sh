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
	"$pkg/vc4infer/ort" \
	"$stage/usr/lib/$triplet" \
	"$stage/usr/lib/vc4infer/kgen" \
	"$stage/usr/share/vc4infer/models" \
	"$stage/usr/share/doc/vc4infer" \
	"$stage/usr/share/man/man1" \

# ---- entry point ------------------------------------------------------------
install -m 0755 "$root/cli/vc4infer" "$stage/usr/bin/vc4infer"

# ---- the Python package (same code as the wheel) ----------------------------
install -m 0644 "$root/python/vc4infer"/*.py "$pkg/vc4infer/"
install -m 0644 "$root/python/vc4infer/kgen"/*.py "$pkg/vc4infer/kgen/"
install -m 0644 "$root/python/vc4infer/contrib"/*.py "$pkg/vc4infer/contrib/"
install -m 0644 "$root/python/vc4infer/ort"/*.py "$pkg/vc4infer/ort/"

# ---- engine library + header ------------------------------------------------
install -m 0755 "$so" "$stage/usr/lib/$triplet/libvc4infer.so.1"
# the header, the .so symlink and the build files live in vc4infer-dev

# ---- kernel generator for C clients (the Python package has its own copy) ---
for f in kgen.py kops.py qenc.py gen.py vpmwords.py; do
	install -m 0644 "$root/tools/$f" "$stage/usr/lib/vc4infer/kgen/$f"
done

# ---- release integrity ------------------------------------------------------
# The measured artefact must be the shipped one.  The 1.2.2 deb shipped 35
# ops while the tree had 51+ (the deb was built from an older tree than the
# one that was measured), which no check caught.  Three checks make that class
# of failure impossible to ship:
#   * the packaged ONNX op table must equal the tree's, name for name and note
#     for note (this is the count the sweep and `vc4infer nodes` measure);
#   * the packaged __version__ must equal the deb version;
#   * the engine library must be newer than every C source and header it is
#     built from (a stale libvc4infer.so is the same bug one layer down).
python3 - "$root" "$pkg/vc4infer" "$ver" <<'PYEOF'
import importlib.util
import os
import sys

root, payload, ver = sys.argv[1], sys.argv[2], sys.argv[3]


def load(d, name):
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(d, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tree = load(os.path.join(root, "python", "vc4infer"), "coverage").OPS
stand = load(payload, "coverage").OPS
if tree != stand:
    print("refusing to build: the packaged op table (%d ops) is not the "
          "tree's (%d ops)" % (len(stand), len(tree)), file=sys.stderr)
    print("  only in tree: %s" % sorted(set(tree) - set(stand)),
          file=sys.stderr)
    print("  only in deb:  %s" % sorted(set(stand) - set(tree)),
          file=sys.stderr)
    for k in sorted(set(tree) & set(stand)):
        if tree[k] != stand[k]:
            print("  note differs for %s" % k, file=sys.stderr)
    sys.exit(1)
import re
with open(os.path.join(payload, "__init__.py")) as fh:
    m = re.search(r'__version__ = "([^"]+)"', fh.read())
modver = m.group(1) if m else "?"
if modver != ver:
    print("refusing to build: packaged __version__ %s != deb version %s"
          % (modver, ver), file=sys.stderr)
    sys.exit(1)
print("  op table: %d ops, packaged == tree; __version__ %s" % (len(stand),
                                                               modver))
PYEOF

src_newest=$(ls -t "$root"/src/*.c "$root"/include/*.h 2>/dev/null | head -1)
if [ -n "$src_newest" ] && [ ! "$so" -nt "$src_newest" ]; then
	echo "refusing to build: $so is older than $src_newest" >&2
	echo "rebuild the engine first: make -C infer" >&2
	exit 1
fi
echo "  library: $so newer than $(basename "$src_newest")"

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
install -m 0644 "$root/../docs/ONNXRUNTIME.md" \
	"$stage/usr/share/doc/vc4infer/ONNXRUNTIME.md"
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
Suggests: vc4d, vc4infer-dev, python3-onnxruntime
Section: misc
Priority: optional
Homepage: https://github.com/rughinnit/vc4-stack
Description: ONNX inference on the Raspberry Pi VideoCore IV QPUs
 vc4infer executes fixed-shape fp32 ONNX graphs on the VideoCore IV (V3D) QPUs
 of the Raspberry Pi 1/2/3, Zero and CM1/CM3.  It ships the libvc4infer engine
 library, a CLI (run, bench, nodes, selftest), the QPU kernel
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

# Python bytecode is created at import time and is not owned by the
# package; the postrm removes it on purge (hand-assembled package, no
# dh_python3/py3clean).
install -m 0755 "$here/vc4infer.postrm" "$stage/DEBIAN/postrm"

( cd "$stage" && find . -type f ! -path './DEBIAN/*' -printf '%P\0' \
	| xargs -0 md5sum > DEBIAN/md5sums )
chmod 0644 "$stage/DEBIAN/md5sums"

install -d "$out"
rm -f "$out/vc4infer_${ver}_${arch}.deb"
dpkg-deb --root-owner-group -Zxz --build "$stage" \
	"$out/vc4infer_${ver}_${arch}.deb" >/dev/null

echo "built $out/vc4infer_${ver}_${arch}.deb ($(du -h "$out/vc4infer_${ver}_${arch}.deb" | cut -f1))"
dpkg-deb -I "$out/vc4infer_${ver}_${arch}.deb" | sed -n '1,14p'
