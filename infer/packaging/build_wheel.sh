#!/bin/sh
# Build the pure-Python wheel: dist/vc4infer-<version>-py3-none-any.whl
#
#   infer/packaging/build_wheel.sh
#
# The wheel carries the Python API, the CLI, the executor, the ctypes binding
# and the QPU kernel generator (as package data).  It does NOT carry the engine
# shared library: libvc4infer.so.1 is architecture-specific and is installed by
# the vc4infer .deb (or by the engine's own `make`).  Install both:
#
#   sudo apt install ./vc4infer_<ver>_arm64.deb   # libvc4infer.so.1 + CLI
#   pip install vc4infer-<ver>-py3-none-any.whl   # the Python package
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
out=$root/../dist

[ -f "$root/pyproject.toml" ] || { echo "missing pyproject.toml" >&2; exit 1; }
[ -f "$root/python/vc4infer/__init__.py" ] || { echo "missing python/vc4infer package" >&2; exit 1; }
[ -f "$root/build/libvc4infer.so" ] || \
	echo "warning: infer/build/libvc4infer.so is not built; the wheel is" \
	     "still pure Python and will use the installed library" >&2

# The wheel takes its version from pyproject.toml while the deb takes it from
# vc4infer.__version__.  A 1.2.11 tree once produced a 1.2.10 wheel for that
# reason, so the two are compared here instead of trusted.
pyver=$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$root/pyproject.toml" | head -1)
solver=$(sed -n 's/^__version__ *= *"\(.*\)"/\1/p' \
	"$root/python/vc4infer/__init__.py" | head -1)
[ -n "$pyver" ] && [ -n "$solver" ] || {
	echo "cannot read a version from pyproject.toml or __init__.py" >&2
	exit 1
}
[ "$pyver" = "$solver" ] || {
	echo "refusing to build: pyproject.toml says $pyver and" \
	     "vc4infer/__init__.py says $solver" >&2
	exit 1
}

install -d "$out"
rm -f "$out"/vc4infer-*-py3-none-any.whl

cd "$root"
python3 ../tools/check_kgen_sync.py || {
	echo "refusing to build: generator copies have drifted" >&2
	exit 1
}
python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir "$out" .

whl=$(ls "$out"/vc4infer-*-py3-none-any.whl)
echo "built $whl"
python3 - "$whl" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
print("  files: %d" % len(names))
print("  kgen:  %s" % ", ".join(n.split("/")[-1] for n in names
                                 if "/kgen/" in n))
print("  version: %s" % [n for n in names if n.endswith("METADATA")][0])
EOF
sha256sum "$whl"
