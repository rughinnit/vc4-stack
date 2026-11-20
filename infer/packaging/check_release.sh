#!/bin/sh
# Verify that the *installed* vc4infer package is the one this tree measures.
#
#   infer/packaging/check_release.sh [TREE]
#
# Release integrity: the artefact that is measured has to be the artefact that
# is shipped.  vc4infer 1.2.2 shipped an op table of 35 ops while the tree it
# was measured against had 51+, and nothing noticed.  This script compares, by
# name, the ONNX op table (vc4infer.coverage.OPS) of the installed package with
# the tree's and fails on any difference.  It also prints the engine's kernel
# registry count (vc4-query) so the .so that came with the deb is identified.
#
# The installed package is imported from a neutral directory with PYTHONPATH
# cleared, so the tree's python/ directory can never shadow dist-packages.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
tree=${1:-$(cd "$here/../.." && pwd)}
arch=$(dpkg --print-architecture)

count() {   # count TREE|INSTALLED -> "n op1 op2 ..." sorted
	if [ "$1" = tree ]; then
		python3 - "$tree/infer/python/vc4infer" <<'EOF'
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location(
    "cov", os.path.join(sys.argv[1], "coverage.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(len(m.OPS), *sorted(m.OPS))
EOF
	else
		(cd / && env -u PYTHONPATH python3 -c \
			'import vc4infer.coverage as c; print(len(c.OPS), *sorted(c.OPS))')
	fi
}

tree_ops=$(count tree)
inst_ops=$(count installed)
tree_n=${tree_ops%% *}
inst_n=${inst_ops%% *}

inst_ver=$(cd / && env -u PYTHONPATH python3 -c \
	'import vc4infer; print(vc4infer.__version__)')
deb_ver=$(dpkg-query -W -f='${Version}' vc4infer 2>/dev/null || echo "not installed")

echo "version:   installed module $inst_ver, deb $deb_ver"
echo "ops:       tree $tree_n, installed $inst_n"
if command -v vc4-query >/dev/null 2>&1; then
	echo "engine:    $(vc4-query 2>/dev/null | grep -m1 '^ops:' || echo 'vc4-query: no op line')"
fi

status=0
if [ "$tree_n" != "$inst_n" ]; then
	echo "FAIL: op count differs (tree $tree_n, installed $inst_n)" >&2
	status=1
fi
if [ "$tree_ops" != "$inst_ops" ]; then
	echo "FAIL: op tables differ in content" >&2
	for o in ${tree_ops#* }; do
		case " ${inst_ops#* } " in
			*" $o "*) ;;
			*) echo "  only in tree: $o" >&2 ;;
		esac
	done
	for o in ${inst_ops#* }; do
		case " ${tree_ops#* } " in
			*" $o "*) ;;
			*) echo "  only in installed: $o" >&2 ;;
		esac
	done
	status=1
fi
if [ "$inst_ver" != "$deb_ver" ]; then
	echo "FAIL: deb version $deb_ver != module version $inst_ver" >&2
	status=1
fi
[ "$status" = 0 ] && echo "OK: installed package matches the tree ($inst_n ops, $inst_ver)"
exit "$status"
