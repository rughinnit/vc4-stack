#!/bin/sh
# Build libvc4_tvm.so, the JSON runtime module of the VC4 Relax BYOC backend.
#
#   ./build.sh            # engine path enabled when vc4infer.h is installed
#   ./build.sh --no-engine # host/record modes only (no libvc4infer link)
#
# Temporary files stay in $TMPDIR (default: ~/scratch/tmp), never /tmp: the
# board is 1 GiB and /tmp is tmpfs.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${VC4_TVM_BUILD_DIR:-$(dirname "$HERE")/build}
ENGINE=auto
for a in "$@"; do
    case "$a" in
        --no-engine) ENGINE=no ;;
        --with-engine) ENGINE=yes ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

: "${TMPDIR:=$HOME/scratch/tmp}"
export TMPDIR
mkdir -p "$TMPDIR" "$OUT"

PY=${TVM_PYTHON:-python3}
TVM_PKG=$("$PY" - <<'PYEOF'
import os, tvm
print(os.path.dirname(os.path.abspath(tvm.__file__)))
PYEOF
)
TVM_FFI_PKG=$("$PY" -c "import os, tvm_ffi; print(os.path.dirname(os.path.abspath(tvm_ffi.__file__)))")
TVM_LIB=$TVM_PKG/lib
TVM_LIB=${TVM_LIB:-}

if [ "$ENGINE" = auto ]; then
    if [ -f /usr/include/vc4infer.h ]; then ENGINE=yes; else ENGINE=no; fi
fi

FLAGS="-std=c++17 -O2 -fPIC -shared -Wall -Wextra -Wno-unused-parameter"
LIBS="-L$TVM_LIB -ltvm_runtime -Wl,-rpath,$TVM_LIB"
if [ "$ENGINE" = yes ]; then
    FLAGS="$FLAGS -DVC4_TVM_ENGINE"
    LIBS="$LIBS -lvc4infer"
fi

# shellcheck disable=SC2086
g++ $FLAGS -o "$OUT/libvc4_tvm.so" "$HERE/runtime_vc4.cc" \
    -I"$TVM_PKG/include" -I"$TVM_FFI_PKG/include" $LIBS

echo "built $OUT/libvc4_tvm.so (engine=$ENGINE)"
