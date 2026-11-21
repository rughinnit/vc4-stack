#!/bin/sh
# Build the PyTorch backend wheel into the stack's shared dist/.
#
#   packaging/build_torch_wheel.sh          # or: make -C torch
#   PYTHON=/path/to/venv/bin/python packaging/build_torch_wheel.sh
#
# The backend is a compiled extension, so this needs the engine and the
# vc4d client archive built in *this* checkout:
#
#   make -C infer        -> infer/build/libvc4infer.a
#   make -C vc4d         -> vc4d/build/libvc4client.a
#
# The interpreter must be the one with torch 2.15 +cpu installed; the
# extension is cp313/aarch64 and statically links both archives, so the wheel
# does not depend on the engine tree at run time.
#
# Output: dist/torch_vc4-<version>-cp313-cp313-linux_aarch64.whl
#
# The filename is part of the published interface: vc4-apt/pip already serves
# torch_vc4-0.3.0-cp313-cp313-linux_aarch64.whl, and packaging/publish_apt.sh
# copies this file there unchanged.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
torch=$root/torch
out=$root/dist
py=${PYTHON:-python3}

if [ ! -f "$torch/setup.py" ]; then
	echo "error: no PyTorch backend at $torch/setup.py." >&2
	echo "       This checkout does not carry the torch/ subtree.  The" >&2
	echo "       backend is part of vc4-stack; a split tree must build it" >&2
	echo "       from its own checkout with VC4_ENGINE_DIR/VC4D_DIR set." >&2
	exit 1
fi

if ! "$py" -c 'import torch; assert torch.__file__; import torch.utils.cpp_extension' >/dev/null 2>&1; then
	echo "error: $py cannot import torch, so the backend cannot be built." >&2
	echo "       Install the +cpu wheel (torch-2.15.0+cpu, cp313/aarch64)" >&2
	echo "       into that interpreter, then re-run this target." >&2
	echo "       (A bare torch/ directory on sys.path is not enough: the" >&2
	echo "       C++ headers and cpp_extension must be importable.)" >&2
	exit 1
fi

for f in "$root/infer/build/libvc4infer.a" "$root/vc4d/build/libvc4client.a"; do
	if [ ! -f "$f" ]; then
		echo "error: missing $f" >&2
		echo "       Build the engine from this checkout first:" >&2
		echo "         make -C infer && make -C vc4d" >&2
		exit 1
	fi
done

# The one translation unit includes torch's headers and peaks near 1 GB of
# address space (measured).  Keep it serial and leave TMPDIR off tmpfs.
export TMPDIR=${TMPDIR:-$HOME/scratch/tmp}
mkdir -p "$TMPDIR"
export MAX_JOBS=1
ulimit -v 1800000 2>/dev/null || true

stage=$torch/build/wheel
rm -rf "$stage"
install -d "$stage" "$out"

cd "$torch"
"$py" setup.py build_ext -b "$stage"

"$py" - "$torch" "$stage" "$out" <<'PY'
import base64
import hashlib
import os
import re
import sys
import sysconfig
import zipfile

torch, stage, out = sys.argv[1:4]
with open(os.path.join(torch, "setup.py"), encoding="utf-8") as handle:
    version = re.search(r'^\s*version="([^"]+)"', handle.read(), re.M).group(1)

py_tag = "cp%d%d" % sys.version_info[:2]
abi_tag = py_tag
platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
name = "torch_vc4-%s-%s-%s-%s.whl" % (version, py_tag, abi_tag, platform)
dist_info = "torch_vc4-%s.dist-info" % version

metadata = """Metadata-Version: 2.1
Name: torch_vc4
Version: %s
Summary: PyTorch PrivateUse1 backend for the Raspberry Pi VideoCore IV QPUs
Home-page: https://github.com/rughinnit/vc4-stack
License: MIT
Requires-Python: >=3.9
Requires-Dist: torch
Description-Content-Type: text/markdown

torch_vc4 is a subset PyTorch backend for the VideoCore IV QPU.  See
torch/COMPLETENESS.md in the vc4-stack source tree for the op table,
measured accuracy and the exact reason for every unsupported op.
""" % version

wheel = """Wheel-Version: 1.0
Generator: torch_vc4-manual
Root-Is-Purelib: false
Tag: %s-%s-%s
""" % (py_tag, abi_tag, platform)

files = []
package = os.path.join(torch, "torch_vc4")
for entry in sorted(os.listdir(package)):
    if entry.endswith(".py"):
        files.append((os.path.join(package, entry), "torch_vc4/" + entry))
built = os.path.join(stage, "torch_vc4")
for entry in sorted(os.listdir(built)):
    if entry.startswith("_C") and entry.endswith(".so"):
        files.append((os.path.join(built, entry), "torch_vc4/" + entry))
if not any(dest.endswith(".so") for _, dest in files):
    sys.exit("error: build_ext produced no _C*.so under %s" % built)

members = list(files)
members.append((None, dist_info + "/METADATA", metadata.encode()))
members.append((None, dist_info + "/WHEEL", wheel.encode()))
members.append((None, dist_info + "/top_level.txt", b"torch_vc4\n"))

def digest(data):
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

record = []
blobs = []
for member in members:
    src, dest = member[0], member[1]
    if src is None:
        data = member[2]
    else:
        with open(src, "rb") as handle:
            data = handle.read()
    blobs.append((dest, data))
    record.append("%s,sha256=%s,%d" % (dest, digest(data), len(data)))
record.append(dist_info + "/RECORD,,")

path = os.path.join(out, name)
with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
    for dest, data in blobs:
        archive.writestr(dest, data)
    archive.writestr(dist_info + "/RECORD", "\n".join(record) + "\n")

print("built %s" % path)
print("  extension: %s" % [d for _, d in files if d.endswith(".so")][0])
print("  files: %d" % len(blobs))
PY

whl=$(ls "$out"/torch_vc4-*.whl)
sha256sum "$whl"
