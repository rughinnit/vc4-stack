#!/bin/sh
# Regenerate the wheel copy of the kernel generator from the canonical tools/
# copy.  tools/check_kgen_sync.py fails the build if they ever drift.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
for f in kgen.py kops.py qenc.py gen.py vpmwords.py; do
	cp "$root/infer/tools/$f" "$root/infer/python/vc4infer/kgen/$f"
done
echo "synced infer/tools -> infer/python/vc4infer/kgen"
