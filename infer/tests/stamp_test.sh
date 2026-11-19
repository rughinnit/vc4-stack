#!/bin/sh
# Prove the blob cache self-invalidates: unstamped and wrongly-stamped blobs are
# discarded and regenerated, correctly-stamped ones are reused.
set -e
cd /home/raspi/work/vc4-stack
export VC4INFER_DAEMON=1 PYTHONPATH=infer/python
PY=/home/raspi/work/torch-vc4/.venv/bin/python
D=/tmp/kc_stamp
rm -rf $D && mkdir -p $D
# start from blobs built by an older run, with no stamp sidecar
cp /tmp/kc_t1/uop_abs_v16_q1.bin $D/ 2>/dev/null || true
echo "blob before:        $(stat -c %Y $D/uop_abs_v16_q1.bin) $(md5sum $D/uop_abs_v16_q1.bin | cut -c1-8)"
sleep 1.1
VC4INFER_CACHE=$D $PY /tmp/vc4probe/genprobe.py abs >/dev/null 2>&1
echo "after unstamped:    $(stat -c %Y $D/uop_abs_v16_q1.bin) $(md5sum $D/uop_abs_v16_q1.bin | cut -c1-8) stamp=$(test -f $D/uop_abs_v16_q1.bin.stamp && echo yes || echo no)"
sleep 1.1
VC4INFER_CACHE=$D $PY /tmp/vc4probe/genprobe.py abs >/dev/null 2>&1
echo "after valid stamp:  $(stat -c %Y $D/uop_abs_v16_q1.bin)  (must be unchanged = reused)"
sleep 1.1
echo "0000000000000000000000000000000000000000000000000000000000000000" > $D/uop_abs_v16_q1.bin.stamp
VC4INFER_CACHE=$D $PY /tmp/vc4probe/genprobe.py abs >/dev/null 2>&1
echo "after bad stamp:    $(stat -c %Y $D/uop_abs_v16_q1.bin)  (must change = regenerated)"
