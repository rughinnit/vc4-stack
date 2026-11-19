#!/bin/sh
# Paired A/B: alternate the two runner modes so board drift hits both equally.
# usage: sh ab160.sh <model> <in> <ref> <rounds>
set -e
M=$1; IN=$2; REF=$3; N=${4:-4}
cd "$(dirname "$0")"
i=1
while [ $i -le $N ]; do
    VC4_NO_SILU_FUSE=1 VC4INFER_DAEMON=1 /home/raspi/work/torch-vc4/.venv/bin/python \
        yolo_bench.py "$M" --input "$IN" --ref "$REF" --repeat 2 \
        --atol 5e-4 --rtol 5e-4 --json "/tmp/ab_off_$i.json" >/dev/null 2>&1
    VC4INFER_DAEMON=1 /home/raspi/work/torch-vc4/.venv/bin/python \
        yolo_bench.py "$M" --input "$IN" --ref "$REF" --repeat 2 \
        --atol 5e-4 --rtol 5e-4 --json "/tmp/ab_on_$i.json" >/dev/null 2>&1
    i=$((i + 1))
done
echo done
