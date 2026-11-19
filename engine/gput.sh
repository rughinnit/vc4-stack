#!/bin/sh
# gput.sh - run one GPU command under the lock, then report the board state
cd /home/raspi/work/engine-fast
flock -w 900 /tmp/vc4-gpu.lock timeout -k 5 120 sudo -n env VC4INFER_TMO_MS=20000 "$@"
rc=$?
sudo -n /home/raspi/vc4cl/tools/read_v3d_status | grep -E "STATE|SRQCS|V3D_ERRORS"
echo "rc=$rc"
