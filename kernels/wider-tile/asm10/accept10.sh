#!/bin/bash
# asm10 acceptance: new packer (pack10: grid-stride row blocks, 4 QPUs, 64 B
# VPM reads) + the asm9 GEMM kernel (gemm6, unchanged), full element + panel
# verification.  One launch per attempt, one flock around the whole sequence.
cd /home/raspi/work/asm10 || exit 1
LOG=/tmp/accept10.log
: > $LOG
say() { echo "$@" | tee -a $LOG; }
export PACK_QPUS=4 QRUN_PACK_STRIDE=1
say "== asm10 acceptance $(date -Is) =="
say "-- clocks:"; sudo -n vcgencmd measure_clock v3d 2>/dev/null | tee -a $LOG
sudo -n vcgencmd get_throttled 2>/dev/null | tee -a $LOG
say "-- 1 QPU control, N=128"
PACK_BIN=pack10_128_k16_single_vp1_us.bin timeout -k 5 300 python3 g6.py gemm6_128.bin 1 128 c1q 2>&1 | tee -a $LOG
say "-- 12 QPUs, N=64, 5 runs"
for i in 1 2 3 4 5; do
    PACK_BIN=pack10_64_k16_single_vp1_us.bin timeout -k 5 300 python3 g6.py gemm6_64.bin 12 64 a64_$i 2>&1 | tee -a $LOG
done
say "-- 12 QPUs, N=128, 5 runs"
for i in 1 2 3 4 5; do
    PACK_BIN=pack10_128_k16_single_vp1_us.bin timeout -k 5 300 python3 g6.py gemm6_128.bin 12 128 a128_$i 2>&1 | tee -a $LOG
done
say "-- 12 QPUs, N=256, 5 runs (asm10 pack + asm9 kernel)"
for i in 1 2 3 4 5; do
    PACK_BIN=pack10_256_k16_single_vp1_us.bin timeout -k 5 300 python3 g6.py gemm6_256.bin 12 256 a256_$i 2>&1 | tee -a $LOG
done
say "-- 12 QPUs, N=256, 3 runs (asm9 pack8 + asm9 kernel, A/B)"
unset QRUN_PACK_STRIDE
for i in 1 2 3; do
    PACK_BIN=../asm9/pack8_256_vp1.bin timeout -k 5 300 python3 g6.py ../asm9/gemm6_256.bin 12 256 b256_$i 2>&1 | tee -a $LOG
done
say "-- after: $(date -Is)"; uptime | tee -a $LOG
sudo -n vcgencmd get_throttled 2>/dev/null | tee -a $LOG
sudo -n /home/raspi/vc4cl/tools/read_v3d_status 2>/dev/null | grep -E "STATE|SRQCS" | tee -a $LOG
