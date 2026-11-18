#!/bin/bash
# asm8 acceptance: 1-QPU control, 5x N=64, 5x N=128, 3x N=256 at 12 QPUs,
# plus 3x the asm7 kernel at N=256 for an interleaved A/B.  One launch per
# g6 call, one flock around the whole sequence.
cd /home/raspi/work/asm8 || exit 1
LOG=/tmp/accept8.log
: > $LOG
say() { echo "$@" | tee -a $LOG; }

say "== asm8 acceptance $(date -Is) =="
say "-- clocks:"; sudo -n vcgencmd measure_clock v3d 2>/dev/null | tee -a $LOG
sudo -n vcgencmd get_throttled 2>/dev/null | tee -a $LOG
say "-- 1 QPU control, N=128"
PACK_QPUS=1 timeout -k 5 300 python3 g6.py gemm5_128.bin 1 128 a1q 2>&1 | tee -a $LOG
say "-- 12 QPUs, N=64, 5 runs"
for i in 1 2 3 4 5; do
    PACK_QPUS=4 timeout -k 5 300 python3 g6.py gemm5_64.bin 12 64 a64_$i 2>&1 | tee -a $LOG
done
say "-- 12 QPUs, N=128, 5 runs"
for i in 1 2 3 4 5; do
    PACK_QPUS=4 timeout -k 5 300 python3 g6.py gemm5_128.bin 12 128 a128_$i 2>&1 | tee -a $LOG
done
say "-- 12 QPUs, N=256, 5 runs (asm8)"
for i in 1 2 3 4 5; do
    PACK_QPUS=4 timeout -k 5 300 python3 g6.py gemm5_256.bin 12 256 a256_$i 2>&1 | tee -a $LOG
done
say "-- 12 QPUs, N=256, 3 runs (asm7 gemm4, A/B)"
for i in 1 2 3; do
    PACK_QPUS=4 timeout -k 5 300 python3 g6.py ../asm7/gemm4_256.bin 12 256 b256_$i 2>&1 | tee -a $LOG
done
say "-- after: $(date -Is)"; uptime | tee -a $LOG
sudo -n vcgencmd get_throttled 2>/dev/null | tee -a $LOG
