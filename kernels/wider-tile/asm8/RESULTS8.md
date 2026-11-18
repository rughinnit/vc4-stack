# asm8 results — 6-word rows are correct and 15 % cheaper to issue, but the wall time is TMU-bound

Work tree `/home/raspi/work/asm8` (copy of `asm7`).  Board: Pi 3B, stock clocks
(V3D 300 MHz, `get_throttled=0x0`), no reset during the session.  One launch per
attempt, one `flock -w 900 /tmp/vc4-gpu.lock` per sequence.

## Verdict: implementable, correct, but REJECT for the speed bar

* The change is real and verified: the k-loop row drops from 7 to 6 words
  (118 -> 104 words per k-pair), `exec_cycles` 681 k -> 577 k per QPU
  (0.47 -> 0.40 cycles/MAC), 1-QPU cyc/k-pair 505.6 -> 429.7.
* At 12 QPUs the wall time is **unchanged within noise** at N=256 and ~2 %
  better at N=128, because the kernel is bound by TMU request latency/bytes,
  not by issue slots: the stall grows 273 k -> 371 k per QPU and cancels the
  exec gain exactly.
* The r3-as-second-temp mechanism from NOTES7 cannot exist on this ISA (all six
  accumulators are pinned); the 6 words come from a register-port pairing rule
  instead (vectors moved into the same file as their C accumulators).

## Verified (12 QPUs, one flock, 15/15 clean)

`accept8.sh`: `Cmm=0`, `Pmm=0`, `SRQCS=0x000c0c00` (done=12, no error),
`v3d_errors=0x1000` (VCD idle only), `STATE: LIVE` on every run.

| N | runs | gemm ms | total ms | GFLOP/s gemm (best/median) | GFLOP/s total (best) | cyc/k-pair |
|---|------|---------|----------|---------------------------|----------------------|------------|
| 64 | 5 | 0.1086-0.1095 | 0.1139-0.1169 | 4.826 / 4.807 | 4.605 | 299.6 |
| 128 | 5 | 0.4731-0.4896 | 0.5587-0.5733 | 8.865 / 8.722 | 7.563 | 382.6 |
| 256 | 5 | 3.2637-3.4134 | 3.7268-3.8715 | 10.281 / 10.108 | 9.004 | 410.1 |

Controls: 1 QPU N=128 -> `Cmm=0 Pmm=0`, 3.0474 ms, **429.7 cyc/k-pair**
(asm7: 3.5650 ms, 505.6 -> -15 %).

## Interleaved A/B against asm7 at N=256 (same session, same flock)

| kernel | run 1 | run 2 | run 3 | best | median |
|--------|-------|-------|-------|------|--------|
| asm8 gemm ms | 3.3058 | 3.4134 | 3.2637 | **3.2637** (10.281) | 3.3196 (10.108) |
| asm7 gemm ms | 3.2727 | 3.3493 | 3.3605 | 3.2727 (10.253) | 3.3493 (10.018) |

Bars: gemm-only > 10.55 GFLOP/s (3.18 ms), total > 9.48 (3.74 ms).  Best
measured: 10.281 and 9.004 — **both missed**, by 2.5 % and 5 %.

## Where the time goes (N=256, 12 QPUs, per QPU)

| kernel | exec | tmu_stall | idle | total | wall |
|--------|------|-----------|------|-------|------|
| asm7 | 681 k | 273 k | 2.6 k | 954 k | 3.43 ms |
| asm8 | 577 k | 371 k | 2.2 k | 951 k | 3.36 ms |

Bytes are identical (0.5625 B/MAC; 512 B of B + 128 B of A per k-pair) and so
are the L2 statistics (asm7 31 k hits / 105 k misses, asm8 27 k / 109 k).  The
stall follows the request/byte rate, so removing 14 words of issue work just
moves the time into the stall.

## Keep / reject

* **Keep** `gemm5.py` as the reference port-aware 6-word schedule: correct,
  15 % cheaper to issue, and it wins at N <= 128 where the kernel is latency-
  rather than bandwidth-bound.  Keep the schedule simulator inside it (it
  catches the r4 aliasing rules that cost two runs).
* **Keep** `QRUN_L2=1` in `qrun_gemm` (L2C hit/miss counters) and `diag8`
  (result-structure analyser).
* **Reject** the NOTES7 plan itself: "free r3 by moving BADR to a general
  register" is not implementable without losing the issue slots, and the
  v8min rotate cannot read a general register.
* **Reject** for the speed bar as it stands: at N=256 the wall time is asm7's.
  The next step is the third B vector set (3-phase k loop) so a request can be
  issued a half before its use; see NOTES8 §6.
