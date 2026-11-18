# asm7 results — multi-QPU reset fixed; 12 QPUs correct 10/10; speed bar missed by 3.5 %

Work tree `/home/raspi/work/asm7` (copy of `asm6`; `asm6` and the engine untouched).
Board: Pi 3B, stock clocks (V3D 300 MHz, `get_throttled=0x0`).  All GPU work in
this run: one launch per attempt, one `flock -w 900 /tmp/vc4-gpu.lock` per
sequence, `timeout -k 5 300`, `STATE: LIVE` after every launch.  No reset
occurred during this run (board up throughout).

## Verdict

* **The fault is found, proven and fixed**: a host-side uniform-row stride
  mismatch in `qrun_gemm.c` (`uint32_t uni[12][9]` / `up[12][9]` read through a
  `const uint32_t (*)[8]` parameter, introduced in asm5 when `u7`/`u8` were
  appended).  QPU *i* was given host words `8i..8i+8` instead of `9i..9i+8`.
* The GEMM kernel (`gemm4.py`) is **unchanged** — it was innocent.
* The reset came from the **pack launch**: with the shifted rows, pack QPU1/2/3
  got panel bases `0x1000` / `0x80`, row-block strides `0xde660000`, and VPM row
  bases `16*4 = 64` / `16*0xde700000`.  VPM DMAs aimed at unrelated low DRAM /
  outside the dma-buf, with out-of-range VPM rows — the two documented
  board-killers on this unit.  Printed evidence in `NOTES7.md` §2.1
  (`QRUN_DRYRUN=1 QRUN_OLDSTRIDE=1`, no launch needed).
* The GEMM launch's shifted rows were only wrong, not fatal (tile0 = a bus
  address -> the QPU exits at the tile guard), which is why 1 QPU was always
  correct and why the GEMM-only probes never crashed the board.

## Verified at 12 QPUs (5/5 + 5/5)

| run | N | pack ms | gemm ms | total ms | GFLOP/s gemm | cyc/k-pair | Cmm | Pmm | errs |
|-----|---|---------|---------|----------|--------------|-----------|-----|-----|------|
| 5x | 64 | 0.0042-0.0057 | 0.1157-0.1164 | 0.1202-0.1218 | 4.506-4.530 | 350.2 | 0 | 0 | 0x1000 |
| 5x | 128 | 0.0833-0.0868 | 0.4893-0.4966 | 0.5758-0.5815 | 8.447-8.572 | 450.1 | 0 | 0 | 0x1000 |

`SRQCS=0x000c0c00` on every run (queue 0, no queue error, done=12),
`v3d_errors=0x1000` = the normal VCD-idle bit only (bit 6 never reappeared).

Controls: 1 QPU N=128 -> Cmm=0 Pmm=0, **505.6 cyc/k-pair** (identical to asm6,
no regression); 2 QPUs N=64 with the pack at 2 QPUs -> Cmm=0 Pmm=0, done=2,
LIVE.

## Speed

| N=256, 12 QPUs | best | median | bar |
|---|---|---|---|
| GEMM only | **10.189** GFLOP/s (3.2932 ms) | 10.085 | > 10.55 |
| pack + GEMM | **8.940** (3.7533 ms) | 8.848 | > 9.48 |

Scaling (N=256, gemm ms): 1 QPU 27.79, 2 14.32 (97 %), 4 9.10 (76 %),
6 6.23 (74 %), 12 3.46 (67 %).

Where the 33 % goes at 12 QPUs (V3D counters, per QPU): exec 681 k
(2.27 ms) + **TMU stall 274 k (0.91 ms)** + idle 31 k (0.10 ms) = 986 k.
`exec_cycles` is already at the issue floor (118 words x 4.1 = 484 cycles per
k-pair, 0.47 cycles/MAC, 25 % better than asm3).  The gap is TMU *latency*:
each B vector has only ~112 cycles between issue and pop, while at 12 QPUs the
effective TMU latency is ~136 cycles.  Fixing it needs a second mul temp so two
rows' chains interleave (7 -> 6 words/row) — the next single experiment, spelled
out in `NOTES7.md` §5.

So: the reset fix cost nothing (1-QPU numbers are bit-identical), and the 12-QPU
number is issue-floor + TMU latency, not harness overhead.

## Keep / reject

Keep: `qrun_gemm.c` with `UNI_WORDS=9` and the `QRUN_DUMP_UNI` / `QRUN_DRYRUN` /
`QRUN_OLDSTRIDE` diagnostics, the extra stall/idle/I-cache counters,
`measure7.sh`, `accept7.sh`.  Reject: the asm5/asm6 `(*)[8]` parameter (never
widen a uniform row without changing the pointer-to-array type).
