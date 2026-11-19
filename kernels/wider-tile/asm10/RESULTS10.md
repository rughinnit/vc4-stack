# asm10 results — pack 458 -> 382 µs (-16 %), total 11.65 -> 11.85-11.97 GFLOP/s

Work tree `/home/raspi/work/asm10`.  Board: Pi 3B, stock clocks (V3D 300 MHz,
`get_throttled=0x0`), no reset during the acceptance session.  One launch per
attempt, one `flock -w 900 /tmp/vc4-gpu.lock` per sequence.

## Verdict: KEEP the new packer (`pack10`, grid-stride, KL=16).  REJECT the
## prime work (measured: it is 0.9 % of the N=64 wall).

* 19/19 clean launches: 5x N=64, 5x N=128, 5x N=256 with the new pack, 1-QPU
  control, 3x asm9 A/B, plus a final 3-run default-path check.  Every run:
  `Cmm=0 Pmm=0` (C element check **and** panel bit-exactness), `SRQCS=0x000c0c00`
  (done=12 error=0), `STATE: LIVE`.
* The GEMM kernel is **byte-identical to asm9** (`cmp` clean), so `exec cyc/k`
  is bit-for-bit unchanged: 291.8 / 372.1 / 398.5 at N=64/128/256.
* N=256 total: **2.803-2.839 ms -> 11.82-11.97 GFLOP/s total** (median 11.85),
  versus **11.52-11.57 total for asm9's own pack8 measured in the same
  session** (+2.7 %) and 11.649 as recorded in RESULTS9.
* The pack pass itself: **457 -> 382 µs at N=256 (-16 %)**, 85.6 -> 80 µs at
  N=128, 5.8 -> 4.6 µs at N=64.

## Verified (12 QPUs, stock clocks, 19/19 clean)

| N | runs | pack ms (range) | gemm ms (range) | total ms (range) | GFLOP/s gemm best/med | GFLOP/s total best/med |
|---|------|-----------------|------------------|------------------|------------------------|-------------------------|
| 64  | 5 | 0.0044-0.0069 | 0.0849-0.0866 | 0.0894-0.0935 | **6.174** / 6.099 | **5.864** / 5.747 |
| 128 | 5 | 0.0791-0.0809 | 0.3960-0.3969 | 0.4753-0.4777 | **10.592** / 10.579 | **8.825** / 8.813 |
| 256 | 5 | 0.3614-0.3870 | 2.4412-2.4557 | 2.8026-2.8386 | **13.745** / 13.719 | **11.972** / 11.851 |

Control: 1 QPU N=128 -> `Cmm=0 Pmm=0`, 3.3460 ms, **417.9 exec cyc/k-pair**
(asm9: 417.9 — identical, as expected for an unchanged kernel).

Mismatches 0 on every run; `nonzero` as expected (4096/16384/65536); maxabs
1.964e-05 / 4.817e-05 / 1.556e-04, bit-identical to asm9 (same data).

## Interleaved A/B, same session, same flock (N=256, 12 QPUs)

| pack | run 1 | run 2 | run 3 | pack median | total best | total median |
|------|-------|-------|-------|-------------|------------|--------------|
| asm10 `pack10` grid-stride | 0.3807 | 0.3838 | 0.3614 | **0.3829** | **11.972** | 11.851 |
| asm9 `pack8` contiguous   | 0.4469 | 0.4655 | 0.4578 | **0.4578** | 11.568 | 11.536 |

GEMM-only is identical in both rows (2.441-2.456 ms, 13.66-13.75 GFLOP/s): the
whole difference is the pack.

## Cost breakdown at N=256 (median, per QPU where applicable)

| item | ms | share of total |
|------|----|----------------|
| pack pre-pass (512 KB of DRAM traffic) | 0.383 | 13.5 % |
| per-tile prime + setup (11 tiles x 0.0008) | 0.009 | 0.3 % |
| GEMM k loop + store (exec 561 k cyc + stall/idle) | 2.446 | 86.2 % |
| **total** | **2.829** | |

N=64 breakdown (the same items, per the SKIP-build measurement): launch
~5.8 µs + non-loop body ~1 µs + k loop ~46.7 µs of exec + ~30 µs of cold-read
time (the harness flushes L2 between the launches, so the 16 KB panel and 16 KB
B are DRAM-cold for a 32-k-pair loop) + pack 4.6 µs = 90 µs.

## Why the pack stopped at 382 µs (the wall, measured)

| probe (N=256, 4 QPUs) | µs | rate |
|---|---|---|
| read half only | 314.2 (6 QPUs: 314.5) | 0.83 GB/s — **shared cap, QPU-count independent** |
| write half only | 371.8 | |
| read+write (adopted) | 380.8-386.8 | 1.34 GB/s combined = 84 % of the 1.6 GB/s LPDDR2 peak |
| contiguous-pitch copy (no transpose) | 465.7 | pattern is not the limiter |
| 1 KB VDR reads (pair mode) | 380.8 | command size is not the limiter |

Acceptance needs pack <= 234 µs to reach 12.5 GFLOP/s total; the mandatory
256 KB read of A alone needs 314 µs.  **12.5 is unreachable with any separate
pack pass** — the remaining route is a fused in-GEMM pack hidden under the k
loop, which needs a 4-row VPM transpose (48-row window vs 8 rows/QPU at 12
QPUs) and therefore a new, unverified VDW-stride write mode.

## Keep / reject

* **Keep** `pack10.py` and `pack10_<N>_k16_single_vp1_us.bin`: -16 % pack,
  bit-exact panel on all three N, built only from proven VPM shapes.
* **Keep** the grid-stride assignment (`QRUN_PACK_STRIDE=1`, u7 unit step):
  17 % of the win comes from it alone, and it is what makes 4 QPUs read a
  contiguous 32 KB window at a time.
* **Reject** `pair` mode (1 KB VDR reads): same wall (380.8), 16 VPM rows/QPU,
  more code — no reason to take the risk.
* **Reject** back-to-back VDWs without a per-write wait (`_now`): correct
  (Pmm=0) but not faster (391 vs 381 µs), and it depends on the DMA engine
  latching its setup words at the address write.
* **Reject** 5/6-QPU packs: no gain (381/387 µs) — the read path is capped
  regardless of parallelism.  8 QPUs is *invalid* anyway: the VPM window is
  48 rows (`VPMBASE=16`), so 8 QPUs x 8 rows corrupts the panel (measured).
* **Reject** the prime changes: the whole non-loop body is <= launch overhead
  (~1 µs; the prime's instructions are 0.8 µs per tile = 0.9 % of the N=64
  wall).  The GEMM kernel was therefore left byte-identical to asm9.
* **Not adopted** (harness-level, and it would move the pack's writeback into
  the GEMM's window): dropping the L2 flush between the pack and the GEMM,
  which is what makes the N=64 loop start on cold DRAM.

## Encoder correction re-verification (2026-11-19)

`qenc.py` was corrected after the XCHECK cross-validation (NOTES10 §6): the
rotate words now use the vc4asm-canonical B mux, and pack/PM/unpack follow the
header widths. Behaviour and speed are unchanged. Three fresh launches after a
reboot, N=256, 12 QPUs, stock clocks, `get_throttled=0x0`, one `flock`:

| run | pack ms | gemm ms | total ms | GFLOP/s total | cyc/k | Cmm | Pmm |
|---|---|---|---|---|---|---|---|
| 1 | 0.3857 | 2.4438 | 2.8294 | 11.859 | 398.5 | 0 | 0 |
| 2 | 0.3861 | 2.4503 | 2.8364 | 11.830 | 398.5 | 0 | 0 |
| 3 | 0.3855 | 2.4510 | 2.8365 | 11.829 | 398.5 | 0 | 0 |

`SRQCS=0x000c0c00`, `STATE: LIVE`, `maxabs=1.557e-4` on every run - identical
to the pre-change numbers above. `qpu_disasm.py`: 754 words, 75 rotates,
0 unknown/illegal.
