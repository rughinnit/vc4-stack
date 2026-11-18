# asm10 — the A-panel pack is DRAM-bound: a 16 % cheaper pack, and proof of the wall

Work tree `/home/raspi/work/asm10` (copy of `asm9`).  Board: Pi 3B, stock clocks
(V3D 300 MHz, `get_throttled=0x0`).  Every GPU sequence: one launch per attempt,
a single `flock -w 900 /tmp/vc4-gpu.lock` around the whole sequence, short
`QRUN_TMO_MS` (15-60 s) so a hang is caught immediately, `read_v3d_status` ->
`STATE: LIVE` after every launch.

The GEMM kernel is **unchanged** (`gemm6.py`, and the built kernels are
byte-identical to asm9's — `cmp` clean), so the k-loop numbers cannot regress.
All the work is in the packer plus the harness's work assignment.

## 1. What changed: `pack10.py` (new packer) + grid-stride row-block assignment

Two independent changes, both in the pack pass:

1. **KL = 16-word VDR rows (64 B) instead of 8.**  One memory->VPM command now
   reads a full 64 B line per A row, so every line of A is read exactly once
   (pack8 read each line as two 32 B halves = 2x the read commands) and one
   command feeds two k strips.  The write side stays on the one shape that is
   proven exact on this silicon (`Depth=8 Units=8`, one contiguous 256 B
   vector per command) — the earlier probe showed any other Depth/Units pair
   steps the destination by 64 B and pads, so a KL=16 read feeds **two** VDWs
   (`col=0` and `col=8`).
2. **Grid-stride work assignment instead of contiguous slabs.**  QPU *i* now
   takes row blocks *i, i+pnq, i+2·pnq, …* instead of a contiguous 1/pnq
   slice.  With 4 QPUs the four concurrent read streams are then 8 KB apart
   instead of 64 KB apart, which is worth **17 % on its own** (464 -> 386 µs
   for the identical kernel).  Implemented in `qrun_gemm.c`
   (`QRUN_PACK_STRIDE=1`: `u1 = i*rbsz`, `u5 = units for this QPU`,
   `u7 = rbsz*pnq` as the unit step, which the kernel pops as an 8th uniform).
   The kernel is `USTRIDE=1 python3 pack10.py <N> 16 single`.

Also added (screening only, not adopted): `pair` mode (16 VPM rows/QPU, one
1 KB VDR covering two 8-row panel blocks, 4 VDWs), `_ro` / `_wo` builds that
drop the write / read half, and a `_cp` build whose VDR pitch is 64 B
(contiguous, no transpose) to measure the streaming ceiling.

## 2. Measured: the pack is at the memory-system wall, not at a QPU limit

N=256, pack only, 4 QPUs (unless stated), panel verified bit-exact (`Pmm=0`):

| kernel / probe | pack µs | what it shows |
|---|---|---|
| `pack8` (asm9) | 451.7 / 457.1 / 464.7 / 467.3 | the baseline |
| `pack10` kl=8, contiguous | 450.7 | the rewrite alone changes nothing |
| `pack10` kl=16, contiguous | 464.1 | wider bursts alone change nothing |
| **`pack10` kl=16, grid-stride** | **380.8 / 384.8 / 386.8** (+6 acceptance runs 361-387) | **adopted** |
| `pack10` kl=16 grid-stride, 5 QPUs | 381.4 | more QPUs: no change |
| `pack10` kl=16 grid-stride, 6 QPUs | 387.0 | more QPUs: no change |
| `pack10` pair (1 KB VDR, 16 rows) | 380.8 | 1 KB reads: no change |
| read half only (`_ro`), 4 QPUs | 314.2 | |
| read half only (`_ro`), 6 QPUs | 314.5 | **reads are a shared, hard cap** |
| write half only (`_wo`), 4 QPUs | 371.8 | |
| contiguous copy (`_cp`, no transpose) | 465.7 | the access pattern is not the issue |

Reading of the numbers:

* the combined rate is 512 KB / 382 µs = **1.34 GB/s**, i.e. ~84 % of the
  board's 1.6 GB/s LPDDR2 peak (32-bit, 400 MHz);
* the **read half alone needs 314 µs** (0.83 GB/s) and does not improve with
  4 -> 6 QPUs, i.e. it is an L2-miss/DRAM-read throughput cap, not per-QPU
  latency.  So a *write-free* pack would still cost 314 µs, and no amount of
  extra VPM buffering, QPU count or command size can go below it;
* the write side adds only 66 µs of exposed time on top of the read (380 vs
  314), so the read and write streams already overlap well.

**VPM row budget (measured, matters for any fused design):** 8 QPUs x 8 VPM
rows corrupts the panel (V1: 5 of 8 QPUs' blocks garbage, V2: 1 of 8), while
6 QPUs x 8 rows and 4 QPUs x 16 rows are exact.  The usable window is therefore
**48 rows**, not 64 (`V3D_VPMBASE=16`).  That is why the pack cannot simply use
all 12 QPUs, and why a per-QPU fused pack (which needs 8 consecutive rows per
QPU) does not fit 12 QPUs at all.

## 3. The 12.5 GFLOP/s target is not reachable with a separate pack pass

At N=256 the acceptance arithmetic is: total = gemm + pack, so 12.5 GFLOP/s
needs total <= 2.68 ms, i.e. **pack <= 234 µs**.  The 256 KB read of A alone
takes 314 µs on this board.  12.5 is therefore unreachable for any design that
materialises an A panel with a separate pass — this is a measurement, not a
tuning failure.  The reachable best is 2.446 + 0.382 = 2.83 ms = 11.85-11.97.

The only remaining structural idea is to **fuse the pack into the GEMM and hide
it under the k loop**: the GEMM currently uses only 0.31 GB/s of DRAM
(768 KB / 2.45 ms), so 512 KB of pack traffic could in principle be absorbed
in its slack.  The blocker is the 48-row VPM window vs 8 rows per packing QPU
(12 QPUs = 96 rows).  A variant that fits would need a 4-row transpose unit,
i.e. half-vector (16 B) panel writes via a VDW stride — unverified on this
silicon — plus DMA issue/wait words spliced into the instruction-tight k loop.
That is a new campaign, not a tweak; it was not attempted here because the k
loop must not be perturbed in this run (~and one wrong VPM field is a
board reset on this unit).

## 4. The per-tile prime: measured, and it is not the N=64 problem either

Measured with the existing `SKIP` builds (SKIP=prime / SKIP=loop run the whole
tile body — setup, tile addressing, 32-word accumulator zeroing, the 27-word
prime and the store — but not the k loop), N=64, 12 QPUs, with the pack:

| build | wall | note |
|---|---|---|
| SKIP=prime (no k loop) | 5.7 µs | whole non-loop body |
| SKIP=loop (no k loop) | 4.7 µs | same body, different branch site |
| bare 12-QPU launch (4-word nop) | 5.8 µs | launch overhead |
| full N=64 GEMM | 84.9-86.6 µs | |

So the entire non-loop body is **at or below the launch-overhead noise floor**
(~1 µs of real work): the prime's own instruction cost is 27 words + the
32 zero words = 59 words ≈ 242 cycles ≈ **0.8 µs per tile** (0.9 % of the
86 µs N=64 wall; at N=256, 11 tiles/QPU, 0.36 % of the GEMM).  "The prime is
60 % of the N=64 wall" does not survive measurement.

What the N=64 wall really is: exec 14 k cycles per busy QPU (46.7 µs, 54 %) +
~2.3 k stall + a further ~8 k cycles/QPU of cold-read time.  The panel
(16 KB) and B (16 KB) are **flushed out of L2 by the harness between the two
launches**, so the first k pairs read them from DRAM at the 0.83 GB/s read cap
— that is the ~25-30 µs block, and it is 32 k pairs long at N=64, far too
short to hide it.  Nothing inside the prime can fix that; the fix would be to
not flush L2 between the pack and the GEMM (harness-level, and it would move
the pack's writeback into the GEMM's window, so it was not adopted).

Consequence: the prime was left **unchanged** (all changes to it are worth
<0.3 % and would put the byte-identical k-loop guarantee at risk).

## 5. Files

| file | change |
|---|---|
| `pack10.py` | new packer: KL=8/16 VDR rows, `single`/`pair` modes, per-write or back-to-back VDWs, grid-stride unit step (u7), `_ro`/`_wo`/`_cp` screening builds |
| `pack10_<N>_k16_single_vp1_us.bin` | **adopted** packer (N = 64/128/256), 8 VPM rows/QPU, grid-stride |
| `pack10_*_{ro,wo,cp,now,pair}.bin` | probes: read-only, write-only, contiguous-pitch, no per-write wait, 1 KB reads |
| `qrun_gemm.c` / `qrun_gemm` | `QRUN_PACK_STRIDE=1` grid-stride pack uniforms (u5 units, u7 unit step); GEMM path untouched |
| `g6.py` | default pack bin is now `pack10_<N>_k16_single_vp1_us.bin` and it adds `QRUN_PACK_STRIDE=1` for `pack10*` |
| `accept10.sh` | acceptance: 1-QPU control, 5x N=64/128/256 with the new pack, 3x asm9 pack8 A/B |
| `packtest.py`, `packscreen*.sh`, `screen*.sh` | pack-only timing + `check_panel8` driver, and the probe batches |
| `gemm6.py`, `gemm6_*.bin` | unchanged from asm9 (byte-identical) |
| `gemm6_*_skip{prime,loop}.bin` | diagnostic builds used for §4 |
