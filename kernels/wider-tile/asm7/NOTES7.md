# asm7 — the multi-QPU "hard reset" was a harness uniform-stride bug, not the kernel

Work tree `/home/raspi/work/asm7` (copy of `asm6`; `asm6` and the engine untouched).
Board: Pi 3B, stock clocks (`vcgencmd measure_clock v3d` -> 300000000,
`get_throttled=0x0`).  Every GPU sequence: one launch per attempt, a single
`flock -w 900 /tmp/vc4-gpu.lock` around the whole sequence, `timeout -k 5 300`,
`read_v3d_status` -> `STATE: LIVE` after each launch.

**Status: FIXED and verified at 12 QPUs.**  The GEMM kernel (`gemm4.py`) is
unchanged from asm6 — it was innocent.  The fault was in `qrun_gemm.c`.

---

## 1. The fault (exact)

`qrun_gemm.c` stores the per-QPU uniform blocks as 9-word rows

    uint32_t uni[12][9];        /* main() */
    uint32_t up[12][9];         /* pack launch */

but `launch_prog()` was declared with an **8**-word row:

    static int launch_prog(const char *path, int nqpu, const uint32_t (*uni)[8], ...)
    ...
        for (int j = 0; j < 9; ++j)
            u[j] = uni[i][j];           /* pointer arithmetic on [8]! */

`uni[i][j]` is `*(*(uni + i) + j)` and `sizeof(uint32_t[8])` is 32 bytes, so QPU
*i* was fed **host words `8i .. 8i+8`** instead of `9i .. 9i+8`.  The GPU-side
uniform blocks are 64 B apart, so this is purely a host-side source-row
misalignment: every QPU except 0 (and every QPU where `8i % 9 == 0`, i.e. i=9)
received another QPU's fields shifted by `i` words.

Regression history: `asm3` had 8-word rows (`uint32_t uni[12][8]`) and a
matching `(*)[8]`, so it agreed.  `asm5` appended `u7`/`u8` (C base and arena
base, to remove the hard-coded bus address) and widened both arrays to `[9]`
but left the parameter type at `[8]`.  From then on every multi-QPU launch was
broken; asm5 never ran it (broken sudo), asm6 hit it.

## 2. Evidence

### 2.1 Printed evidence of the delivered blocks (no launch: `QRUN_DRYRUN=1 QRUN_OLDSTRIDE=1`)

Pack launch (`PACK_QPUS=4`, N=128, `per_qpu=4`), what asm6 actually delivered:

    old_uni[0]: 00000080 00000000 de660000 00001000 de700000 00000004 00000000 de700000 00000000
    old_uni[1]: 00000000 00000080 00004000 de660000 00001000 de700000 00000004 00000001 de700000
    old_uni[2]: de700000 00000000 00000080 00008000 de660000 00001000 de700000 00000004 00000002
    old_uni[3]: 00000002 de700000 00000000 00000080 0000c000 de660000 00001000 de700000 00000004

pack8's u2=A base, u3=row-block stride, u4=panel base, u6=VPM row base index.
So pre-fix the packer ran with

* QPU1: panel base `0x1000`, row-block stride `0xde660000`, VPM row base `16*4 = 64`
* QPU2: A base `0xde700000`, panel base `0x00000080`, VPM row base `16*0xde700000`
* QPU3: row-block stride `0xde660000`, VPM row base `16*4 = 64`

i.e. **VPM DMAs aimed at bus `0x1000` (unrelated low DRAM) and at
`0xde660000 + n*…` (outside the dma-buf), plus VPM row bases of 64 and ~3.7e9**
(the VPM row field is 6 bits / the window is 64 rows, so 64 overflows into the
neighbouring setup fields).  Both sub-classes are documented board-killers on
this unit: asm5 NOTES §4 (a QPU DMA into unrelated DRAM) and asm1/asm2 (a VPM
setup field overflow -> the DMA never completes -> SoC reset).  That is the
hard reset; it needs a power cycle because the ARM has already lost low memory
or the VPM engine is wedged.

GEMM launch (12 QPUs), what asm6 actually delivered:

    old_uni[1]: de600000 00000080 de610000 de660000 de6a0000 de700000 00000001 0000000c de610000
    old_uni[9]: 00000080 de610000 de660000 de6a0000 de700000 00000008 0000000c de610000 de600000

QPU1 got `u1 = 0x80` (C base = N) and `u5 = 0xde700000` (tile0 = panel bus
address) -> the tile-loop guard `t - ntiles` is positive, so the QPU exits
immediately.  QPU9's block is byte-identical to the correct one (`8*9 == 9*8`),
so one QPU ran correctly.  This is why the GEMM side was "benign but wrong"
(0 work or work into the arena) — and why 1 QPU was always correct.

### 2.2 Journal evidence that these were not kernel-timeouts

`journalctl --list-boots`: the four pre-fix boots (-4..-1) each ended ~2 min
after boot with **no kernel message at all** (last line = an ssh session start),
i.e. an abrupt loss of the ARM, not a GPU hang, panic or OOM.  Consistent with
a QPU DMA writing into low DRAM / a wedged VPM engine.

### 2.3 Reproduction of the arithmetic (host-side, no board)

    QPU i reads host words 8i..8i+8:   QPU1 -> [u8, u0, u1, ..., u7] of row 1
    expected:                          QPU1 -> [u0, u1, ..., u8] of row 1

## 3. The fix (`qrun_gemm.c`, asm7)

    #define UNI_WORDS 9
    static int launch_prog(..., const uint32_t (*uni)[UNI_WORDS], ...)
        for (int j = 0; j < UNI_WORDS; ++j) u[j] = uni[i][j];
    ... (const uint32_t (*)[UNI_WORDS])uni / up

plus two zero-cost diagnostics used for the evidence above:
`QRUN_DUMP_UNI=1` (print the 9 words each QPU is actually given) and
`QRUN_DRYRUN=1` (copy uniforms, launch nothing).  `qrun_gemm` rebuilt with
`gcc -O2 -Wall`.  No change to `gemm4.py`, no change to any kernel binary.

## 4. Verification (12 QPUs, one launch per run, single flock)

Accepted: **5/5 clean at N=64 and 5/5 clean at N=128**, 0 mismatches in C *and*
in the packed panel, `v3d_errors=0x1040`-free (`0x1000` = the normal VCD-idle
bit only), `SRQCS=0x000c0c00` (queue 0, no err, **done=12**).

| N | run | pack ms | gemm ms | total ms | GFLOP/s gemm | cyc/k-pair | Cmm | Pmm | v3d_errors |
|---|-----|---------|---------|----------|--------------|-----------|-----|-----|------------|
| 64 | 1 | 0.0045 | 0.1157 | 0.1202 | 4.530 | 350.2 | 0 | 0 | 0x1000 |
| 64 | 2 | 0.0057 | 0.1161 | 0.1218 | 4.516 | 350.2 | 0 | 0 | 0x1000 |
| 64 | 3 | 0.0053 | 0.1163 | 0.1216 | 4.508 | 350.2 | 0 | 0 | 0x1000 |
| 64 | 4 | 0.0042 | 0.1162 | 0.1204 | 4.512 | 350.2 | 0 | 0 | 0x1000 |
| 64 | 5 | 0.0047 | 0.1164 | 0.1211 | 4.506 | 350.2 | 0 | 0 | 0x1000 |
| 128 | 1 | 0.0868 | 0.4893 | 0.5761 | 8.572 | 450.1 | 0 | 0 | 0x1000 |
| 128 | 2 | 0.0841 | 0.4966 | 0.5807 | 8.447 | 450.1 | 0 | 0 | 0x1000 |
| 128 | 3 | 0.0860 | 0.4955 | 0.5815 | 8.465 | 450.1 | 0 | 0 | 0x1000 |
| 128 | 4 | 0.0835 | 0.4923 | 0.5758 | 8.519 | 450.1 | 0 | 0 | 0x1000 |
| 128 | 5 | 0.0833 | 0.4932 | 0.5765 | 8.504 | 450.1 | 0 | 0 | 0x1000 |

Control runs:

| run | result |
|-----|--------|
| 1 QPU, N=128 | Cmm=0 Pmm=0, 3.5650 ms, **505.6 cyc/k-pair** (identical to the asm6 number -> no regression) |
| 2 QPUs, N=64, pack at 2 QPUs | Cmm=0 Pmm=0, 0.2853 ms, 520.2 cyc/k, `done=2`, LIVE |
| 12 QPUs, N=256 | 3.2932 / 3.3273 / 3.3878 ms gemm -> 10.189 / 10.085 / 9.905 GFLOP/s; total 3.7533 / 3.7925 / 3.8456 -> 8.940 / 8.848 / 8.725 |

## 5. Speed: 1.43x per QPU at 1 QPU does not survive 12 QPUs (and why)

QPU scaling at N=256, same code, same session:

| QPUs | gemm ms | GFLOP/s | efficiency |
|------|---------|---------|-----------|
| 1 | 27.794 | 1.207 | — |
| 2 | 14.315 | 2.344 | 97 % |
| 4 | 9.095 | 3.689 | 76 % |
| 6 | 6.233 | 5.384 | 74 % |
| 12 | 3.456 | 9.710 | 67 % |

The counters (`qrun_gemm` now maps type 17/13/21 as well) say exactly where the
missing 33 % is (12 QPUs, N=256, per QPU): exec_cycles 681 k (2.27 ms) +
**tmu_stall 274 k (0.91 ms)** + idle 31 k (0.10 ms) = 986 k = 3.29 ms wall.
At 1 QPU the TMU stall is 0.6 %; at 12 QPUs it is 26 % of the wall.

`exec_cycles` is already at the issue floor: 118 words x 4.1 cycles = 484
cycles per k-pair, i.e. the k loop itself is 25 % faster per MAC than asm3
(0.47 vs 0.63 cycles/MAC) and 2 QPUs scale perfectly.  The stall is pure TMU
latency: each B vector is issued at `w4` of rows 0..3 of a half and popped 4
rows later (~112 cycles of cover), while at 12 QPUs the effective TMU latency
is ~136 cycles.  asm3 does not have this because its pops sit a full
39-word iteration (156 cycles) after the issue.

Closing that needs 4 B register sets (only 2 exist; the B file has no free
registers) or a re-schedule with a second mul temp.  Next single experiment,
in order of expected value:

1. Free accumulator `r3` (`ACC_BADR` -> a B-file register) as a second `fmul`
   temp so two rows' `fmul->fadd` chains can interleave: 7 words/row -> 6
   (4 fmul + 1 rot = 5 mul slots, 4 fadd + 1 rep = 5 add slots, chain needs 6)
   => ~14 % fewer issue cycles if the stall does not grow => ~11.5 GFLOP/s.
   Encode with `AUDIT=1 LIST=1`, verify 1 QPU first (safety: BADR feeds a TMU
   *read* address only, so a mistake corrupts data, it cannot write DRAM).
2. Then, if the stall grows, split the 4 B requests 2/2 over TMU0/TMU1 with the
   A pop moved to half 0 (it only has to precede the branch-delay-slot copy) so
   TMU0 stays <= 4 in flight, and pop one half later (224 cycles of cover).

**Acceptance status**: correctness bar met 5/5 + 5/5.  Speed bar *not* met:
best GEMM-only 10.189 vs 10.55 GFLOP/s (3.5 % short), total 8.940 vs 9.48
(5.7 % short), at N=256.  Nothing in the reset fix cost speed — 1-QPU
cyc/k-pair is bit-identical at 505.6 and the 12-QPU numbers are limited by the
TMU stall described above, not by the harness fix.

## 6. Files

| file | change |
|------|--------|
| `qrun_gemm.c` / `qrun_gemm` | **the fix** (`UNI_WORDS`, `QRUN_DUMP_UNI`, `QRUN_DRYRUN`, `QRUN_OLDSTRIDE`) + stall/idle/I-cache counters |
| `measure7.sh` | one launch per QPU count, prints counters, C and panel check |
| `accept7.sh` | the acceptance sequence (5x N=64, 5x N=128, 1-QPU control, 3x N=256) in one flock |
| `gemm4.py`, `g6.py`, `*_*.bin` | unchanged from asm6 |
