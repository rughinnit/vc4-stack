# asm8 — port-aware 6-word k-loop rows (the r3 interleave, corrected)

Work tree `/home/raspi/work/asm8` (copy of `asm7`).  Board: Pi 3B, stock clocks
(`vcgencmd measure_clock v3d` -> 300 MHz, `get_throttled=0x0`).  Every GPU
sequence: one launch per attempt, a single `flock -w 900 /tmp/vc4-gpu.lock`
around the whole sequence, `timeout -k 5 300`, `STATE: LIVE` after each launch.
No reset occurred; the board was up throughout.

## 1. what this variant sets out to do, and why it cannot be built as described

NOTES7 §5 proposed: move `ACC_BADR` to a general register to free accumulator
r3 as a second fmul temp, so two rows' chains interleave, 7 -> 6 words/row.

**Moving BADR out of an accumulator destroys the issue slots.**  The add ALU
has ONE A-side (`raddr_a`) and ONE B-side (`raddr_b`) register port.  With BADR
in an accumulator a B issue with an offset is `add tmu1_s, r3, rb_off` —
raddr_b only — so it can share a word with an fmul whose vector reads an A-file
register (raddr_a).  With BADR in a general register it becomes
`add tmu1_s, ra_badr, rb_off`: BOTH ports, and an fmul always occupies one of
them.  The 8 issues and 2 BADR bumps would then each need a dedicated word: 10
extra words per iteration, more than the 14 words the 7 -> 6 change buys back.

**A second temp is impossible anyway: all six accumulators are pinned.**

| acc | role | why it cannot move |
|-----|------|--------------------|
| r0 | A vector | the v8min rotate only works on an accumulator operand.  Tested: `v8min r1, ra31, imm(50)` with the A vector copied to A-file r31 gives rotations that are only correct modulo 4 lanes (C rows r and r+4 came out identical — period 4 in the row map).  GP source = per-4-lane-group rotate. |
| r1 | rot | the rep reads it in the same word; that is port-free only for an accumulator (a GP read needs raddr_a, which the rot's own A-vector read already occupies). |
| r3 | BADR | see above. |
| r4 | TMU read | every TMU pop, on either TMU, lands in accumulator r4 (ISA). |
| r5 | broadcast | the rep writes it (ISA). |

## 2. What actually buys the 6 words: the register-port pairing rule

A word may hold one fmul and one fadd only if their operands are in OPPOSITE
register files (the fmul reads its B vector, the fadd reads its C accumulator,
one port each).  asm7 has `vector_file(j) = opposite(accum_file(j))`, so only
*same-group* pairs are legal (j0/j1 fmul with j0/j1 fadd) and each group needs
3 words — fmul_a, fmul_b|fadd_a, fadd_b — hence 7 words with the rep.

Swapping the vector registers — j0/j1 vectors in the **A** file (16,17 P /
18,19 Q), j2/j3 in the **B** file — makes `vector_file(j) = accum_file(j)`, so
*cross-group* pairs are legal and 4 fmul + 4 fadd + rep + rot fit in 6 words:

    w0: rot(row+1)   | rep(row r)          w1: fmul(j0) ra | issue / bump / copy
    w2: fmul(j2) rb  | fadd(j0) ra         w3: fmul(j1) ra | fadd(j2) rb
    w4: fmul(j3) rb  | fadd(j1) ra         w5: <A-file copy> | fadd(j3) rb

One temp (r2) is enough: each fadd reads the temp written one word earlier by
the fmul in the *same* word (same-word WAR, the idiom asm7 already relies on).
6 words is the floor: rep -> 4 fmul -> last fadd is 6, and the rep needs a word
of its own (the fmuls read r5 one word later).

Iteration: 104 words instead of 118 (word 0 = A issue, word 1 = first B issue,
16 rows x 6 words, 2 utility words, branch + 3 delay slots).

## 3. Two correctness traps found on the way (each cost a run)

1. **Every TMU pop lands in r4, so the A pop must come after the last B copy.**
   The first build put the A pop at the end of half 1 but *before* the last B
   pop; the A-vector copy then read B data.  Fix: the A pop rides on the AADR
   bump word, after the last B copy, 2 words before the delay-slot copy.
2. **A pop's copy must precede the NEXT pop.**  Re-arranging the pops for more
   cover moved one copy past the next pop, which destroyed exactly one column
   block (`Cmm` = 1/4 of the elements, maxabs 30).  The pops/copies are now at
   30/33, 36/39, 42/43, 48/49 (half 0) and 78/81, 84/87, 90/91, 96/97 (half 1).

## 4. Verification

`gemm5.py` carries a schedule simulator that walks the emitted words and
asserts: raddr_a/raddr_b conflicts (the same register twice is allowed), the
temp chain, broadcast and rot read gaps, A/B-file RAW >= 2 words, TMU FIFO
order per TMU, issue->pop >= 8 words, <= 4 in flight, pop+imm and issue+pop.
All clean for N = 64/128/256.

Acceptance (`accept8.sh`, one flock): 1-QPU control + 5x N=64, 5x N=128,
5x N=256 at 12 QPUs — 15/15 clean — plus 3x asm7 at N=256 as an interleaved
A/B.

## 5. Speed: the issue count falls 15 %, the wall time does not

Per QPU, N=256, 12 QPUs (V3D counters, this run):

| kernel | exec | tmu_stall | idle | total | wall |
|--------|------|-----------|------|-------|------|
| asm7 | 681 k | 273 k | 2.6 k | 954 k | 3.43 ms |
| asm8 | 577 k | 371 k | 2.2 k | 951 k | 3.36 ms |

`exec` is 0.40 cycles/MAC instead of 0.47 (410 vs 484 cycles per k-pair; 1-QPU
cyc/k-pair 429.7 vs 505.6) — exactly the predicted 14 % — but the TMU stall
absorbs all of it.  The bytes moved are unchanged (identical request pattern:
512 B of B + 128 B of A per k-pair = 0.5625 B/MAC, and the L2 hit/miss counts
are the same for both kernels), and the stall tracks requests/bytes, not
instructions.

Two things that did NOT help (measured, then reverted):

* column-block-outer tile order (all 12 QPUs share one 64 KB B block): L2
  misses 109 k -> 83 k and L2 accesses 138 k -> 95 k, but the wall time went
  3.45 -> 3.57 ms — the 12 QPUs then contend on the same lines.
* a 2/2 split of the B requests over TMU0/TMU1 needs an extra word for the A
  issue (the A issue needs raddr_a, so it cannot ride on the w1 word whose
  fmul reads an A-file vector) and the A pop must stay last (r4 aliasing), so
  it does not fit in 104 words; the counter evidence says the stall is
  bytes/latency, not per-TMU queue depth.

The cover (issue->pop distance) is 29/33/33/33 words in half 0 and
27/27/27/27 in half 1.  It is capped by the free add slots (only a row's w1
word can take an issue) and by the copy slots: the last copy must be word 49,
so the last pop is 48.

## 6. Next single experiment

**A third B vector set, with the k loop unrolled over 3 phases.**  The cover is
capped at ~33 words (~130 cycles) while the 12-QPU effective latency is ~200+
cycles, so every B pop is exposed.  With a request issued a full half before
its use the stall should collapse: exec + idle = 442 cycles/iteration
(2.1 ms, ~16 GFLOP/s) versus 707 today.

Registers: the 3rd set needs 4.  Free them by dropping the debug-only A-file
registers RA_COFF/RA_UP (28/29, used only under DBGSTORE) and RA_UB (30, never
used), and moving the colblk mask back to the small immediate `imm(ncb-1)`
(0/1/3, already used by the tile header) so B-file 28 is free too: A 28,29,30 +
B 28.  The vector-file split then has to be re-drawn (3 A-file + 1 B-file per
set), and because the fmul register operand changes per phase the k body must
be triplicated (N/2 = 32/64/128 k pairs; 32 is not a multiple of 3, so the tail
needs a 1- or 2-phase remainder loop, or accept 2 wasted k pairs).

A cheaper but smaller step (about +4 %, probably inside the noise): move the
half-0 pops to 31/38/42/48 with copies 33/39/43/49, which raises the minimum
cover from 29 to 30-33 words.

## 7. Files

| file | change |
|------|--------|
| `gemm5.py` | the 6-word port-aware schedule + the schedule simulator/verifier |
| `gemm5_{64,128,256}.bin` | built kernels (351 words) |
| `qrun_gemm.c` / `qrun_gemm` | + `QRUN_L2=1`: counters 3/4 read L2C hits(28)/misses(29); prints `L2 hits=.. misses=..` |
| `accept8.sh` | the acceptance sequence in one flock |
| `diag8.c` / `diag8` | result-structure analyser (row/col map, k-subset, scale) |
| `NOTES7.md`, `RESULTS7.md` | asm7 (multi-QPU uniform-stride fix) |
