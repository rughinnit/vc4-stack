# asm9 — three B vector sets, 3-phase k loop: the TMU stall is gone

Work tree `/home/raspi/work/asm9` (copy of `asm8`; only `gemm6.py` is new, plus
`sim9.py`, `probe9.py` and the `g6.py` counter print).
Board: Pi 3B, stock clocks (`vcgencmd measure_clock v3d` -> 300000000,
`get_throttled=0x0` throughout).  Every GPU sequence: one launch per attempt, a
single `flock -w 900 /tmp/vc4-gpu.lock` around the whole sequence, `timeout -k
5 300`, `read_v3d_status` -> `STATE: LIVE` after every launch.  No reset; the
board was up across the whole session (uptime 1:15 at the end).

## 1. What changed: one k-pair of request lookahead

asm8 issued a B request in one half and popped it in the same half, i.e. 57-69
words of issue->pop (measured 27-45 in asm8 itself), against a 12-QPU effective
TMU latency of ~250 cycles (see asm8 RESULTS §5).  asm9 gives every request a
**full k pair** of lookahead: three B vector sets, and a k loop unrolled over
three k pairs (six halves), so

    phase p (48 words):   use      set u(p)   = p mod 3
                          pop+copy set u(p+1)  (issued at phase p-1)
                          issue    set u(p+2)  (popped at phase p+1, used at p+2)

The A vector is issued one k pair ahead too (in the k-even phase, popped at the
next k-pair boundary).

Why the body has to be six halves
---------------------------------
The A vector is accumulator r0 and `v8min` can only rotate an accumulator, so
the A vector can never be double buffered: it changes once per k pair, at a
fixed point of the loop body.  That makes the body a whole number of k pairs (2
phases each).  A rotating fill distance of two phases needs 3 sets (at any
instant: one in use, one ready, one being filled), and the set index is
(p mod 3), so the body period must be a multiple of 3.  lcm(2, 3) = 6 halves.

Per iteration the body is 303 words for 3 k pairs = **101 words/k pair**
(asm8: 104), because the 5-word A boundary is amortised over two k pairs and
the back-edge A change rides in the branch delay slots.

### Per-phase word schedule (48 words, row r = words 6r..6r+5)

| word | mul slot | add slot |
|---|---|---|
| w0 | rot(row r+1), or the boundary | rep(row r), ws=1 |
| w1 | fmul(j0) | B issue (rows 0-3) / copy / BADR bump |
| w2 | fmul(j2) | fadd(j0) |
| w3 | fmul(j1) | fadd(j2) |
| w4 | fmul(j3) | fadd(j1)  (+ TMU pop) |
| w5 | copy from r4 | fadd(j3) |

**issues** j0,j2,j1,j3 at w1 of rows 0..3 (TMU0, TMU1, TMU0, TMU1);
**BADR bump** at w1 of row 5; **pops** j0 w4r1, j2 w4r3, j1 w4r5, j3 w4r6;
**copies** j0 w5r1 (mul slot), j2 w1r4, j1 w5r5 (mul slot), j3 w1r7 — each
after its own pop and before the next pop, because *every* pop lands in r4.
Row 6 uses the other valid alternating fmul order (j2,j0,j3,j1) so its w1 mul
reads a B-file vector and raddr_a stays free for the A issue in the same word.

### The k-pair boundary (A vector change), 5 words

    B0: A pop (sig)         | AADR += 64        (raddr_b = RB_O64)
    B1: r0 <- r4 (mul)      | -
    B2: pad                 | -
    B3: rot(next phase r0)  | (mul_b = small immediate)
    B4: k counter (sub imm 1, sf=1)             [between phases 3 and 4]

Back edge: `[p5 w4r7: A pop] [branch] [d1: r0<-r4] [d2: AADR+=64] [d3: rot(p0
r0)]`, then the one-word prologue (which keeps rot->rep >= 2 across the back
edge).  Cost: 4 words per iteration.

### Register budget: how the third set fits (12 vectors)

Freed without touching any hot port path:

| register | was | now |
|---|---|---|
| B28 | `RB_NCB1` | column-block mask is the small immediate `imm(ncb-1)` (0/1/3) |
| B30 | `RB_UB` (B base) | moved to an A-file register (A30); its only uses are `add S3, UB, r3` = raddr_a + accumulator, so the ports are unchanged |
| A28/A29/A30 | `RA_COFF`, `RA_UP`, `RA_UB` (debug only) | vectors |

That leaves 6 A-file + 6 B-file vectors = three sets of four, each 2 A-file +
2 B-file (forced: C(j0),C(j1) live in the A file and C(j2),C(j3) in the B file,
so v(j2),v(j3) must be B-file, v(j1) A-file, and v(j0) A-file because the B
issues ride on the w1 word whose fmul is j0).  `DBGSTORE` builds are not
carried over (asm8/gemm5.py keeps that tooling).

### TMU occupancy (never above the FIFO depth of 4)

TMU0: A requests + j0/j1; TMU1: j2/j3.  Per phase each TMU issues 2 and pops 2
one phase later, and TMU0 also carries exactly one A request, so the peak is

    TMU0: 2 carried + 1 new = 3, +1 A  = 4
    TMU1: 2 carried + 2 new          = 4

with `<= 4` enforced by the verifier at every word.  Covers (issue -> pop):

    B j0 57 words (234 cyc)   j2 63 (258)   j1 69 (283)   j3 69 (283)
    A    60 words (246 cyc)   in flight: exactly 1

### Tail: KPAIRS % 3

N/2 is 32/64/128, so `NBODY = KPAIRS // 3` = 10/21/42 full bodies and a
remainder of 2/1/2 k pairs.  The remainder is a build-time constant:

* REM = 1 (N=128): two phases that only consume — the last body prefetched set
  0 and set 1 and the A vector for exactly this k pair, so nothing is issued
  and the FIFO drains to empty by construction;
* REM = 2 (N=64, N=256): four phases that consume the prefetched sets and fill
  set 2 (at the first phase) for their second k pair; the last two phases issue
  nothing, so the FIFO is again empty at the exit.

No drain loop, no wasted k pairs, no zero page.

## 2. Two bugs the simulator caught before any GPU run

`sim9.py` walks one tile (setup -> prime -> k loop -> tail) with the real
control flow (branch + 3 delay slots) and tags every request with the address
the issue actually computes: B -> (k, column block), A -> k pair.  It then
checks that every phase uses all four column blocks of one k, that the k
sequence in program order is exactly 0..N-1, that every row's broadcast and A
vector carry the A tag of k//2, and that the FIFO ends empty.

1. **The A request was in the wrong FIFO position.**  Priming `A(kp1)` before
   the loop put it ahead of phase 0's B issues, so phase 1's pops received the
   A data (and shifted every later pop by one).  The A issue now rides in the
   k-even phase's w1r6, i.e. *after* that k pair's B issues and before the next
   phase's, which is exactly the pop order.  (The first attempt also put it on
   a w1r6 whose fmul read an A-file vector — the verifier's raddr_a conflict
   check caught that, which is why row 6 uses the B-first fmul order.)
2. **The loop-exit path had rot -> rep = 1 word.**  The branch delay slot
   writes the rotate result and the tail's first word read it.  One pad word at
   the tail entry fixed it.

Both were found offline; the first GPU build was already 0-mismatch.

## 3. Verification summary

* `gemm6.py`: encoder checks + the schedule simulator (ports, temp chain,
  broadcast, rot->rep, A-copy->rot, A/B-file RAW >= 2, per-TMU FIFO/order,
  issue->pop >= 8, depth <= 4, pop+imm, issue+pop, and a new check that every
  pop is copied out of r4 before the next pop) — clean for N = 64/128/256.
* `sim9.py`: functional tag simulation — clean for N = 64/128/256 (64/128/256
  phases, k order 0..N-1 exactly, FIFO empty at the end).
* `accept9.sh` (one flock): 1-QPU control + 5x N=64, 5x N=128, 5x N=256 at 12
  QPUs, 15/15 clean (`Cmm=0 Pmm=0`, `SRQCS=0x000c0c00` done=12 error=0,
  `v3d_errors=0x1000`, `STATE: LIVE`) + 3x asm8 interleaved A/B.

## 4. Why 4.1 cycles per instruction — the new ceiling

`probe9.py` runs three 1-QPU loops of known instruction count (100 nop words;
100 fully independent `or`s; asm9's own 6-word row x 8) with u0 = 2000
iterations:

| body | words/iter | exec cycles | cycles/instruction | stall |
|---|---|---|---|---|
| nops | 102 | 840024 | 4.12 | 0 |
| independent ors | 102 | 840024 | 4.12 | 0 |
| row-like (rep/fmul/fadd) | 50 | 424024 | 4.24 | 0 |

So on this board one 64-bit instruction costs ~4.1 V3D cycles whatever it is,
with no dependency or fetch component (`icache_miss` 8-14 for 200k
instructions).  exec_cycles/300 MHz reproduces `wall_ns` exactly, i.e. the
counter is in the V3D clock domain and the QPU retires one instruction per four
V3D cycles.  Consequences:

* the kernel is now **pure instruction-count bound**: 101 words/k pair ->
  398.5 exec cycles/k pair, and exec is 91 % of the wall;
* the remaining TMU stall (19 cycles/k pair) and idle (15) together are 8 % of
  the wall, so no further request pipelining is worth doing;
* the only lever left is fewer instructions per MAC.

## 5. Next single experiment

**A 5-word row.**  The row is 8 ALU ops (4 fmul + 4 fadd) + the rep; dual issue
puts the floor at 5 words (rep, then 4 words each holding an fmul and an fadd
from opposite register files), and asm8/asm9 reach 6 because there is only one
mul temp (r2) and all six accumulators are pinned (asm8 NOTES §1).  If the
fadd could read its accumulator from the *accumulator* file instead of a GP
register, every word would be port-free; on this ISA only r0-r5 are
accumulators.  The cheapest untried variant is to give up the last fadd's
pairing (w5 = fadd only) and instead put two fmuls in one word with two temps
(r2 and r3, moving BADR back to a GP register and paying 5 extra words per
phase for the offset issues) — that is 5 words/row + 5 issue words = 85 words
per 8-row phase vs 48 today, i.e. 1.06 words/row-MAC worse, so it is not worth
it.  A 5-word row needs a *new* accumulator-like operand path, which this
silicon does not have.  Therefore the realistic next step is not the row but
the two things around it:

1. **the pack pre-pass** is 0.44 ms of the 2.89 ms total at N=256 (18 %); a
   fused single-launch pack (or packing B only once per k block) is worth more
   than any remaining kernel change;
2. **the per-tile prime** is 41 words + ~2 phases of latency for N=64, where
   the fixed overhead is 60 % of the wall.

## 6. Files

| file | change |
|---|---|
| `gemm6.py` | three B sets, 6-half body, A boundary in 5 words / delay slots, REM tail, extended verifier |
| `gemm6_{64,128,256}.bin` | built kernels (754 / 653 / 754 words) |
| `sim9.py` | functional request-tag simulation of one tile |
| `probe9.py`, `probe9_*.bin`, `probe9.sh` | the instruction-issue-rate probe |
| `accept9.sh` | acceptance sequence in one flock |
| `g6.py` | now prints `stall= idle= icmiss=` from the CTR line |
| `NOTES8.md`, `RESULTS8.md`, `gemm5.py` | asm8 (the 6-word row) |
