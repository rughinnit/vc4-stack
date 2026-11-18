# asm9 results — 3-phase k loop kills the TMU stall: 13.7 GFLOP/s GEMM, both bars beaten

Work tree `/home/raspi/work/asm9`.  Board: Pi 3B, stock clocks (V3D 300 MHz,
`get_throttled=0x0`), no reset during the session.  One launch per attempt,
one `flock -w 900 /tmp/vc4-gpu.lock` per sequence.

## Verdict: KEEP

* 15/15 clean (5x N=64, 5x N=128, 5x N=256 at 12 QPUs) plus the 1-QPU control;
  `Cmm=0 Pmm=0`, `SRQCS=0x000c0c00` (done=12, error=0), `STATE: LIVE` on every
  run.
* N=256: **2.446 ms GEMM -> 13.72 GFLOP/s GEMM-only, 11.65 GFLOP/s total**,
  i.e. **+38 %** over asm8 measured in the same session (3.386 ms / 9.91) and
  well past both asm3 bars (10.55 GEMM-only / 9.48 total) and the 12 GFLOP/s
  stretch target (GEMM-only 13.72).
* The TMU stall drops 13.4x: 362 k -> 27 k cycles/QPU, and the kernel becomes
  instruction-count bound (4.1 cycles per instruction, measured).

## Verified (12 QPUs, stock clocks, 15/15 clean)

| N | runs | gemm ms (range) | total ms (range) | GFLOP/s gemm best/median | GFLOP/s total best | exec cyc/k |
|---|------|-----------------|------------------|--------------------------|--------------------|------------|
| 64 | 5 | 0.0846-0.0869 | 0.0917-0.0948 | **6.199** / 6.142 | **5.720** | 291.8 |
| 128 | 5 | 0.3963-0.4112 | 0.4827-0.4992 | **10.584** / 10.545 | **8.690** | 372.1 |
| 256 | 5 | 2.4459-2.4520 | 2.8806-2.9134 | **13.719** / 13.712 | **11.649** | 398.5 |

Control: 1 QPU N=128 -> `Cmm=0 Pmm=0`, 3.2441 ms, **417.9 exec cyc/k-pair**
(asm8: 429.7, asm7: 505.6).

Mismatches: 0 on every run, `nonzero` as expected (4096/16384/65536), maxabs
1.96e-05 / 4.82e-05 / 1.56e-04 (identical to asm8, same data).

## Interleaved A/B against asm8, same session, same flock (N=256, 12 QPUs)

| kernel | run 1 | run 2 | run 3 | gemm best | gemm median |
|--------|-------|-------|-------|-----------|-------------|
| asm9 gemm ms | 2.4460 | 2.4459 | 2.4520 | **2.4459** (13.719) | 2.4471 (13.712) |
| asm8 gemm ms | 3.3858 | 3.3968 | 3.3669 | 3.3669 (9.966) | 3.3858 (9.910) |

In the same session that is **+38 % throughput** (-27.7 % time).  Against
asm8's best run ever recorded (3.2637 ms / 10.281 GFLOP/s) it is still
**+33 %** (-25 % time), and against the asm3 GEMM-only bar (3.18 ms / 10.55)
**+30 %** (-23 % time).

## Where the time goes (N=256, 12 QPUs, per QPU, per k pair)

Counters are totals over the 12 QPUs; divided by 12 QPUs and by 1408 k pairs
(11 tiles x 128 k pairs, the busiest QPU).

| kernel | exec | tmu_stall | idle | icache_miss | wall cycles/k pair |
|--------|------|-----------|------|-------------|--------------------|
| asm8 | 577 k (410.1/k) | 362 k (256.9/k) | 43.9 k (31.2/k) | ~0 | ~707 |
| asm9 | 561 k (398.5/k) | **27.0 k (19.2/k)** | 21.5 k (15.3/k) | 0.2/k | ~521 |

The stall fell 13.4x and now costs 2.1 cycles per request instead of 28.6.  The
covers that did it (issue -> pop): B j0 57 words, j2 63, j1 69, j3 69, A 60 —
i.e. ~234-283 V3D cycles, versus 27-45 words (110-185 cycles) in asm8.  The
12-QPU effective latency (~250 cycles) is now inside the cover, and one A
request is in flight at a time.

Sanity check of `B/MAC`: identical to asm8 (0.5625 B/MAC, same request pattern:
512 B of B + 128 B of A per k pair), and the L2 statistics are unchanged — so
the win is latency coverage, not traffic.

## Why exec is now the whole story (measured, not assumed)

Three 1-QPU probe loops of known instruction count (`probe9.py`, u0 = 2000):

| body | words/iter | exec cycles | cycles/instruction | tmu_stall |
|---|---|---|---|---|
| 100 nops | 102 | 840024 | 4.12 | 0 |
| 100 independent `or`s | 102 | 840024 | 4.12 | 0 |
| row-like rep/fmul/fadd | 50 | 424024 | 4.24 | 0 |

One 64-bit instruction costs ~4.1 cycles of this board's V3D clock regardless
of the mix, dependency or fetch.  So: 101 words/k pair x 4.1 = 414 cycles,
measured exec 398.5.  exec is 91 % of the wall, stall 4 %, idle 4 %.

## Keep / reject

* **Keep** `gemm6.py` and `gemm6_*.bin`: the 3-phase loop, the A-boundary word
  budget, and the REM tail.  13.72 GFLOP/s GEMM-only is the new best on this
  board and the last block of asm8's stall is gone.
* **Keep** `sim9.py` — the functional request-tag simulation.  It found the
  A-request FIFO-position bug and the 1-word rot->rep gap on the loop-exit
  path, both before any GPU run.
* **Keep** the extended verifier rules (r4-copy-before-next-pop, A-copy->rot
  >= 2, raddr port conflicts) — the raddr check caught the first A-issue slot.
* **Keep** `g6.py` printing `stall=/idle=/icmiss=`, and `probe9` (it settles
  the "is it TMU or issue" question for any future kernel).
* **Reject** nothing here; asm8's `gemm5.py` stays as the 2-set reference (it
  is still the right kernel for a single QPU or a latency-limited variant where
  code size matters).
* **Not worth doing**: any further request pipelining (stall is 19 cycles/k
  pair), moving pops later, or a fourth set.  The next real lever is the
  0.44 ms pack pre-pass (18 % of the N=256 total) or the 41-word per-tile prime
  (60 % of the N=64 wall), not the k loop.
