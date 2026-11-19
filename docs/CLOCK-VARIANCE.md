# Board performance variance: check the core clock before comparing runs

The SoC core clock (which drives L2 and the memory path) is not fixed at 400 MHz on
this board. Under sustained 12-QPU load it was observed to drop to 250 MHz while
V3D stayed at 300 MHz and get_throttled reported 0x0. The effect is a ~1.6-1.7x
wall-time increase in memory-bound kernels with *identical* cycle counts - exactly
the discrepancy seen between runs recorded at different times.

Rule: every comparison must record `vcgencmd measure_clock v3d` and
`vcgencmd measure_clock core`, plus the cycle counters, not just wall time.
If core reads 250, idle until it returns to 400 before measuring.

## 2026-11-19: external fan fitted -- no droop in 89/89 samples

Setup: 10 minutes of sustained mixed load on one boot, stock clocks: a
`make clean && make -j4` rebuild loop of the engine (19 full rebuilds)
concurrent with a continuous 256^3 matmul loop on the GPU, every GPU job
serialised by `/tmp/vc4-gpu.lock`, sampled every ~7 s.

| quantity | result |
|---|---|
| SoC core clock | **89/89 samples at 400 MHz**; no 250, 275 or 300 MHz sample |
| V3D clock | 89/89 at 300 MHz |
| temperature | 52.1 -> 61.2 C, mean 59.1 C |
| core voltage | 1.3188 V, constant (same value as the drooping trace) |
| `get_throttled` | `0x0` in all 89 samples |
| load average | 2.8 - 6.7 |
| memory | MemAvailable 60-560 MiB; swap peaked at 731 MiB from a concurrent compile, no OOM |

Two follow-up probes, each one continuous GPU process of 4000 matmuls
(~16 s of QPU work) with the core clock sampled at 1 Hz (32 samples):

| probe | core clock | ms/rep |
|---|---|---|
| GPU only, ARM idle | 32/32 at 400 MHz | 3.95 (steady) |
| GPU + 4 ARM spinner threads | 32/32 at 400 MHz | 5.23 (steady) |

**Verdict: the droop did not occur in this run, with the fan fitted.**
It is *not* a die-temperature threshold: this run held 400 MHz at 59-61 C,
hotter than the 55 C of the original drooping trace, with the same 1.3188 V and
no throttle bits.  That points at power/board-level headroom (the fan also cools
the PMIC and the board, not just the SoC) or a firmware DVFS decision, rather
than the temperature sensor.  With one board and no fan-off A/B the honest
statement is "gone with the fan fitted, and not explained by die temperature".

**Caveat that cost an hour to find: 250 MHz is also the idle clock.**  With no
GPU work the core settles at 250 MHz after ~4 s (and returns to 400 MHz later);
a single `measure_clock core` taken *between* two GPU jobs therefore looks
exactly like the old droop.  Always sample while the work is in flight, and
prefer the cycle counts (which are load-independent) over one clock reading.
An earlier attempt in this run did exactly that: a GPU process that had
failed at startup (engine `kgen` looks for `kcache/` relative to the *cwd*, so
running it from the wrong directory is a no-op) produced a convincing 18 s
"droop" at 43 C that was simply an idle board.

What remains under sustained load is ARM contention, with the core at 400 MHz:
the compiled VC4CL GEMM loses ~25 % (3.085 -> 2.31 GFLOP/s; its launch/wait path
is ARM-driven), the hand-written asm path ~7 % (11.85 -> 11.20 GFLOP/s) and the
engine ~8 % (8.50 -> 7.80 GFLOP/s).  With four pure ARM spinners the engine's
polling path loses 32 % (3.95 -> 5.23 ms/rep).  So compare performance numbers
between runs with the same background load -- but the "same binary, half the
speed" clock effect is no longer part of that variance.
