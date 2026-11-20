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

## It came back, and the cause was firmware DVFS (2026-11-20, second occurrence)

A "same binary, half the speed" effect reappeared: `mm 256^3` measured
**3.73 GFLOP/s** where the recorded best is 8.40, with `throttled=0x0`,
1.3125 V and 43-45 C.  Sampling the clock *during* the load explained it:

```
core=400000000 thr=0x0 temp=43.5
core=400000000 thr=0x0 temp=44.0
core=400000000 thr=0x0 temp=43.5
core=250000000 thr=0x0 temp=42.9   <- after ~4 s of QPU load
core=250000000 thr=0x0 temp=42.9
```

The core drops **400 -> 250 MHz** mid-benchmark, at a die temperature that is
not the cause.  On this board the firmware's DVFS decides the core clock from
the V3D's activity, and **our QPU work goes through `vc4d`, which the firmware
cannot see**: the V3D is idle while the QPUs are saturated, so the core is
downclocked to its minimum.  The Python/VC4CL paths show the same thing
whenever their GL/V3D component is idle.

`/boot/firmware/config.txt` had **no** `core_freq` line, so the clock was
firmware-managed.  It now pins the **stock** frequency:

```
# Pin the core (QPU) clock to its stock 400 MHz. The firmware DVFS cannot
# see QPU work issued through vc4d, so it drops the core to 250 MHz in the
# middle of a benchmark. This is the stock maximum, not an overclock.
core_freq=400
core_freq_min=400
```

`core_freq_min` is the line that matters: `core_freq` alone still allowed the
drop after ~4 s.  Neither value is above the board's stock 400 MHz and no
voltage was changed -- this is a *floor* at the stock maximum, not
overclocking.

Verified after a reboot:

| check | before | after |
|---|---:|---:|
| core clock, sustained 400-rep load | 250 MHz after ~4 s | **400 MHz in 5/5 samples** |
| `mm 256^3` | 3.73 GFLOP/s (8.99 ms) | **8.47 GFLOP/s (3.96 ms)** |
| YOLOv8n 160 | 767 ms / 1.30 FPS | **487 ms / 2.05 FPS** |
| temperature under load | 42.9 C | 44.5-46.2 C |
| `throttled` | 0x0 | 0x0 |

**Consequence for the record:** any measurement taken on this board between
the previous reboot and this fix may have been at 250 MHz without saying so.
The numbers in `docs/COMPLETENESS.md`, `docs/ONNX.md` and the release notes
should be read with the clock check in mind; `vc4infer util` and `vc4-smi`
print the clock for exactly this reason.  The lesson is the same as the first
occurrence: **record the clock with every number, and pin it before comparing
two runs.**
