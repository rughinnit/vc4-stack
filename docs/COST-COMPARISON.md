# The cost of a delivered frame: QPUs versus the four ARM cores

The question this page answers is narrow: **YOLOv8n inference is about six
times slower on the VideoCore IV QPUs than on the four ARM cores at 4 threads,
so is that latency worth paying because the QPU path frees the CPU for other
work?**  The answer is measured as CPU seconds per *delivered frame* at equal
frame rate, not as average CPU share during a run.  The two readings disagree,
and the disagreement is the whole result.

On the build this page was written around, vc4infer consumed **more** CPU per
delivered frame than ncnn at 3 threads, and no less than ncnn at 4 threads with
passive waiting: 0.255–0.313 CPU-s per frame against 0.157–0.265.  The
impression that the QPU path frees the CPU came from comparing different frame
rates: a VC4 frame is idle for most of its 0.4–0.5 s, while ncnn uses all four
cores for 0.05–0.09 s and then sleeps.  Per second the QPU path leaves three
cores free; per frame it did not.

Two host-side changes followed from breaking that 0.313 s down by phase (see
[CPU-BREAKDOWN.md](CPU-BREAKDOWN.md)): one launch per concat input instead of
one per 16-channel block, and a per-run node plan in the Python executor.  They
remove 24–29 % of the CPU and 16–21 % of the wall time.  After them the QPU
path uses **0.193–0.241 CPU-s per frame** and 2.31–3.16 FPS.  At equal delivered
FPS that is below ncnn at 4 threads with passive waiting at 128 and 160, level
with ncnn at 3 threads at 160, and still 5–7 % above both at 96.  Both row sets
are below; §2 and §3 label which is which.

## 1. Conditions

| | |
|---|---|
| board | Raspberry Pi 3B (BCM2837), VideoCore IV, 905 MiB RAM, fan fitted |
| kernel | 6.18.50+rpt-rpi-v8, Debian 13 arm64 |
| GPU clock | VideoCore core 400 MHz throughout (`vcgencmd measure_clock core`) |
| ARM clock | **1.0 GHz**, governor `ondemand`, `scaling_min_freq = scaling_max_freq = 1.0 GHz` |
| thermal guard | `tools/thermal-guard.sh --max 75 --interval 1`, every cell |
| peak temperature | 53.2 °C over the suite; phase-A cells 47.2–51.5 °C; the after cells 44.0–50.5 °C |
| throttling | `throttled=0x50000` sticky (under-voltage and throttling occurred before the suite); current bits clear, `in0_lcrit_alarm=0` |
| GPU client | one at a time; `vc4d` 96 MiB arena on `/run/vc4d.sock`, restarted before each cell |
| engine | vc4infer 1.2.10 from this tree, daemon backend, `VC4_OPT=all`; the after rows add the two host-side changes in [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) |
| CPU engine | ncnn 1.0.20261020 wheel, 4 threads and a 3-thread control |
| model | official Ultralytics `yolov8n.pt`, same checkpoint at 96, 128 and 160 |

**The ARM clock is 1.0 GHz, not 1.2 GHz.**  At 1.2 GHz the board logged
under-voltage at 255 s and 312 s after boot and the firmware dropped the ARM
clock to 600 MHz under 4-thread inference, which would silently change every
CPU-second in this page.  1.0 GHz is the highest cap tested that held 1.0 GHz
with no new under-voltage event; the two events below 1.2 GHz are the only two
in `dmesg` for the run.  Numbers here are therefore about 17 % slower in wall
time than the 1.2 GHz figures in [BENCHMARKS.md](BENCHMARKS.md), and relative
comparisons are unaffected.

The governor is `ondemand` with the minimum pinned to the same 1.0 GHz.  Left
free, `ondemand` drops to 700 MHz between the GPU waits of a QPU frame (the
client is blocked, so the load looks idle) and client CPU changes by about 20 %
between repeats.  Pinning the floor keeps the DVFS decision out of a
CPU-second measurement; the 1.0 GHz cap is what the supply allows, and that is
the constraint this page reports.

Each ncnn size uses its own PPNX export from that checkpoint.  The packaged
160 export is static: fed a 96 or 128 tensor it still returns the 160 grid
(44,100 values), so a size-mismatched ncnn number would not be an inference of
the smaller graph.  The 96 and 128 exports return 15,876 and 28,224 values as
expected, and the engine returns the same counts.

## 2. Unconstrained: wall, CPU and memory

Three warm-ups, then 30–50 timed runs for ncnn and 10 for vc4infer, median of
per-run wall.  CPU is `utime+stime` from `/proc/<pid>/stat` around the timed
loop; for vc4infer it includes both the client and the `vc4d` process that
launches the kernels.  Share is that CPU time divided by wall and by four.

| runtime | imgsz | wall ms | FPS | client CPU-s/frame | daemon CPU-s/frame | total CPU-s/frame | share of 4 cores | peak RSS MiB | arena / pool / peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ncnn 4t | 96 | 53.2 | 17.42 | 0.219 | — | 0.219 | 95.3 % | 93.1 | no arena API |
| ncnn 4t | 128 | 68.5 | 14.15 | 0.272 | — | 0.272 | 96.2 % | 94.1 | no arena API |
| ncnn 4t | 160 | 86.3 | 11.44 | 0.337 | — | 0.337 | 96.2 % | 96.4 | no arena API |
| ncnn 3t | 96 | 58.4 | 16.12 | 0.175 | — | 0.175 | 70.6 % | 92.6 | no arena API |
| ncnn 3t | 128 | 66.5 | 14.95 | 0.205 | — | 0.205 | 76.5 % | 94.4 | no arena API |
| ncnn 3t | 160 | 84.5 | 11.77 | 0.260 | — | 0.260 | 76.6 % | 96.6 | no arena API |
| vc4 | 96 | 384.6 | 2.60 | 0.196 | 0.059 | 0.255 | 16.6 % | 185.6 | 64.0 / 31.7 / 25.1 |
| vc4 | 128 | 416.1 | 2.40 | 0.200 | 0.065 | 0.265 | 15.9 % | 185.8 | 64.0 / 35.5 / 25.6 |
| vc4 | 160 | 503.4 | 1.95 | 0.231 | 0.082 | 0.313 | 15.3 % | 185.7 | 64.0 / 41.6 / 26.3 |

Read the share column and the QPU path looks obviously cheap: 15–17 % of the
four cores against 95–96 % for ncnn at 4 threads.  Read CPU seconds per frame
and the gap is much smaller, or reversed: 0.255–0.313 against 0.219–0.337, and
against 0.175–0.260 for ncnn at 3 threads.  The next section holds the frame
rate constant and asks the question that matters.

### The same table on the current build

The vc4 rows above are the build the page was written around.  The current
build adds the two host-side changes from [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md);
nothing about the graph, the kernels or the arena plan moved, so the memory
columns stand.  These cells restart `vc4d` before the cell, so the client gets
the full 96 MiB arena.

| runtime | imgsz | wall ms | FPS | client CPU-s/frame | daemon CPU-s/frame | total CPU-s/frame | share of 4 cores |
|---|---:|---:|---:|---:|---:|---:|---:|
| vc4 | 96 | 315.7 | 3.16 | 0.146 | 0.047 | **0.193** | 15.3 % |
| vc4 | 128 | 348.2 | 2.86 | 0.151 | 0.053 | **0.204** | 14.7 % |
| vc4 | 160 | 432.5 | 2.31 | 0.175 | 0.066 | **0.241** | 13.9 % |

The share of the four cores is about the same, because both the CPU and the
wall time fell: the path is not cheaper per second, it is cheaper per frame and
delivers more frames.  Per frame the client moves 0.062–0.066 s less and
`vc4d` 0.012–0.014 s less than the rows above, which is 24–29 % of the total.

### Memory

ncnn has no arena interface.  Its process high-water mark is 92.6–96.6 MiB,
the model files are 12.6 MiB, and the load delta is about 78 MiB: the weights
plus ncnn's per-blob buffers, which are freed and reused by its pool allocator
but not exposed as a plan.  The engine's client high-water mark is 185.6–185.8
MiB, including the 64 MiB client arena mapping; the graph pool itself is 31.7,
35.5 and 41.6 MiB at the three sizes, with a measured peak of 25.1, 25.6 and
26.3 MiB.  The arena is not the constraint on this curve.

## 3. Equal throughput: CPU seconds per delivered frame

ncnn is throttled with a sleep so that it delivers exactly the frame rate the
engine delivered naturally on the earlier build: 2.60, 2.40 and 1.95 FPS.
Each cell delivers 12–18 frames at the target rate, and the final frame period
is completed before the wall time is taken, so the delivered rate equals the
target.  "Default" is
ncnn with its bundled libgomp default wait policy; "passive" sets
`OMP_WAIT_POLICY=passive`, which removes the spin before each barrier.

| imgsz | delivered FPS | vc4infer | ncnn 4t default | ncnn 4t passive | ncnn 3t default | ncnn 3t passive |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 2.60 | **0.255 (16.6 %)** | 0.231 (15.0 %) | 0.172 (11.2 %) | 0.169 (11.0 %) | **0.157 (10.2 %)** |
| 128 | 2.40 | **0.265 (15.9 %)** | 0.293 (17.6 %) | 0.243 (14.5 %) | 0.216 (12.9 %) | **0.212 (12.7 %)** |
| 160 | 1.95 | **0.313 (15.3 %)** | 0.358 (17.5 %) | 0.265 (12.9 %) | 0.273 (13.3 %) | **0.265 (12.9 %)** |

Cells are total CPU seconds per delivered frame, with the average share of the
four cores in brackets.  Bold marks the cheapest configuration in each row.

That was the build this page was written around.  On it the QPU path was the
most CPU-expensive of the five configurations at 96 and 128, and tied the best
ncnn configuration at 160: 0.255 against 0.157 CPU-s/frame at 96 (+62 %), 0.265
against 0.212 at 128 (+25 %), 0.313 against 0.265 at 160 (+18 %).  Its per-frame
CPU was as high as ncnn's despite the QPU doing the arithmetic because the
client spent 0.196–0.231 s marshalling 462–536 launches and `vc4d` spent
another 0.059–0.082 s launching them: 0.255–0.313 CPU-seconds of host work, for
0.157–0.265 CPU-seconds of ncnn inference.

### After the host-overhead work

The current build delivers 3.16, 2.86 and 2.31 FPS, so ncnn is throttled to
those rates for the comparison.  These are fresh ncnn cells, not the numbers
scaled: an ncnn frame's CPU per frame depends slightly on how deeply its worker
threads sleep between frames, and at the higher rates the passive cells cost
more per frame than they did at the old rates.

| imgsz | delivered FPS | vc4infer | ncnn 4t passive | ncnn 3t passive |
|---:|---:|---:|---:|---:|
| 96 | 3.16 | 0.193 (15.3 %) | 0.184 (14.5 %) | **0.181 (14.3 %)** |
| 128 | 2.86 | **0.204 (14.6 %)** | 0.250 (17.9 %) | 0.196 (14.0 %) |
| 160 | 2.31 | **0.241 (13.9 %)** | 0.303 (17.5 %) | 0.243 (14.0 %) |

Cells are total CPU seconds per delivered frame, share of four cores in
brackets, bold the cheapest in each row.  At equal delivered frames the QPU
path is now cheaper than ncnn at 4 threads with passive waiting at 128
(−18 %) and 160 (−20 %), and still 5 % above it at 96.  Against ncnn at 3
threads it is 7 % above at 96, 4 % above at 128 and level at 160 (−1 %).

The offload case therefore holds where it was weakest before.  A frame
delivered by the QPUs no longer costs more host CPU than the same frame
delivered by ncnn's passive workers at 128 and 160, while the QPU path still
leaves 3.4–3.5 cores free for the whole 0.32–0.43 s frame instead of ncnn's
1.8–2.3 under one competing thread.  At 96 it does not hold: ncnn's CPU per
frame is at its smallest there (0.181 CPU-s), and the engine's launch count
barely shrinks with resolution (291 launches at 96 against 365 at 160), so the
fixed per-launch cost dominates a smaller tensor.

The shares tell the same story from the other side.  At equal FPS the engine
uses 13.9–15.3 % of the cores and ncnn at 3 threads uses 14.0–14.3 %.  It
leaves a large block of each *second* idle because its frame takes 0.32–0.43 s;
ncnn leaves small gaps between 0.05–0.09 s frames.  Co-work sees the gaps, not
the average.

## 4. Co-run: the same CPU work under continuous inference

**The co-run tables in this section were measured on the earlier build and were
not re-run after the host-overhead work.**  They are kept because the question
here (what the engine's idle gaps buy a competing workload) is not the same as
§3, and nothing below is used in the conclusion's CPU-per-frame comparison.
The current build's own co-run numbers would differ: it delivers 2.31–3.16 FPS
alone instead of 1.95–2.60, so the inference column and the displaced-CPU
column would move with it.

The burner is a C program with a fixed instruction count per thread: 320 M
iterations of a dependent xorshift/rotate/accumulate chain, `gcc -O2 -pthread`,
one or four threads.  Its checksum is identical across alone and co-run cells
(`f2b02d50ea060e33` for one thread, `4be6d80e14563e86` for four), so the work
is the same and the ratio is a throughput ratio, not a timing guess.  Alone, it
takes 4.842 s on one thread and 4.964 s on four.  "Lost" is
`1 − alone_wall / corun_wall`; "displaced" is the burner CPU-seconds the same
work would have completed alone in the co-run window minus what it actually
received, divided by the inference wall time.

### One competing thread

This is a co-work thread that wants one core, not the whole machine.

| runtime | imgsz | burner lost | inference FPS | inference CPU-s/s | headroom cores | displaced CPU-s per inference-s |
|---|---:|---:|---:|---:|---:|---:|
| vc4infer | 96 | 0.4 % | 2.55 | 0.66 | 3.34 | 0.00 |
| vc4infer | 128 | 0.4 % | 2.36 | 0.63 | 3.37 | 0.00 |
| vc4infer | 160 | 0.5 % | 1.96 | 0.61 | 3.39 | 0.00 |
| ncnn 4t passive | 96 | 2.1 % | 12.44 | 1.73 | 2.27 | 0.01 |
| ncnn 4t passive | 128 | 4.2 % | 8.71 | 1.66 | 2.34 | 0.03 |
| ncnn 4t passive | 160 | 1.7 % | 8.90 | 1.99 | 2.01 | 0.00 |
| ncnn 3t passive | 96 | 1.5 % | 16.15 | 2.20 | 1.80 | 0.00 |
| ncnn 3t passive | 128 | 2.2 % | 12.38 | 2.17 | 1.83 | 0.01 |
| ncnn 3t passive | 160 | 1.8 % | 10.13 | 2.23 | 1.77 | 0.00 |

With one competing thread every runtime leaves it running: the burner loses
0.4–4.2 %, and the QPU path keeps its 2.0–2.6 FPS while ncnn keeps 8.7–16.2.
The engine has more headroom (3.3–3.4 cores against 1.8–2.3), but it is
delivering four to eight times fewer frames.

### Four competing threads

This is the case where co-work wants all four cores.

| runtime | imgsz | burner lost | inference FPS | inference CPU-s/s | headroom cores | displaced CPU-s per inference-s |
|---|---:|---:|---:|---:|---:|---:|
| vc4infer | 96 | 20.2 % | 1.86 | 0.52 | 3.48 | 0.71 |
| vc4infer | 128 | 15.9 % | 1.65 | 0.48 | 3.52 | 0.55 |
| vc4infer | 160 | 39.7 % | 1.10 | 0.48 | 3.52 | 1.49 |
| ncnn 4t passive | 96 | 41.7 % | 7.39 | 0.90 | 3.10 | 1.60 |
| ncnn 4t passive | 128 | 40.6 % | 5.46 | 0.90 | 3.10 | 1.53 |
| ncnn 4t passive | 160 | 26.6 % | 5.60 | 1.00 | 3.00 | 1.00 |
| ncnn 3t passive | 96 | 28.1 % | 9.57 | 0.97 | 3.03 | 1.07 |
| ncnn 3t passive | 128 | 28.2 % | 7.76 | 1.02 | 2.98 | 1.06 |
| ncnn 3t passive | 160 | 28.4 % | 6.05 | 1.03 | 2.97 | 1.08 |

Under full contention the engine keeps the burner at 16–40 % loss and holds
1.1–1.9 FPS; ncnn keeps the burner at 27–42 % loss and holds 5.5–9.6 FPS.  The
engine ends with 3.5 cores free against ncnn's 3.0, a gain of about half a core
for giving up four to eight frames per second.  That half core is the only
currency the latency buys.

The 160 + four-burner engine cell is the noisiest: it was run twice after the
daemon had to be restarted to recover a fragmented arena, giving 1.19 FPS
(884.0 ms) and 1.10 FPS (887.5 ms); the second is tabulated.  The 96 and 128
cells are stable to a few percent.

### The default ncnn thread pool is not a fair co-run citizen

ncnn's bundled libgomp spins before each barrier by default.  With any
competing thread the spin collapses ncnn's own latency, before the arithmetic
cost is even considered:

| config | alone FPS | under 1 burner | under 4 burners |
|---|---:|---:|---:|
| ncnn 4t, 96 | 17.42 | 1.61 | 2.15 |
| ncnn 4t, 160 | 11.44 | 1.39 | 1.14 |
| ncnn 3t, 160 | 11.77 | 1.52 | 0.71 |

`OMP_WAIT_POLICY=passive` restores 8.9–16.2 FPS with one burner and 5.5–9.6
with four.  The passive numbers are the ones used for the break-even because
they measure the runtime's CPU demand rather than libgomp's reaction to
preemption.  The default numbers are still what a deployment gets without the
environment variable, and they are a real reason the QPU path can look better
than it is.

## 5. The break-even

**This section uses the earlier build's co-run result and was not re-measured.**
On that build, at a fixed frame rate the QPU path already used more CPU per
frame, so there was no positive amount of co-work that made it win on CPU cost.
The break-even existed only when the comparison was *unequal throughput* under
saturated CPU co-work: the engine gave up frames and bought headroom.  The
price of that headroom, using the passive ncnn control (the cheapest credible
CPU engine at each point):

| imgsz | ncnn config | ncnn FPS | vc4 FPS | frames lost/s | extra free cores | CPU-s given up per lost frame | co-work value needed, frames per CPU-s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | 4t passive | 7.39 | 1.86 | 5.53 | 0.38 | 0.069 | 14.6 |
| 96 | 3t passive | 9.57 | 1.86 | 7.71 | 0.45 | 0.058 | 17.1 |
| 128 | 4t passive | 5.46 | 1.65 | 3.81 | 0.42 | 0.110 | 9.1 |
| 128 | 3t passive | 7.76 | 1.65 | 6.11 | 0.54 | 0.088 | 11.3 |
| 160 | 4t passive | 5.60 | 1.10 | 4.50 | 0.52 | 0.116 | 8.7 |
| 160 | 3t passive | 6.05 | 1.10 | 4.95 | 0.55 | 0.111 | 9.0 |

The break-even sentence: **on that build, for every frame the QPU path gave up
it had to buy 58–116 ms of CPU work for the competing workload, or
equivalently that workload had to be worth 8.7–17.1 frames per CPU-second it
consumes.**  A detector frame is the unit here; if the co-work is worth less
than that per CPU-second, running ncnn and accepting its CPU demand delivered
more frames and more co-work at the same time.  At equal FPS on the current
build the required value is negative at 128 and 160, because the QPU path now
uses less CPU per frame there; at 96 it is still positive and small.

The one regime where the engine wins on its own terms is a workload that needs
cores in a contiguous burst and cannot be scheduled around ncnn's 50–90 ms
frames: the engine leaves 3.4–3.5 cores free for the whole 0.32–0.43 s frame,
while ncnn at 4 threads leaves 0.15 cores across its run and then sleeps.  That
is a latency-shape argument, not a CPU-cost argument, and it comes with a
five-fold frame-rate penalty (unconstrained ncnn at 4 threads is 17.42, 14.15
and 11.44 FPS at 96/128/160 against 3.16, 2.86 and 2.31 for the engine).

## 6. Limits of these numbers

- The ARM cap is 1.0 GHz, not the 1.2 GHz used in earlier pages, because the
  board's supply logged under-voltage at 1.2 GHz and the firmware capped to
  600 MHz.  Absolute wall and CPU seconds are not comparable with 1.2 GHz
  tables; the ratios in this page are.
- The after cells use governor `ondemand` with the floor pinned to the 1.0 GHz
  cap.  Unpinned, `ondemand` drops to 700 MHz between a frame's GPU waits and
  client CPU moves by about 20 % between repeats.  `vcgencmd measure_clock arm`
  read 999.998–1000.002 MHz in every after cell.
- The after cells restart `vc4d` before the cell.  The daemon can come up with a
  degraded (48 MiB) arena while the previous block is still held, then replaces
  itself with a fresh one; a client that connects in that window falls back to a
  16 MiB region and refuses the 160 graph.  Every tabulated after cell had the
  full 96 MiB arena and a 64 MiB client region.
- The account in [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) was taken with the
  launch-path profiler enabled (`VC4INFER_PROF=1`).  The profiled 160 cell reads
  0.169 s client CPU against 0.175 s unprofiled, so the instrument is inside the
  repeat-to-repeat spread; the tables here are unprofiled.
- The fan keeps the peak at 53.2 °C, so none of the cells throttled thermally
  (`in0_lcrit_alarm=0`, current throttle bits clear).  Under-voltage history is
  sticky in `0x50000`; it does not indicate a cap during the cells.
- ncnn's exported graph is static at each size.  Feeding the 160 model a 128
  tensor returns the 160 grid, so each size uses a separate PPNX export from
  the same checkpoint.
- The engine's 160 + four-burner cell varies by 8 % between repeats and is the
  least certain row in the co-run tables; the 96 and 128 cells are stable.
- One GPU client at a time.  The daemon was restarted when CMA fragmentation,
  while another client held a region, left it with a smaller arena; the
  tabulated cells all ran against a 96 MiB arena.  Cells that raced another
  client were discarded.
- CPU time includes the Python client and, for the engine, `vc4d`; it excludes
  GPU time by construction.  Per-frame CPU varies by about 5–10 % between
  repeats, mostly because the fixed-point tick is 10 ms and the run lengths are
  a few seconds.

## Conclusion

Measured as CPU seconds per delivered frame at equal FPS, the earlier build did
not free the CPU: 0.255–0.313 CPU-s per frame against 0.157–0.265 for ncnn at 3
threads and 0.172–0.265 for ncnn at 4 threads with passive waiting, while
delivering the same 1.95–2.60 FPS.  Breaking that down
([CPU-BREAKDOWN.md](CPU-BREAKDOWN.md)) put 19 % of the client's CPU in the
executor loop, 35 % in per-node host code and 41 % in the C op and launch
paths, and showed that 43 % of all launches were concat copies issued one
16-channel block at a time.

The current build removes those copies and most of the per-node planning.  It
uses **0.193–0.241 CPU-s per frame** (24–29 % less) at **2.31–3.16 FPS** (16–21 %
less wall).  At the rates it now delivers, that is below ncnn at 4 threads with
passive waiting at 128 (−18 %) and 160 (−20 %), 5 % above it at 96, and within
±7 % of ncnn at 3 threads at all three sizes.  **The offload case therefore
holds at 128 and 160 and does not yet hold at 96**, where the engine's
launch-bound fixed cost dominates the smallest tensor: 291 launches at 96
against 365 at 160, and ncnn's own CPU per frame is at its lowest.

The remaining 0.235 s at 160 is the daemon's poll loop (0.065 s), the padded
images and GEMM operand padding (0.055 s), per-node host code (0.045 s) and the
executor (0.035 s).  Halving the original 0.313 s would need a C-side execution
plan that removes the last two together; this change does not attempt it.
