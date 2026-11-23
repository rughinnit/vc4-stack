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
remove 24–29 % of the CPU and 16–21 % of the wall time.  With the C execution
plan of [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) §6 on top, the path now uses
**0.1675–0.2013 CPU-s per frame** at the 3.30/3.00/2.44 FPS it delivers.  At
equal delivered rate that is 16.5–32.9 % below ncnn at 4 passive threads at
every size, 9.8 % and 16.1 % below ncnn at 3 threads at 128 and 160, and 8.8 %
above it at 96.

## 1. Conditions

| | |
|---|---|
| board | Raspberry Pi 3B (BCM2837), VideoCore IV, 905 MiB RAM, fan fitted |
| kernel | 6.18.50+rpt-rpi-v8, Debian 13 arm64 |
| GPU clock | VideoCore core 400 MHz throughout (`vcgencmd measure_clock core`) |
| ARM clock | **1.0 GHz**, governor `ondemand`, `scaling_min_freq = scaling_max_freq = 1.0 GHz` |
| thermal guard | `tools/thermal-guard.sh --max 75 --interval 1`, every cell |
| peak temperature | 53.2 °C over the suite; phase-A cells 47.2–51.5 °C; the 2026-11-22 cells 44.0–50.5 °C; the 2026-11-23 cells 43.5–47.2 °C |
| throttling | `throttled=0x50000` sticky (under-voltage and throttling occurred before the suite); current bits clear, `in0_lcrit_alarm=0` |
| GPU client | one at a time; `vc4d` 96 MiB arena on `/run/vc4d.sock`; the 2026-11-22 cells restart it per cell, the 2026-11-23 cells do not |
| engine | vc4infer 1.2.10 from this tree, daemon backend, `VC4_OPT=all`; the 2026-11-22 rows add the two host-side changes in [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md), and the 2026-11-23 rows add its execution plan |
| 2026-11-23 run | one packaged `vc4d` 1.2.9 (protocol 2), 96 MiB arena on `/run/vc4d.sock`, uptime 696 s at the start and not restarted; load average 0.17 at the start, 0.88 immediately after the run (the measurement itself), 0.16 sixty seconds later; 43.5–47.2 °C; `throttled=0x0`; ARM pinned at 1.0 GHz |
| ncnn cells | 15 delivered frames after three warm-ups, five cells per configuration, median; `OMP_WAIT_POLICY=passive`, one thread count per cell |
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

### The same table with the plan on (2026-11-23)

The vc4 rows above are the build the page was written around.  The current
build adds the two host-side changes of [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md)
§3 and its C execution plan (§6).  Neither changes the graph, the kernels or
the arena plan, so the memory columns stand.  These cells are on an idle
board with one packaged `vc4d`; §4.1 of that page has the preconditions, the
load and the spread.

| runtime | imgsz | wall ms | FPS | client CPU-s/frame | daemon CPU-s/frame | total CPU-s/frame | share of 4 cores |
|---|---:|---:|---:|---:|---:|---:|---:|
| vc4 | 96 | 303.3 | 3.30 | 0.1225 | 0.0450 | **0.1675** | 13.8 % |
| vc4 | 128 | 333.5 | 3.00 | 0.1262 | 0.0488 | **0.1750** | 13.1 % |
| vc4 | 160 | 409.0 | 2.44 | 0.1388 | 0.0613 | **0.2013** | 12.3 % |

The share of the four cores falls a little because the CPU falls faster than
the wall: the path is not cheaper per second, it is cheaper per frame and
delivers more frames.

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

### At the rates this build delivers (2026-11-23)

The current build delivers 3.30, 3.00 and 2.44 FPS (the wall medians above).
ncnn is throttled to 3.31, 3.01 and 2.45 FPS, the rates the build delivers,
and each cell delivers exactly that; the achieved rate is in the table.  The
vc4 column is the instrumented `breakdown.py` run of
[CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) §4.1; its per-call wrappers add a small
cost to the client, so the margins below are, if anything, conservative for
the QPU path.

| imgsz | delivered FPS | vc4infer | ncnn 4t passive | ncnn 3t passive |
|---:|---:|---:|---:|---:|
| 96 | 3.31 | 0.1675 (13.8 %) | 0.2007 (16.6 %) | **0.1540 (12.7 %)** |
| 128 | 3.01 | **0.1750 (13.1 %)** | 0.2447 (18.4 %) | 0.1940 (14.6 %) |
| 160 | 2.45 | **0.2013 (12.3 %)** | 0.3000 (18.4 %) | 0.2400 (14.7 %) |

Cells are total CPU seconds per delivered frame, share of four cores in
brackets, bold the cheapest in each row.  At equal delivered rate the QPU path
uses 8.8 % more host CPU per frame than ncnn at 3 passive threads at 96, and
9.8 % and 16.1 % less at 128 and 160; against ncnn at 4 passive threads it is
16.5 %, 28.5 % and 32.9 % less.  The smallest margin, 8.8 % at 96, is more than
three times the largest run-to-run spread in either side of that row (0.8 % over
the three vc4 rounds, 0.8 % over the five ncnn cells), so no margin here is
inside the spread.

The offload case holds at 128 and 160, where a frame delivered by the QPUs
costs less host CPU than the same frame delivered by ncnn's workers, and it
does not hold at 96, where ncnn's own CPU per frame is at its smallest
(0.1540 CPU-s) and the engine's launch count barely shrinks with resolution
(279 launches at 96 against 332 at 160), so the fixed per-launch cost dominates
a smaller tensor.

The shares tell the same story from the other side.  At equal FPS the engine
uses 12.3–13.8 % of the cores and ncnn at 3 threads 12.7–14.7 %.  It leaves a
large block of each *second* idle because its frame takes 0.30–0.41 s; ncnn
leaves small gaps between 0.05–0.09 s frames.  Co-work sees the gaps, not the
average.

### Batch launch submission

A later change makes the per-launch request itself optional: protocol 3 sends
an ordered list of launches in one request and `vc4d` executes it in order,
each launch completing before the next is kicked.  This is a same-build A/B
(`VC4INFER_NO_BATCH=1` against the batch path), not a new comparison with
ncnn.  The cells use `tools/cost/breakdown.py`, which instruments each binding
call and node dispatch; the absolute CPU figures are therefore higher than the
`measure.py` rows above and the table is an A/B, not a replacement for them.
Conditions are §3's: ARM pinned at 1.0 GHz, `thermal-guard.sh --max 75`, three
interleaved rounds per size, a fresh 96 MiB `vc4d` per cell, and
`VC4_NO_CPLAN=1` so that both sides are the batch measurement alone with the
plan held out.  It is a protocol A/B, not the shipping path of §3.

| imgsz | requests before → after | wall ms before → after | FPS before → after | total CPU-s before → after |
|---:|---|---|---|---|
| 96 | 279 → **131** (−53 %) | 382.5 → **372.4** | 2.61 → **2.69** | 0.249 → **0.238** (−4.4 %) |
| 128 | 276 → **127** (−54 %) | 412.6 → **402.8** | 2.42 → **2.48** | 0.257 → **0.245** (−4.7 %) |
| 160 | 332 → **131** (−61 %) | 492.3 → **478.6** | 2.03 → **2.09** | 0.288 → **0.272** (−5.6 %) |

Client CPU falls 3.4–3.6 % and the daemon's 9.4–10.6 %; the socket round trip
falls from 13.0/13.0/15.2 ms per frame to 8.6/8.6/9.1 ms.  Wall moves less
(2.4–2.8 %) because the frame is GPU-bound at these sizes.  That A/B ran at
2.69, 2.48 and 2.09 FPS and is kept as the batch measurement; the equal-rate
cells in §3 are fresh ones on the shipping build at 3.31, 3.01 and 2.45 FPS,
not scaled from it.

A whole-graph request is not attempted.  A 160 frame performs 8 uploads, 2
downloads and 17 explicit L2 flushes, and its handlers read arena data between
nodes; each is a point where the queue must complete before the CPU proceeds.
The measured granularity is therefore the node (131 requests for 332 launches),
and outputs are byte-identical to the single-launch path for the three shipped
models and YOLOv8n at 96/128/160.

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
build the required value is negative at 128 and 160, because the QPU path uses
less CPU per frame there (-9.8 % and -16.1 % against ncnn at 3 threads); at 96
it is still positive and small (+8.8 %).

The one regime where the engine wins on its own terms is a workload that needs
cores in a contiguous burst and cannot be scheduled around ncnn's 50–90 ms
frames: the engine leaves 3.4–3.5 cores free for the whole 0.32–0.43 s frame,
while ncnn at 4 threads leaves 0.15 cores across its run and then sleeps.  That
is a latency-shape argument, not a CPU-cost argument, and it comes with a
five-fold frame-rate penalty (unconstrained ncnn at 4 threads is 17.42, 14.15
and 11.44 FPS at 96/128/160 against 3.30, 3.00 and 2.44 for the engine).

## 6. Limits of these numbers

- The ARM cap is 1.0 GHz, not the 1.2 GHz used in earlier pages, because the
  board's supply logged under-voltage at 1.2 GHz and the firmware capped to
  600 MHz.  Absolute wall and CPU seconds are not comparable with 1.2 GHz
  tables; the ratios in this page are.
- The after cells use governor `ondemand` with the floor pinned to the 1.0 GHz
  cap.  Unpinned, `ondemand` drops to 700 MHz between a frame's GPU waits and
  client CPU moves by about 20 % between repeats.  `vcgencmd measure_clock arm`
  read 999.998–1000.002 MHz in every after cell.
- The 2026-11-22 after cells restart `vc4d` before the cell.  The daemon can
  come up with a degraded (48 MiB) arena while the previous block is still
  held, then replaces itself with a fresh one; a client that connects in that
  window falls back to a 16 MiB region and refuses the 160 graph.  Every
  tabulated cell of that window had the full 96 MiB arena and a 64 MiB client
  region.
- The 2026-11-23 cells do not restart it: one packaged `vc4d` 1.2.9 (protocol
  2), 96 MiB arena, uptime 696 s at the start, so every cell met the same
  daemon and the same arena.  Load average read 0.17 at the start and 0.88
  immediately after the run (the run itself); 60 s later it was 0.16.  The
  cells are `breakdown.py` (profiled and instrumented) for vc4infer and
  `measure.py` for ncnn.
- The account in [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) was taken with the
  launch-path profiler enabled (`VC4INFER_PROF=1`).  The profiled 160 cell reads
  0.169 s client CPU against 0.175 s unprofiled, so the instrument is inside the
  repeat-to-repeat spread.  The 2026-11-22 tables here are unprofiled; the
  2026-11-23 vc4 cells are `breakdown.py` (profiled and instrumented, which
  makes them, if anything, pessimistic) and its ncnn cells are `measure.py`.
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
  repeats in the long ncnn cells, mostly because the fixed-point tick is 10 ms
  and the run lengths are a few seconds.  In the 2026-11-23 cells it varies by
  0.6–1.7 %: three vc4 rounds per size and five ncnn cells per configuration.

## Conclusion

Measured as CPU seconds per delivered frame at equal FPS, the earlier build did
not free the CPU: 0.255–0.313 CPU-s per frame against 0.157–0.265 for ncnn at 3
threads and 0.172–0.265 for ncnn at 4 threads with passive waiting, while
delivering the same 1.95–2.60 FPS.  Breaking that down
([CPU-BREAKDOWN.md](CPU-BREAKDOWN.md)) put 19 % of the client's CPU in the
executor loop, 35 % in per-node host code and 41 % in the C op and launch
paths, and showed that 43 % of all launches were concat copies issued one
16-channel block at a time.

The current build removes those copies, most of the per-node planning and, with
the execution plan of CPU-BREAKDOWN §6, the per-node Python itself.  It uses
**0.1675–0.2013 CPU-s per frame** at **3.30/3.00/2.44 FPS**.  At the rates it
delivers that is below ncnn at 4 passive threads at every size (−16.5 %, −28.5 %
and −32.9 %), below ncnn at 3 passive threads at 128 (−9.8 %) and 160
(−16.1 %), and 8.8 % above ncnn at 3 threads at 96.  **The offload case
therefore holds at 128 and 160 and does not hold at 96**, where the engine's
launch-bound fixed cost dominates the smallest tensor.

The 0.201 s frame at 160 is the daemon's poll loop, the padded images and GEMM
operand padding, the remaining per-node host code and the launch path.  The
plan removes the per-node Python; what it does not touch is the daemon and the
C setup, which are the next two rows in [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md)
§5.
