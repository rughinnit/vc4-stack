# Where a frame's CPU time goes

[COST-COMPARISON.md](COST-COMPARISON.md) measures **0.313 CPU-seconds per
delivered YOLOv8n-160 frame** on the QPU path: 0.231 s in the Python client and
0.082 s in `vc4d`.  More host CPU than the 0.263 s of QPU busy time, and more
than ncnn at the same frame rate.  This page accounts for those 0.313 s by
category, describes the two host-side changes that followed from the account,
and measures what is left.

The account is CPU time, not wall time: a phase that blocks on the GPU (the
socket `recvmsg` in a launch) contributes only the microseconds it spends
runnable.  Wall time appears in the result tables because it moved too.

## 1. Method

Four independent readings are combined; each one is exact for the part it
covers.

* **`/proc/<pid>/stat`** `utime+stime` around the timed loop gives client and
  daemon CPU in seconds, exactly as the cost page reads them.
* **A launch-path profiler in the remote backend** (`VC4INFER_PROF=1`,
  `infer/src/vc4_dev_remote.c`) timestamps each phase.  The socket round trip
  is timed with `CLOCK_THREAD_CPUTIME_ID`; the phases that cannot block (dirty
  range cleaning, the uniform address mask, upload/download copies, the padded
  image) use `CLOCK_MONOTONIC` (vDSO, about 20 ns).  The counters are exposed
  through `vc4_prof_ptr()` and read by `tools/cost/breakdown.py`.
* **`time.thread_time()` around every C binding method** gives the C total, so
  `C total - remote phases` is the C work that never reaches the socket
  (kernel choice, uniform fill, scratch bookkeeping, operand padding).
* **`time.thread_time()` around the executor and each node dispatch**, plus
  wrappers on the layout helpers, splits the Python remainder into layout
  copies and the rest of the node body.

The client rows are a partition of the client figure by construction: C rows +
Python rows = `thread_time` of the timed loop, and that agrees with the
`/proc` reading to the 10 ms tick.  The daemon figure is added unchanged.  A
sample-based cross-check (`SIGPROF` at 1 kHz CPU, leaf Python frame) agreed
with the phase counters to within a few percent; its numbers are not tabulated
because the exact counters are better.

`tools/cost/breakdown.py` reproduces the table:

```
VC4INFER_PROF=1 python3 tools/cost/breakdown.py --size 160 --runs 8
```

Run it under `tools/thermal-guard.sh --max 75`, with one GPU client.

## 2. The account at 160

One YOLOv8n inference (`yolov8n_160.onnx`, `passes="all"`, 64 MiB arena,
252 nodes, 365 launches after the changes below).  Seconds per frame; a row is
a category, and the client categories sum to the client reading.

| category | what it is | before | after |
|---|---|---:|---:|
| Python executor | the node loop: liveness, buffer recycling, names, per-node stats | 0.0455 | 0.0353 |
| node host code | per-node shape/attribute work and the numpy inside handlers | 0.0851 | 0.0454 |
| layout copies (Python side) | NCHW/NHWC conversion, re-pad, slice/expand helpers | 0.0114 | 0.0113 |
| C op setup | padded image, operand padding, uniform fill, kernel choice | 0.0676 | 0.0550 |
| — conv pad image, zero fill | 24.8 MB written per frame | 0.0146 | 0.0145 |
| — conv pad image, row copy | 6.6 MB copied per frame | 0.0060 | 0.0059 |
| — `vc4__invalidate_off` | 9.5 MB invalidated per frame | 0.0009 | 0.0009 |
| — other C setup | GEMM operand padding, im2col uniforms, scratch | 0.0461 | 0.0337 |
| launch path, client | dirty-range clean, address mask, `sendmsg`/`recvmsg`, tail | 0.0305 | 0.0234 |
| **client total** | Python client process | **0.240** | **0.169** |
| daemon | socket, validation, kick, poll, L2 write-back | 0.080 | 0.065 |
| **total per frame** | client + daemon | **0.320** | **0.235** |

The C rows that are measured directly: dirty-range cleaning 0.0061 s (1.00 M
cache lines per frame), uniform address mask 0.0004 s, socket round trip
0.0157 s, uploads/downloads 0.0006 s.  Kernel blob cache lookups are **zero
per frame**: the 115 blobs a 160 frame launches are loaded and uploaded on
first use (about 1.5 s once), then the engine's kernel table and `vc4d`'s
program slots are hit by index.

Two rows deserve a note.  The padded image is materialised once per conv and
**zeroed in full before only its border is used**: 24.8 MB of zero stores per
frame, 0.0146 s, more than the dirty-range cleaning and the address mask
together.  The remaining 0.0337 s of "other C setup" is dominated by the GEMM's
alignment copies (`M` to 8, `N` to 64, `K` to 16) and the few hundred words of
uniform data each launch needs.

## 3. The two changes

Both are host software.  No kernel, no arithmetic and no data layout changed,
and the output is byte-identical to the previous path on every model tested.

### 3.1 One launch per concat input

A channel concat of two NHWC tensors issued one `copy2d_x` **per 16-channel
block per input**: 231 launches for the 17 concat nodes of YOLOv8n, 43 % of the
frame's 536 launches.  The row copy kernel already has the fields for a block
walk (`nblk`, `sbstep`, `dbstep`), and `_slice_ch` was already using them.  The
concat now issues one call for all full 16-channel blocks of an input and one
more only for a partial tail (where the store depth must equal the lane count,
or it would spill into the next pixel).

```
before: for each 16-channel block, for each image: copy2d_x(...)
after:  for each input: copy2d_x(..., nblk=cs/16, sbstep=64, dbstep=64)
        plus copy2d_x(..., nlane=tail, depth=tail) when cs % 16 != 0
```

The bytes moved are the same; the launches are not.  Launches per frame fall
from 536 to 365 at 160, 468 to 297 at 128, and 462 to 291 at 96.

### 3.2 A per-run node plan in the executor

`_dispatch` used to build a 55-entry dictionary of bound methods and lambdas on
**every node call**, `_attr` walked the protobuf attribute list on every read,
and the loop read `eng.stats` twice per node (a ctypes call and a C struct
copy) even when nothing asked for a profile.  All three are now decided once
per load:

* `Runner._build_op_table()` builds the handler table once, merged with the
  family tables under the same override rule as before;
* node names are computed once (`Runner._names`);
* `_attr` memoises parsed attributes, keyed on the node's declared identity
  (`name`, `op_type`, inputs) - *not* `id()`, which protobuf recycles between
  wrapper objects, and which was measured to make modelA's Gemm read a
  Flatten's cached `axis`;
* the per-node `eng.stats` snapshot is taken only when `profile=True` or the
  runner is hybrid (it is only used by the profile and the plan report).

C binding calls per frame fall from 403 to 232, and `op host code` - the row
that held the attribute walk and the dispatch table - falls from 0.0851 s to
0.0454 s.

### 3.3 Layout assignment

The executor chose a tensor's layout per node.  A 1-channel conv output came
back packed NHWC with `pitch == W`, which is the same byte order as NCHW, but
the `Reshape` that read it did not know that and converted the plane: 33
`copy2d_x` launches of four rows by 16 words each, on the DFL head of
YOLOv8n-160.  The layout pass names that tensor NCHW at its producer and the
executor relabels the buffer; no byte moves.

The pass removes one copy row and 33 launches; the other rows are copies of
real data (a concat or a split is not a relabel):

| copy row | before | after | what it is |
|---|---:|---:|---|
| `/model.22/dfl/Reshape_1` (from `dfl/conv`) | 33 | 0 | NHWC->NCHW of a 1x1x4x525 plane, identical bytes |
| 13 C2f/SPPF `Concat` | 38 | 38 | channel concat, a real copy |
| 8 `Split` | 16 | 16 | channel split, a real copy |
| 3 SPPF `MaxPool` | 6 | 6 | border fill + interior copy |
| 2 `Resize` | 2 | 2 | nearest upsample |
| 6 box-head `Reshape` | 8 | 8 | NHWC->NCHW before the anchor concat |
| `dfl/Transpose` | 1 | 1 | permute before the class softmax |
| 2 `Slice` | 2 | 2 | box-coordinate slices |
| **total** | **106** | **73** | 36 copy nodes -> 35 |

The number of calls removed scales with the anchor count, because the plane
is 4x525 at 160 but 4x336 at 128 and 4x189 at 96: `ceil(anchors/16)` is 33,
21 and 12.  The A/B on the rebased tree (`920a683` plus this pass), three
interleaved rounds per size, median of 8 timed runs after 2 warm-ups, one
client, `tools/thermal-guard.sh --max 75`, ARM min/max 1.0 GHz, client and
daemon CPU from `/proc/<pid>/stat`:

| imgsz | launches | wall ms | client CPU-s | daemon CPU-s | total CPU-s |
|---:|---|---:|---:|---:|---:|
| 96 before | 291 | 358.8 | 0.176 | 0.047 | 0.224 |
| 96 after | 279 | 358.7 | 0.176 | 0.046 | 0.222 |
| 128 before | 297 | 388.8 | 0.179 | 0.051 | 0.229 |
| 128 after | 276 | 387.4 | 0.180 | 0.050 | 0.230 |
| 160 before | 365 | 470.5 | 0.199 | 0.065 | 0.264 |
| 160 after | 332 | 465.9 | 0.198 | 0.064 | 0.260 |

The output is **bit-identical** in every size and round (`np.array_equal`);
the pass changes a label and the recorded plan, no arithmetic.  At 96 and 128
the wall and CPU difference is inside the run-to-run spread (wall +/-1 ms,
total CPU +/-0.002 s); at 160 it is outside it: **wall -4.6 ms (-1.0 %),
total CPU -0.004 s (-1.4 %), of which daemon CPU is -1.9 %**.  Client CPU
barely moves because the removed calls are small device copies with almost no
Python work on either side of them: the win is the daemon round trips.

The remaining 73 copy launches are not relabels.  A channel concat or split
moves each input into a new contiguous buffer, and the box-head `Reshape`
reorders pixel-major conv output into the `[1,64,525]` the anchor concat is
written on.  Removing those needs a consumer that reads several source
buffers (concat) or an anchor-major head rewrite, not a layout choice.

Include the pass in `all`: it removes launches at every size, its output is
bit-identical, and the 160 frame is 1.0 % faster in wall and 1.4 % cheaper in
CPU.  The gain is small because the removed calls are the cheapest in the
frame; the case for it is that the cost is zero and the decision is now
recorded instead of being re-derived per node.

## 4. Result at the three sizes

Same protocol as the cost page: one GPU client, fan fitted,
`tools/thermal-guard.sh --max 75`, median of 10 timed runs after three warm-ups.
The ARM cap is **1.0 GHz** because of the board's marginal supply; the governor
is `ondemand` with `scaling_min_freq = scaling_max_freq = 1.0 GHz`, so it cannot
drop to 700 MHz between GPU waits (letting it do so changed client CPU by about
20 % between repeats).  Each cell starts a fresh `vc4d` so the client gets the
full 96 MiB arena.

| imgsz | launches | wall ms | FPS | client CPU-s | daemon CPU-s | total CPU-s/frame |
|---:|---:|---:|---:|---:|---:|---:|
| 96 before | 462 | 402.0 | 2.49 | 0.212 | 0.060 | 0.272 |
| 96 after | 291 | 315.7 | 3.16 | 0.146 | 0.047 | **0.193** |
| 128 before | 468 | 432.6 | 2.31 | 0.216 | 0.065 | 0.281 |
| 128 after | 297 | 348.2 | 2.86 | 0.151 | 0.053 | **0.204** |
| 160 before | 536 | 514.9 | 1.94 | 0.237 | 0.080 | 0.317 |
| 160 after | 365 | 432.5 | 2.31 | 0.175 | 0.066 | **0.241** |

CPU seconds per frame fall 29 %, 27 % and 24 %; wall time falls 21 %, 19 % and
16 %.  The frame rate rises because a launch costs about 200 us of host CPU
(client plus daemon) and the removed launches were short copy kernels: the
wall time was paying for the round trips, not for the bytes.

Correctness: modelA, modelB, modelC and `yolov8n_160` (shipped, with their
recorded inputs) and the 96/128/160 exports produce outputs **byte-identical**
to the pre-change engine.  Against the shipped recorded outputs, modelA/B/C are
within 4.8e-6 absolute and `yolov8n_160` is at max abs 2.04e-2, max rel
2.23e-4, 0/44100 elements over the recorded tolerance - the same figures the
shipped reference records.

## 5. What is left

The goal was to halve the host CPU.  The measured reduction is 24-29 %, so it
did not halve, and the remaining 0.235 s (0.241 s unprofiled) at 160 is
dominated by four costs:

* **daemon, 0.065 s.**  Most of it is the poll loop: `vc4d` reads `SRQCS` and
  sleeps 10 us per iteration while the kernel runs.  On the previous build the
  loop measured about 88 us of daemon CPU per launch; with the short copy
  launches gone, the 365 that remained at that measurement are the long
  kernels, so the average cost per remaining launch is higher (0.065 s / 365 =
  178 us).  The layout pass takes 33 of them (sec. 3.3).  A coarser poll
  interval would cut this and add latency; it is a trade, not a free win.
* **C op setup, 0.055 s.**  The padded image is 0.021 s of it (24.8 MB zeroed,
  6.6 MB copied) and the GEMM alignment copies are most of the rest.  Zeroing
  only the border is the obvious next change and was not made here.
* **node host code, 0.045 s.**  Per-node shape decisions and the remaining
  numpy in handlers.  The concat host fallback for tensors whose layout is not
  NHWC and the 6x6 decomposition are the larger contributors.
* **executor, 0.035 s.**  The loop still does per-node dictionary and liveness
  work in Python; the plan removed the largest parts of it, not all of it.

A C-side execution plan that walked the node list and called the existing C op
entry points would remove the last two rows together.  Section 6 starts that
for the node classes that dominate the launch count; its figure is host-only
and the client/daemon rows still need a run on the board.  Reducing the launch
count further
(Softmax still issues 40 launches, Add 38; the Reshape row was 41 before the
layout pass and is 8 after it, sec. 3.3) would attack the first two.

At these numbers the QPU path now spends less CPU per delivered frame than
ncnn at 4 threads with passive waiting at 128 and 160, and is about level with
ncnn at 3 threads at 160.  It is still 5-7 % above both at 96, where ncnn's
own CPU per frame is smallest.

## 6. An execution plan in C

The two rows this page called addressable - the Python executor (0.035 s) and
the per-node host code (0.045 s) - are per-node Python work for 332 launches.
What they compute is the same on every frame: which engine call to make, with
which buffers, for a graph whose shapes and layout decisions are already
fixed.  This section describes a plan that records that answer once and
replays it in C, scoped to the node classes that dominate the launch count
(elementwise, unary, concat, split, copy/layout, reshape-class).  Convolution
and GEMM stay on the Python path in this pass.

The implementation is `infer/src/vc4_plan.c`,
`infer/python/vc4infer/_plan.py` and the plan hooks in
`infer/python/vc4infer/_onnxrun.py`.

### 6.1 What a plan is

A plan is a straight line of device calls over **arena offsets**, not over
tensors.  Each record is a fixed-size POD (`vc4_plan_op`, 152 bytes) with:

| field | meaning |
|---|---|
| `kind` | one of `EW`, `UNARY`, `COPY2D`, `COPY2D_N`, `COPY2D_X`, `FILL2D`, `GATHER`, `BLIT`, `REDUCE`, `CUMSUM`, `ALIAS` |
| `a`, `b`, `out` | engine buffer `(offset, bytes)` for the operands and the store; `b` only for `EW` |
| `n` | element count, or byte count for `BLIT` |
| `p[14]` | kind-specific integers: the elementwise op code, the unary op and its three float parameters (carried as IEEE-754 bits, so nothing is rounded), the copy strides, the fill value |

There is no function pointer, no allocation and no host address in a record.
`vc4_plan_run(ctx, ops, n)` decodes each record and calls the entry point the
executor itself called - `vc4_elementwise`, `vc4_unary`, `vc4_copy2d_x`,
`vc4_fill2d`, `vc4_gather`, `vc4_blit`, `vc4_reduce`, `vc4_cumsum` - so the
bytes moved, the launch count and the kernel choice are the recording's, not
the walker's.  An `ALIAS` record makes no call: it is a reshape, a flatten, an
identity or a fused activation's relabel, whose output name maps to a buffer
another record already produced.

The Python side mirrors the struct (`_binding.PlanOp`) and refuses to build a
plan when `vc4_plan_op_size()` disagrees with its own `sizeof`; the two
writers cannot drift silently.

### 6.2 How it is built

The first strict, non-profile frame that creates no persistent buffer runs
exactly as before, with two additions:

* the engine is wrapped in `_plan.Trace`, which copies each call's decoded
  arguments into a list (a call it does not recognise is recorded as
  "other" and marks its node unrepresentable);
* the pool labels every allocation, so the frame's offsets can be named.

After the frame the trace is split per node.  A node becomes a run of C
records when **every** call it made is on the representable list and **every**
output it produced is an arena buffer.  A node with no calls whose outputs
alias existing buffers becomes an `ALIAS`.  Everything else becomes a
`("p", index)` step: the replay driver hands that index to the ordinary
`Runner._dispatch`, exactly as the loop did before, and continues after it.
Skipped and folded nodes (the optimiser's fused activations, the SiLU pairs)
become `("s", index)` steps and cost nothing.

A frame is only recorded when it allocated no buffer that outlives it (a
packed weight panel, a broadcast expansion).  Such a buffer takes a range an
earlier node had already used and released, which is correct for that frame
but not for a replay: the replay hands the earlier node its recorded offset
back and that node writes over the panel.  Section 6.7 is the frame where
that happened.  For a conv model the recording frame is therefore the
second one, and `runner.cplan_deferrals` counts the frames that built a panel
instead.

The pool is then switched, in place, from bump/free allocation to the
recording: every non-persistent allocation the replay still asks for (a
Python node's buffers, the input uploads) is served its recorded offset.  The
segment allocations are the same engine buffers, so constants and packed
panels keep their addresses.  `Pool.check` is off for this mode, because the
C steps write buffers the Python allocator never sees and an overlap check
could only report false conflicts; the recording came from a frame that
already ran correctly.

Panels are a plan-time concept, and the allocator now says so: a buffer that
is created once and kept (a broadcast expansion, a conv weight panel, a GEMM
operand) is allocated under its own `panel:` scope rather than the node's, and
its upload is marked static.  Without that, the panel's ordinal would vanish
from the node's allocation sequence on replay and the pinned offsets after it
would name the wrong buffers.  This is the invariant a node scope has to
keep: every allocation under it must happen on every frame.

### 6.3 The boundary, and what happens when the plan is wrong

The rule is about calls, not op types, which is what keeps the boundary
honest as handlers change:

* a node is a C step if all its calls are representable and all its outputs
  are buffers;
* otherwise it is a Python step, unchanged, with the same allocator keys and
  the same dispatcher;
* the plan is not built or not used for a profile run (per-node stats and the
  backend report would stop being per node), a hybrid run (`host_policy`
  tiers), an engine artifact (its own pinned arena plan already owns the
  offsets) or when `$VC4_NO_CPLAN=1`;
* a feed whose shape or element count differs from the recording is refused
  by name, because a static plan pins the buffers;
* if a Python node allocates differently on a later frame than it did while
  recording - a handler that branches on activation values, such as
  Softmax's range check - the pinned pool raises `ArtifactPlanError`, the
  plan is dropped for that Runner, the pool returns to the ordinary
  allocator and **the frame is re-run from the feeds on the Python path**.
  The failure costs the plan, not the answer.  `runner.plan_error` records
  why, and later frames do not try again.

### 6.4 Coverage on YOLOv8n-160

With `passes="all"`, the optimised graph has 252 nodes: 138 execute and 114
are skipped (fused activations and the epilogue).  Of the 138, 47 become C
nodes - 17 `Concat`, 8 `Split`, 8 `Add`, 8 `Reshape`, 2 `Slice`, 2 `Mul`,
1 `Transpose`, 1 `Sigmoid` - with 105 records between them, 92 of them
`COPY2D_X`.  That is the copy work this page counts as structural: the 17
channel concats, the 8 channel splits and the DFL permute.

The 91 Python steps are 64 `Conv`, 19 `Constant`, 3 `MaxPool`, 2 `Resize`,
2 `Sub` and 1 `Softmax`.  Conv, MaxPool and Resize are the stated scope
boundary.  `Constant` is a node that re-uploads its data on every frame - a
pre-existing cost, not one the plan introduces, and folding it to an
initialiser is a graph pass, not a plan feature.  `Sub` is a runtime tensor
minus a broadcast constant: `_scaled` uploads a full-size constant buffer per
frame, which is exactly the host traffic that keeps its node in Python.

### 6.5 Measured on silicon

The plan is default again.  Three interleaved rounds per size, plan off
(`VC4_NO_CPLAN=1`) then on, median of the three; eight timed frames after
three warm-ups per cell, one client, packaged `vc4d` at `/run/vc4d.sock`,
ARM pinned at 1.0 GHz, `tools/thermal-guard.sh --max 75`, peak 49.4 C,
`throttled=0x0`.  `tools/cost/breakdown.py` reads client and daemon CPU from
`/proc/<pid>/stat` and the phases from the launch-path profiler.

| imgsz | plan | wall ms | client CPU-s | daemon CPU-s | total CPU-s | launches |
|---:|---|---:|---:|---:|---:|---:|
| 96 | off | 383.8 | 0.2025 | 0.0475 | 0.2500 | 279 |
| 96 | on | 302.3 | 0.1225 | 0.0450 | **0.1675** | 279 |
| 128 | off | 413.8 | 0.2050 | 0.0525 | 0.2575 | 276 |
| 128 | on | 332.7 | 0.1263 | 0.0488 | **0.1750** | 276 |
| 160 | off | 490.9 | 0.2225 | 0.0650 | 0.2875 | 332 |
| 160 | on | 407.8 | 0.1400 | 0.0613 | **0.2013** | 332 |

CPU per frame falls **33.0 %, 32.0 % and 30.0 %**, wall time falls 81-83 ms.
The Python side is where it comes from: the same breakdown splits 160 into
0.1421 s of node Python work on the Python path and 0.0694 s through the
plan, and the executor skeleton (everything outside the per-node dispatch)
from 0.0830 s to 0.0402 s.  C time falls by 0.009 s as well, because one
`plan_run` call replaces a ctypes call and a dictionary lookup per C node.
The daemon changes by only -0.002 to -0.004 s: the same 332 launches reach it.

Two caveats about the absolute column.  The 160 baseline here is 0.2875 s, not
the 0.241 s of section 4.  The measurement ran against the daemon that was
already serving the board rather than a fresh one per cell, with idle
processes from an unrelated pipeline experiment on the same ARM (load average
1.3 of 4 cores), so the client figure is inflated.  The on/off delta is
measured inside the same environment, cell by cell, which is the comparable
number; the absolute baseline has to be re-taken on an idle board before it
replaces section 4.

The command for that window is:

```
tools/cost/baseline.py --check      # preconditions only, opens no device
tools/cost/baseline.py --out ~/scratch/logs/baseline-plan.json
```

`baseline.py` repeats the three-round A/B above (plan off then on, eight timed
frames after three warm-ups per cell) with each cell under the thermal guard,
and refuses to start until one ARM clock at 1.0 GHz, no other Python process,
load average <= 0.20, <= 55 C, `throttled=0x0` and the packaged `vc4d` on
`/run/vc4d.sock` all hold.  It prints the daemon's uptime and leaves a
restart to the window owner: a long-lived daemon may hand the client a
smaller arena than a fresh one, which is the remaining difference from the
section 4 protocol.

### 6.6 The first run on silicon, and what it found

The gate that mattered, in order:

```
PYTHONPATH=infer/python python3 infer/tools/op_matrix.py     # 152/152
PYTHONPATH=infer/python python3 tools/cost/check_plan_equiv.py
python3 infer/tests/test_cplan_gpu.py                        # real models
VC4INFER_PROF=1 python3 tools/cost/breakdown.py --size N --runs 8
```

`op_matrix` must be given `PYTHONPATH=infer/python`: it imports `vc4infer`
without adding the tree to `sys.path`, so a run from a checkout silently
tests the *installed* package.  The first gate run did exactly that and
reported 147/152 with a wide-matmul mismatch from the installed, older
engine; against the tree it is 152/152 with no mismatch.

`check_plan_equiv.py` reports all seven models byte-identical (modelA/B/C,
`yolov8n_160`, and the 96/128/160 exports).  `test_cplan_gpu.py` runs
modelB/modelC and YOLOv8n at 96/128/160 with the plan on and off, four and
three frames respectively, and compares every output byte; all identical,
same launch counts.

### 6.7 The defect the first silicon run found

The first version recorded the plan from the model's first frame.  For
YOLOv8n that frame packs each conv's weight panel on first use.  The panel is
persistent, but it was allocated in the middle of the frame, and the
allocator gave it a range an earlier node had used and released - valid for
that frame, because the earlier node was finished.  The recording then maps
that earlier node's allocation to the same range.  On replay the node runs
again at its recorded offset, which is now the panel, and writes over it
before the conv reads it.  The first conv after the stem read a corrupted
panel; the values grew through the backbone and the head's `dfl/Softmax`
received `±inf`, which the executor refuses by design.

What the host proof could not see: it compared the recorded call list with
the replayed call list and the final bytes, with a stub engine.  The call
lists were identical, because a corrupted panel does not change which calls
are made; and the stub's conv ignored the panel and its allocator never
returned a freed range to one, so neither half could fail.  The equivalence
was real but not sufficient: it never asserted that a *recording* is
replayable.

Two rules close it.  A frame that allocates a persistent buffer is not
recorded; the next frame has the panel live from its start and the allocator
keeps clear of it.  And `Runner._replay_safe` refuses a recording in which
any per-frame allocation overlaps a buffer that outlives the frame, naming
the key and both ranges; `runner.plan_error` carries the refusal and the
Runner stays on the Python path.  `_replay_safe` is stated rather than
re-derived, so a future persistent buffer cannot reintroduce the class
quietly.

The host check is `test_panel_deferral` in `test_cplan.py`: a graph whose
conv output is larger than its input, so the panel lands in the input's freed
range; the stub conv reports the panel's first word, so a corrupted panel
changes the answer.  It asserts that the panel frame was deferred, that the
replay matches the Python path, and that no pinned allocation overlaps a
panel.  The silicon check is `test_cplan_gpu.py`, which runs a real model's
plan-enabled frame, not a `profile=True` one: a profiling run never replays
the plan, which is how the defect escaped the first selftest.
