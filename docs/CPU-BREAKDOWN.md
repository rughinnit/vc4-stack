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
  launches gone, the 365 that remain are the long kernels, so the average cost
  per remaining launch is higher (0.065 s / 365 = 178 us).  A coarser poll
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
entry points would remove the last two rows together; that is the larger
rewrite this measurement does not attempt.  Reducing the launch count further
(Softmax still issues 40 launches, Reshape 41, Add 38) would attack the first
two.

At these numbers the QPU path now spends less CPU per delivered frame than
ncnn at 4 threads with passive waiting at 128 and 160, and is about level with
ncnn at 3 threads at 160.  It is still 5-7 % above both at 96, where ncnn's
own CPU per frame is smallest.
