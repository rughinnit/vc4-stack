# Overlapping the ARM with the QPUs

A YOLOv8n-160 frame takes 433 ms of wall time and 250-263 ms of QPU time: a
57-61 % duty cycle.  The rest is the ARM preparing the next launch while the
GPU sits idle.  This page describes the protocol and the engine changes that
let the client prepare request N+1 while the daemon executes request N, states
what that can and cannot hide, and the CPU-side gates it passes before it is
allowed near the QPUs.  Silicon numbers are the next section to be filled in;
the first-run checklist and the stop conditions are at the end.

## 1. The arithmetic

| imgsz | wall ms | QPU ms | duty | FPS | launches | requests |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 353.9 | 179.8 | 50.8 % | 2.83 | 462 | - |
| 128 | 384.9 | 204.0 | 53.0 % | 2.60 | 468 | - |
| 160 | 433.0 | 262.7 | 60.7 % | 2.31 | 536 | 131 |

Launches and requests are from [CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) after the
concat and executor work (536 launches become 365, 131 requests at 160); wall,
QPU and FPS are from [ACCURACY-FPS.md](ACCURACY-FPS.md).  At 160 the client
spends 0.169 CPU-seconds per frame and the daemon 0.065, for 0.235 s of host
work against 0.262 s of QPU work.  If every microsecond of host work could be
hidden, the frame would be `QPU + daemon launch gaps`, about 275-290 ms.  The
target of 260 ms assumes the gaps are small as well.

## 2. Why one connection with two requests, and not two daemons

Two frames in flight need two of everything a frame writes, and the two ways to
get that are not equivalent:

* **Two daemons, two arenas, alternating.**  Each daemon would have its own
  arena, its own locks and its own `/tmp/vc4-gpu.lock` acquisition, so two
  launches from two daemons still serialise on the GPU and on the lock.  The
  second daemon buys nothing the first cannot do, doubles the CMA footprint at
  a time when one 64 MiB client arena plus a 72 MiB daemon arena is already
  most of what the board will give, and turns every failure (timeout, arena
  restore, reset) into a two-daemon state problem.  Not built.

* **One connection, `SUBMIT` without waiting, a completion poll.**  This is
  what is built.  Protocol 4 adds `VC4D_OP_SUBMIT`, a `LAUNCH_BATCH` whose
  completion the client may collect later, and states that a version-4 client
  may have `VC4D_SUBMIT_MAX` (2) of them outstanding on one connection.  The
  daemon executes each request exactly as it always did - in order, each
  launch to completion, one reply - so its execution path, validation, error
  statuses and timeouts are unchanged.  The only difference is on the client's
  side of the socket: the request for node N+1 is already in the socket buffer
  while node N executes, so the client is free to spend the QPU's busy time
  building it.

The alternative inside option two - `SUBMIT` returning an id after validating
the list, with a separate `WAIT id` that executes it - was rejected.  A
single-threaded daemon cannot start request N+1 while request N is executing,
so the only thing an accepted-but-unrun queue could add is an earlier
validation error.  It would cost an id table and, worse, a new lifetime rule:
an accepted request names program slots and arena regions that must then be
pinned until it runs, which is the class of bug the batch work just removed
(an evicted program slot under a queued launch).  The socket is already an
ordered queue with flow control; using it keeps the daemon stateless per
request.

The version bump is 4 with `VC4D_VERSION_MIN` still 2.  A version-4 client
descends one version at a time when HELLO is refused, so it settles on 3
against a version-3 daemon and keeps the batch op; `vc4_pipeline_begin()`
reports `VC4_ERR_UNSUPPORTED` when the daemon does not advertise
`VC4D_F_ASYNC_SUBMIT`.  The single-launch and batch messages, their validation
and their replies are byte-for-byte unchanged.

## 3. The pipeline in the engine

Two requests in flight means two of everything the second request is writing
while the first is reading:

| resource | why it cannot be shared | cost |
|---|---|---|
| uniform shadow | the daemon reads a launch's 64 B/QPU block when it kicks it | 48 KiB per set, 96 KiB total (`0x290000`, below `VC4_DATA_BASE`) |
| scratch windows | a conv zero-pads its whole input into scratch, then launches on it | 1/3 of the free arena per window |
| pool blocks | a block freed by node N+1's liveness may still be read by node N's request | the quarantine's high-water; measured +0.06 to +0.17 MiB |

The state machine is two `struct vc4_batch`, each EMPTY, FILLING or IN_FLIGHT.
`b[cur]` is the one being filled; submitting it marks it IN_FLIGHT, gives it an
epoch and moves `cur` to the other one.  Opening a batch whose set was last
used by an in-flight request reaps that request first - that is the "no buffer
reused while in flight" rule at the scratch and uniform level, and it is the
only place the pipeline ever blocks on its own.  Three submits in flight cannot
happen: `remote_submit()` reaps the oldest first, which is the depth-2 limit.

Buffer recycling is the caller's, because only the caller knows which
allocation a node is about to overwrite.  `vc4_pipeline_epoch()` returns the
newest submitted epoch and `vc4_pipeline_reaped()` the newest reaped one; the
Python `Pool` holds a freed block until `reaped >=` the epoch that was current
when it was freed, and only then puts it back on the free list.  That is the
whole of the pool's change, and it is what makes two frames of the same graph
safe on one arena.

Everything that touches memory a request may still be using - `vc4_upload`,
`vc4_download`, `vc4_invalidate_buf`, a kernel load, the L2 flush an op asks
for, a scratch reset that would reuse a live window - completes the outstanding
requests first.  These are the same points that drained the queue before; the
synchronous API is unchanged and `run()` never arms the pipeline.

## 4. What does not fit: two whole frames

The memory plan is per graph, and two frames of a 160 graph do not fit in one
client arena.  Measured with the pool's own estimators and its own high-water
mark (64 MiB client arena, 3 MiB of code and uniform slots at the base, so
61 MiB of data):

| imgsz | pool plan MiB | measured peak MiB | 2 x pool MiB | + 2 scratch sets | usable MiB |
|---:|---:|---:|---:|---:|---:|
| 96 | 31.69 | 25.13 | 63.4 | 67.9 | 61 |
| 128 | 35.48 | 25.70 | 71.0 | 78.6 | 61 |
| 160 | 41.60 | 26.82 | 83.2 | 94.7 | 61 |

Two contexts would be worse, not better: each has its own 3 MiB code/uniform
base and its own copy of every weight and packed panel.  So frame-level double
buffering is not what gets built; per-request depth 2 is, and it fits:

| imgsz | one pool + 2 windows | free for the workspace and growth |
|---:|---:|---:|
| 96 | 31.69 + 2 x 2.24 = 36.2 | 24.8 |
| 128 | 35.48 + 2 x 3.78 = 43.0 | 18.0 |
| 160 | 41.60 + 2 x 5.76 = 53.1 | 7.9 |

The three-way split of the free arena (one third per window, one third left
above them) is what keeps a Softmax's training workspace from failing with
"need 64 B, have 0 B": the windows are capped at two thirds of the free arena
and everything above them is still the workspace, exactly as before.

## 5. What this can hide, and what it cannot

Between two barriers of one graph node the client does work that does not
depend on any GPU result: the executor's liveness and bookkeeping, attribute
parsing, the next node's allocations, kernel choice, uniform fill, the
dirty-range clean, the address mask and the `sendmsg`.  All of that is
hideable.  Per frame at 160 it is about 0.080 s (Python), 0.023 s (launch
path) and part of the 0.055 s of C op setup.

What is not hideable is the part of the CPU's work that reads the previous
node's output.  The NHWC conv path zero-pads its whole input into scratch with
a `memcpy` on the CPU (`nhwc_pad_image`, 24.8 MB zeroed and 6.6 MB copied per
frame, 0.020 s), and that input is the previous launch's result.  Every conv
node therefore has to wait for the node before it, whatever the pipeline
depth; the same is true of an elementwise op whose element count is not a
multiple of 256, which pads its operands on the CPU.  This is the single
reason the frame cannot reach `max(QPU, host)` on this build, and it is worth
being explicit about the sizes: 171 ms of host time at 160, of which the
measured drag is the pad image plus the Python that runs after the last
barrier of each node.

The expectation to measure, not a promise:

* pipelined 160: 335-375 ms, 2.7-3.0 FPS, duty 0.70-0.78;
* pipelined 96 and 128: similar gains in absolute milliseconds, so less in
  duty, because their QPU time is smaller and the fixed pad cost is the same;
* single-frame latency: unchanged (the synchronous path is untouched) or
  +1-2 ms (one extra flush at the end of a pipelined frame).

Two levers would move it past that, in increasing order of work:

1. **Border-aware `im2col_nhwc`** (or the flat panel kernel that was refused
   until it passes silicon).  Reading the unpadded input with bounds in the
   gather removes the CPU copy, and with it the per-conv dependency.  The frame
   then approaches `QPU + daemon launch gaps`.  This is a kernel change, not an
   engine change.
2. **A GPU-side pad** (`fill2d` + `copy2d_x`, both existing) instead of the
   CPU copy.  It breaks the dependency the same way and costs two launches per
   conv, about +0.18 s of daemon dispatch per 100 convs at the measured
   per-launch cost - probably a wash, which is why the kernel change is the
   better lever.

Even with either, the floor is not 250 ms: the daemon builds launch N+1 only
after launch N has completed, so the QPU idles for the daemon's per-launch
setup (validation, register writes, kick, L2 write-back) between launches.
That cost is inside the 0.065 s of daemon CPU per frame and is serial with the
GPU.  Removing it needs a daemon that pre-builds the next launch while the
current one runs, which is a larger change than this branch.

## 6. The CPU-side gates

`infer/tests/test_pipeline.py` drives the whole engine through a protocol
stand-in for vc4d (`vc4d/test/fake_daemon.py`: real wire protocol, memfd arena
over `SCM_RIGHTS`, configurable per-request delay, JSONL record of every
request).  No QPU is touched.

* **A. Same request stream.**  For modelA, modelB, modelC and YOLOv8n at 96,
  128 and 160, the pipelined run sends the same launches in the same order, in
  the same number of requests, with the same kernel, QPU count, uniform count
  and every non-pointer uniform word as the synchronous run.  Pointer words are
  wildcarded: the two paths place buffers at different offsets on purpose (the
  pipeline reserves its windows above the pool), and no consumer reads an
  offset from anywhere but its uniform block.  Result: 45/22/10/547/647/896
  launches identical, 21/17/10/304/294/308 requests on both sides.
* **B. Nothing changes under a running request.**  With a 20 ms delay per
  request, the fake snapshots each request's uniform shadow and 4 KiB at every
  address its launches name, and compares after the delay.  ModelA, modelB,
  modelC and YOLO-96: 0 shadow changes and 0 probe changes over 352 requests.
  This is the direct test of the two-set rule and of the pool quarantine.
* **C. The quarantine.**  A block freed while an epoch is outstanding is not
  handed out again, and is handed out once that epoch is reaped; held bytes
  count as used; three live blocks never overlap.
* **D. Accounting.**  After a pipelined run nothing is left on the hold list,
  `used_bytes` equals the sum of live blocks, no free range holds a live block,
  and runs two and three reach the same high-water.  The pipeline's pool cost
  at YOLOv8n is +0.06/+0.11/+0.17 MiB at 96/128/160 against the synchronous
  path's high-water.
* **E. The old daemon.**  Against a version-3 daemon, `run_pipelined()` raises
  with "needs protocol 4" and no `SUBMIT` is sent; `run()` still works.
* **F. The synchronous path did not move.**  The launch stream from this build
  with the pipeline off is identical to `dc2e79d`'s on modelA (45 launches) and
  YOLOv8n-96 (547 launches), on the same fake daemon.

What these gates do not prove is that the outputs are byte-identical: the fake
computes nothing, so an output tensor there is whatever bytes happen to be at
its offset, and the two paths deliberately choose different offsets.  Byte
identity is the silicon gate (`tools/overlap.py --check`), and it is the first
thing the window below runs.

## 7. The first silicon run

Preconditions: no other client on the board, the replay debugging that owns it
finished, `systemctl is-active vc4d` for the packaged daemon, ARM pinned to
1.0 GHz, starting temperature below 60 C.  Nothing is installed.

1. Record the packaged unit's command line (`systemctl cat vc4d`), build
   `vc4d/` and `infer/` in `~/work/vc4-overlap`, then `systemctl stop vc4d` and
   start the built daemon on the same socket and `--arena-mb`.  Acceptance:
   `vc4d/build/test_submit` passes against it, and `test_batch` still passes
   (protocol 3 is still served).
2. Wire smoke with one graph: `tools/overlap.py --sizes 96 --frames 1
   --warmup 1 --check`.  Acceptance: "byte-identical", and the pipelined cell
   reports a duty above the synchronous one.
3. Correctness at the other sizes, one frame each: `--sizes 128,160 --frames 1
   --warmup 1 --check`.
4. Timing, under `tools/thermal-guard.sh --max 75`: `tools/overlap.py --sizes
   96,128,160 --frames 20 --warmup 3`.  This prints wall, FPS, QPU ms, duty,
   requests and launches per frame, client and daemon CPU per frame, wall
   blocked in `poll()` (the un-overlapped residual) and single-frame latency,
   for both paths.
5. A/B the old path: `VC4INFER_NO_BATCH=1` (protocol-2 style, no batch, no
   submit) for one cell, to confirm the fallback still runs and to have the
   pre-batch wall on the same build.

Stop, discard the cell and report rather than retry:

* any `V3D_ERR_STAT` bit, any `E_GPU`, any timeout, any daemon
  "restart"/"TIMEOUT" line, any `E_BUSY` that is not a transient GPU-lock wait;
* any output byte different in the `--check` step (that is a wrong answer, and
  it means the uniform or pool rule above is incomplete - the next step is the
  fake daemon's per-request snapshot, not a retry);
* dmesg showing a CMA allocation failure, OOM or a vc4 reset;
* the guard firing or 75 C reached;
* single-frame latency up by more than 2 % against the synchronous path;
* the pipelined duty not above the synchronous duty by at least two points on
  the first size: that would mean the dependency analysis in section 5 is
  wrong, and the next step is profiler attribution (`wait_ms`, `pad_zero`,
  `pad_copy`, `rt`), not a deeper pipeline.

What the window is expected to establish, honestly: how much of the 171 ms of
host time at 160 is hideable on this engine, and therefore whether the 260 ms
target is reachable by overlap alone (it is not, by the analysis above) or only
after the pad image leaves the CPU.
