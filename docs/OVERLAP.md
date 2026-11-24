# Overlapping the ARM with the QPUs

A YOLOv8n-160 frame takes 433 ms of wall time and 250-263 ms of QPU time: a
57-61 % duty cycle.  The rest is the ARM preparing the next launch while the
GPU sits idle.  This page describes the protocol and the engine changes that
let the client prepare request N+1 while the daemon executes request N, the
CPU-side gates it passes, and the silicon measurement.  The short version: it
is correct and it is worth 1.8-2.6 % of wall on the tree that ships, because
the frozen plan already removed most of the host work it was built to hide and
the residual sits at the pipeline's own two-window depth, not at the conv input
pad.  The numbers and the attribution are in section 6.

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

**What the wire actually carries (measured later, `feat/daemon-pipe`).**  The
second bullet above is the design; the shipped `remote_submit()` reaps the
previous request *before* it sends the next one, so the connection carries one
request at a time.  Request N+1 is built during N's execution and handed over
after N's reply.  Over 107 652 launches a daemon that checks the socket during
every launch found a datagram there **zero** times.  `VC4INFER_SUBMIT_EAGER=1`
(off by default) sends before reaping and makes the connection depth 2; even
then, daemon-side preparation of the next request is worth 0.5 % of wall,
because the daemon's whole share of the per-request period is about 30 µs.  See
[DAEMON-PIPE.md](DAEMON-PIPE.md).

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
| uniform shadow | the daemon reads a launch's 64 B/QPU block when it kicks it | 48 KiB per set, 96 KiB total (`0x510000` with protocol 5, `0x290000` with 2..4, below `VC4_DATA_BASE`) |
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

## 5. What it was expected to hide, and what it hides

Between two barriers of one graph node the client does work that does not
depend on any GPU result: the executor's bookkeeping, attribute parsing, the
next node's allocations, kernel choice, uniform fill, the dirty-range clean,
the address mask and the `sendmsg`.  That is the work the pipeline was built
for.

The analysis that set the target was: the NHWC conv zero-pads its whole input
into scratch with a `memcpy` on the CPU (`nhwc_pad_image`, 24.8 MB zeroed and
6.6 MB copied per frame), and that input is the previous launch's result, so
every conv node must wait for the node before it and the frame cannot reach
`max(QPU, host)`.  From the pre-plan host time of 171 ms at 160 it predicted
335-375 ms pipelined, against 433 ms serial.

**The measurement refutes that.**  On the shipping tree the conv input pad is
not where the pipeline waits: 0.4 ms of 180.5 ms blocked at 96, 0.8 ms of
286.5 ms at 160.  Section 6 has the numbers and the attribution; the residual
is the scratch-window boundary, and the ceiling is the two-window depth, not
the pad.

## 6. The silicon run

Measured on `2689a76` + this branch, private protocol-4 daemon (96 MiB arena,
`--gpu-lock`), one client, ARM 1.0 GHz, thermal guard at 75 C (peak 49.4 C,
throttled 0x0), 20 frames per cell after 3 warm-up frames.  "QPU" is the sum of
per-launch GPU time, "req/f" the daemon requests per frame, "wait" the wall
spent blocked in `poll()`, "pad" the part of that wait spent inside
`nhwc_pad_image`, "lat" a single frame after an idle gap:

| imgsz | path | wall ms | FPS | QPU ms | duty | req/f | launch/f | cli CPU ms | dmn CPU ms | wait ms | pad ms | lat ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | sync | 296.1 | 3.38 | 169.5 | 0.573 | 187 | 483 | 110 | 50 | 0 | 0 | 295.5 |
| 96 | pipelined | 290.7 | 3.44 | 169.5 | 0.583 | 187 | 483 | 120 | 50 | 180.5 | 0.4 | 290.3 |
| 128 | sync | 333.7 | 3.00 | 200.8 | 0.602 | 183 | 583 | 110 | 60 | 0 | 0 | 334.4 |
| 128 | pipelined | 325.9 | 3.07 | 200.8 | 0.616 | 183 | 583 | 120 | 60 | 213.6 | 0.3 | 330.1 |
| 160 | sync | 424.4 | 2.36 | 267.5 | 0.630 | 187 | 832 | 130 | 80 | 0 | 0 | 426.1 |
| 160 | pipelined | 416.0 | 2.40 | 268.2 | 0.645 | 187 | 832 | 140 | 80 | 286.5 | 0.8 | 417.3 |

The launch counts are higher than the 365 in [ACCURACY-FPS.md] because that
table ran the optimiser (`passes="all"`); this window ran the shipping default,
which is no optimisation, and that is the graph the plan replays.  More
launches is the harder case for the pipeline (more requests to overlap), so
the gain above is not an artefact of a cheaper graph.

A second 160 cell, with the wait split by the barrier that caused it:

| path | wall ms | QPU ms | duty | wait ms | at scratch | at depth | at invalidate | at l2 flush | at host | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sync | 421.8 | 267.5 | 0.634 | 0 | - | - | - | - | - | - |
| pipelined | 410.9 | 267.9 | 0.652 | 285.3 | 268.3 | 1.9 | 0.0 | 0.0 | 0.1 | 14.2 |

The byte-identical gate passed: `tools/overlap.py --sizes 96 --frames 1
--warmup 1 --check` printed "outputs byte-identical" for the plan-path frame,
sync against pipelined.

**The gain is 1.1-1.8 points of duty, 1.8-2.6 % of wall, and it does not pass
the two-point rule set for this run.**  The numbers above are recorded as a
measurement, not as a throughput result: `run_pipelined()` stays an explicit
opt-in that nothing in the engine calls by default.

Where the blocked time is: 99 % of it is at the **scratch-window boundary**,
not at the pad and not at a data dependency.  An op opens its batch, and to
write its scratch it must reuse the window of the request two submissions
back, so it waits for that request (`remote_open_batch` -> `remote_reap_until`,
attributed as `scratch`).  At 160 that wait is 268 ms over 187 requests, 1.43
ms per request, which is the per-request QPU time: the client is waiting for
the QPU, which is the intended steady state, but with only two windows it
cannot *submit* op N before it has waited, so its own preparation of op N+1
cannot start either.  The wall is therefore `QPU + the preparation that two
windows cannot hide + the daemon's per-request serial cost`, and the pipeline
hides only the small part that fits in the other window's flight.

That changes the answer to the question this branch was asked to settle:

* the pad is **not** the pipeline's residual (0.4-0.8 ms of 180-286 ms), so
  the flat-panel or border-aware `im2col` path would not be a pipeline fix.  It
  remains a panel-row and CPU saving on its own merits, against a CPU cost that
  is largely hidden already.
* what would move the wall is more depth, and depth is bought with scratch: a
  third window needs another 5.76 MiB at 160, and only 19.4 MiB is free beside
  the 41.6 MiB pool plan.  Depth 3 fits at 96 (free 29.3 MiB, 2.24 MiB per
  window) and would need a larger arena at 160 - or a smaller pool estimate,
  or a smaller scratch footprint per op.
* the daemon's per-request serial cost is the other half: 80 ms of daemon CPU
  per frame at 160 over 187 requests, plus the idle gaps between launches
  while it prepares the next one.  Removing it needs a daemon that builds
  launch N+1 while launch N runs, which is a larger change than this branch.

  That daemon exists now (`vc4d --prefetch`, `feat/daemon-pipe`) and it
  measures as no change: of the 0.79 ms per request that is not the kernel,
  about 0.03 ms is the daemon's register sequence, about 0.13 ms is its
  completion notice, and the rest is the client's own host work.  The frame at
  160 is 404 ms = 268 ms of QPU + ~25 ms of daemon tail + ~111 ms of client CPU.
  [DAEMON-PIPE.md](DAEMON-PIPE.md) §6 has the counters and the explanation.

**Single-frame latency: no cost.**  The pipelined cell is 5.2/4.3/8.8 ms
*faster* on one frame after an idle gap (96/128/160), because the frame's own
early preparation is hidden under its first launches, and the final flush is
one request deep.  A reader gets both numbers in the table; neither path is
made slower.

CPU per frame: the client costs 10 ms more with the pipeline (the extra
`poll`/reap syscalls), the daemon is unchanged.  Total host CPU per frame is
therefore unchanged at 160 and +10 ms at 96 and 128, for 1.8-2.6 % of wall.

## 7. The CPU-side gates

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
  offset from anywhere but its uniform block.  The frame that is compared is a
  **frozen-plan replay**, because the plan is the shipping path and replays
  from the second frame; a single frame would only exercise the recording.
  Result: 43/19/9/483/583/832 launches identical, 13/10/6/187/183/187 requests
  on both sides.  A2 repeats the check on the executor path
  (`VC4_NO_CPLAN=1`): 43 launches, 13/13 requests, identical.
* **B. Nothing changes under a running request.**  With a 20 ms delay per
  request, the fake snapshots each request's uniform shadow and 4 KiB at every
  address its launches name, and compares after the delay.  Three frames per
  model, so the replay is covered.  ModelA, modelB, modelC and YOLO-96: 0
  shadow changes and 0 probe changes over 786 requests.  This is the direct
  test of the two-set rule; on the plan path it is also the test of the barrier
  set, because a pinned pool cannot hold a block back.
* **C. The quarantine.**  A block freed while an epoch is outstanding is not
  handed out again, and is handed out once that epoch is reaped; held bytes
  count as used; three live blocks never overlap.
* **D. Accounting** (executor path, `VC4_NO_CPLAN=1`, because only an
  unpinned pool has a hold list).  After a pipelined run nothing is left on the hold list,
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

## 8. The window as run

The packaged unit and a protocol-4 daemon cannot share `/run/vc4d.sock`, so the
window used a private one.  Nothing is installed and nothing is copied into
`/usr/sbin`.

1. Record `systemctl cat vc4d` (the packaged unit runs `--socket
   /run/vc4d.sock --arena-mb 96 --group vc4 --drop-user nobody
   --max-timeout-ms 10000` with `VC4D_GPU_LOCK=1`).  Check that no client is
   connected (`ss -xp | grep /run/vc4d.sock`), `systemctl stop vc4d`, and start
   the built daemon:
   `sudo vc4d/build/vc4d --socket /run/vc4d-overlap.sock --arena-mb 96
   --group vc4 --drop-user nobody --max-timeout-ms 10000 --gpu-lock`.
   Acceptance: a `vc4c_connect`/`vc4c_info` against that socket reports
   protocol 4, `VC4D_F_ASYNC_SUBMIT` set and a 96 MiB arena.  The output of the
   run's transcript is `docs/logs/overlap-window.log`.
2. Byte-identical gate: `tools/overlap.py --sizes 96 --frames 1 --warmup 1
   --check`, under `tools/thermal-guard.sh --max 75`.  Acceptance:
   "byte-identical".
3. Measurement: `tools/overlap.py --sizes 96,128,160 --frames 20 --warmup 3`
   under the guard.  The table in section 6 is that run.
4. Restore, unconditionally, from a shell trap: **kill the private daemon
   first**, wait for it to exit so its arena is released, then `systemctl start
   vc4d` and poll `vc4-query` until it prints `daemon: vc4d 1.2.9, protocol 2,
   arena 100663296 bytes (96.0 MiB)`.  Polling `vc4-query`'s exit status is not
   enough: it exits 0 while printing "daemon: not reachable", so the check has
   to match the daemon line.  Leaving the private daemon running while the unit
   starts costs the unit its arena: the first window did that, the unit's 96 MiB
   request failed with `-16` twice and its ladder fell back before recovering.

What the window establishes:

* the pipeline is correct on the shipping path (byte-identical, and the fake's
  per-request snapshots are clean on the replay frame);
* it is worth 1.1-1.8 points of duty on this tree, which does not pass the
  two-point rule.  The 335-375 ms prediction is not what happened: the frame is
  416 ms at 160, and the prediction's premise - that the pad is the serialising
  edge - is refuted by the attribution;
* the residual is the two-window depth at the scratch boundary (268 of 285 ms
  of blocked time at 160), plus the daemon's serial per-request cost.  Depth 3
  is the next thing to try, and at 160 it needs ~5.8 MiB more scratch than the
  arena has beside a 41.6 MiB pool estimate: it fits at 96, not at 160, without
  a larger arena or a smaller scratch footprint;
* there is no single-frame latency cost: the pipelined path is 4-9 ms faster on
  one frame after an idle gap, and 1.8-2.6 % faster over 20.
