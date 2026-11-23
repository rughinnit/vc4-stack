# Pipelining the daemon: preparing request N+1 while request N runs

Status: **built and measured. The change is correct and it buys nothing on the
shipping client, because the client never has a second request on the wire.**
This page gives the design, what must never overlap, the gates it passed, and
the measurement that says where the frame's non-QPU time actually is. The
prepared path is opt-in (`--prefetch`) and off by default.

## 1. What this branch was asked to achieve, and what it found

Measured elsewhere (`5961b76`, depth 2, one client, YOLOv8n-160): a frame is
416 ms, 187 daemon requests, 268 ms of QPU, and the per-request period is
2.22 ms against 1.43 ms of kernel time. The remaining 0.79 ms per request was
attributed to the daemon: "187 requests per frame, a per-request period of
2.22 ms against 1.43 ms of kernel time, about 148 ms per frame of daemon work
and launch gap".

The premise of this branch was that the daemon reads request N+1 only after it
has replied to N, so everything it does to make N+1 runnable happens with the
QPU idle, and that preparing it during N would bring the period down to the
kernel time.

The first measurement refutes the premise's *first half*:

> **The daemon never had request N+1 to prepare: over 107 652 launches, the
> wait loop found a datagram in the socket zero times.**

The reason is in the client, not the daemon. `remote_submit()` reaps the
previous request before it sends the next one:

```c
int other = r->cur ^ 1;
if (r->b[other].state != VC4_BATCH_EMPTY) {   /* the other request is in flight */
        remote_reap_one(ctx);                 /* ... so wait for it, then send */
}
remote_send(ctx, b, 0, 1, NULL, NULL);
```

So the connection carries one request at a time. The client's *build* of N+1
is hidden (that is the 10 ms the pipeline is worth), but N+1 is on the wire
only after N's reply has arrived, and no amount of daemon-side preparation can
change that.

The branch therefore has two parts: the daemon path, which is what was asked
for, and a client flag (`VC4INFER_SUBMIT_EAGER`) that lets a client hand over
the next request before reaping the previous one, which is the condition under
which the daemon path can do anything at all. Both are measured below.

## 2. The design chosen

The daemon is single-threaded, and that is a security property as much as a
simplicity one: it drops to `nobody` with an empty capability set before it
serves anything, serialises clients on one launch in flight, and validates a
request immediately before using it. Three ways to overlap were considered.

**(a) A preparation thread that does everything except touch the registers.**
Rejected.  The register window, the arena mapping and the program table are
plain writable memory with no boundary a second thread cannot cross by
accident.  A thread re-creates the window this change exists to avoid: between
"this uniform word is inside the arena" and "the QPU reads it" there would be a
scheduler.  It also has to hand over the program slot with a memory-ordering
rule, take the GPU lock from a third context, wake up inside a kernel's flight
(a condition-variable wakeup is 10-30 µs here, the same order as the gain, and
a spin burns one of four cores the client is using), and keep the failure paths
(timeout → context reset → probe → exit 75) from interleaving with another
context's MMIO.

**(b) An accepted-but-unrun queue**: validate a request, return an id, execute
it later from `WAIT id`.  Rejected in [OVERLAP.md](OVERLAP.md) §2 and still
rejected: an id table, a new lifetime rule for pinned programs and regions, and
a promise to the client that the current protocol does not make.

**(c) A state machine in the poll loop, single-threaded** — chosen.  The same
thread that polls the QPU also reads the next datagram and validates it, stages
the result, and when the current launch completes performs only the register
sequence for the staged request.  No second context, no new lock, and the
validation is the same code run earlier.  It is (b)'s benefit without (b)'s
promise: a staged request is **prepared, not accepted**.  It owns no reply, no
launch id and no client-visible status until the previous request has completed
and it is its turn; if the daemon cannot run it (it exits after a timeout, or
it is signalled) the client sees what it sees today — EOF.

## 3. What must never happen concurrently

Even single-threaded, "prepare during a launch" puts two pieces of work in one
iteration.  These are the pairs that must never overlap, and the rule that
keeps each apart.  Every rule is a property of the function that does the work,
so it can be checked by reading it rather than by trusting a schedule.

| must never overlap | rule | where it is enforced |
|---|---|---|
| **Register writes and QPU kicks** | only the execution path writes `G.regs`; exactly one launch in flight | `launch_prepare()` performs no MMIO at all: it reads the message, the arena and the tables and writes only the plan.  Every `G.regs` store is in `launch_kick()` (via `gpu_kick`), `gpu_finish`/`l2_writeback()`, `handle_sync`, the batch fence or `try_recover` |
| **Arena mapping** | the dma-buf mapping is created once before the privilege drop and never re-created; the region table changes only on `ALLOC`/`FREE`/`CLOSE` | preparation *reads* arena bytes and nothing else; `ALLOC`, `FREE` and `CLOSE` are never prepared — a datagram of another op is held in the inbox and executed after the current request |
| **L2 maintenance** | `L2CACTL` writes and the `dsb` stay in the execution path | `l2_writeback()` is called from `await_launch`, the batch fence and `SYNC`; never from preparation |
| **The GPU lock** | taken immediately before a kick, released when the launch completes; never held across a prepared request or while idle | `gpu_lock_take()` is called only in `launch_kick()`; preparation never takes it, so a staged request cannot delay a direct-root client |
| **Program-slot eviction** | a slot named by a prepared entry cannot be freed or rewritten | `struct prog.pins` is incremented by preparation and decremented when the plan is dropped; `prog_slot_for()` skips pinned slots in both the same-name reuse and the LRU pass; the plan also re-checks `gen` and the region offset before the kick |
| **Per-client accounting** | `inflight`, `launch_id`, `st_launches`, the LRU touch, arena bytes | written only in `launch_kick()`; preparation allocates nothing and counts nothing the client can see |
| **Two requests' kernels** | one launch in flight, still | `gpu_kick()` refuses when `G.inflight`; a plan is not run until the previous request has completed and been replied to |

The one thing this change deliberately allows is a wider *validation* window:
the words a client put in its uniform block are read when the request is
prepared, while the QPU still reads them at the kick.  A client that rewrites
its own uniform block between submitting and the kick can therefore have words
validated that are not the words used — but such a client can already race the
check inside one message, and the daemon's rules (address mask, arena
containment) are about accidents, not about a client attacking itself.  The
engine writes a uniform block, submits it, and does not touch it again until it
reaps the request.

## 4. What was built

* `launch_one()` split into `launch_prepare()` (validation: no MMIO, no lock,
  no allocation; refusals returned, not logged) and `launch_kick()` (LRU touch,
  GPU lock, `gpu_kick`, bookkeeping, wait).  The synchronous path calls them
  back to back, so the two paths cannot drift in what they accept.
* `handle_batch()` split the same way into `batch_prepare()` (every entry
  validated and staged; the first refusal recorded on its entry) and
  `batch_run()` (BUSY check, entries executed in order, one reply).  The
  denial text of an entry is logged when `batch_run()` reaches that entry, so
  the log and the reply are identical to the synchronous path.
* `await_launch()`'s completion loop reads the next datagram into an inbox and
  prepares it, at most one request ahead, into one of two plan buffers (so a
  prepared request can prepare the next one while it runs).  Only
  `SUBMIT`/`LAUNCH_BATCH` are prepared; anything else is held and executed by
  the normal path in order.
* The main loop's `recvmsg` and its validation block became
  `inbox_fill()`/`inbox_process()`, one path for a datagram read at the top of
  the loop and one read during a wait.
* `struct prog.pins`; `prog_slot_for()` and `prog_release()` respect it.
* Counters printed on exit, not in the protocol: datagrams read early, requests
  prepared under a launch, and the per-launch gap split into completion notice,
  register kick and the remainder.
* Flags: `--prefetch` (default off; `VC4D_PREFETCH=1`), `--no-prefetch`,
  `--poll-us N` (default 10, the interval the daemon always used).

No wire change: no new op, no new field, no version bump, and `HELLO`
advertises what it advertised before.  A client that never sends a batch cannot
reach the prepared path.

## 5. Gates

Run on this board in one boot, 44-49 C, `throttled=0x0`, ARM 1.0 GHz, a private
daemon of this build on `/run/vc4d-daemon.sock` (96 MiB, `--gpu-lock`), one
client.  The packaged unit was stopped for the window and restored afterwards,
verified with `vc4-query`.

| gate | result |
|---|---|
| build, `-Wall -Wextra` | clean, no new warning |
| **byte-identical outputs**, modelA/B/C and YOLOv8n 96/128/160, sync and pipelined, `--prefetch` against `--no-prefetch` | **all identical** (`tools/daemonpipe_check.py --suite`, 12 of 12 digests) |
| per-launch statuses, refusal attribution, launch ids | `test_prefetch` transcripts identical in both modes (22 checks); `test_batch` and `test_uni_mask` transcripts identical in both modes |
| a client that dies mid-request | `test_prefetch` case 4: the child submits two launches and exits without reading; the daemon completes them, reclaims the region and program, answers `PING`, and a new client runs a launch immediately |
| two clients still serialise | `test_prefetch` case 3: a request's launches are contiguous in id (8 launches, ids 6..13), so no other client's kernel ran inside it |
| the prepared path actually ran | daemon exit line: `3 datagram(s) read early, 3 prepared under a launch` in `--prefetch` mode, `0 ... 0` without it |
| the installed unit after the window | `vc4-query`: `vc4d 1.2.9, protocol 2, arena 100663296 bytes (96.0 MiB)` |

Reproduce:

```sh
cd ~/work/vc4-daemonpipe/vc4d && make
TMPDIR=$HOME/scratch/tmp ./test/test_prefetch.sh     # two private daemons, no unit
VC4D_SOCKET=/run/vc4d-daemon.sock VC4INFER_DAEMON=1 \
  python3 ../tools/daemonpipe_check.py --suite      # against either daemon mode
```

## 6. The measurement

One boot, 20 frames after 3 warm-up, private daemon per cell, one client.
"cli CPU" and "dmn CPU" are `utime+stime` per frame; "prepared\*" is the
daemon's own count of requests prepared under a launch over the whole run
(~5 000 requests), not per frame.

| imgsz | daemon | path | wall ms | FPS | QPU ms | duty | req/f | launch/f | cli CPU | dmn CPU | lat ms | prepared\* |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | default | sync | 291.9 | 3.43 | 169.6 | 0.58 | 187 | 483 | 110 | 40 | 295.9 | 0 |
| 96 | default | pipelined | 288.0 | 3.47 | 169.4 | 0.59 | 187 | 483 | 120 | 40 | 289.1 | 0 |
| 96 | `--prefetch` | sync | 292.5 | 3.42 | 170.3 | 0.58 | 187 | 483 | 110 | 50 | 297.0 | 0 |
| 96 | `--prefetch` | pipelined | 288.1 | 3.47 | 170.3 | 0.59 | 187 | 483 | 120 | 50 | 288.9 | 3 |
| 128 | default | sync | 330.1 | 3.03 | 200.9 | 0.61 | 183 | 583 | 120 | 50 | 329.0 | 0 |
| 128 | default | pipelined | 321.4 | 3.11 | 201.0 | 0.63 | 183 | 583 | 120 | 50 | 321.1 | 0 |
| 128 | `--prefetch` | sync | 329.9 | 3.03 | 202.4 | 0.61 | 183 | 583 | 120 | 50 | 330.0 | 0 |
| 128 | `--prefetch` | pipelined | 322.0 | 3.11 | 202.2 | 0.63 | 183 | 583 | 120 | 50 | 322.1 | 0 |
| 160 | default | sync | 414.0 | 2.42 | 267.6 | 0.65 | 187 | 832 | 130 | 60 | 413.6 | 0 |
| 160 | default | pipelined | 404.4 | 2.47 | 268.0 | 0.66 | 187 | 832 | 140 | 60 | 403.4 | 0 |
| 160 | `--prefetch` | sync | 415.7 | 2.41 | 269.4 | 0.65 | 187 | 832 | 130 | 70 | 415.0 | 0 |
| 160 | `--prefetch` | pipelined | 405.3 | 2.47 | 269.6 | 0.67 | 187 | 832 | 135 | 70 | 405.8 | 3 |

**There is no gain.**  The pipelined cell is 0.1/0.6/0.9 ms *slower* with
`--prefetch` at 96/128/160 (+0.3 %, at or below this board's run-to-run spread),
and the daemon costs 10 ms more CPU per frame at 96 and 160.  Requests per
frame, launches per frame, kernel time, QPU duty and single-frame latency do
not move.  The mechanism engaged for **3 requests in a whole run of ~4 700** —
the frame's first submits, before the client settles into reap-then-send.

### 6.1 The client does not queue ahead

The client-side condition was tested directly with `VC4INFER_SUBMIT_EAGER=1`
(the flag added in this branch): send request N+1 before reaping N, and let
`remote_open_batch()` — which already waits for the set two submissions back
before writing into it — do the waiting after the send instead of before.  That
is what the depth-2 pipeline was described as doing; the default path does not
do it.

| 160, pipelined | wall ms | vs baseline | read early | prepared | gap ns/launch | kernel ns/launch |
|---|---:|---:|---:|---:|---:|---:|
| baseline: client reaps before it sends | 407.8 | — | 0 | 0 | 230 867 | 326 515 |
| client sends first, daemon prepares | 405.9 | -0.5 % | 66 | 66 | 229 568 | 326 535 |
| client sends first, daemon reads when idle | 401.3 | -1.6 % | 0 | 0 | 228 626 | 321 974 |

The daemon's prepared path was enabled in all three cells of this table (they
were taken before it became opt-in); the "prepared" column says what it
actually did.  The third cell is the client change alone.

With the eager client the daemon prepares 66 of ~250 requests under a launch —
and the period is unchanged.  Even the client change alone (401.3) is inside
noise; what it does do is move the client's wait from before the send to after
it, which is what the pipeline was supposed to do all along.

### 6.2 Where the period actually goes

The daemon's own counters, per launch, on the 160 pipelined cells: kernel
326 µs, gap 223 µs.  The gap split: **register kick 1.4 µs**, completion notice
about 29 µs, and the rest is the client's turnaround (reply → the client's next
request).  The kick and the notice are measurable directly:

| `--poll-us` | wall ms | daemon CPU ms/frame | kernel ns/launch | gap ns/launch |
|---:|---:|---:|---:|---:|
| 10 (default) | 406.9 | 80 | 326 881 | 223 893 |
| 1 | 399.4 | 80 | 320 060 | 222 029 |
| 0 (`sched_yield`) | 385.5 | 270 | 297 778 | 225 917 |

Polling tighter notices completion ~29 µs sooner per launch, which is 24 ms of
a 407 ms frame, and cuts the daemon's `kernel` figure by the same amount
because that figure is kick-to-notice.  It also quadruples the daemon's CPU:
0.27 s of a core per frame instead of 0.08.  That is the only lever the daemon
has on this frame, and it is 6 % of it at a large CPU cost (the trade
[CPU-BREAKDOWN.md](CPU-BREAKDOWN.md) §5 already flagged).

So the 148 ms of "daemon work and launch gap" is not 148 ms of daemon work:

* ~24 ms is the completion notice (removable at 3.4× the daemon's CPU);
* ~1 ms is the register sequence (832 × 1.4 µs);
* ~2 ms is the daemon's request preparation, which is what this branch set out
  to hide;
* the remaining ~110 ms of the 407 ms frame that is not QPU is the **client's
  own host work**: 130-135 ms of client CPU per frame, of which the per-frame
  dirty-range cache maintenance (`clean_ns` in the launch-path profile) is
  70-95 ms at 160 — `dc civac` over roughly 15 MiB of dirty ranges per frame.
  It is on the critical path because the client cannot build request N+1 until
  the scratch window of request N-2 is free, which is the 262 ms it spends
  blocked at `scratch` in the same run.

That is the honest answer to the question the branch was asked: **the daemon
is not the remaining serial cost, and neither is the launch gap.**  At 160 the
frame is 407.9 ms = 271.7 ms of QPU + ~25 ms of daemon tail + ~111 ms of client
host work.  Reaching 270-300 ms needs the client's per-frame CPU — mostly the
cache maintenance volume and the scratch-window serialisation — not a faster
daemon.

## 7. What is landed, and what is not

Landed, all measured here:

* the prepared-request path in `vc4d`, **off by default** (`--prefetch` turns it
  on).  It is correct, it is gated, and it costs about 10 ms of daemon CPU per
  frame when on, because a client that does not queue ahead makes the socket
  check find nothing.  A daemon that no pipelined client talks to must not pay
  for it;
* `--poll-us N`, default 10 (unchanged behaviour), so the completion-notice
  trade is measurable rather than folklore;
* `VC4INFER_SUBMIT_EAGER=1` in the engine, **off by default**: send before
  reaping, which is the condition under which the daemon's prepared path can
  ever find a request, and which is what the depth-2 pipeline was documented as
  doing;
* `tools/daemonpipe_check.py` (per-tensor digests through the daemon),
  `tools/overlap.py --pipe-only`, `vc4d/test/test_prefetch.{c,sh}`.

Not done, and now known to be the next thing that matters:

* the client's cache maintenance volume (70-95 ms/frame at 160).  Coalescing
  the dirty ranges, cleaning only what the GPU will read, or pushing the clean
  into the launch that consumes it are the candidates;
* the scratch-window serialisation (262 ms blocked at 160).  A third window
  does not fit at 160 ([OVERLAP.md](OVERLAP.md) §6), so this is a footprint
  question, not a pipeline one;
* reordering the reply behind the staged kick, which was sized at one
  `sendmsg` and is not worth a protocol-visible ordering change.

## 8. Notes for the next person

* Turn the prepared path on for a client that actually queues ahead
  (`VC4INFER_SUBMIT_EAGER=1` plus `--prefetch`), and measure the *client's* CPU
  in the same run: the two changes are only ever worth the sum of the smaller
  of (daemon preparation, client turnaround), and on this tree that sum is
  about 30 µs per launch.
* The daemon's per-launch gap counter is the cheapest way to answer "is the
  daemon the bottleneck" for any future change: kernel, gap, notice and kick
  are printed at exit, no protocol change and no client instrumentation.
* `--poll-us 0` is a trap: it makes the daemon spin with `sched_yield` and
  shows what the notice costs, but it triples the daemon's CPU and steals time
  from the client on a four-core board.
* The daemon cannot unlink its socket after it drops privileges (`/run` is
  root-owned).  A private daemon started with a socket in `/run` leaves the
  file behind if it is killed; remove it with `sudo` before the next start, or
  the next daemon's `bind()` meets a stale socket and clients get `E_IO`.
