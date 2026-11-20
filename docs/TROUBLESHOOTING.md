# Troubleshooting the VC4 stack

Everything here is a failure we actually hit on the reference Pi 3B, with the
symptom as it appeared, the cause, and the fix.  Commands assume the board
layout from [TUTORIAL.md](TUTORIAL.md): `/home/raspi/work/vc4-stack`, user
`raspi`, `vc4d` active, stock clocks.

## 0. Quick triage

| symptom (first line you see) | go to |
|---|---|
| `kernel jit failed ... status=2` | 1 |
| `UnsupportedNode: node ...` | 2.1 |
| `HostNotAllowed: node ... may not run on the host` | 2.2 |
| `NeedsDeviceDispatch: node ... needs device-side dispatch` | 2.3 |
| `GPU timeout (context is dead)`, `daemon marked dead`, `TIMEOUT` | 3 |
| `arena exhausted` while nothing is running | 3b |
| every launch takes ~60 s, then `E_BUSY` | 4 |
| output changes between runs, more elements over tolerance on repeats | 5 |
| `graph pool exhausted`, `conv_nhwc: out of arena memory` | 6 |
| `cma_heap_allocate` page-allocation failure, `vc4d` restart loop | 7 |
| `ARENA DEGRADED`, arena smaller than `--arena-mb` | 7 |
| `parameter -2 has bit pattern 0xc0000000` | 7a |
| `kernel cache full (128 entries)`, `out of arena/slots` | 7b |
| last bits differ from a CPU/float64 reference | 8 |
| build or venv dies from RAM, `/tmp` fills | 9 |
| `vc4_init` ok but the client arena is exactly 3 MiB (`VC4_DATA_BASE`) and `vc4_alloc` fails `have 0 B` | 14 |

## 1. `kernel jit failed ... status=2`

**Symptom.** The first run of a shape not already in the kernel cache fails
with a message built by `infer/src/vc4_kern.c`.  The kind and the generator
path vary; the form is:

```
kernel jit failed for conv_nhwc (kgen=/usr/lib/python3/dist-packages/vc4infer/kgen/kgen.py, status=2)
```

`status=2` is the generator's exit code, not a QPU error.  `kgen.py` exits 2
for:

* an unknown kernel *kind* (`unknown kernel kind '...'`), or
* an option the generator does not know (`kgen: <kind>: unknown option(s)
  --dil: the caller expects a kernel this generator cannot build, refusing to
  emit a silently different one`).

**Cause.** A client built against a newer engine is running an older
generator.  The two common forms:

1. A wheel or `.deb` from before the op-long-tail work shipped an older
   `kgen.py`; the older generator parsed options generically and silently
   ignored the new ones (that produced wrong kernels, which is why the
   generator now refuses them with rc=2).
2. A `.so` built in a development tree baked that tree's `kgen.py` path; after
   the tree moved or was deleted, the JIT fell back to a generator that did
   not match the library.  The resolution order is now
   `$VC4INFER_KGEN` -> the path baked at build time if it exists ->
   `/usr/lib/vc4infer/kgen/kgen.py`, but an old build can still have a bad
   baked path.

**Fix.** Match the generator to the engine, then clear the cache:

```sh
sudo apt update && sudo apt install --reinstall vc4infer vc4top
# or rebuild the engine from the matching source tree:
make -C infer
python3 -c "import vc4infer; print(vc4infer.__version__)"
```

Point the engine at the installed generator explicitly if needed:

```sh
export VC4INFER_KGEN=/usr/lib/python3/dist-packages/vc4infer/kgen/kgen.py
rm -rf ~/.cache/vc4infer/kcache
vc4infer selftest
```

The cache is keyed by a SHA-256 of `kgen.py` + `kops.py`, the ABI tag and the
generator argument vector, so a changed generator invalidates its old blobs
automatically.  An *unstamped* or stale blob is refused rather than run:

```
kernel <name> not in the cache (or stale) and JIT failed
```

If you are iterating on `kops.py`, use a fresh cache directory instead of
trusting the invalidation:

```sh
VC4INFER_CACHE=$HOME/scratch/tmp/kcache-test vc4infer run ...
```

A stale blob is what wedged the QPU during the op-long-tail work; do not
"fix" this by deleting the stamp check.

## 2. The three policy errors: `UnsupportedNode`, `HostNotAllowed`, `NeedsDeviceDispatch`

These are not crashes.  They are the contract that the engine never silently
moves arithmetic to the ARM.

### 2.1 `UnsupportedNode` - there is no kernel for this op/shape/attribute

**Symptom** (verified with a one-node `Sin` graph and `Engine(...).plan()`):

```
UnsupportedNode: node 'Sin_0' (Sin): no implementation
```

For a graph with an unsupported control-flow node it can be raised as:

```
UnsupportedNode: unsupported graph: graph input cond: only fp32 inputs
```

(the constructor rejects the graph before the policy check; non-fp32 graph
inputs are not supported).

**Cause.** `mode="strict"` (the default) refuses any node the executor cannot
dispatch.  The node name, ONNX op and the engine's reason are in the message
(`node`, `op`, `detail` attributes).  A shape-dependent constraint can pass the
static `nodes` report and still raise in `run()`; `run()` is authoritative.

**Fix.** Read the reason, then choose:

* re-export the model without the op/attribute (opset 13, static shapes,
  `.eval()` before export - see TUTORIAL.md section 6);
* replace the op with the supported set (e.g. `Sin` is not implemented at all;
  `Pow` only for exponent 2; `BatchNormalization` must be folded or run in
  eval mode);
* use the ONNX engine instead of `torch_vc4` if the kernel exists there but not
  in the PyTorch backend.

`vc4infer nodes model.onnx -v` names every unsupported node without touching
the GPU.  `vc4infer run`/`plan()` is the final authority for the concrete
input.

### 2.2 `HostNotAllowed` - a numeric op with no VC4 kernel, under the device policy

**Symptom** (verified with the same `Sin` graph, `mode="hybrid",
host_policy="device"`):

```
HostNotAllowed: node 'Sin_0' (Sin) may not run on the host under
host_policy='device': numeric op with no VC4 kernel: no implementation
(host_policy='any' runs it in the numpy reference evaluator; the operator still
needs a VC4 kernel)
```

`vc4infer nodes --mode hybrid --host-policy device` exits 4 and prints the
node under `host DEBT`.

**Cause.** Hybrid mode is opt-in.  `host_policy="device"` (the default for
hybrid) allows only plan-time constant/shape folding and orchestration; every
runtime *numeric* node must be VC4.  A `host-unwritten` node - a numeric op with
no kernel - raises instead of running in numpy.

**Fix.** This is the correct behaviour for a device-invariant run.  Either:

* keep `host_policy="device"` and remove/fold the node, or
* accept host arithmetic explicitly: `host_policy="structural"` allows runtime
  metadata and control flow on the host and still refuses `host-unwritten`;
  `host_policy="any"` additionally runs `host-unwritten` nodes in the numpy
  reference evaluator.  `any` is a compatibility escape, not a performance
  mode; the debt stays in the report.

A hybrid timing is never a VC4-only number.  The tool prints that caveat; do
not quote it as one.

### 2.3 `NeedsDeviceDispatch` - control flow the host-scheduled executor cannot run

**Symptom** (verified with a graph containing an `If` whose condition is cast
from a float input, `mode="hybrid", host_policy="device"`):

```
NeedsDeviceDispatch: node 'If_1' (If) needs device-side dispatch:
If/Loop/Scan branch execution on the device
(persistent QPU kernel + device-side work queue; the current executor is
host-scheduled, so control flow cannot be emulated without the host making the
branch decision)
```

**Cause.** `If`/`Loop`/`Scan` need the device to make the branch/iteration
decision.  The executor is host-scheduled and cannot pretend otherwise under
the device policy.  This is a named missing runtime feature, not "slow" and not
a missing kernel.

**Fix.** The same two choices as 2.2: remove the control flow from the exported
graph (unroll/constant-fold it before export), or opt into
`host_policy="structural"`/`"any"` and accept that the host is in the loop.
There is no device-side dispatch in this build.

## 3. GPU wedge after a bad kernel or a killed client

**Symptom.** A launch fails with a timeout:

```
engine error: copy2d_x: GPU timeout (context is dead)
(LAUNCH: GPU timeout)
```

`vc4top --once` may then show `ERRSTAT` with `0x40` set (e.g. `0x1040` instead
of `0x1000 VCD-idle`), and launches may keep timing out.

**Cause.** The QPU context is wedged.  The usual triggers, all measured here:

* killing a client in the middle of a launch - a `SIGPIPE` from
  `vc4infer ... | head`, `timeout` expiring during a launch, Ctrl-C;
* `sudo systemctl restart vc4d` issued while another process has a launch in
  flight;
* a genuinely bad kernel (the op-long-tail and two-groups-per-QPU experiments
  both did it);
* a legitimate launch that simply outran its timeout.

**What the daemon does now.**  Since `vc4d 1.2.4` (B9) a launch timeout no
longer kills the daemon.  It resets the V3D context - queue, VPM base, L2/SLC,
error latch, counters - under the board GPU lock and then **proves** the reset
with a launch of its own 4-instruction probe program (the last code slot is
reserved for it).  Only a probe that completes with no new error bits lets
launches through again; the journal shows which happened:

```
vc4d: TIMEOUT uid=1000 pid=4242 launch=17
vc4d: after a launch timeout: GPU context reset and verified with a probe
      launch (0.1 ms); accepting launches again
```

If the reset cannot be verified (the QPU is really stuck, the V3D is powered
down, the queue will not drain), the daemon does not become a zombie: it logs
`GPU context reset FAILED` and exits 75, and `Restart=always` starts a clean
one about two seconds later.  So:

1. **Retry the command once.**  A transient timeout is recovered in-process; a
   single failed launch is not evidence of a dead daemon any more:

   ```sh
   vc4infer run model.onnx --input img.jpg
   # only if that still fails:
   systemctl status vc4d --no-pager -l | tail -20
   ```

2. If the journal shows `GPU context reset FAILED` repeatedly, or `vc4top
   --once` shows `ERRSTAT` with `0x40` set, the QPU itself is wedged - **reboot**:

   ```sh
   sudo reboot
   # after it comes back:
   vc4top --once --no-color      # ERRSTAT must be 0x1000 VCD-idle
   vc4infer selftest             # 4/4
   ```

3. Never try to reset the V3D by unbinding the driver:

   ```sh
   # DO NOT DO THIS
   echo 3fc00000.v3d | sudo tee /sys/bus/platform/drivers/vc4_v3d/unbind
   ```

   It took the whole board down and needed a watchdog reboot in our tests.

Prevention: one GPU client at a time; never pipe a GPU command through
`head`/`grep -m`; do not restart `vc4d` while a launch may be in flight; give
long runs a timeout that is comfortably longer than the expected work, and do
not `kill -9` them mid-launch.

## 3b. `arena exhausted` with nothing running

**Symptom.** Every new client fails while `vc4top`/`top` show no GPU work:

```
vc4d: DENY uid=1000 pid=4242 op=ALLOC: arena exhausted
```

`vc4d` logs `clients=N` in STATS with no matching processes.

**Cause.** A client's arena region was tied to its *socket*, not to the process
that allocated it.  A client that forked (or leaked its descriptor) kept the
connection open after the allocating process exited, and the region was pinned
until the socket closed - forever, if the holder lived on.  This was G9.

**Fix.** `vc4d 1.2.4` anchors liveness on the peer pid from `SO_PEERCRED` and
reclaims a region when that process is gone, even if the socket is still open;
it sweeps every poll iteration, on `HELLO`, and before refusing a connection
for lack of a slot.  It logs what it reclaimed:

```
vc4d: reclaim pid=4242 uid=1000 (peer process is gone): releasing 1 region(s)
```

If you are on an older daemon, `sudo systemctl restart vc4d` remains the
workaround.  A client that forks and keeps *using* the inherited connection
after the parent exits is reaped by design: reconnect in the child instead.

## 4. `flock` deadlock

**Symptom.** Every launch blocks and then fails with `E_BUSY` or `GPU timeout
(context is dead)`; the GPU is idle the whole time.  If you inspect the lock,
something other than `vc4d` holds `/tmp/vc4-gpu.lock`.

**Cause.** `vc4d` takes `flock(LOCK_EX)` on `/tmp/vc4-gpu.lock` **per launch**,
to serialise all clients.  Since `vc4d 1.2.4` / `vc4infer 1.2.5` the engine
takes the same lock for the same window in **both** modes (that is what makes
G11 fixed: a `--backend direct` run and a live daemon now take turns instead of
overwriting each other's queue).  If the client also holds that lock (a test
harness or shell wrapper that copied the old direct-root pattern), the engine
waits for its own lock and fails after 30 s.  Older releases only hit this in
daemon mode.

**Fix.** Do not hold the lock in a client.  The engine and the daemon own it.
Remove the `flock` from wrappers and test harnesses:

```sh
# wrong in direct mode and in daemon mode (both take it internally):
flock /tmp/vc4-gpu.lock vc4infer run model.onnx --input img.jpg

# right:
vc4infer run model.onnx --input img.jpg
```

In this repository, `tests/util.h` no longer takes the lock in either mode for
exactly this reason.  If a stale process still holds the lock, find it with
`fuser -v /tmp/vc4-gpu.lock` (or `lsof`) and stop that process; do not delete
the lock file while the daemon is running.  If you must measure a single
direct-mode launch without the ~ms lock acquisition in the number, set
`VC4INFER_NO_GPU_LOCK=1` - but then nothing else may be using the QPUs.

The daemon's own unit needs `ReadWritePaths=/tmp` to create the lock at boot;
the shipped unit has it.

## 5. Cacheable-arena coherency: output changes with pool reuse

**Symptom.** The same model and the same input produce different outputs in
different processes, or the first run looks plausible and repeats drift.  With
the YOLOv8n 160 reference the counts were:

* 12-80 elements over the 5e-4 tolerance on a first run;
* ~22,700 elements over on repeats in one process;
* a direct probe: 400-600 of 4096 words stale after the QPU wrote a range,
  and still 0-432 stale after 2 MiB of unrelated churn (proving it was the ARM
  D-cache, not a late V3D L2 write-back);
* after `DC CIVAC` on the range: 0 stale.

**Cause.** The client's dma-buf arena mapping is **cacheable** on this kernel,
not write-combining.  `vc4d` maps its own `/dev/mem` view uncached, but the
client's `mmap` of the dma-buf is a normal write-back mapping.  Two host access
patterns need maintenance:

1. the host writes a word into a cache line the QPU has written; the
   pre-launch clean writes back the *whole* line, reverting the other words to
   the ARM's older copy (no host read required - a constant uploaded into a
   recycled pool offset is enough);
2. the host reads arena bytes the QPU wrote (pad streams, elementwise/softmax
   copies, the conv pad image, weight-panel reads) and gets a stale clean line.

**Fix.** Use a build with the coherency fix (1.2.x).  The client now:

* invalidates the destination range before `vc4__write`/`vc4__remote_upload`;
* invalidates the source before every arena->host copy (`vc4__invalidate_off`);
* keeps the `CIVAC` in `vc4_download`;
* exposes `vc4_invalidate_buf()` for zero-copy users (`Engine.view()`);
* has `infer/tests/test_coherency.c` in `run_tests.sh` so the bug cannot come
  back silently.

Do **not** try to fix this with `sleep`, a full-arena clean on every launch, or
by disabling the cache in user space.  A full 62 MiB clean costs milliseconds;
the engine tracks the dirty ranges (a 256³ matmul dirties ~1.5 MiB).  If you
are on an older build, upgrade; the 1.2.1 `selftest` reports 0/44,100 over in
20/20 fresh processes and 20/20 repeats after the fix.

## 6. `--arena-mb` too small for 320x320

**Symptom.** 320x320 fails at plan time with exit 3, and the message now names
both sides:

```
vc4infer: unsupported: unsupported graph: graph pool needs at least 24.8 MiB
  (weights + packed conv panels + the largest tensor) and this model's peak is
  about 41.6 MiB, but the client arena is 16.0 MiB with 5.8 MiB reserved for
  conv/elementwise scratch: start vc4d with a larger --arena-mb (or set
  VC4INFER_ARENA_MB) - about 38.5 MiB would do - or lower VC4_POOL_RESERVE_MB
```

(those are the YOLOv8n-160 numbers against a 16 MiB client region; a 320x320
model reports its own.  Older releases printed `graph pool exhausted: 589888 B
needed, 7845248 B used of 8388608 B` instead, because the pool was floored at
8 MiB - G10)

**Cause.** The default `vc4d` unit serves 96 MiB, which leaves the client its
64 MiB region.  A 320x320 YOLOv8n needs a ~68 MiB graph pool plus >=23 MiB of
per-conv scratch; the default cannot hold it.  Since `vc4infer 1.2.5` the pool
is sized from the model (weights + live activations + conv panels) and the
message above is what you get when the arena cannot supply it.

**Fix.** Use the 150 MiB recipe in TUTORIAL.md section 7.2:

```sh
sudo systemctl stop vc4d
sudo /usr/local/sbin/vc4d --socket /run/vc4d.sock --arena-mb 150 \
     --group vc4 --drop-user nobody --max-timeout-ms 10000 &
sleep 1
VC4INFER_ARENA_MB=145 VC4_POOL_RESERVE_MB=32 \
  vc4infer bench yolov8n_320.onnx --reps 3
```

`VC4INFER_ARENA_MB` must match what the daemon serves (145 client + 5 daemon
overhead).  If the allocation fails, go to section 7; a 150 MiB contiguous CMA
block is not always available.  Restore the unit's 72 MiB default when you are
done (`sudo systemctl restart vc4d`).

## 7. CMA exhaustion (VC4CL + the engine, or a too-large arena)

**Symptom.** `vc4d` starts and immediately falls back or restarts; the kernel
log has a page-allocation failure from `cma_heap_allocate`:

```
vc4d: page allocation failure: order:5, mode:0x40cc0(GFP_KERNEL|__GFP_COMP)
...
 cma_heap_allocate+0x120/0x290
 dma_heap_ioctl+0x208/0x318
```

`vc4top --once` shows `cma: CMA 256 MiB / 3 MiB free`, and
`/proc/meminfo` has `CmaFree` in the low megabytes while `CmaTotal` is 262144
kB.  A requested 150 MiB arena fails even though 150 < 256.

**Cause.** CMA is a single contiguous pool shared by the DRM driver, `vc4d`
and any other dma-buf user.  It fragments and is held for the lifetime of the
allocations:

* the `vc4` KMS driver holds framebuffers and scanout buffers;
* `vc4d` holds its arena (72 MiB by default);
* a VC4CL/OpenCL process holds its own buffers;
* the engine's dma-buf is a large contiguous block, not pages.

A 128 MiB *contiguous* block was already unreliable at boot in our tests: the
daemon's shrink ladder fell back to 64 MiB after the DRM driver had taken its
share.  A 150 MiB block needs an unusually clean boot.

**Fix.**

1. Do not run VC4CL and the engine at the same time.  They are two GPU users
   with two CMA allocations; the board has 256 MiB total.  Stop the OpenCL job
   (or the engine) and wait for the process to exit before starting the other.
2. Stop `vc4d` before allocating a larger arena, so its 72 MiB is released:

   ```sh
   sudo systemctl stop vc4d
   grep -E 'CmaTotal|CmaFree' /proc/meminfo
   ```

   If `CmaFree` is still too small, reboot.  There is no way to defragment CMA
   from user space.
3. Prefer the default 72 MiB unit for 160x160.  Use 150 MiB only for the 320
   recipe and restore 72 afterwards.
4. If the daemon logs an allocation failure but stays active, check which
   arena it actually got:

   ```sh
   sudo journalctl -u vc4d -n 30 --no-pager
   vc4infer --version          # prints the live backend and socket
   ```

   The daemon's shrink ladder tries 32/16/8 MiB if the requested size fails;
   a client that asks for 64 MiB will then fail with `E_NOMEM` even though the
   daemon is "running".

   Since `vc4d` 1.2.7 the requested size is retried (0/200/400 ms) and only
   then does the daemon fall back, with a line that names the size obtained,
   the size requested and `CmaTotal`/`CmaFree`:

   ```
   vc4d: arena attempt 1/3 for 96 MiB: 96 MiB: cma_heap_allocate failed
         (Cannot allocate memory; CmaFree 19 MiB of 256 MiB)
   vc4d: ARENA DEGRADED: obtained 32 MiB of the 96 MiB requested.
   vc4d:   CMA: 19 MiB free of 256 MiB total (/proc/meminfo); CMA is one
   ...
   ```

   **The degraded arena is no longer permanent.**  Once the arena is idle and
   CMA has the full block again, the daemon exits 75 and systemd restarts it
   with the requested size:

   ```
   vc4d: arena was 32 MiB of the 96 MiB requested; CMA now has 180 MiB free and
         the arena is idle - exiting so a fresh daemon claims the full block
         (no reboot needed)
   vc4d: exit: restarting ...
   vc4d: restart reason: the arena was degraded and CMA has recovered
   ```

   If CMA never recovers the daemon stays up at the smaller size and logs that
   it is waiting, instead of looping — free CMA (steps 1-3) and it restores
   itself.  Repeated failed attempts are spaced out (1 min doubling to 6 h):

   ```
   vc4d: arena restore: 1 attempt(s) already failed; next try in 55 s (a
         fragmented CMA can refuse the full block even when enough is free in
         pieces).  Stop the other CMA user and restart vc4d to retry now.
   ```

   To retry immediately, free CMA and run `sudo systemctl restart vc4d` (a
   restart clears the stamp).  The state is one file per requested size,
   `/run/vc4d/arena-restore-<MiB>.state`; delete it to clear the backoff by
   hand.  A board whose CMA is fragmented so that the block cannot be had at
   any `CmaFree` value (measured here: 107 MiB free, a 96 MiB block refused)
   needs a reboot — CMA fragmentation is not defragmentable from user space.

## 7a. A parameter of −2.0 or below is refused in managed mode

**Symptom (before `vc4d` 1.2.7).** Any op whose scalar attribute is at or below
−2.0 fails in managed mode while the same call works with `--backend direct`:

```
vc4infer: unsupported: LeakyRelu: unary: vc4_unary leakyrelu: parameter -2 has
  bit pattern 0xc0000000, which vc4d refuses as a possible GPU bus address
  (every float <= -2.0 is in 0xC0000000..0xFFFFFFFF) ...
vc4infer: unsupported: Clip: unary: vc4_unary clip: parameter -3 has bit
  pattern 0xc0400000 ...
```

**Cause.** Every fp32 value ≤ −2.0 has a bit pattern ≥ `0xC0000000`, which is
the GPU bus window.  The daemon validated each materialised uniform word as
"inside this arena, or not an address", so it could not tell `-2.0f` from a bus
address; the engine refused the call first so the message would be honest.

**Fix.** Upgrade to `vc4d`/`vc4infer` 1.2.7 or later and restart the daemon
(`sudo systemctl restart vc4d`).  `LAUNCH` now carries a per-slot address mask
(`VC4D_LAUNCH_UNI_MASK`): the engine marks the uniform slots that hold GPU
addresses, the daemon range-checks exactly those, and every other word is a
value.  Old clients that send no mask keep the old behaviour unchanged.

## 7b. `kernel cache full (128 entries)` or `out of arena/slots`

**Symptom.** A long session — `vc4infer util model.onnx` on a full YOLO graph,
a training run, a benchmark over many shapes — fails on a shape it has not run
before:

```
vc4infer: unsupported: ... UPLOAD: out of arena/slots
kernel cache full (128 entries)
```

and every later shape fails too, until `sudo systemctl restart vc4d` (or a new
process).  The shapes that were already loaded keep working, which makes it look
like a model problem.

**Cause (before 1.2.7).** Both caches were fixed arrays with no eviction: the
daemon had 128 program slots (127 usable, one is the recovery probe) and the
engine a 128-entry kernel table.  "More distinct kernels than slots" was
therefore a permanent failure instead of a cache miss.

**Fix.** Upgrade to 1.2.7 or later.  Program slots are an LRU keyed by the
kernel name, the engine's kernel table is an LRU that keeps each kernel's code,
and a launch whose daemon slot was evicted re-uploads it transparently.  The
daemon counts evictions in `STATS.evictions` and logs them at `--verbose`.  A
session may now touch hundreds of distinct shapes; the cost is a re-upload per
evicted kernel, not a restart.  If you still see `kernel cache full`, the client
is older than the daemon: `vc4infer --version` shows both versions.

## 8. FPU truncation and denormal flushing

**Symptom.** A result differs from a CPU/float64 reference in the last bits;
a tolerance that is tight enough fails on a handful of elements, and the
difference is not reproducible as a logic error.  YOLOv8n 160 reports
`0/44,100 over 5e-4` but `max abs 2.04e-02` on large-magnitude outputs; a
single `fmul`/`fadd` can be one ulp low.

**Cause.** Two hardware facts, measured on silicon:

* **The FPU truncates (round toward zero).** `1.0 + 0.75 ulp -> 1.0`,
  `-1.0 + -0.75 ulp -> -1.0`, and a product closer to the next float up
  returns the lower one.  Both `fadd` and `fmul` round toward zero; there is no
  IEEE round-to-nearest in the ALU.
* **The FPU flushes denormals.**  The datapath and bitwise ALU ops preserve
  them (`Abs` returns 1.4e-45), but `fmax`/`fmin`/`fmul`/`fadd` return 0 for
  denormal operands and results.  Any design that produces or consumes a
  denormal diverges from an emulator that keeps them.  This is why the engine
  does not use the magic-exponent trick near zero.

**Fix.** Verify with a documented tolerance, not bit equality, wherever
truncation can bite:

* elementwise `mul` is implemented as a correctly-rounded product (Veltkamp
  split + exact residual + integer round-to-nearest-even), so `mul` can be
  bit-exact;
* `add`/`relu` are bit-exact only for exactly representable sums;
* accumulation order changes the last ulp.  The engine's tests use
  `atol=rtol=5e-4` for the graph, `1e-3` relative for gradients, and report
  the actual max abs/rel next to the pass/fail.  Do the same.

If a reference differs by more than a few ulps, it is not this: check the
coherency section (5), the input layout (NCHW vs NHWC), and whether your
reference used the same padding/phase decomposition.  Full detail:
[NUMERICS.md](NUMERICS.md), and [ONNX.md](ONNX.md) section 11 for the denormal
probe.

## 9. `/tmp` is tmpfs (RAM), not disk

**Symptom.** A build, `pip install`, venv or kernel-cache directory under
`/tmp` consumes hundreds of megabytes of RAM; `MemAvailable` collapses, the
board swaps, a compile is killed, or the board resets.  On a fresh Raspberry Pi
OS, `df -hT /tmp` shows `tmpfs` (about 452 MB), not ext4.

**Cause.** Raspberry Pi OS mounts `/tmp` as tmpfs.  On a 1 GB board that is
half the memory, and tmpfs pages cannot be reclaimed to disk.  The 2 GB
swapfile does not help tmpfs.

**Fix.** Move `/tmp` to the root filesystem once:

```sh
sudo systemctl mask tmp.mount
sudo reboot
findmnt /tmp          # prints nothing: /tmp is on /
df -hT /tmp           # /dev/mmcblk0p2  ext4  58G
```

*(Already applied on the reference board; the commands are from
[BUILD.md](BUILD.md) and were verified there.)*

This board is already masked:

```sh
$ systemctl is-enabled tmp.mount
masked
$ findmnt /tmp
$ echo $?
1
$ df -hT /tmp
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/mmcblk0p2 ext4   58G   21G   36G  37% /
```

Use `$HOME/scratch/{tmp,refs,evidence,logs}` for anything large and export
`TMPDIR`; `mktemp -d` follows it.  Lock files and sockets
(`/tmp/vc4-gpu.lock`, `/run/vc4d.sock`) are tiny and fine either way.

Measured effect on YOLOv8n 160 (`vc4infer util`, stock clocks):

| | before | after |
|---|---:|---:|
| wall / FPS | 539-543 ms / 1.84 | **478.7 ms / 2.09** |
| GPU (pack+kernel) | 321 ms | **259 ms** |
| QPU duty | 59% | **54%** |
| `MemAvailable` at run time | ~189 MB | **647 MB** |

That is a before/after observation, not a controlled A/B (the board cannot be
returned to the old layout without re-enabling tmpfs); the mechanism is
ordinary memory pressure, and the spread across 3 reps was small.

## 10. `out of arena memory` on the second GPU process

**Symptom.** The first process exits, the second one (torch or the engine)
fails with `out of arena memory` / `arena exhausted` even though nothing is
running.  The backend retries for ~10 s and may still fail.

**Cause.** On a `vc4d` older than 1.2.3 the arena region of a client that
exits is not reclaimed promptly (G9/B10); the daemon still accounts for it when
the next client asks.  This is separate from CMA exhaustion (section 7).
Since 1.2.3 the region is released from the peer pid as soon as the process is
gone, so the second client gets the full arena.

**Fix.** On 1.2.3+ nothing; on an older daemon, restart it between long-lived
GPU processes:

```sh
sudo systemctl restart vc4d
```

The torch test runner does this per test file (`tests/run_all.sh`) for exactly
this reason.  Do not run two GPU clients concurrently and then blame the
second one; the `vc4` group shares one arena by design.

## 11. `operator torchvision::nms does not exist` (Ultralytics)

**Symptom.** `YOLO(...).predict(...)` aborts during warmup:

```
RuntimeError: operator torchvision::nms does not exist
```

**Cause.** Ultralytics' `AutoBackend.warmup()` unconditionally imports
`torchvision` to bind NMS.  This board's torchvision 0.29.2 registers no
operators against its torch 2.15.0+cpu build.

**Fix.** Let `vc4infer.contrib.ultralytics.register()` install its warmup shim
(the default `patch_warmup=True`):

```python
import vc4infer.contrib.ultralytics as vu
vu.register()
```

On a normal torch/torchvision build, pass `patch_warmup=False` so stock
behaviour is preserved.  If you do not want any Ultralytics patch, use
`vc4infer.yolo.Detector` or `Engine.run` + `decode_yolo`; it is the same QPU
graph.

## 12. `torch.device("vc4")` raises `Invalid device string`

**Symptom.**

```
RuntimeError: Invalid device string: 'vc4'
```

**Cause.** `torch.utils.rename_privateuse1_backend("vc4")` runs at
`import torch_vc4`, after torch's device-string parser has been populated.
The parser still accepts only the built-in `"privateuseone"` spelling in torch
2.15.

**Fix.** Use `torch_vc4.device()` (or `torch.device("privateuseone")`).
Tensors still report `device(type='vc4')`.  This is documented, not a bug to
patch around.

## 13. `torch_vc4` cannot allocate (`out of arena`)

**Symptom.**

```
torch_vc4: out of arena ... (bytes ...)
```

**Cause.** `torch_vc4` allocates its tensors from a fixed 24 MiB region inside
the engine's 64 MiB client arena.  A training loop that leaks tensors, or a
model whose activations exceed the region, runs out.  The backend does not
silently spill to the CPU.

**Fix.**

* Check `torch_vc4.arena()` for `region_bytes` / `arena_used`.
* Keep the model and batch small; the region is 24 MiB, not 64 MiB.
* `TORCH_VC4_ARENA_MB` overrides the region size, but the engine's client
  arena is still 64 MiB by default; a larger value needs the 150 MiB daemon
  recipe.
* A leak shows as steadily growing `arena_used` across iterations.  The
  allocator's `alloc_info()` and the 3000-tensor churn check in
  `tests/test_ops.py` guard the known leak (a `DataPtr` deleter that never ran
  because its context was null); update if you are on an old build.

## 14. `vc4_init` succeeds but the client arena is 3 MiB and nothing allocates

**Symptom.** A plain C client (the test binaries, a probe) prints:

```
init 0 (ok)
arena_bytes=3145728 used=3145728
alloc: st=-2 out of arena memory
  last_error: arena exhausted: need 262144 B, have 0 B
              (arena 3145728 B, 3145728 B in pool, 0 B in scratch)
```

`3145728` is `VC4_DATA_BASE`: the region the engine keeps for its code and
uniform slots.  A client that reports *that* as its whole arena has no data
space at all.  Python and the installed `.deb` are unaffected, which is the
give-away.

**Cause.** A **mixed build**.  `struct vc4_ctx` gained a field (the G11
`lock_fd`), the Makefile had no header prerequisites, and `vc4_dev_remote.o`
was left compiled against the old struct.  It wrote `arena_bytes`,
`arena_top` and `scratch` 8 bytes below where the freshly compiled objects
read them, so the client saw the old `arena_top` (3 MiB = `VC4_DATA_BASE`) as
its arena size.  Nothing fails at link time: both objects define the same
symbols with the same signatures.

**Fix.** Rebuild from clean, in this order:

```sh
make -C vc4d
make -C infer clean && make -C infer
```

Since `vc4infer 1.2.6` a mixed build is refused instead of silently producing a
3 MiB arena: `vc4_init` checks the context it got and prints
`vc4infer: vc4d context inconsistent after init: arena 3145728 B … stale .o
files`, and the grant itself is validated against the daemon's HELLO reply.  If
you see either message, run the two `make` commands above.

## 15. Exit codes and where to look

| exit | meaning | first thing to run |
|---:|---|---|
| 0 | success | - |
| 1 | selftest element over tolerance, or a model failed | `vc4infer selftest` |
| 3 | `UnsupportedNode` / arena exhausted | `vc4infer nodes model.onnx -v` |
| 4 | `HostNotAllowed` / `NeedsDeviceDispatch` / host debt under `device` | `vc4infer nodes --mode hybrid --host-policy device` |
| 6 | `NeedsDeviceDispatch` (control flow) | the node name in the message |
| 1/255 | daemon down, socket missing, permission | `systemctl status vc4d`, `id` |

`vc4infer --version` prints the live backend and socket; if it says `vc4d is
not running`, start the daemon rather than running as root.  The client should
never open `/dev/mem`; if it does, you are on the wrong backend
(`VC4INFER_DAEMON=0` is the direct-root path used only by the low-level test
suite).

