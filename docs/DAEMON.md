# Running the inference engine unprivileged through vc4d

Status: **done**.  `VC4INFER_DAEMON=1` in `libvc4infer` selects a backend that
talks to `vc4d`; the direct `/dev/mem` path is unchanged when the flag is
unset.  The full suite passes as user `raspi` with every shape and gradient
check clean.

Environment: Raspberry Pi 3B, Debian 13 arm64, **stock clocks** (core 400 MHz,
V3D 300 MHz, `throttled=0x0`), one boot for every number below.  The board is
shared with concurrent work; every GPU run was serialised by
`/tmp/vc4-gpu.lock` and the runs reported here were taken in a quiet window.

## What changed

- `infer/src/vc4_dev_remote.c` (~480 lines): connect to `/run/vc4d.sock`,
  `ALLOC` one 64 MiB region and `mmap` the dma-buf fd, upload each kernel blob
  on first use, launch with `UNI_IN_ARENA | WAIT` in one round trip, and do
  the client-side cache maintenance.
- `infer/src/vc4_dev.c`: dispatch at `vc4_init`, `vc4__launch`,
  `vc4_upload`, `vc4_download`, `vc4__l2_flush`, `vc4__write`,
  `vc4__kernel_load_file`, `vc4__wait_idle`, `vc4__health`, `vc4_shutdown`.
- `vc4d` protocol version 5 (see "Protocol 5" below; the version-2 `LAUNCH`
  message is unchanged and still served by the client ladder): `ALLOC` replies
  with the region's GPU bus address; `LAUNCH` gains `UNI_IN_ARENA` (uniform
  words read from the client's arena) and `WAIT` (kick → idle → counters → L2
  write-back in one reply).  Since 1.2.7 it also gains `UNI_MASK` (a per-slot
  mask saying which uniform words are addresses, which is what makes
  parameters ≤ −2.0 possible), and `UPLOAD` may carry the kernel name that
  keys the program-slot LRU.  Both are opt-in.  Protocol 5 raises the program
  table to 256 slots (255 usable) and moves the code/uniform zones: a real
  detector needs 133 distinct kernels, and against 128 the LRU freed five of
  them on every frame.  The unit uses `--arena-mb 96` (CMA is 256 MiB; 96
  leaves the engine's full 64 MiB client region).
- Op code (`vc4_gemm.c`, `vc4_ew.c`, `vc4_conv.c`, `vc4_convbwd.c`,
  `vc4_train.c`) is unchanged except for dirty-range marks where the CPU
  writes buffers that the GPU later reads.

## Security model

The client holds only an `AF_UNIX SOCK_SEQPACKET` connection and a dma-buf
mapping.  `/dev/mem`, the V3D register window, the CMA allocation and the
physical-base discovery are all in `vc4d`, which closes those descriptors and
drops to `nobody` with an empty capability set before serving requests.  The
daemon validates every uniform word as "inside this arena" or "a literal
outside the GPU bus, V3D MMIO and arena-physical windows", keeps one launch in
flight, bounds the timeout, and goes permanently dead after a GPU timeout.
`vc4d --strict` adds a SHA-256 program allowlist and refuses raw SRQ writes for
appliances.  Honest limits: a QPU program is raw machine code and can DMA
anywhere; the `vc4` group is a mutual-trust group and all clients share one
arena.

Client evidence, captured while a suite client (`test_train`) was running:

```
```
client pid=1312
Uid:  1000 1000 1000 1000
Gid:  1000 1000 1000 1000
CapEff: 0000000000000000
NoNewPrivs: 0
open fds:
 0 -> /dev/null
 1 -> /tmp/vc4-client-evidence.log
 2 -> /tmp/vc4-client-evidence.log
 3 -> /dev/null          (the dummy lock fd: the daemon owns the real lock)
 4 -> socket:[11314]     (AF_UNIX to vc4d)
PASS: uid 1000, CapEff=0, no /dev/mem
```

The daemon's own state (same boot): uid 65534, `CapEff=0000000000000000`,
`NoNewPrivs=1`, and no `/dev/mem` fd after startup.
```

## Test results

`REPS=5 VC4INFER_DAEMON=1 ./run_tests.sh` as user `raspi` (no sudo):

```
`test_ew` 5/5 (add/mul/relu/sigmoid/silu/relu_bwd/rsqrt, 0 mismatches),
`test_matmul` 6 shapes × 5/5 (128³, 256³, 64×48×32, 20×36×44, 16³,
512×64×288; 0/16384 etc.), `test_conv` 6 cases × 5/5 (3×3 and 1×1, bias and
no bias, pad 0/1), `test_train_ops` (transpose, relu_bwd, rsqrt, matmul_bt,
softmax+CE, 0 mismatches), `test_grad` GRADCHECK PASSED, `test_convgrad`
CONV GRADCHECK PASSED (dW/dX 6×6 pad 0/1, pool fwd/bwd), `test_train --opt
sgd|adam` 300 steps TRAIN PASSED, `test_convnet --opt sgd|adam` 150 steps
CONV TRAIN PASSED, and `ALL TESTS PASSED` at the end.  Best suite numbers:
256³ 8.44 GFLOP/s, 128³ 5.37 GFLOP/s (daemon, GPU time).
```

## Measurements

Same binary (`tests/bench_matmul.c`), same boot, 5 reps, best wall.  `gpu` is
the engine's pack+kernel time; `wall` is what the caller waits for.

| size | path | gpu ms | gpu GFLOP/s | wall ms | wall GFLOP/s | overhead (wall) |
|---|---|---:|---:|---:|---:|---:|
| 256³ | direct root | 3.962 | 8.47 | 3.980 | 8.43 | — |
| 256³ | vc4d | 3.998 | 8.39 | 4.289 | 7.82 | **+0.309 ms, 7.8%** |
| 128³ | direct root | 0.706 | 5.94 | 0.723 | 5.80 | — |
| 128³ | vc4d | 0.771 | 5.44 | 0.942 | 4.45 | **+0.219 ms, 30%** |

The GPU work is within 1% (256³) / 9% (128³) of the direct path; the wall
difference is protocol plus cache maintenance.  Two launches cost two round
trips at ~50 µs and one dirty-range clean (~1.5 MiB at 256³, ~320 KiB at
128³).  At 256³ that is 7.8%; at 128³ the fixed cost is 30% of a 0.72 ms call,
which is launch-bound.  First 128³ run after boot is colder (1.25 ms wall).

## Findings worth keeping

- **The CMA dma-buf mapping is cacheable, not write-combining.**  The first
  port launched with all-zero uniforms: the daemon's uncached `/dev/mem` read
  did not see the client's stores even after `dsb`.  `dc civac` on the written
  range fixed it.  The vc4d demo never noticed because it always called
  `vc4c_sync()`.  Client-side clean before launch, invalidate before CPU read;
  every line is cleaned before a launch, so a later CIVAC cannot write a stale
  line over a GPU result.
- Cache maintenance must be precise.  A full 62 MiB sweep costs ~7 ms at the
  measured 8–10 GB/s; the engine's dirty set for a 256³ matmul is ~1.5 MiB.
  Range tracking covers uploads, uniform blocks, per-op scratch and the four
  persistent output copies.  Missed ranges showed up as `conv` bias failures
  (the bias is folded into the packed panel *after* the pack launch) and one
  odd-shaped matmul (scratch reused between ops without a reset).
- The daemon's per-client program table must hold a whole process's kernel
  set: one program per shape.  With 16 slots, later conv-gradient kernels
  failed to upload and the test reported a misleading "out of arena memory".
  64 fixed it; the arena's code region only has room for ~67.
- Arena sizing: a 64 MiB daemon arena leaves a client `64 − 2` MiB, one
  megabyte less than the direct path, which made `test_convgrad` run out
  late.  The unit uses 72 MiB, which yields the client's full 64 MiB region
  and a few MiB spare.  A 128 MiB arena is not reliable at boot: contiguous
  CMA may have fragmented under the vc4 DRM driver, and the shrink ladder then
  fell back to 64 MiB and reintroduced the failure.  The final suite ran with the
  64 MiB client region and the 72 MiB daemon; `test_convgrad` was also verified
  separately against this exact unit.

## Client arenas and CMA accounting

`vc4d` allocates **one** dma-buf from the 256 MiB CMA pool at startup and
divides it into per-client regions, so a managed engine does not allocate CMA.
Closing an engine (`vc4_shutdown` -> `vc4c_close`) sends `CLOSE`, and the
daemon frees the region before it replies; `arena_used` also returns to 0 B
when the peer socket outlives the process that allocated (B10).

Measured on this board with the 96 MiB unit: 16 engine open/close cycles in one
process left `arena_used == 0 B` after every close, and `CmaFree` was flat to
the kilobyte.  Three engines opened at once with `arena_mb=24` also fit, and
all three regions were released.

`CmaFree` is nevertheless a poor leak counter, because the pool is declared
`reusable` and page cache occupies it between CMA allocations.  One short
Python run moved `CmaFree` from 104300 kB to 62256 kB with no GPU client open;
`echo 3 > /proc/sys/vm/drop_caches` moved it back to 104296 kB.  A low
`CmaFree` at idle therefore does not mean an arena leaked.

The kernel lines

    cma: __cma_alloc: alloc failed, req-size: 16384 pages, ret: -16
    cma: __cma_alloc: alloc failed, req-size: 8192 pages, ret: -16

are the *direct* backend's `gpu_alloc()` ladder (64 MiB, then `want/2`), not
vc4d: a managed client asks the daemon for a region and never calls
`DMA_HEAP_IOCTL_ALLOC`.  With two daemons holding 48 + 96 MiB, ~58 MiB of free
pages split by page cache cannot satisfy a 64 MiB contiguous request, and the
32 MiB retry can still fail on fragmentation.  `infer/tests/test_hybrid.py`
refuses to run on the direct backend for this reason.

## G2/G3/G4 (1.2.7): the three that turned a transient condition permanent

Each of these was a case where a moment of bad luck (CMA busy, more shapes
than slots, one parameter below −2.0) became a state the session could not
leave.  Measured on this board, stock clocks, fanless:

| # | before | after | test |
|---|---|---|---|
| G2 | `--arena-mb 16` under CMA pressure came up at **8 MiB**; 8 s after the pressure was released (31 MiB `CmaFree`) it was still **8 MiB**, i.e. until a reboot | retried 0/200/400 ms, then fell back **loudly**; on an idle arena with CMA recovered it exits 75 and the replacement daemon is at the full **16 MiB** | `infer/tests/test_daemon_g2.sh` |
| G3 | 127/250 uploads; `upload #127 failed: out of arena/slots`, every later kernel unavailable | **250/250 uploaded, 250/250 launched**, 374 LRU evictions, 127 resident; the first kernel runs again after a re-upload. Engine side: 124/200 → **200/200** | `vc4d/test/test_prog_lru`, `infer/tests/test_daemon_kernels` |
| G4 | `LeakyRelu(-2)` and `Clip(-3,3)` refused with `parameter -2 has bit pattern 0xc0000000`; `Pow(x,-2)` ran (it lowers to `reciprocal(x*x)` and never carried −2 as a uniform) | all three run through the daemon: `max|err|` 0, 0, 9.5e-07 | `vc4d/test/test_uni_mask`, `infer/tests/test_daemon_uni`, `infer/tests/test_daemon_values.py` |

Why the G2 restore is a restart and not a re-allocation in place: by the time
`vc4d` serves clients it has dropped to `nobody` with an empty capability set
and closed `/dev/mem`, so it cannot open a larger dma-buf or map a new
physical base.  Keeping `/dev/mem` open in the unprivileged process would give
up the property the daemon exists for.  The exit is gated on an idle arena
(`arena_used == 0`, no clients, no launch in flight) **and** on
`CmaFree + arena_bytes >= requested + 8 MiB` — the daemon's own block counts,
because exiting releases it.  Deliberate restarts are spaced by a stamp in
`/run/vc4d/arena-restore-<MiB>.state` (1 min doubling to 6 h, cleared by a
start that gets the full size), so a genuinely fragmented board cannot turn
the restore into a restart loop.  Measured on this board after the upgrade:
48 MiB of 96 → the gate fired, the daemon exited 75, and the replacement got
96 MiB; and in a deliberately fragmented state (107 MiB of `CmaFree`, no
96 MiB block) it stayed at 48 MiB and logged `arena restore: 1 attempt(s)
already failed; next try in 55 s`.

The G4 wire change adds no bytes: `vc4d_launch_req.pad` became `uni_addr`, and
the flag is `VC4D_LAUNCH_UNI_MASK` (bit 2).  Program names ride after the code
in `UPLOAD` (`VC4D_UPLOAD_NAMED`, bit 0) so the request struct is unchanged.
`HELLO.flags` advertises `VC4D_F_NAMED_PROGS`, `VC4D_F_UNI_MASK` and
`VC4D_F_ARENA_RESTORE`; the engine checks them and falls back to the 1.2.6
calls, which is why a new engine works against an old daemon and vice versa.
- Lock topology: the daemon takes `/tmp/vc4-gpu.lock` per launch, so the
  tests must not hold it (otherwise both wait and the launch fails as busy).
  The engine retries `E_BUSY` for a minute, which is what let it coexist with
  long `qrun_gemm` runs from a concurrent session.  Since the G11 fix the direct path
  takes the same lock for the same window, so a direct-mode run next to a live
  daemon takes turns instead of timing out (`conv_nhwc_dil: GPU timeout`);
  `tests/util.h` therefore never takes it in either mode.  The unit needs
  `ReadWritePaths=/tmp` to create the lock at boot.
- No kernel needed a fallback to root.  `vc4_gemm`, `vc4_ew`, `vc4_conv`,
  `vc4_convbwd`, `vc4_train` and the softmax/pool paths all run through
  `LAUNCH_WAIT`.

## Protocol 3: batch launch submission

Protocol 3 adds one op, `VC4D_OP_LAUNCH_BATCH` (13), and one HELLO flag,
`VC4D_F_LAUNCH_BATCH` (bit 5).  The single `LAUNCH` message is unchanged and
still what every other client uses.  A 1.2.10 daemon accepted version 2 as
well as 3 and replied with 3 and the flag, so an old client kept working: it
never sent the op and the daemon never saw it.  The client library mirrors this
- it starts HELLO at its own version, drops one version at a time on
`E_VERSION`, and then adopts whatever version the HELLO reply carries - so a
new engine also works against an old daemon.  The reply matters because 1.2.9
accepts a version-3 HELLO and still expects every later request to say 2;
without honouring the reply a client gets `bad header` on its first `ALLOC`.
Both behaviours are exercised against a protocol-level fake and, separately,
against the packaged daemon.

The request is

```
struct vc4d_batch_req { uint32_t count, flags, reserved[2]; }
count * struct vc4d_batch_ent { kind, prog, timeout_ms, nqpu, nuni, flags,
                               uni_handle, uni_off, uni_addr, handle, dir; }
```

and the reply is the header plus `count * struct vc4d_batch_item { status,
launch, errs, pad, wall_ns, cycles, pad2 }`, in request order.  `detail0` is
how many entries ran and `detail1` the index of the first failure (0xffffffff
when none).  A `LAUNCH` entry carries exactly what `struct vc4d_launch_req`
carries; `VC4D_LAUNCH_UNI_ARENA` is required because the batch has no room for
a per-word uniform array, `VC4D_LAUNCH_UNI_MASK` is optional and means the same
thing, and `WAIT` is implied.  A `FENCE` entry is a `struct vc4d_sync_req`
(region + direction) applied at that point in the order.

Execution is the important part:

* **Ordered and complete.**  Entry `i+1` is kicked only after entry `i` has
  completed, with the same L2 write-back, the same timeout accounting and the
  same GPU lock acquisition as a separate `LAUNCH_WAIT`.  The GPU sees the
  same sequence it would have seen from one message per launch.
* **Per-entry validation.**  `launch_one()` is literally the body that
  `LAUNCH` used before; a batch entry and a single launch cannot drift apart in
  what they accept.  `nuni` must match the count declared at `UPLOAD`, the
  uniform block must lie in the client's region, and the address mask is
  checked per word.
* **Stop on the first failure.**  Entries before it have completed, it carries
  its own `VC4D_*` status, and every later entry is `VC4D_E_NOTRUN` and never
  touched.  A timeout therefore leaves the daemon in the same state (dead
  context, recovery attempted) it reaches after a single timed-out launch, and
  the client never has to infer which entries ran.
* **What it is not.**  It is not a way to overlap host work with the GPU, and
  it does not let a client put CPU work between entries.  A launch whose input
  the client computes, or whose result the client reads, is a batch boundary.
  The engine enforces the memory side of this in the remote backend: uploads,
  downloads, dirty marks, scratch reuse, explicit invalidations and the L2
  flush all drain the queue first.

On the client side, `vc4c_launch_batch()` takes an array of
`vc4c_batch_ent` and one uniform region.  The engine's remote backend snapshots
each launch's 64 B x nqpu uniform block into a private shadow slot before
queueing it (96 KiB at 0x510000 with protocol 5, between the last kernel
uniform area and the data heap; 0x290000 with protocol 2..4), so a queued
launch cannot be corrupted by the next launch's uniform writes.  `vc4_batch_begin()` / `vc4_batch_end()` are explicit: a caller
that does not arm them gets the synchronous path byte for byte.  The Python
binding arms around one C API call and reads the result in `_check`, which
makes the C API call the unit of batching and keeps per-call error and timeout
attribution.  `VC4INFER_NO_BATCH=1` turns it off for A/B runs on the same
daemon.

## Protocol 5: the 256-slot arena map

Version 5 changes the arena map and nothing else: no message grew, no op
changed meaning, and the single-launch, batch and SUBMIT paths are byte for
byte the protocol-4 ones.  The client also only lays out its *own* region, so
the daemon and the client must agree on where the fixed zones sit before the
first `ALLOC`.

| zone | protocol 2..4 ("slots128") | protocol 5 ("slots256") |
|---|---|---|
| code slots (16 KiB each) | `0x001000`, 128 -> `0x201000` | `0x001000`, 256 -> `0x401000` |
| per-kernel uniforms (4 KiB each) | `0x210000`, 128 -> `0x290000` | `0x410000`, 256 -> `0x510000` |
| batch uniform shadows (2 x 64 x 768 B) | `0x290000` -> `0x2a8000` | `0x510000` -> `0x528000` |
| data heap | `0x300000` | `0x600000` |

The extra 3 MiB of reserved prefix is the whole cost.  With the 96 MiB arena
the client still asks for its 64 MiB region and keeps 58 MiB instead of 61
MiB of data; the largest planned graphs on this board are YOLOv8n-320
(22.96 MiB pool + 22.20 MiB scratch) and the bccd 320 detector (25.17 +
22.20 MiB), so both still fit.

Why 256: a code slot cannot exceed 16 KiB (`kgen` refuses a larger blob), a
16 KiB slot is what makes the table size the only lever, and the measured
model that exposed the limit needs 133 kernels (bccd YOLO11n-160, 3 classes).
Against 128 entries the LRU freed 5 slots per frame; the UPLOAD that restores
them is a serialised round trip, and the frame went to 1631 ms / 0.61 FPS
with the QPU at 17 % duty.  With 256 the set stays resident: 428 ms / 2.3 FPS
for the same class of model, zero evictions in the journal.  256 also leaves
room for a second input size in one session without turning the cache back
into the bottleneck.

**HELLO.**  The reply now carries the map's name in a 12-byte field that used
to be reserved: `"slots256"` from this daemon, empty from a daemon older than
5.  A version-5 client compares the name before it allocates and refuses a
mismatch by printing both names.  The request is unchanged, which is what
keeps the fallback ladder working:

- a version-5 client against a protocol-2..4 daemon starts at HELLO 5, gets
  `E_VERSION`, descends one version at a time, and settles on the version the
  old daemon serves.  The protocol version *is* the map's name there
  (protocol 4 means slots128), so the client switches its uniform base, batch
  shadows and heap start to the protocol-2..4 values and caps its kernel
  cache at 128.  Everything else - batching, SUBMIT, the single-launch path -
  keeps working.
- an old client against this daemon is refused.  `VC4D_SERVE_MIN` is 5, so a
  version-4 HELLO gets `E_VERSION` and the denial names both maps:
  `protocol 4 (layout slots128) refused: this daemon serves protocol 5,
  layout slots256 (256 slots, data base 0x600000)`.  The same refusal applies
  to every non-HELLO request from an old protocol, so a client that ignores
  the HELLO reply cannot send an `ALLOC` under a 3 MiB heap assumption
  either.  This is deliberate:
  an old client would place its heap at a data base the daemon did not
  reserve, and no field in the old protocol could tell it otherwise.

The tests that hardcode the map were updated rather than special-cased:
`vc4d/test/fake_daemon.py` picks data base 6 MiB / 3 MiB from its
`FAKE_VERSION`, so the protocol-3 fallback in `infer/tests/test_pipeline.py`
still models a real old daemon; `vc4d/test/test_submit.c` asserts the
negotiated protocol against `VC4D_VERSION`; `test_prog_lru` runs 320 programs
so the 256-slot LRU is exercised; `test_daemon_kernels` runs 300; and
`vc4d/test/test_hello_version` pins the negotiation itself (HELLO 5 names the
map, HELLO 4 and a version-4 ALLOC are refused, the connection survives for
the ladder).

## Reproduce

```sh
cd /home/raspi/work/vc4-stack && make -C vc4d && make -C infer
packaging/vc4d/build_deb.sh 1.4.0 && infer/packaging/vc4infer/build_deb.sh 1.2.12
cd /home/raspi/work/engine-daemon
make
REPS=5 VC4INFER_DAEMON=1 ./run_tests.sh                       # as raspi
./build/bench_matmul 256 5                                    # daemon (flag on)
sudo -n ./build/bench_matmul 256 5                            # direct (flag off)
```
