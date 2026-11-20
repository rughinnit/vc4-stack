# Running the inference engine unprivileged through vc4d

Status: **done**.  `VC4INFER_DAEMON=1` in `libvc4infer` selects a backend that
talks to `vc4d`; the direct `/dev/mem` path is unchanged when the flag is
unset.  The full suite passes as user `raspi` with every shape and gradient
check clean.

Environment: Raspberry Pi 3B, Debian 13 arm64, **stock clocks** (core 400 MHz,
V3D 300 MHz, `throttled=0x0`), one boot for every number below.  The board is
shared with concurrent jobs; every GPU run was serialised by
`/tmp/vc4-gpu.lock` and the runs reported here were taken in a quiet window.

## What changed

- `infer/src/vc4_dev_remote.c` (~480 lines): connect to `/run/vc4d.sock`,
  `ALLOC` one 64 MiB region and `mmap` the dma-buf fd, upload each kernel blob
  on first use, launch with `UNI_IN_ARENA | WAIT` in one round trip, and do
  the client-side cache maintenance.
- `infer/src/vc4_dev.c`: dispatch at `vc4_init`, `vc4__launch`,
  `vc4_upload`, `vc4_download`, `vc4__l2_flush`, `vc4__write`,
  `vc4__kernel_load_file`, `vc4__wait_idle`, `vc4__health`, `vc4_shutdown`.
- `vc4d` protocol version 2: `ALLOC` replies with the region's GPU bus
  address; `LAUNCH` gains `UNI_IN_ARENA` (uniform words read from the client's
  arena, each word validated) and `WAIT` (kick → idle → counters → L2
  write-back in one reply).  The daemon's per-client program limit is 64 and
  the unit uses `--arena-mb 72` (CMA is 256 MiB; 72 leaves the engine's full 64 MiB client region).
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
  64 fixes it; the arena's code region only has room for ~67.
- Arena sizing: a 64 MiB daemon arena leaves a client `64 − 2` MiB, one
  megabyte less than the direct path, which made `test_convgrad` run out
  late.  The unit uses 72 MiB, which yields the client's full 64 MiB region
  and a few MiB spare.  A 128 MiB arena is not reliable at boot: contiguous
  CMA may have fragmented under the vc4 DRM driver, and the shrink ladder then
  fell back to 64 MiB and reintroduced the failure.  The final suite ran with the
  64 MiB client region and the 72 MiB daemon; `test_convgrad` was also verified
  separately against this exact unit.
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

## Reproduce

```sh
cd /home/raspi/work/vc4d && make && sudo ./tools/install.sh   # protocol v2
cd /home/raspi/work/engine-daemon
make
REPS=5 VC4INFER_DAEMON=1 ./run_tests.sh                       # as raspi
./build/bench_matmul 256 5                                    # daemon (flag on)
sudo -n ./build/bench_matmul 256 5                            # direct (flag off)
```
