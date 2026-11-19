# vc4d — privileged helper for unprivileged VC4 compute

Source tree: `/home/raspi/work/vc4d`, mirrored in this repo as `vc4d/`.
Installed as `vc4d.service` (active, protocol version 2).

Goal: move `/dev/mem` and V3D dispatch out of client processes, so engines and
tools run unprivileged without raw memory access.

## Protocol
AF_UNIX SOCK_SEQPACKET `/run/vc4d.sock` (0660 root:vc4), 28-byte header
(`magic "VC4D"`, len == datagram size, op, version, seq, status, two detail
words), 64 KiB cap.  Ops: HELLO, ALLOC(+fd via SCM_RIGHTS), FREE, UPLOAD,
LAUNCH, WAIT, READ_REGS, WRITEREG, SYNC, STATS, PING, CLOSE.

- **ALLOC** returns `{handle, off, len, bus}`; `bus` is the GPU bus address of
  the region, so a client can build pointer uniforms itself.
- **UNI_PTR** uniforms are resolved to `arena_bus + base + off` only after
  alignment, length and 64-bit bounds checks.  **UNI_LIT** is refused if it
  lands in the GPU bus, V3D MMIO or the arena's physical range.  UPLOAD
  declares the uniform count; LAUNCH must match it.
- **UNI_IN_ARENA** (`LAUNCH.flags & 1`): the client has materialised the
  `nqpu × 64 B` uniform blocks in its own mapped region.  `uni_handle` names
  the region, `uni_off` is the 64-byte-aligned first block; the daemon checks
  ownership/alignment/range, then validates every declared word as "a bus
  address inside this arena" or "a literal outside the bus/arena/MMIO
  windows" and points SRQUA at the block.  Containment is per-arena instead of
  per-buffer.  The client must have cleaned its cache (see below).
- **LAUNCH_WAIT** (`flags & 2`): kick, poll to idle, counters and L2
  write-back in one round trip; the reply is the WAIT struct.  The engine uses
  `UNI_IN_ARENA | WAIT`, so one kernel = one message.

## Security
- Root only until it maps `/dev/mem` and one CMA dma-buf (72 MiB in the unit,
  so a 64 MiB client region fits); then drops to uid 65534, CapEff=0,
  NoNewPrivs=1, dumpable=0; never execs; no network transport.
- Register writes allowlisted (SRQPC/SRQUA/SRQUL/SRQCS/VPMBASE/L2CACTL/
  SLCACTL/VPACNTL/INTCTL/INTENA/INTDIS); reads and block dumps allowed.
- Per-client handle and program ownership; one launch in flight; bounded
  timeouts; dead-after-timeout; all denials logged with SO_PEERCRED.
- Arena physical base verified from three offsets before first use.
- `--strict`: SHA-256 program allowlist, no raw SRQ writes, 2 s timeout.
- Honest limit: a QPU program is raw machine code, so a client allowed to
  upload arbitrary code can still DMA outside the arena; structural checks
  stop accidents and naive attacks, `--strict` closes it for an appliance.
  The `vc4` group is a mutual-trust group (shared arena).

## Verification
90/90 hostile cases rejected with the daemon alive (`test_hostile --fuzz
500`), including the new UNI_IN_ARENA cases (unknown region, unaligned block,
block past the region, unknown flag, trailing payload, word in the bus window
outside the arena, word = V3D MMIO).  The old 82 cases plus 2000-datagram
fuzzing are unchanged.  No fd leak; V3D LIVE.

The engine suite also runs through the daemon: `REPS=5
VC4INFER_DAEMON=1 ./run_tests.sh` as user raspi, 0 mismatches everywhere; the
client has uid 1000, CapEff=0 and no `/dev/mem`.  See `docs/DAEMON.md`.

## Performance (same boot, stock 400/300 MHz)
256³ matmul, engine `bench_matmul`, 5 reps, best wall:

| path | GPU ms | GFLOP/s (GPU) | wall ms | GFLOP/s (wall) | overhead |
|---|---|---|---|---|---|
| direct root | 3.949 | 8.50 | 3.976 | 8.44 | — |
| via vc4d | 3.985 | 8.42 | 4.271 | 7.86 | +0.295 ms (7.4%) |

128³: direct 0.708 ms / 5.80 GFLOP/s wall, daemon 0.770 ms GPU / 0.941 ms
wall / 4.46 GFLOP/s, i.e. +0.218 ms (30%) — two round trips plus the
fixed-size cache maintenance on a 0.72 ms call.  The QPU work itself is
within 1% (256³) and 9% (128³) of the direct path.  The demo (six messages
per GEMM iteration) measures 5–13% at 256³; the engine's one-round-trip
launch is better.

## Platform findings
- The CMA dma-buf mapping is **cacheable**, not write-combining (measured: the
  daemon's uncached read of fresh client stores returned zeros until the
  client did `dc civac`).  `vc4c_sync()` is required before/after a launch.
  Running out of a cached mapping also means the engine tracks dirty ranges
  instead of sweeping 62 MiB.
- `DC IVAC` is undefined at EL0 on ARMv8 (GNU as emits it; it SIGILLs).  Use
  `DC CIVAC`; cleaning every line before a launch makes the later CIVAC safe.
- `/dev/mem` writes go through a cacheable kernel alias; arena self-checks must
  write uncached and read back a page.
- A 128 MiB contiguous CMA arena is not reliable at boot (the vc4 DRM driver
  plus fragmentation); the daemon's shrink ladder then silently fell back to
  64 MiB, one megabyte short for the engine's conv-grad suite.  72 MiB is
  reliable and yields the full 64 MiB client region.
- The unit needs `ReadWritePaths=/tmp` (with `ProtectSystem=strict`) or the
  daemon cannot create `/tmp/vc4-gpu.lock` at boot.
- Per-client program slots must cover a whole process's kernel set (one per
  shape): 64, not 16.

## Remaining
None for the engine port.  `VC4INFER_DAEMON=1` in `libvc4infer` selects the
daemon backend; unset keeps the direct `/dev/mem` path.  The daemon's
`--strict` mode is the deployment profile for an appliance.
