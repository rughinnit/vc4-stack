# vc4d — privileged helper for unprivileged VC4 compute

Source tree: `/home/raspi/work/vc4d`. Installed as `vc4d.service` (active,
protocol version 2).

Goal: move `/dev/mem` and V3D dispatch out of client processes, so engines and
tools run unprivileged without raw memory access.

## Protocol
AF_UNIX SOCK_SEQPACKET `/run/vc4d.sock` (0660 root:vc4), 28-byte header
(`magic "VC4D"`, len == datagram size, op, version, seq, status, two detail
words), 64 KiB cap.  Ops: HELLO, ALLOC(+fd via SCM_RIGHTS), FREE, UPLOAD,
LAUNCH, WAIT, READ_REGS, WRITEREG, SYNC, STATS, PING, CLOSE.

- **ALLOC** returns `{handle, off, len, bus}`; `bus` is the GPU bus address of
  the region, so a client can build pointer uniforms itself.
- **UPLOAD** (`flags & 1`, `VC4D_UPLOAD_NAMED`): the payload is
  `upload_req + code + name` (no NUL, 1..63 printable bytes).  The name is the
  key of the program-slot LRU: re-uploading a name reuses its slot, and when
  the table is full the least recently used slot that is not executing is
  freed (the daemon's probe slot and the in-flight program are never
  evicted).  A client that sends no name is keyed by the code sha256, which
  gives it the same behaviour for a repeated upload.  Slots are 128, 127
  usable; `STATS.evictions` counts the frees.
- **UNI_PTR** uniforms are resolved to `arena_bus + base + off` only after
  alignment, length and 64-bit bounds checks.  **UNI_LIT** is refused if it
  lands in the GPU bus, V3D MMIO or the arena's physical range.  UPLOAD
  declares the uniform count; LAUNCH must match it.
- **UNI_IN_ARENA** (`LAUNCH.flags & 1`): the client has materialised the
  `nqpu × 64 B` uniform blocks in its own mapped region.  `uni_handle` names
  the region, `uni_off` is the 64-byte-aligned first block; the daemon checks
  ownership/alignment/range, then validates the words and points SRQUA at the
  block.  Containment is per-arena instead of per-buffer.  The client must
  have cleaned its cache (see below).
- **UNI_MASK** (`flags & 4`): `launch_req.uni_addr` is authoritative.  Bit *u*
  means "uniform word *u* is a GPU bus address the program consumes": such a
  word must lie inside this client's arena, and a word whose bit is clear is a
  **value** and is not compared with any address window.  Bits ≥ `nuni` must be
  zero.  This is what makes `-2.0f` (`0xC0000000`), `-3.0f`
  (`0xC0400000`), `-inf` and every other float ≤ −2.0 legal; the old
  per-word guess ("inside the arena or not an address-looking literal")
  cannot tell such a value from an address and is still what a launch
  **without** the flag gets, so pre-1.2.7 clients are unchanged.  The engine
  sets the flag on every launch it makes (`vc4_dev_remote.c:remote_uni_mask`,
  a word is an address exactly when it points into the client's arena).  This
  is a correctness aid, not a security boundary: a client that marks a real
  address as a value can already make the QPU read anywhere by uploading raw
  machine code (see Security).
- **LAUNCH_WAIT** (`flags & 2`): kick, poll to idle, counters and L2
  write-back in one round trip; the reply is the WAIT struct.  The engine uses
  `UNI_IN_ARENA | WAIT | UNI_MASK`, so one kernel = one message.
- **STATS** gained `evictions`, `arena_want` and `arena_degraded` (bytes / 0|1),
  so a tool can tell a daemon that is serving less than it was asked for.
  **HELLO.flags** advertises `VC4D_F_NAMED_PROGS`, `VC4D_F_UNI_MASK` and
  `VC4D_F_ARENA_RESTORE`; a client must not use a feature the daemon does not
  advertise (the engine does not, and therefore works with 1.2.6 daemons).
- **Arena**: one contiguous CMA dma-buf.  The requested size is retried
  three times (0/200/400 ms, each attempt dropping its reservation) before the
  daemon falls back to half/quarter/eighth, and the fallback is logged with
  the obtained and requested sizes and `CmaTotal`/`CmaFree`.  A degraded arena
  is restored: when it is idle and CMA can supply the full block, the daemon
  exits 75 and `Restart=always` starts a clean one (it cannot re-allocate
  itself after dropping privileges and closing `/dev/mem`).

## Security
- Root only until it maps `/dev/mem` and one CMA dma-buf (72 MiB in the unit,
  so a 64 MiB client region fits); then drops to uid 65534, CapEff=0,
  NoNewPrivs=1, dumpable=0; never execs; no network transport.
- Register writes allowlisted (SRQPC/SRQUA/SRQUL/SRQCS/VPMBASE/L2CACTL/
  SLCACTL/VPACNTL/INTCTL/INTENA/INTDIS); reads and block dumps allowed.
- Per-client handle and program ownership; one launch in flight; bounded
  timeouts; dead-after-timeout; all denials logged with SO_PEERCRED.  Program
  slots are an LRU keyed by name, so a session is not limited to 127 kernels.
- Uniform words: with `UNI_MASK` the launcher declares which slots are
  addresses and those are checked against the arena; the rest are values.  A
  client that declares an address as a value gains nothing it did not already
  have (raw QPU code can DMA anywhere), and one that sends no mask still gets
  the old strict rule.
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

The 1.2.7 fixes are covered by `vc4d/test/test_prog_lru` (G3: 250 named
programs, LRU eviction, the first kernel re-run), `vc4d/test/test_uni_mask`
(G4: values ≥ `0xC0000000`, declared addresses in and out of the arena),
`infer/tests/test_daemon_kernels` (G3 through the engine, 200 kernels),
`infer/tests/test_daemon_uni` (G4 through the engine, 0/1024 wrong),
`infer/tests/test_daemon_values.py` (G4 through the ONNX runner) and
`infer/tests/test_daemon_g2.sh` (G2: retry, loud fallback, restore after the
pressure is released).  Before/after numbers are in `docs/BUGS.md` B13-B15.

The engine suite also runs through the daemon: `REPS=5
VC4INFER_DAEMON=1 ./run_tests.sh` as user raspi, 0 mismatches everywhere; the
client has uid 1000, CapEff=0 and no `/dev/mem`.  See `docs/DAEMON.md`.

## Performance (same boot, stock 400/300 MHz)
256³ matmul, engine `bench_matmul`, 5 reps, best wall:

| path | GPU ms | GFLOP/s (GPU) | wall ms | GFLOP/s (wall) | overhead |
|---|---|---|---|---|---|
| direct root | 3.962 | 8.47 | 3.980 | 8.43 | — |
| via vc4d | 3.998 | 8.39 | 4.289 | 7.82 | +0.309 ms (7.8%) |

128³: direct 0.706 ms / 5.80 GFLOP/s wall, daemon 0.771 ms GPU / 0.942 ms
wall / 4.45 GFLOP/s, i.e. +0.219 ms (30%) — two round trips plus the
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
  plus fragmentation); the shrink ladder then silently fell back to 64 MiB, one
  megabyte short for the engine's conv-grad suite.  72 MiB is reliable and
  yields the full 64 MiB client region.  Since 1.2.7 the fallback is preceded
  by three retries and is logged with both sizes and `CmaFree`, and a degraded
  arena is restored (exit 75 → `Restart=always`) once it is idle and CMA has
  the block again — on this board a 32 MiB block can fail with 60 MiB of
  `CmaFree` when the free pages are fragmented, which is exactly why the
  daemon must not settle for the smaller size silently.
- The unit needs `ReadWritePaths=/tmp` (with `ProtectSystem=strict`) or the
  daemon cannot create `/tmp/vc4-gpu.lock` at boot.
- Per-client program slots must cover a whole process's kernel set (one per
  shape): 128, not 64 - and the last one is reserved for the daemon's recovery
  probe, so clients get 127.  Since 1.2.7 that is a cache, not a limit: slots
  are evicted LRU-wise (keyed by the kernel name the client sends with
  UPLOAD) and the engine re-uploads a kernel whose slot was evicted, so a
  session can touch 250 distinct kernels (measured: 250/250 uploads and
  launches, 374 evictions) where 127 was previously fatal.
- A launch timeout is no longer terminal (G1): the daemon resets the V3D
  context under the GPU lock, verifies it with the probe program and resumes;
  if the reset cannot be verified it exits 75 for `Restart=always`.  A departed
  client's arena is reclaimed from its peer pid (G9), not from its socket.

## Remaining
None for the engine port.  `VC4INFER_DAEMON=1` in `libvc4infer` selects the
daemon backend; unset keeps the direct `/dev/mem` path.  The daemon's
`--strict` mode is the deployment profile for an appliance.
