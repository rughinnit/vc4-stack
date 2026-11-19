# vc4d — privileged helper for unprivileged VC4 compute

Source tree: `/home/raspi/work/vc4d`. Installed as `vc4d.service` (active).

Goal: move `/dev/mem` and V3D dispatch out of client processes, so engines and tools run
unprivileged without raw memory access.

## Protocol
AF_UNIX SOCK_SEQPACKET `/run/vc4d.sock` (0660 root:vc4), 28-byte header (`magic "VC4D"`, len ==
datagram size, op, version, seq, status, two detail words), 64 KiB cap.
Ops: HELLO, ALLOC(+fd via SCM_RIGHTS), FREE, UPLOAD, LAUNCH, WAIT, READ_REGS, WRITEREG, SYNC,
STATS, PING, CLOSE. Uniforms are tagged: PTR(handle, off, len) resolved to bus+base+off only after
alignment, length and 64-bit bounds checks; LIT refused if it lands in the GPU bus, V3D MMIO or the
arena's physical range. UPLOAD declares the uniform count; LAUNCH must match it.

## Security
- Root only until it maps `/dev/mem` and one 64 MiB CMA dma-buf; then drops to uid 65534,
  CapEff=0, NoNewPrivs=1, dumpable=0; never execs; no network transport.
- Register writes allowlisted (SRQPC/SRQUA/SRQUL/SRQCS/VPMBASE/L2CACTL/SLCACTL/VPACNTL/
  INTCTL/INTENA/INTDIS); reads and block dumps allowed.
- Per-client handle ownership; one launch in flight; bounded timeouts; dead-after-timeout;
  all denials logged with SO_PEERCRED.
- Arena physical base verified from three offsets before first use.
- `--strict`: SHA-256 program allowlist, no raw SRQ writes, 2 s timeout.
- Honest limit: a QPU program is raw machine code, so a client allowed to upload arbitrary code
  can still DMA outside the arena; structural checks stop accidents and naive attacks, `--strict`
  closes it for an appliance. The `vc4` group is a mutual-trust group (shared arena).

## Verification
82/82 hostile cases rejected with the daemon alive (truncations at every length 1-23, bad
magic/version/len/opcode, 70 KiB and 2 MiB datagrams, bad ALLOC/FREE/UPLOAD/LAUNCH, overflow and
misaligned pointers, bogus handles, cross-client use, 2000 fuzz datagrams, disconnect mid-launch,
strict-mode one-byte program change). No fd leak; V3D LIVE.
Client: uid 1000, CapEff=0, no `/dev/mem` fd.

## Performance (256^3, same boot)
| path | GPU ms | GFLOP/s | overhead | V3D cycles |
|---|---|---|---|---|
| direct root | 4.28 | 7.84 | - | 6,773,184 |
| via vc4d (unprivileged) | 4.43 | 7.58 | 0.24-0.42 ms (5-13%) | 6,773,184 |

Identical cycle counts: the difference is six protocol round-trips at 43-60 us, not compute.

## Platform findings
- `DC IVAC` is undefined at EL0 on ARMv8 (GNU as emits it; it SIGILLs). Use `DC CIVAC`.
- `/dev/mem` writes go through a cacheable kernel alias; arena self-checks must write uncached and
  read back a page.

## Remaining
Engine backend `vc4_dev_remote.c` behind `VC4INFER_DAEMON=1` (see the engine docs).
