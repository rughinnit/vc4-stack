# Status

Tracks in flight: wider tile, int8 probe/GEMM, engine foundation.
Accepted compiler work lives in the VC4C branches (product-acc, pipeline2/3/4,
spill-retry) and the VC4CL docs branch; this repo collects the hand-written stack.

## Engine unprivileged through vc4d (2026-11-19)

`libvc4infer` has a second backend selected by `VC4INFER_DAEMON=1`; unset
keeps the direct `/dev/mem` path.  `vc4_dev_remote.c` connects to
`/run/vc4d.sock`, ALLOCs one 64 MiB region and mmaps the dma-buf fd, uploads
kernels on first use, and launches with `UNI_IN_ARENA | WAIT` in one round
trip.  The op code is unchanged apart from cache dirty marks.  Full suite as
user raspi: 5/5, 0 mismatches, client CapEff=0 and no `/dev/mem`.  Overhead vs
direct in the same boot: 7.4% wall at 256³ (GPU time within 1%), 30% at 128³
(launch-bound).  Details and the coherency correction (the CMA mapping is
cacheable, not write-combining) in `docs/DAEMON.md` and `docs/VC4D.md`.
