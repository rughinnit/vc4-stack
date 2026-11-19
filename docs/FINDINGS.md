# Findings catalogue — what this work unfolded

Everything below is measured on this board (Pi 3B, Debian 13 arm64) at stock clocks unless
stated. Each item points to the document with the evidence.

## Hardware and ISA

1. **The QPU issues one instruction every 4 clocks**, not one per clock. Structural peak is
   28.8 GFLOP/s at 300 MHz (12 QPUs x 16 lanes x 2 ALU ops x 1/4). Confirmed against Broadcom's
   24 GFLOP/s at 250 MHz. -> NUMERICS.md
2. **Both ALUs in a word**: one add + one mul per 64-bit instruction, but a word has only one
   `raddr_a` and one `raddr_b`, so at most one file-A read per word. `fadd` and `fmul` (both
   file-A readers) can never co-issue in the GEMM. -> XCHECK.md, postra-pack rejection
3. **The FPU truncates (round toward zero)** for both `fadd` and `fmul`. A single multiply
   cannot satisfy bit-exact IEEE tests; correctly-rounded multiply needs Veltkamp splitting +
   integer round-to-even. -> NUMERICS.md
4. **TMU allows only 2 outstanding requests per channel.** A third overwrites. This is why the
   3-phase k loop was needed. -> int8-RESULTS.md, asm9
5. **VPM window with VPMBASE=16 is 48 rows, not 64.** Using 8 QPUs for the packer corrupts the
   panel. -> asm10
6. **VPM fields that would be 16 are encoded as 0**; a literal 16 overflows into MPITCH and
   wedges the DMA. -> VC4MON / asm notes
7. **Rotate semantics**: 48 = rotate by r5 (dynamic), 49-63 = rotate 1..15, rotate-by-0 is a
   plain move, `1.0f` = 32, and 48-63 alias constants -16..-1. `ws=1` replicates 16 lanes,
   `ws=0` is per-quad. -> XCHECK.md
8. **`v8muld` is lossy** (per-byte round(a*b/255)), and there is **no exact packed byte MAC**, so
   int8 equals fp32 in MAC/s; int8 is a footprint optimisation only. -> int8-RESULTS.md
9. **Register hazards**: A/B file write->read needs >=2 words; add-ALU accumulator results need
   >=3 words before the mul ALU reads them. -> NUMERICS.md
10. **`DC IVAC` is undefined at EL0** on ARMv8 (assembles, SIGILLs); use `DC CIVAC`.
    `/dev/mem` writes go through a cacheable kernel alias, so arena self-checks must write
    uncached. -> VC4D.md
11. **No MMU between the QPU and main memory**: a stray DMA address can write any physical page
    (we corrupted libcrypto's page cache this way; files on disk intact). -> VC4D.md

## Compiler (VC4C)

12. **Silent wrong-code path**: an unplaceable local left the allocator in an intermediate state
    and code was emitted. The fix fails closed. -> M20
13. **Spill/free-and-retry** lets 130/130 corpus kernels compile (8 previously refused, 0
    regressions). -> perf/spill-retry
14. **Product-accumulator allocation** cuts the k loop from 517 to 453 words (+15-17%). -> 
    perf/product-acc
15. **Corrected TMU pipelining** (after `group-memory`, both TMUs, 2-deep, address slice moved
    with the request) gives +24.6% at N=128, +15.5% at N=96, +32% at N=256. -> perf/pipeline2-4
16. **Flag-setter pop relaxation** +2%; **gather-base default with allocation fallback** +1.2%.
    -> perf/pipeline3/4
17. Net compiled result: **2.668 / 3.085 GFLOP/s** at N=128/256, CPU parity or better, zero
    mismatches, `bench_vc4cl` 5/5. -> release docs
18. Rejected with evidence: post-RA packing (unencodable), cross-iteration prefetch (no gain),
    first pipelining pass (inert, then wrong), 4x4 lane-block stores (layout/allocator).

## Hand-written kernels

19. **GEMM 13.72 GFLOP/s GEMM-only / 11.97 total** at N=256, 15/15 verified runs. -> asm9/10
20. **3-phase k loop** removed the TMU stall (256.9 -> 19.2 cycles/k-pair); the kernel is now
    instruction-bound (exec = 91% of wall). -> asm9
21. **Pack pre-pass** is DRAM-bound at 1.34 GB/s (84% of LPDDR2); 16-word VDR rows + grid-stride
    assignment made it 16% faster. -> asm10
22. Ceilings measured: ALU-only **28.36**, coalesced memory-fed **16.42**, real GEMM **9.48**
    (asm3), improved **11.97** total. -> README
23. **Encoder validated by four independent tools**; 5,329 words, 0 unknown opcodes. -> XCHECK.md

## Engine, training, privilege

24. Engine **29/38 ops** with strict dispatch (no CPU fallback); matmul **8.49 GFLOP/s** at 256^3.
25. **Training works**: MLP 64-32-10 and conv3x3->relu->pool->fc both reach **100% train/test**
    with SGD and Adam, gradient checks 4e-7..9e-5 relative, bit-identical reruns.
26. **Unprivileged compute** via `vc4d`: drop to uid 65534, CapEff=0, allowlisted registers,
    90/90 hostile inputs rejected, 7.8% overhead at 256^3 (30% at 128^3, launch-bound).
27. **Unprivileged telemetry** via `vc4mon` (DKMS): 78 registers, exact parity with DRM debugfs,
    no `/dev/mem`; counters via a two-file write surface, values within 0.8% of root.
28. **YOLO-class ops** are in place (NHWC, stride-2/depthwise conv, 5x5 pool, concat, upsample);
    synthetic 160x160 forward: 202.6 ms/image, 4.9 FPS, 258 launches/image. No real weights yet.

## Packaging and productivity

29. vc4top **1.0.4**: redesigned TUI (43+2 fields, parity proven), no flicker (single write per
    frame, alternate screen), unprivileged counters, signed apt repo, DKMS module.
30. Compiler release installed into VC4CL and gated; rollback documented.

## What is still unknown or unproven

- Our GEMM has **not been benchmarked against a working third-party SGEMM** on this board
  (V3DLib blocked by KMS/gpu_mem; py-videocore blocked by VCSM). We cannot claim fastest.
- No ONNX/graph loader and no real weights -> no YOLOv8n number.
- Autograd/graph layer for training is not written; training is hand-coded per network.
- The residual ~10% stall at N=256 is unexplained (cross-iteration prefetch did not remove it).
- A safe compute UAPI/driver was designed but not written; privileged helpers are used instead.
