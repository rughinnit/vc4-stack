# vc4-stack — VC4 compute stack: kernels, engine, training

Target: Raspberry Pi 3B, Broadcom VideoCore IV (V3D), Debian 13 arm64, stock clocks.

Everything here runs on the VC4 QPUs for arithmetic. The ARM cores run the control
plane only (framework, allocation, I/O); there is no CPU compute fallback.

## Layout

- `kernels/`  hand-written QPU kernels and the verified harnesses
              (GEMM 16x16 packed-A, wider tile, int8 experiments)
- `engine/`   libvc4infer: dma-buf arena, kernel upload/cache, strict op registry,
              matmul, conv2d (im2col), elementwise. Unsupported ops raise.
- `training/` forward+backward kernels and a toy trainer (MLP/conv) demonstrating
              gradient computation on the VC4 data plane
- `docs/`     measurements, ceilings, coverage tables, per-track results
- `tools/`    QPU encoder/launcher, verification harnesses

## Measured ceilings (stock 300/400 MHz, no overclock)

| workload | GFLOP/s | % of 28.8 structural peak |
|---|---|---|
| ALU-only microbenchmark | 28.36 | 98.5% |
| coalesced memory-fed loop | 16.42 | 57% |
| hand-written GEMM (verified) | 9.48 | 33% |
| compiled VC4CL path | 3.09 | 10.7% |

Structural peak: 12 QPUs x 16 lanes x 2 ALU ops x 1/4 instruction per clock x 300 MHz.

## Scope and honest limits

- Inference: a fixed model family at low resolution; int8 is the performance path.
- Training: a toy trainer only. No fp16/bf16, no FMA, 76 MiB device memory, and the
  board browns out under sustained load. General training is out of scope.
- Not CUDA/ROCm parity: no tensor cores, no fp16, no cuBLAS/cuDNN, no autograd library.

Every reported number must have a run log and a mismatch count.
