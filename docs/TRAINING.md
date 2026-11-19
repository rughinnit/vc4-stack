# Training on VC4 — design and hard limits

Goal: forward **and** backward on the QPUs, with no CPU compute fallback.
Time is not a constraint; capacity is.

## Architecture

- Tape-based autograd over a fixed op set. Every op has a forward kernel and a
  backward kernel (hand-written, same rules as the GEMM: packed ALU, both TMUs,
  2-deep, VPM->DMA stores, verified against a CPU oracle *as a checker*).
- Ops with gradients: matmul/linear, conv2d (im2col), relu, silu, sigmoid,
  softmax + cross-entropy, add/mul (broadcast), reshape/transpose, batchnorm
  (optional), SGD, Adam.
- Memory planner over the 76 MiB dma-buf arena: weights, gradients, optimizer state
  (SGD: 1x params; Adam: 2x), activations. Gradient checkpointing to trade compute
  for memory; micro-batches with gradient accumulation.
- Numerics: fp32 only, no FMA. Gradient checks against finite differences on small
  tensors; loss curves recorded per step.

## What "train anything" can and cannot mean

- CAN: any model whose weights + gradients + optimizer state + one micro-batch of
  activations fit in the device arena, using the implemented op set. Speed:
  forward ~9.5 GFLOP/s hand-written, backward ~2-3x forward, minus memory stalls.
  Small models are feasible; training time is large but finite.
- CANNOT: models that exceed ~76 MiB of combined device state, or that use ops not
  implemented, or that need fp16/bf16/tensor cores. No amount of time fixes capacity.
  Workarounds: smaller batch, gradient accumulation, checkpointing, model sharding to
  host memory with streaming (slower, but extends the limit).

## Deliverables

1. Backward kernels for the op set with finite-difference gradient checks.
2. A toy trainer (MLP on a small dataset, then a small conv net) with loss curves.
3. A capacity calculator: given a model spec, decide whether it fits.
4. Honest throughput numbers: samples/s and time-per-epoch for the toy models.

## Implementation status (2026-11-19, `~/work/train`)

Deliverable 1 (backward kernels + gradient checks) and 2 (toy trainer) are
built and green; 3 (capacity calculator) is still a paper formula.  Measured
on a Pi 3B at stock clocks (400 MHz idle; the trainer runs fit in the 400 MHz
bin, see `work/train/RESULTS-train.md`).

- **Kernels added** (all QPU; no CPU fallback): dense 16x16 transpose (the
  packer with a strided VDW destination), `matmul_bt` (C = A*B^T via the asm3
  packer + bmode=1 GEMM), `relu_bwd` (bit-exact mask), `rsqrt` (bit-trick +
  4 Newton), and the fused softmax+cross-entropy backward (one 8-row group per
  QPU: max, exp, 1/S, dlogits, loss, argmax).
- **Gradchecks** (`tests/test_grad.c`, central differences, CPU checker only):
  matmul dA and dB at 16^3 and 32x32x16, relu mask, softmax dlogits.  Global
  relative L2 7e-7 (matmul), 9e-7 (relu), 9e-5 (softmax); every element passes
  |err| <= 1e-3|g| + 1e-4.  The per-element worst softmax case (~2e-3) is a
  gradient of magnitude 0.01 where the finite-difference noise floor (~1e-5
  absolute, from the fp32/truncating forward) dominates.
- **Trainer** (`tests/test_train.c`): 64-32-10 MLP, 640 training / 160 test
  samples of a deterministic 10-cluster synthetic set, full batch, bias folded
  as augmented weight rows, SGD and Adam.  SGD: 2.32 -> 9.3e-4 loss in 300
  steps, 100% test accuracy, ~107 steps/s (68k samples/s).  Adam: 2.32 ->
  3.0e-5, 100%, ~92 steps/s (59k samples/s).  Two runs are bit-identical.
- **Coverage**: 8 of 22 entries -> 15 of 27 (relu_bwd, rsqrt, transpose,
  matmul_bt, softmax_ce_bwd, softmax added; the unsupported list keeps the
  real gaps: batched matmul, pooling, layernorm, f16, arbitrary permute).
- **Limits that remain**: fp32 only; the softmax kernel's row width is exactly
  16 (the model output is padded, the toy uses 10+6); 76 MiB-class arena
  (64 MiB default) with no checkpointing; no conv backward yet (the im2col
  path has no gradient kernel), so the "conv net" toy is not built.
