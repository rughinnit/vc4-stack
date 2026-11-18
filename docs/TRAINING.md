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
