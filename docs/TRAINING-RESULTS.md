# Training on VC4 — first working path (2026-11-19)

All forward and backward arithmetic runs on the QPUs. The ARM only does layout bytes,
scalar hyperparameters and the accuracy print. No CPU compute fallback.

## New kernels
transpose (VDR + vertical VDW, bit-exact 16x16), matmul_bt (C = A*B^T via pack + the
verified asm3 GEMM with bmode=1), relu_bwd (bit-exact mask), rsqrt (bit-trick seed +
4 Newton steps), softmax_ce_bwd (fused 16-wide row: rotate+fmax, deg-5 2^f exp,
rotate+fadd sum, bit-trick+3 Newton 1/S, dlogits=(p-onehot)/B, loss, argmax).

## Gradient checks (CPU checker only; forwards and analytic grads are QPU results)
dA/dB 16^3: rel L2 7.5e-7 / 6.8e-7, 0/256 over tol
dA/dB 32x32x16: 9.6e-7 / 7.7e-7, 0/512
relu mask: 9.1e-7, 0/32
softmax dlogits 8x16: 9.1e-5, 0/128 ; 64x16: 8.3e-5, 0/1024
Kernel vs CPU oracle: dlogits 5.2e-9 abs, loss rel 2.2e-7, argmax exact.

## Toy trainer: MLP 64-32-10, 640 train / 160 test
SGD  lr 0.5 : loss 2.321555 -> 0.000933 train, 0.000884 test, 100% both,
              300 steps in 2.898 s = 103.5 steps/s = 66254 samples/s
Adam lr 0.01: 2.321555 -> 0.0000296 train, 0.0000283 test, 100% both,
              300 steps in 3.317 s = 90.5 steps/s = 57890 samples/s
Optimizer state matches a CPU oracle after one step (6.9e-8 / 8.1e-8, incl. m and v).
Two runs per optimizer are byte-identical (timing aside).

## Op coverage: 8/22 -> 15/27
Added: relu_bwd, rsqrt, transpose, matmul_bt, softmax_ce_bwd, softmax.
Still unsupported: conv 5x5/stride2/depthwise/grouped, f16, batched matmul, maxpool,
layernorm, concat/slice, arbitrary permute, upsample, quantize.

## Honest limits
- The hardware vector rotate is 1-ulp lossy (2.50054908 -> 2.50054741).
- An add-ALU accumulator result is not reliably visible to the mul ALU in the next
  word; dependent chains keep 3 words. Small immediates 127/512 do not fit.
- Softmax rows are fixed 16-wide (10+6 padding here); more classes need a second
  reduction pass. No conv backward yet, so no conv-net toy.
- fp32 only, no FMA, 64 MiB arena, no checkpointing/gradient accumulation.
