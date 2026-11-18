# VC4 QPU numerics — measured facts

## The FPU truncates (round toward zero)

Measured on silicon: `1.0 + 0.75 ulp -> 1.0`, `-1.0 + -0.75 ulp -> -1.0`, and a
product closer to the next float up returns the lower one. Both `fadd` and `fmul`
round toward zero; there is no IEEE round-to-nearest in the ALU.

Consequences:
- A single `fmul`/`fadd` cannot satisfy a bit-exact IEEE test for arbitrary inputs.
- The elementwise `mul` is implemented as a correctly-rounded product: Veltkamp
  12-bit split (all sub-products exact even on a truncating multiplier), exact
  residual, then integer round-to-nearest-even on the result bits. Cost ~1.8x a
  plain fmul (754 vs 416 instructions for a 16-vector body).
- `add`/`relu` are bit-exact only for exactly representable sums.
- Accumulation order changes results in the last ulp; verification uses a
  documented tolerance, not bit equality, wherever truncation can bite.

## Other engine facts learned the hard way

- VPM-DMA setup word is read from **accumulator 5**, not register r5; writing it to
  r5 makes the DMA read garbage rows (V3D_ERRORS bit 5, VPM read range).
- VPM fields that would be 16 are encoded as 0.
- The dma-buf bus base is not stable across boots; take it from a runtime uniform.
- One group per QPU per launch; long elementwise bodies hang with a second group.
- Unary elementwise kernels must still pass a valid second operand address: a TMU
  load from 0 wedges the QPU.
- Pack/im2col must keep Q in an A-file register; a B-file read used as an A address
  makes every QPU use row 0.
- im2col needs a horizontal VDW ([channel][pixel]); vertical stores [pixel][channel].
- GEMM must reset B's column offset per row block, and B blocks advance by 64*K.

## Engine test status (2026-11-19, stock clocks)

ALL TESTS PASSED, three runs (incl. clean rebuild): add/mul/relu/add+tail/sigmoid/
silu 5/5 with 0 mismatches; matmul 128^3 5.8, 256^3 6.2, 512x64x288 6.8 GFLOP/s;
conv 3x3 and 1x1, six shapes, 5/5.
