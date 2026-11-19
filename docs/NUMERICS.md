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

## Training-kernel facts measured on silicon (2026-11-19)

These were found while building the backward kernels; they are the reasons the
softmax kernel looks the way it does.

- **The vector-rotate primitive is 1-ulp lossy.** `mul rX, rY, imm(48+n)`
  (MUL_V8MIN + small immediate) rotates the 16 lanes, but a lane value can come
  back with the low byte perturbed (measured: 2.50054908 -> 2.50054741).  A
  rotate-based max/sum is therefore *not* bit-exact.  Consequence: the
  softmax kernel's argmax test is `|z_label - max| <= 1e-5` (absolute, logits
  O(10)), not a bit comparison.  The loss is affected by ~1e-7 relative.
- **An add-ALU accumulator result is not reliably visible to the mul ALU in
  the next word.**  A rotate 1 word after an add saw a stale operand; the
  packing worked only with 3 words between them.  All dependent accumulator
  chains in the softmax kernel keep 2 nops (3 words).  The same applies when
  the accumulator was written from a *register file* operand.
- **Small immediates are limited.**  Values 0..31 behaved correctly in
  `ADD_SHR/ADD_SUB` (23 and 31 are used); 127 and 512 do not fit and silently
  become the rotate encodings.  Use a register constant (`ldi 127, rX`).
- **One group per QPU per launch.**  A counter loop around the 1100-instruction
  softmax body hung the V3D (context timeout, board LIVE but queue dead; a
  reboot recovered it).  The adopted shape is branch-free: one 8-row group per
  QPU, ceil(groups/12) launches from the host.
- **Fused softmax+CE accuracy (measured).**  dlogits vs CPU: 5.2e-9 max abs
  (64x16), 1.3e-9 (512x16 with -1e30 padding columns); total loss relative
  2.2e-7 / 3.3e-7; argmax count exact.  Gradcheck against central differences:
  global relative L2 9e-5 (softmax), 7e-7 (matmul dA/dB), 9e-7 (relu).
