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

## conv backward / pooling (added 2026-11-19, train-conv)

- **col2im accumulates in fixed tap order.**  A destination 16x16 tile is owned
  by one task; the 16 channel rows are summed over the taps in the order
  (ky,kx) = (0,0),(0,1),... in registers and stored once, so the result is
  independent of QPU scheduling and reproducible byte-for-byte (two runs
  compare equal).  The sum itself is a chain of 9 QPU `fadd`s, which truncate
  toward zero: it is not the IEEE-ordered sum a CPU would produce.  Measured
  against a float64 oracle the relative L2 error is ~1e-7 (input O(1)); the
  gradient check tolerance is 1e-3, dominated by the finite-difference noise
  floor, not by this.
- **A destination tile must not be written by two tasks.**  The kernel is
  organised as dst-tile-owns-gather (dX[d] = sum_tap dA[d-tap]), never as a
  scatter, which is what makes the no-read-modify-write scheme possible.
  Source blocks outside [0, M) land in zeroed guard row blocks of the packed
  panel (the same trick the packer uses for the A panel).
- **maxpool backward ties:** every element equal to the 2x2 maximum receives
  dy (a valid subgradient).  Random gradient checks never hit a tie.
- **4-row DMA shape:** both pool kernels compute four 64 B rows per task and
  use the GEMM's verified `units=4, depth=16` VPM->DMA store.  A single-row
  `units=1` store is *not* a proven shape on this board.
- One task per QPU, no branches: the host enumerates tasks.  A missing uniform
  word in an early pool wrapper produced a wild TMU address and killed the
  V3D queue (board LIVE, `SRQCS` done never increments, reboot required);
  every address is now checked in `asmq/vsim.py` before hardware runs.

## NHWC / YOLO path (added 2026-11-19, `~/work/yolo`)

Measured while bringing up the NHWC data plane and the synthetic YOLO forward.
Kernel source: `tools/kgen.py` (`im2col_nhwc`, `pool_nhwc`, `up2`, `copy`);
offline checker: `tools/qsim_nhwc.py` (decodes the generated words, runs the
integer/control flow on lane 0 and reconstructs every VDR/TMU/store address).

- **`r4` is the TMU result register *and* the TMU configuration latch.**  The
  `tmu_noswap` word writes `r4 = 1`.  A mul (or any ALU op) that writes `r4`
  between gathers changes the per-lane address mode of every later load: the
  measured per-lane byte step went from 4 (one word per lane) to 24, i.e. the
  16 channels of a pixel were read as a stride-6 float pattern.  Keep products
  in `r5` (or re-latch the config) in any kernel that gathers and multiplies.
- **`r5` is clobbered by the verified 4-row VPM->DMA store** (`ldi 0x1A00, r5`).
  A value that must survive a store cannot live in `r5`: the first 5x5 pool
  used `r5 = lane*4` and every quad of lanes then read its leader's channel
  (out[c] = in[c & ~3]).
- **TMU gathers pop into either file.**  A pop is a normal ALU write selected
  by `ws`, so `add_or(Bn, acc(4), acc(4), ws=1)` is a valid way to keep 16
  channel values in the B file (used by the pool fmax and the depthwise mul).
- **A transposed (horizontal=0) 16x16 VDW block with stride 0 is a legal 1 KB
  panel write.**  Combined with a VDR over `[pixel][channel]` (rows = 16
  consecutive output pixels at pitch `stride*Cin*4`) it produces the k-major
  GEMM panel directly, with no im2col matrix and no ALU data movement.
- **A 16-row VDR tile per QPU needs 16 VPM rows**, so an im2col launch is
  capped at 4 QPUs (64 VPM rows); 12-QPU kernels must fit 4 rows per QPU
  (the 4-row store shape).
- **conv stride 2** is the same panel builder with the VDR pitch set to
  `2*Cin*4` and the tap offsets `kx*Cin*4`: consecutive output pixels are two
  source pixels apart, and the row step is `stride*row_pitch` (using the row
  pitch for the row step is the bug the simulator caught).
- Verification: every NHWC op 0 mismatches over 5 runs against a float64
  oracle (conv max rel 5.2e-5 at k=576, pool/concat/upsample byte-exact); the
  160x160 synthetic forward is 0 mismatches on 13 stages, max rel 2.3e-7.
