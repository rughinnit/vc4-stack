# Training on VC4 — the backward op set, measured

This document describes the training core that was added under
`infer/src`, `infer/include`, `training/` and records, honestly, which
backward paths are verified on the board and which are not.

**Status summary (2026-11-20):** the *API*, the *kernel generation* and the
*gradient-check harness* are in place and the elementwise-derivative family
and the GEMM-based reductions verify to ~3e-7.  The composite ops (softmax,
batchnorm, layernorm, pooling, resize) pass in isolation but wedge the QPU
when they run inside the full gradient-check sequence; the cause was not
isolated before this run ended.  `REPS=2 ./run_tests.sh` is therefore
**not** green.  The engine defects that were found on the way (and which any
framework plugin must know about) are listed in section 6.

## 1. Architecture

Autograd lives on the host (PyTorch/JAX/TF): the framework records the forward
ops, and for each one calls a `vc4_*_bwd` entry point with the saved forward
input and the incoming gradient.  The library provides

* a `vc4_*_bwd` entry point per differentiable op (section 3),
* `vc4_bwop_run()`, the elementwise-derivative family,
* `vc4_op_differentiable()` plus a `differentiable` column in the registry, so
  a plugin can decide up front whether a graph is trainable here,
* `vc4_affine()` (`a*p0 + p1`) and the reductions below, which are what a
  chain-rule helper needs.

Every backward is a *composition* of primitives that the test suite already
verifies element by element, or a single kernel on the same verified harness as
`build_ew`:

| primitive | use in the training path | verified by |
|---|---|---|
| `vc4_matmul` (asm10 GEMM) | reductions (× ones), broadcasts (ones × ones), and any irregular layout as a 0/1 or weight matrix × vector | `test_matmul` |
| `vc4_elementwise` add/mul/relu/relu_bwd/sigmoid/silu/rsqrt | the atoms every derivative is built from | `test_ew` |
| `vc4__ewb` exp, rcp | exp/reciprocal on the 16-vector store path | this work (probe, ≤3e-7) |
| `vc4_affine` (uop affine) | scalar scale/offset, `1/B`, `1/M` | this work (probe, 6e-8) |
| `vc4_transpose` | dense 16x16 transpose backward | `test_matmul` |
| `vc4_maxpool2x2` / `vc4_maxpooln` | max-pool forward, used by its backward | `test_convgrad`, `test_onnxops` |
| `vc4_conv2d_backward` | conv dX/dW | `test_convgrad` |
| `vc4_softmax_ce` | 16-class softmax + cross-entropy | `test_grad` |

### Why GEMM is the layout primitive

The two kernels that were written for irregular layouts, `vc4_gather` (a
table-driven copy) and `vc4_reduce` (a strided reduction), are **not usable on
this board** (section 6).  Everything irregular in the training path is
therefore expressed as

```
out[i] = sum_j W[i][j] * in[j]
```

with `W` built on the host from *geometry only* (window bounds, tap indices,
concatenation offsets) and multiplied by the verified GEMM.  This is exact for
any layout, needs no new QPU code, and costs `n_out * n_in` MACs.  The cost is
capped by `VC4_MATVEC_MAX` (32 MiB of table): above it the op returns
`VC4_ERR_UNSUPPORTED` with a message telling the caller that this op's
irregular form is only supported for small tensors.  For the regular cases
(the same layout on both sides) no table is used at all: a reduction over the
last axis is `A * ones`, a broadcast is `ones^T * a`, both O(n) work.

## 2. API

```c
/* elementwise derivatives */
typedef enum { VC4_BW_SIGMOID, VC4_BW_SILU, VC4_BW_TANH, VC4_BW_EXP,
               VC4_BW_LOG, VC4_BW_SQRT, VC4_BW_ABS, VC4_BW_CLIP,
               VC4_BW_LEAKYRELU, VC4_BW_HARDSWISH, VC4_BW_HARDSIGMOID,
               VC4_BW_GELU, VC4_BW_GELU_ERF, VC4_BW_ERF, VC4_BW_COUNT } vc4_bwop;
vc4_status vc4_bwop_run(ctx, op, x, g, out, n, p0, p1);
vc4_status vc4_affine(ctx, a, out, n, p0, p1);

/* softmax */
vc4_status vc4_softmax(ctx, x, rows, cols, y);
vc4_status vc4_softmax_bwd(ctx, y, g, dx, rows, cols);
vc4_status vc4_softmax_ce_bwd(ctx, logits, labels, dlogits, loss, y_scratch,
                              rows, cols, inv_batch);

/* normalisation */
vc4_status vc4_batchnorm_fwd(ctx, x, gamma, beta, y, save_mean, save_invstd,
                             running_mean, running_var, N, C, HW, momentum,
                             eps, update_running);
vc4_status vc4_batchnorm_infer(ctx, x, gamma, beta, rm, rv, y, N, C, HW, eps);
vc4_status vc4_batchnorm_bwd(ctx, x, gamma, dy, save_mean, save_invstd, dx,
                             dgamma, dbeta, N, C, HW);
vc4_status vc4_layernorm_fwd(ctx, x, gamma, beta, y, save_mean, save_invstd,
                             rows, cols, eps);
vc4_status vc4_layernorm_bwd(ctx, x, gamma, dy, save_mean, save_invstd, dx,
                             dgamma, dbeta, rows, cols);

/* pooling */
vc4_status vc4_avgpool_bwd(ctx, dy, dx, C, H, W, N, S, pad, count_include_pad);
vc4_status vc4_adaptive_avgpool_bwd(ctx, dy, dx, C, H, W, outH, outW);
vc4_status vc4_global_avgpool_bwd(ctx, dy, dx, C, H, W);
vc4_status vc4_maxpooln_bwd(ctx, x, dy, dx, C, H, W, N, S, pad);

/* broadcasting binary ops */
typedef enum { VC4_BIN_ADD, VC4_BIN_SUB, VC4_BIN_MUL, VC4_BIN_DIV } vc4_binop;
vc4_status vc4_binop_bwd(ctx, op, x, y, g, odims, xdims, ydims, ndim, dx, dy);

/* layout */
vc4_status vc4_cat_bwd(ctx, dy, dx[], axisdim[], nsrc, outer, inner);
vc4_status vc4_split_bwd(ctx, dy[], dx, axisdim[], nsrc, outer, inner);
vc4_status vc4_pad_bwd(ctx, dy, dx, N, C, H, W, pt, pl);
vc4_status vc4_transpose_bwd(ctx, dy, dx, rows, cols);
vc4_status vc4_reduce_bwd(ctx, g, dx, dims, ndim, axis, mean);
vc4_status vc4_resize_nearest(ctx/x, y, planes, H, W, Ho, Wo);
vc4_status vc4_resize_nearest_bwd(ctx, dy, dx, planes, H, W, Ho, Wo);
vc4_status vc4_resize_bilinear(...);       /* half-pixel, edge clamp */
vc4_status vc4_resize_bilinear_bwd(...);
vc4_status vc4_gather_bwd(ctx, g, dx, idx, n_out, n_in);
vc4_status vc4_scatter_bwd(ctx, g, dx, idx, n, n_out);
```

Layouts are fp32 row-major; NCHW tensors are `[N][C][H][W]` contiguous (with
`N == 1` this is the per-image form `vc4_conv2d` takes).

`vc4_softmax_bwd` implements `dx = y*(g - sum_c g*y)` with the row sum from the
GEMM, for any class count.  `vc4_softmax_ce_bwd` gives `dlogits` for any class
count and the loss scalar for the 16-class fused path.

## 3. Op table (forward / backward / measured accuracy)

`ok` = element-verified on the board with the printed error.  `isolated` =
passes in the single-op probe (`probe_atoms`, see section 5) but not yet inside
the full sequence.  `-` = not implemented.

| op | fwd | bwd entry | accuracy measured | state |
|---|---|---|---|---|
| matmul / matmul_bt | QPU | `vc4_matmul_bt` + transpose | rel_L2 5.8e-7 … 8.0e-7 | **ok** (`test_grad`) |
| conv2d 3x3 / 1x1 | QPU | `vc4_conv2d_backward` | 1e-3-relative | **ok** (`test_convgrad`) |
| relu | QPU | `vc4_relu_backward` | 9.1e-7 | **ok** |
| softmax+CE (16 cls) | QPU | `vc4_softmax_ce` | rel_L2 8.4e-5 | **ok** |
| sigmoid | QPU | `vc4_bwop_run(VC4_BW_SIGMOID)` | 2.9e-7 vs exact | **ok** |
| silu | QPU | `VC4_BW_SILU` | 2.8e-5 (probe 9.6e-8) | isolated |
| tanh | (via 2σ(2x)−1) | `VC4_BW_TANH` | 4.5e-7 (probe) | isolated |
| exp | ewb exp | `VC4_BW_EXP` | 2.2e-7 (probe) | isolated |
| log | - (no fwd kernel) | `VC4_BW_LOG` = g·rcp(x) | 2.4e-7 (probe) | isolated |
| sqrt | - | `VC4_BW_SQRT` = g·0.5·rsqrt(x) | 4.7e-8 (probe) | isolated |
| abs | - | `VC4_BW_ABS` = relu_bwd(x,g) − relu_bwd(−x,g) | 0 (probe) | isolated |
| clip | - | `VC4_BW_CLIP` | 0 (probe) | isolated |
| leakyrelu | - | `VC4_BW_LEAKYRELU` | 1.5e-9 (probe) | isolated |
| hardsigmoid | - | `VC4_BW_HARDSIGMOID` (relu_bwd masks) | 5.0e-9 (probe) | isolated |
| hardswish | - | `VC4_BW_HARDSWISH` (same masks) | not measured separately | unverified |
| gelu (tanh form) | - | `VC4_BW_GELU` | 5.8e-7 (probe) | isolated |
| erf | - | `VC4_BW_ERF` | 1.1e-7 (probe) | isolated |
| gelu (erf form) | - | `VC4_BW_GELU_ERF` | **impossible here** | see below |
| softmax (any cols) | QPU (≤16 fused, else generic) | `vc4_softmax_bwd` | 7x13 inf; wedge | **not verified** |
| batchnorm (train) | composition | `vc4_batchnorm_bwd` | dgamma/dbeta 1.3e-6, dx 0.34 | **dx not verified** |
| layernorm | composition | `vc4_layernorm_bwd` | wedge | **not verified** |
| avgpool | (runner composes) | `vc4_avgpool_bwd` | wedge | **not verified** |
| adaptive avgpool | (runner composes) | `vc4_adaptive_avgpool_bwd` | wedge | **not verified** |
| global avgpool | (runner composes) | `vc4_global_avgpool_bwd` | wedge | **not verified** |
| maxpool 2x2 / NxN | QPU | `vc4_maxpool2x2_backward` (2x2), `vc4_maxpooln_bwd` | 2x2 ok (`test_convgrad`); NxN wedge | 2x2 **ok**, NxN not verified |
| add/sub/mul/div + broadcast | QPU | `vc4_binop_bwd` | sub atom 0.9 (probe: composed sub is exact; the probe compared against the broken ewb sub) | not verified in sequence |
| cat / split | host bytes | `vc4_cat_bwd` / `vc4_split_bwd` | - | not verified |
| resize nearest | QPU (W×H matrix) | `vc4_resize_nearest_bwd` | - | not verified |
| resize bilinear | QPU (W×H matrix) | `vc4_resize_bilinear_bwd` | - | not verified |
| transpose | QPU | `vc4_transpose_bwd` | - | not verified |
| pad | host bytes | `vc4_pad_bwd` | - | not verified |
| sum / mean over an axis | - | `vc4_reduce_bwd` | - | not verified |
| gather / scatter | - | `vc4_gather_bwd` / `vc4_scatter_bwd` | - | not verified |

Rows that cannot reach 1e-3, with the reason:

* **`gelu_erf_bwd` — impossible on this board.** The derivative is
  `0.5(1+erf(x/√2)) + x·exp(−x²/2)/√(2π)`.  The erf core needs 23 constants
  plus the three extras; at the only working store granularity (16 vectors per
  group) the body is 4222 instructions and the code slot holds 2048, and the
  4-vector variant writes no output on this board.  Use the tanh-form GELU
  (`VC4_BW_GELU`), whose error against the tanh-form derivative is 5.8e-7, and
  document the difference (≤1e-3 in the derivative near |x|≈2).
* **maxpool ties**: ties share the gradient (all positions equal to the window
  maximum receive `dy`), which is the documented subgradient choice; a finite
  difference at an exact tie is therefore half the analytic value.
* **abs / relu / clip / leakyrelu at the kink**: the analytic value is the
  right/left subgradient; the check skips exact hits (they do not occur with
  the random inputs used).

## 4. Autograd composition

A framework plugin does not need to reimplement the chain rule: each backward
above takes `(saved forward input, incoming gradient)` and writes the input
gradient.  The three compositions that need care are

* **broadcasting binary ops** — `vc4_binop_bwd(op, x, y, g, odims, xdims,
  ydims, ndim, dx, dy)` reduces `g·∂out/∂x` over exactly the axes where
  `xdims[a] == 1 < odims[a]`, so a plugin passes the numpy-style shapes it
  already has;
* **reductions** — `vc4_reduce_bwd(g, dx, dims, ndim, axis, mean)` broadcasts
  the gradient back;
* **batchnorm** — the backward must see the *whole batch* (the gradient sums
  over every `N·H·W` element of the channel).  Calling it per image with
  `N = 1` gives a different (wrong) result; the demo does it once per batch.

## 5. Gradient checks

`training/tests/test_grad.c` covers every row above.  The elementwise family is
checked against the exact closed-form derivative (fp64), which is stricter than
a finite difference of an fp32 forward; the composite ops use central
differences (h chosen per op, 1e-2…2e-2) with the QPU forward where one exists
and an fp64 reference where the forward is a pure affine layout.

Measured at the time of writing (stock clocks, one QPU user):

```
PASS dA matmul 16x16x16      rel_L2 5.8e-7
PASS dB (A^T dC) 16x16x16    rel_L2 6.7e-7
PASS dA matmul 32x32x16      rel_L2 8.0e-7
PASS dB 32x32x16             rel_L2 8.0e-7
PASS relu_bwd                rel_L2 9.1e-7
PASS softmax_ce 8x16         rel_L2 8.9e-5
PASS softmax_ce 64x16        rel_L2 8.4e-5
PASS sigmoid_bwd             rel_L2 2.9e-7
--- beyond this point the sequence wedges the QPU (see section 6) ---
```

The single-op probe (`training/tests/probe_atoms.c`, not in `run_tests.sh`)
measures the atoms in a fresh context:

```
exp (g=1)       2.2e-7     sigmoid'  2.1e-8     tanh'   4.5e-7
silu'           9.6e-8     gelu'     5.8e-7     erf'    1.1e-7
sqrt'           4.7e-8     abs'      0          clip'   0
leakyrelu'      1.5e-9     hardsigmoid' 5.0e-9      log'    2.4e-7
matmul row sums 3.1e-7     matmul broadcast 0
affine a*2+1    6.0e-8     copy2d 13 words 0
```

## 6. Engine defects found (and what the training path does about them)

These are in the *inference* library, not in the training core, and none of the
existing tests in `infer/tests` exercises them.  They matter to anyone building
on `vc4_gather`/`vc4_reduce`/`vc4_blit`:

1. **`vc4_gather` wedges the QPU.**  Its body hands the TMU1 setup word an
   address taken from an accumulator (`areg(100)`), and its per-iteration
   load/pop does not match `build_ew`'s 2-deep pipeline.  Two candidate fixes
   (stage the address in A30, with and without an explicit no-forwarding gap)
   both still hang, so the training path does not use it.
2. **`vc4_reduce` wedges the QPU.**  It walks a fixed 256-output group, so it
   reads and writes past the caller's `outer` elements (a TMU read past the
   end of the arena hangs the board), and its accumulator seed is written to
   register 36 (r4) instead of r0.  `vc4_reduce` now pads the input tail in
   scratch, but the seed bug remains; the training path uses
   `matmul-with-ones` instead.
3. **`vc4_blit` returns `VC4_OK` and copies nothing**, and the `pool`/`blit`
   4-vector store path (`_store4` via `build_ew`'s `nvec != 16` arm) writes no
   output either.  Every kernel in the training path is generated with 16
   vectors per group for this reason.
4. **The `ewb` kernels' second stream is not read correctly** (`sub`, `div`,
   `eqmask`, `abs_bwd`, `max`, `min` all see a wrong `b`).  The `a`-only ones
   (`exp`, `rcp`) are fine.  The training path therefore composes `sub`, `div`
   and `eqmask` from the verified `ew` kernels (`a - b = a + (−1)b`,
   `a / b = a · rcp(b)`, `|d| = relu(d) + relu(−d)`).
5. **The uop family's `nvec = 4` bodies write no output** (exp/log/sqrt/
   reciprocal/tanh/erf), so `vc4_unary(VC4_UOP_EXP)` and friends are not usable;
   `vc4_affine` (16 vectors) is.

The initial wedge cost a board reboot; every subsequent one did not (the QPU
recovers after a reboot only).

## 7. The demo

`training/tests/test_cnntrain.c` is a complete CNN trainer:

```
x (18x18, 1 channel padded to 16) -> conv3x3(16 filters) -> batchnorm ->
relu -> maxpool 2x2 -> flatten 512 -> fc(16 classes) -> softmax CE
```

It uses the new ops (`vc4_batchnorm_fwd/bwd`, `vc4_softmax_ce_bwd`), saves the
weights (conv, fc, gamma, beta, running stats) to `vc4_cnntrain.weights`, and
`--reps 2` trains twice from the same seed with freshly zeroed optimizer state
and compares the two saved files byte for byte.

**It has not been run to completion**: it stops at the first batchnorm
backward, which is one of the composite paths that wedge the board
(section 3), so the loss curve, the step time and the bit-identical claim are
*not* measured yet.  `run_tests.sh` does not invoke it while that is true.

## 8. What a framework user can rely on today

* Forward inference through the verified op set, and training through
  matmul/conv/relu/softmax-CE/sigmoid (the rows marked **ok**).
* `vc4_op_differentiable()` tells you per op whether a backward exists at all;
  the table above tells you whether it is *verified* on hardware.
* Everything else is implemented and registered but must be treated as
  unverified until the composite-path defect in section 6 is fixed.
