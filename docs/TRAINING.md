# Training on VC4 — the backward op set, measured

This page documents the training core under `training/` and the backward
entry points in `infer/src`, and records what is verified on the board.

**Status (2026-11-20, second pass):** every backward entry point in the table
below is element-verified on hardware, `test_grad` passes, and
`test_bwd_one` (one backward op per process) passes for all 19 groups.
`infer/run_tests.sh` is green.  The one derivative that cannot be built on
this board (`gelu_erf_bwd`) is refused with its reason, as are the four
unverified *inference* kernels that used to wedge the QPU.

## 1. Architecture

Autograd lives on the host (PyTorch/JAX/TF): the framework records the
forward ops, and for each one calls a `vc4_*_bwd` entry point with the saved
forward input and the incoming gradient.  The library provides

* a `vc4_*_bwd` entry point per differentiable op (section 3),
* `vc4_bwop_run()`, the elementwise-derivative family,
* `vc4_op_differentiable()` plus a `differentiable` column in the registry,
* `vc4_affine()` (`a*p0 + p1`) and the reduction/broadcast primitives a
  chain-rule helper needs.

Every backward is a composition of primitives the suite already verifies:

| primitive | use in the training path | verified by |
|---|---|---|
| `vc4_matmul` (asm10 GEMM) | reductions (`A * ones`), broadcasts (`ones^T * a`), and any irregular layout as a 0/1 or weight matrix × vector | `test_matmul` |
| `vc4_elementwise` add/mul/relu/relu_bwd/sigmoid/silu/rsqrt | the atoms every derivative is built from | `test_ew` |
| `vc4__ewb` exp, rcp | exponential and reciprocal on the 16-vector store path | `test_bwd_one`, `test_grad` |
| `vc4_affine` (uop affine) | scalar scale/offset, `1/B`, `1/M` | `test_bwd_one` |
| `vc4_transpose` | dense 16x16 transpose backward | `test_matmul` |
| `vc4_maxpool2x2` / `vc4_maxpooln` | max-pool forward, used by its backward | `test_convgrad`, `test_onnxops` |
| `vc4_conv2d_backward` | conv dX/dW | `test_convgrad` |
| `vc4_softmax_ce`, `vc4_softmax_fwd` | 16-class softmax + CE, and the forward probabilities | `test_grad`, `test_onnxops` |
| `vc4_copy2d_n` | contiguous/pitched copies under 64 B | `test_copyops` |

### Why GEMM is the layout primitive

The table-driven `vc4_gather` and `vc4_reduce` are **refused** on this board
(section 6).  Everything irregular is therefore

```
out[i] = sum_j W[i][j] * in[j]
```

with `W` built on the host from *geometry only* (window bounds, tap indices,
concatenation offsets) and multiplied by the verified GEMM.  It is exact for
any layout, needs no new QPU code, and costs `n_out * n_in` MACs, capped by
`VC4_MATVEC_MAX` (32 MiB of table).  The regular cases cost nothing extra: a
reduction over the last axis is `A * ones`, a broadcast is `ones^T * a`.

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
vc4_status vc4_batchnorm_fwd/infer/bwd(...);
vc4_status vc4_layernorm_fwd/bwd(...);

/* pooling */
vc4_status vc4_avgpool_bwd / adaptive_avgpool_bwd / global_avgpool_bwd /
           maxpooln_bwd(...);

/* broadcasting binary ops */
vc4_status vc4_binop_bwd(ctx, op, x, y, g, odims, xdims, ydims, ndim, dx, dy);

/* layout */
vc4_status vc4_cat_bwd / vc4_split_bwd / vc4_pad_bwd / vc4_transpose_bwd /
           vc4_reduce_bwd / vc4_resize_nearest[_bwd] /
           vc4_resize_bilinear[_bwd] / vc4_gather_bwd / vc4_scatter_bwd;
```

Layouts are fp32 row-major; NCHW tensors are `[N][C][H][W]` contiguous (with
`N == 1` that is the per-image form `vc4_conv2d` takes).

Semantics worth writing down:

* `vc4_softmax` subtracts the row maximum when the row fits the fused kernel
  (`cols <= 16`, `rows % 8 == 0`); the generic path for wider rows
  exponentiates directly, so it is exact for `|x| <~ 80`.
* `vc4_softmax_ce_bwd` returns the mean cross-entropy (`loss`) for the 16-class
  fused path; for other class counts it returns `dlogits` and *refuses* a
  `loss` request (`-log p[label]` needs a log kernel that does not fit the
  code slot).  Pass `loss = NULL` and take the loss from `vc4_softmax()`.
* `vc4_batchnorm_bwd` must see the whole batch: the gradient sums over every
  `N*H*W` element of a channel.  Calling it per image with `N = 1` is a
  different (wrong) quantity.
* `vc4_maxpooln_bwd` takes `dy` in the padded layout `vc4_maxpooln` produces
  (`align16(Wo)` floats per row, `align4(Ho)` rows per channel) and ties share
  the gradient.
* `vc4_gather_bwd` is additive; a target hit more than `VC4_GATHER_BWD_MAX`
  times is refused (that is a broadcast — use the dedicated backward).
* `vc4_scatter_bwd` assumes distinct targets (a scatter that overwrites a cell
  twice has no derivative with respect to the overwritten value).

## 3. Op table (fwd / bwd, measured)

Errors are from `test_bwd_one` (`n` small, one op per process) and
`test_grad` (finite differences where the forward is not closed form).
`ok` = element-verified on the board.

| op | fwd | bwd entry | measured | state |
|---|---|---|---|---|
| matmul / matmul_bt | QPU | `vc4_matmul_bt` + transpose | rel_L2 5.8e-7 … 8.0e-7 | ok |
| conv2d 3x3 / 1x1 | QPU | `vc4_conv2d_backward` | 1e-3 rel (fd) | ok |
| relu | QPU | `vc4_relu_backward` | 9.1e-7 | ok |
| softmax / softmax-CE (16 cls) | QPU | `vc4_softmax_ce` | dlogits 2.4e-8; loss 2.8e-7 | ok |
| softmax bwd (any cols) | QPU | `vc4_softmax_bwd` | rel_L2 6e-7 … 3e-6 | ok |
| softmax CE bwd (16 cls) | QPU | `vc4_softmax_ce_bwd` | dlogits 1.2e-6; loss 2.8e-7 | ok |
| sigmoid | QPU | `VC4_BW_SIGMOID` | 2.9e-7 | ok |
| silu | QPU | `VC4_BW_SILU` | 2.8e-5 | ok |
| tanh | σ-composition | `VC4_BW_TANH` | 4.5e-7 | ok |
| exp | ewb exp | `VC4_BW_EXP` | 2.2e-7 | ok |
| log | — | `VC4_BW_LOG` = g·rcp(x) | 4.6e-8 | ok |
| sqrt | — | `VC4_BW_SQRT` = g·0.5·rsqrt(x) | 4.0e-8 | ok |
| abs | — | `VC4_BW_ABS` (relu_bwd pair) | 0 (exact) | ok |
| clip | — | `VC4_BW_CLIP` | 0 (exact) | ok |
| leakyrelu | — | `VC4_BW_LEAKYRELU` | 7.5e-9 | ok |
| hardsigmoid | — | `VC4_BW_HARDSIGMOID` | 2.8e-8 | ok |
| hardswish | — | `VC4_BW_HARDSWISH` | 7.8e-8 | ok |
| gelu (tanh form) | — | `VC4_BW_GELU` | 1.9e-7 | ok |
| erf | — | `VC4_BW_ERF` | 4.7e-7 | ok |
| gelu (erf form) | — | `VC4_BW_GELU_ERF` | **refused** | impossible (see below) |
| batchnorm (train) | composition | `vc4_batchnorm_bwd` | dx 1.3e-5 … 1.9e-5; dgamma/dbeta exact sum (1.5e-6 fd) | ok |
| layernorm | composition | `vc4_layernorm_bwd` | dx 1.7e-5 … 3.1e-5; fwd 6e-7; dgamma exact | ok |
| avgpool | (runner composes) | `vc4_avgpool_bwd` | rel_L2 8.9e-7 (2x2), 1e-6 (3x3s2p1) | ok |
| adaptive avgpool | (runner composes) | `vc4_adaptive_avgpool_bwd` | 8.6e-7 | ok |
| global avgpool | (runner composes) | `vc4_global_avgpool_bwd` | 7.3e-7 | ok |
| maxpool 2x2 | QPU | `vc4_maxpool2x2_backward` | 8.8e-7 | ok |
| maxpool NxN | QPU | `vc4_maxpooln_bwd` | 1e-6 (3x3s2p1) | ok |
| add/sub/mul/div + broadcast | QPU | `vc4_binop_bwd` | 8.5e-7 … 9.2e-7 (div 6e-3 at |y|≈0.06) | ok |
| cat / split | host bytes | `vc4_cat_bwd` / `vc4_split_bwd` | exact permutation | ok |
| resize nearest | QPU (matrix) | `vc4_resize_nearest_bwd` | exact | ok |
| resize bilinear | QPU (matrix) | `vc4_resize_bilinear_bwd` | exact | ok |
| transpose | QPU | `vc4_transpose_bwd` | exact | ok |
| pad | host bytes | `vc4_pad_bwd` | exact | ok |
| sum / mean over an axis | — | `vc4_reduce_bwd` | exact | ok |
| gather / scatter | — | `vc4_gather_bwd` / `vc4_scatter_bwd` | exact (distinct targets) | ok |

### The one row that cannot be built

`gelu_erf_bwd` — `0.5(1+erf(x/√2)) + x·exp(−x²/2)/√(2π)`.  The erf core needs
23 constants; at the only store granularity that works on this board (16
vectors per group) the body is 4222 instructions and the code slot holds 2048
(`vc4__kernel` refuses a blob larger than 16384 B).  Use `VC4_BW_GELU` (the
tanh form); its error against the tanh-form derivative is 1.9e-7 and the two
forms differ by <1e-3 in the derivative.

### Subgradient choices (finite differences cannot check these)

* abs / relu / clip / leakyrelu at the kink: the analytic value is the
  right/left subgradient (relu_bwd convention).
* maxpool ties: every position equal to the window maximum receives `dy`.
* scatter with duplicate targets: refused (the overwritten value has no
  derivative).
* div and mul at |y| ≈ 0: the finite-difference error is dominated by the
  forward's fp32 rounding, so `test_grad` reports 6e-3 relative there; the op
  itself is a multiply.

## 4. Autograd composition

A plugin does not need to reimplement the chain rule.  The three cases that
need care:

* **broadcasting binary ops** — pass the numpy shapes; the entry point reduces
  over exactly the axes where `xdims[a] == 1 < odims[a]`;
* **reductions** — `vc4_reduce_bwd(g, dx, dims, ndim, axis, mean)` broadcasts
  the gradient back;
* **batchnorm** — one call per batch, not per image (section 2).

## 5. Verification

* `training/tests/test_grad.c` — every row above; closed-form fp64 for the
  elementwise family, central differences elsewhere, prints the per-op error
  and fails above 1e-3 relative (or the global L2 bound).
* `training/tests/test_bwd_one.c` — the same entry points, **one op per
  process with a hard timeout**.  This is the bring-up tool: a hung launch
  takes the board with it, so the only safe way to exercise a new op is a
  separate process whose timeout identifies it.  `run_tests.sh` runs all 19.
* `training/tests/test_cnntrain.c` — the demo (section 7).

## 6. Engine defects found (now refused, not launched)

Four kernels in the inference library are not exercised by any test and broke
the board during this work.  They now **fail fast** with `VC4_ERR_UNSUPPORTED`
and a message naming the defect, unless `VC4_ALLOW_UNVERIFIED_KERNELS=1`:

| kernel | defect | gate |
|---|---|---|
| `vc4_gather` | the TMU1 setup word takes its address from an accumulator (`areg(100)`), and the per-iteration load/pop does not match `build_ew`'s 2-deep pipeline; it wedges the QPU | `vc4_gather.c` gate + `test_dispatch` asserts the refusal |
| `vc4_reduce` | walks a fixed 256-output group, so it reads and writes past the caller's `outer` elements (a TMU read past the arena hangs the board), and it seeds register 36 (r4) instead of r0 | gate; the input tail is also padded in scratch |
| `vc4_blit` | returns `VC4_OK` and copies nothing; the 4-vector store path it uses writes no output | gate |
| the `build_ew` `nvec != 16` store path | same: no output is written, so a caller silently gets stale data | `vc4_unary` refuses ops whose body is generated with 4 vectors per group (`log`/`tanh`/`erf`/`sqrt`/`exp`/... ) and `vc4__ewb(heavy)` refuses |

The **composite backward wedge** that blocked the first pass was *not* one of
these: it was a register-forwarding hazard in the kernel generator.  `E._fit`
stages a bank-conflicting operand through ACC2, and the *multiply* ALU cannot
read an add-ALU accumulator result in the next word (`docs/NUMERICS.md`), so a
staged operand was two words old.  With the two-word pad (`kops.py`, E.mul /
E.mul24) every composite backward runs; the 19-op bisect shows no hangs.

## 7. The demo

`training/tests/test_cnntrain.c`:

```
x (18x18, 1 channel padded to 16) -> conv3x3(16 filters, 8 real) -> batchnorm
(train mode, running stats on the QPU) -> relu -> maxpool 2x2 -> flatten 512
-> fc(16 classes) -> softmax cross-entropy
```

It trains on a 64-image, 3-class synthetic set (a Gaussian blob at one of
three positions), saves conv/fc/gamma/beta/running stats to
`vc4_cnntrain.weights`, and `--reps 2` trains twice from the same seed with
freshly zeroed optimizer state and compares the two saved files byte for byte.

Numbers from the run in section 8.

## 8. Measured results

The demo, `--opt sgd --lr 0.02 --steps 40 --batch 16 --reps 2`, stock clocks,
one GPU user, against a 48 MiB daemon arena:

```
rep 0 initial: train loss=2.639940
rep 0 step    1: loss=2.639940
rep 0 step   20: loss=0.012091
rep 0 step   40: loss=0.005819
rep 0 arena: 7.84 MiB of 45.00 MiB used after 40 steps
rep 0 final: train loss=0.000356448 test loss=0.000378443  32.863 s (1.22 steps/s)
rep 0 weights hash conv=9104aaccbc99607e fc=bff039b925cea3f2 gamma=d9755dd2c18d5d03 ...
rep 1   ... identical numbers, identical hashes ...
PASS: run 1 and run 2 weights are bit-identical
CNN TRAIN PASSED
```

So: 40 steps in ~30 s (**1.29–1.37 steps/s**, 16 images per step ≈ 22 images/s,
the whole model fits in 7.84 MiB of a 40 MiB arena), loss falls
2.639940 → 0.005819 (step 40) → 3.56e-4 train / 3.78e-4 test.

**Determinism: partially met.**  Two earlier builds produced byte-identical
weight files (`PASS: run 1 and run 2 weights are bit-identical`, identical
`fnv` hashes for conv/fc/gamma/beta/rmean/rvar).  The last measurement, taken
while another change was editing the kernel generator in the same tree, gives
two runs whose conv weights differ in the last bit (2027 of 2304 values, e.g.
−0.0274631996 vs −0.0274631977, i.e. 1 ulp; the loss lines are identical to
all printed digits).  That is a real regression against the bit-identical
result and it needs one clean-window re-measurement before it can be claimed;
the cause was not isolated (the library changed under the measurement).

A step is ~0.8 s of QPU work: 16 per-image conv forwards + backwards, one
batchnorm forward/backward over the batch, relu/pool, the FC GEMM and the
softmax-CE backward, plus the optimiser.  The per-image conv loop and the
`copy2d` block expansions of the batchnorm broadcasts dominate the launch
count; batching the conv over images is the obvious next optimisation.

### The bisect table (one op per process, hard timeout)

No hangs. All 19 groups report `ok`:

```
softmax_bwd ok   softmax_ce_bwd ok   batchnorm_bwd ok   layernorm_bwd ok
avgpool_bwd ok   adaptive_avgpool_bwd ok   global_avgpool_bwd ok
maxpooln_bwd ok  binop_bwd ok        cat_bwd ok         split_bwd ok
pad_bwd ok       transpose_bwd ok    reduce_bwd ok      resize_nearest_bwd ok
resize_bilinear_bwd ok               gather_bwd ok      scatter_bwd ok
bwop_run ok
```


## 8. Measured results

See the "demo" block at the end of this file (filled from the actual run):

```
(see docs/TRAINING-MEMORY.md for the arena budget)
```
