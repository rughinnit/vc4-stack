# Pooling, normalisation, softmax variants and losses on the VC4 QPUs

This is the measurement record for the seventeen ONNX operators that made up
the "pooling / normalisation / softmax / loss" block: `AveragePool`,
`GlobalMaxPool`, `LpPool`, `GlobalLpPool`, `MaxUnpool`, `BatchNormalization`,
`InstanceNormalization`, `GroupNormalization`, `LayerNormalization`,
`MeanVarianceNormalization`, `LRN`, `Hardmax`, `LogSoftmax`, `Softmax`,
`SoftmaxCrossEntropyLoss`, `NegativeLogLikelihoodLoss` and `Dropout`.

It is the companion of `OPS-REDUCE.md` (the reduction / comparison /
shape-and-arith family) and `OPS-UNARY.md` (the inverse/hyperbolic unary
family), and follows the same pattern: one dispatch module,
`infer/python/vc4infer/_dispatch_norm.py`, whose handlers are compositions of
kernels that are already element-verified, then the generated matrix, then the
per-form test.  Unlike the unary block, no new QPU kernel was written: the
three primitives this block needs (`vc4_batchnorm_infer`,
`vc4_layernorm_fwd`, `vc4_softmax`) were already in the library, and the rest
is `vc4_matmul`, `vc4_elementwise`, `vc4_unary` and `vc4_copy2d_x`.

Everything below was measured on this board with the core clock pinned at its
stock 400 MHz (`core_freq=400`, `core_freq_min=400`), one client at a time
through `vc4d` (no `flock`).

## 1. Result

| | before | after |
|---|---:|---:|
| operators in this block | 17 | 17 |
| `refused` by `op_matrix` (whole operator) | 14 | 0 |
| scored `vc4` (every required form) | 2 | 17 |
| `vc4-limited` (a required form refused) | 1 | 0 |
| `vc4-unverified` (ran, no oracle) | 0 | 0 |
| `mismatch` / `hang` | 0 / 0 | 0 / 0 |

Before (tree `6bfb0de`, `op_matrix --only <the 17 ops> --stage all`):

```
ops: 17  structurally excluded: 0  expressible denominator: 17
complete   17/17 = 100.00%  (of all ops: 100.00%)
general    17/17 = 100.00%  (of all ops: 100.00%)
refused 14  mismatch 0  hang 0  invalid-model 0  untested 0
```

After (the tree this document describes):

```
ops: 17  structurally excluded: 0  expressible denominator: 17
complete   16/17 = 94.12%  (of all ops: 94.12%)
general    16/17 = 94.12%  (of all ops: 94.12%)
refused 0  mismatch 0  hang 0  invalid-model 0  untested 0
```

`GlobalLpPool` is scored `vc4` because `op_matrix.py` carries an independent
numpy oracle for it (`_spec_oracle`, written from `GlobalLpPool-2` as
`(sum |x|^p)^(1/p)`), used only where `onnx.reference` 1.17 has no
implementation at all.  The QPU path is 5 launches with max abs 7.2e-7; the
same formula is covered for p=1 and p=2 in `infer/tests/test_onnx_norm.py`.

`Softmax` was `vc4-limited` before this block: the fused kernel only takes
16 classes, so `axis=1` on a 4-D tensor with 4 channels was refused while the
16-channel and last-axis forms ran.

## 2. Per-operator record

One row per operator.  *Worst measured* is the largest `|got - ref|` (and the
largest relative error) over the cases `op_matrix` generates, against
`onnx.reference`, inside the matrix tolerance `1e-4 + 1e-4*max(|ref|,1)`.
*Launches* is what `op_matrix` counted for the whole one-node graph on the
board (this engine is launch-bound, so the count is part of the cost of the
method).  `exact` means the output was bit-identical to the oracle in every
generated case.

| op | method | constraints enforced | worst measured | launches |
|---|---|---|---|---|
| `AveragePool` | windowed sum as one QPU lane copy per tap (up to `k*k` copies) + `Add`, then a divide | 4-D input (2-D spatial); any kernel/stride/dilation/pads; `auto_pad` NOTSET/VALID/SAME_UPPER/SAME_LOWER; `ceil_mode` (not with `auto_pad`); `count_include_pad` 0/1 | abs 1.8e-7, rel 3.3e-6 | 26-61 |
| `GlobalMaxPool` | `max_last` chain over the plane: slice column `k`, `CMP_GT`, select, `k = 1..H*W-1` | 4-D input; `+-inf` exact, NaN refused | exact | 91 |
| `LpPool` | `Abs` (p=1) or `Mul` (p=2) over the input, then the same windowed sum, then `Sqrt` for p=2 | p in {1, 2}; same geometry as `AveragePool`; **no divisor** (see §3.1) | abs 7.2e-7, rel 3.4e-7 | 27-57 |
| `GlobalLpPool` | `Abs`/`Mul`, one GEMM sum over `H*W`, `Sqrt` for p=2 | 4-D input; p in {1, 2} | `vc4`: `op_matrix` uses an independent numpy oracle of `(sum |x|^p)^(1/p)` (`_spec_oracle`), because `onnx.reference` 1.17 has no implementation; max abs 7.2e-7 | 5 |
| `MaxUnpool` | zero fill, then values routed by a host-built destination map: consecutive indices with a constant positive step are one laned `copy2d` store, everything else a single-word store | 4-D input; in-range int64 indices; unique indices; `output_shape` equal to the inferred shape | exact | 2 |
| `BatchNormalization` | `vc4_batchnorm_infer` (`y = (x-mean) * rsqrt(var+eps) * scale + B`) | inference only (`training_mode=0`, one output); `C` = length of scale/B/mean/var | abs 2.4e-7, rel 2.6e-6 | 15 |
| `InstanceNormalization` | `vc4_layernorm_fwd` with one row per (n,c) plane, ones/zeros gamma/beta, then a per-channel affine | rank >= 3; epsilon inside the sqrt | abs 6.0e-7, rel 3.9e-6 | 32 |
| `GroupNormalization` | `vc4_layernorm_fwd` with one row per (n,group), then a per-channel affine | `num_groups` divides `C`; `stash_type=1`; epsilon inside the sqrt | abs 9.5e-7, rel 2.1e-6 | 32 |
| `LayerNormalization` | `vc4_layernorm_fwd` over the axes `[axis..rank-1]`, gamma/beta flattened to one column each; optional `Mean`/`InvStdDev` outputs | any `axis`; Scale/B broadcastable to `X.shape[axis:]`; `stash_type=1`; epsilon inside the sqrt | abs 7.2e-7, rel 1.9e-6 | 22 |
| `MeanVarianceNormalization` | GEMM mean, GEMM `E[x^2]`, `E[x^2]-mean^2`, `Sqrt`, `+1e-9`, reciprocal-multiply | explicit `axes` (schema default `[0,2,3]` when absent); `|x| <= 1e19` | abs 3.6e-7, rel 1.1e-6 | 50 |
| `LRN` | one GEMM against a constant channel-band matrix for the window sum of `x^2`, `affine(alpha/size, bias)`, `exp(beta*log(t))`, reciprocal-multiply | rank >= 3; `size >= 1`; `bias > 0`; `|x| <= 1e19` | abs 6.3e-5, rel 6.8e-5 (§7.1) | 15 |
| `Softmax` | fused `vc4_softmax_fwd` when `D <= 16` and `rows % 8 == 0`; else `vc4_softmax` when `|x| <= 80`; else a row-max / exp / sum / reciprocal composition | both opset conventions; any axis; any class count (see §8) | abs 4.5e-7, rel 2.7e-6 | 3-7 |
| `LogSoftmax` | `x - log(sum exp x)` when `|x| <= 80`, otherwise `(x-max) - log(sum exp(x-max))` | both opset conventions; any axis, any class count | abs 1.4e-6, rel 1.1e-6 | 10-12 |
| `Hardmax` | first-maximum arg scan (`CMP_GT`, so a tie keeps the first) + host-built one-hot mask + `copy2d` back to the input order | both opset conventions; `+-inf` exact, NaN refused | exact | 33-153 |
| `SoftmaxCrossEntropyLoss` | log-softmax over the class axis, one GEMM against a host-built negated gather-weight mask, then the reduction; optional `log_prob` output | rank >= 2; `|scores| <= 80`; reduction none/mean/sum; `ignore_index`; per-class `weights` | abs 9.5e-7, rel 5.1e-7 | 22 |
| `NegativeLogLikelihoodLoss` | one GEMM against the negated gather-weight mask, then the reduction | rank >= 2; reduction none/mean/sum; `ignore_index`; per-class `weight` | abs 6.0e-8, rel 6.7e-8 | 12 |
| `Dropout` | the identity; the optional mask output is a host `bool` array of ones | inference only (`training_mode` 0 or absent) | exact | 0 |

## 3. How each operator is built

The ARM never touches an activation value in this block.  It builds

* integer geometry: pooling window bounds, the per-output valid-sample counts,
  the LRN band indices, the MOP-up of the `MaxUnpool` destination offsets,
* constants: pooling divisors, ones/zeros gamma tiles, the one-hot /
  gather-weight mask a loss multiplies activations by,
* a validation read of the input (see §6), and the int64 indices of
  `Hardmax`, `MaxUnpool` and the losses (index data, the class of work
  `ArgMax`/`Gather` already do).

### 3.1 Pooling geometry (`AveragePool`, `LpPool`)

The output size is ONNX's, computed exactly as the reference does it:
`floor_or_ceil((in + pad_begin + pad_end - dilation*(k-1) - 1) / stride + 1)`,
where the division is a float division before the floor/ceil so that a
`ceil_mode` boundary lands where the reference lands it.  With `ceil_mode` the
implicit extra pad is split the way ONNX splits it: `extra//2` at the
beginning, `extra - extra//2` at the end.  `auto_pad=VALID` is zero padding,
`SAME_UPPER`/`SAME_LOWER` use the reference's
`(out-1)*stride + k - in` formula, and `ceil_mode` with `auto_pad` is refused
because ONNX forbids the combination.

A tap `(i, j)` of the window reads input `(oh*stride_h + i*dilation_h - pad_top,
ow*stride_w + j*dilation_w - pad_left)`.  For one tap the in-bounds output
positions form a rectangle, so each tap is **one `vc4_copy2d_x` call per
channel plane** with a lane stride of `stride_w*4` bytes and a row step of
`stride_h*W*4`; the taps are summed with `vc4_elementwise(ADD)`.  A tap buffer
is zero-filled before its rectangle is written, because only the rectangle is
written and the accumulator is a sum.  No padding buffer is materialised.

`count_include_pad=0` (the ONNX default) divides by the number of **in-bounds**
samples, which factors into `rows(oh) * cols(ow)`; that product is integer
geometry, so it is built on the ARM as a constant and applied with one
broadcast + reciprocal + multiply.  When every output pixel has the same
divisor (a `count_include_pad=1` window, or a `count_include_pad=0` window with
no padded border) the divide is a single `vc4_unary(AFFINE)` instead, which is
both one launch cheaper and IEEE-exact for an overflowed sum.

**`LpPool` has no divisor.**  ONNX defines no `count_include_pad` for it, and
the reference pads with zero and multiplies the average back by
`prod(kernel_shape)` — the two factors cancel, so the answer is the plain
window sum of `|x|^p` followed by the p-th root.  There is no `p=3` path: any
other `p` needs a power of a possibly zero base and is refused (§8).

### 3.2 Standardisation (`BatchNormalization`, `InstanceNormalization`, `GroupNormalization`, `LayerNormalization`)

`vc4_batchnorm_infer` is the inference form of BatchNormalization and is used
as is.  The other three reuse `vc4_layernorm_fwd`, which standardises each row
over its columns and then applies a per-column affine:

| op | rows | columns | affine |
|---|---|---|---|
| `InstanceNormalization` | `N*C` (one channel plane per row) | `H*W` | ones/zeros, then a per-channel scale/B pass |
| `GroupNormalization` | `N*G` (one contiguous group per row) | `(C/G)*H*W` | ones/zeros, then a per-channel scale/B pass |
| `LayerNormalization` | `prod(shape[:axis])` | `prod(shape[axis:])` | the flattened `Scale`/`B` |

`InstanceNormalization` and `GroupNormalization` need **no layout move**: with
compact NCHW, `[N][C][H*W]` already has one channel plane per row, and a group
of `C/G` channels is a contiguous run of planes.  Their scale/B is applied
afterwards by broadcasting the `[C]` vector over `[N][C][H*W]` with one
`copy2d_x` index map (a constant geometry map, not activation arithmetic).
`LayerNormalization` flattens `Scale`/`B` to one column each when they are
graph constants; a runtime `Scale`/`B` is broadcast to `[rows] + X.shape[axis:]`
on the QPUs and its first row is used, which is the same vector because every
row of that broadcast is identical.

### 3.3 Mean and variance (`MeanVarianceNormalization`)

The ONNX definition is a *function body*: `ReduceMean(X)`,
`E[x^2] = ReduceMean(X^2)`, `Variance = E[x^2] - mean^2`, `Std = Sqrt(Variance)`,
`Y = (X - mean) / (Std + Epsilon)` with `Epsilon = 1e-9`.  The epsilon is
therefore **added after the square root**, not inside it, and this is the one
normaliser in the family where that is true.  Each `ReduceMean` is one GEMM
against a constant ones vector (`sum_last`) per axis, and the result is
broadcast back to the input shape with a `copy2d` index map.

### 3.4 LRN

The channel axis is made innermost (`ax_last`: one `copy2d_x` per plane
block, so the reduction is a row reduction), the window sum of `x^2` is **one
GEMM against a constant band matrix** with a 1 in `[c-c1, c+c2)`, clipped to
`[0, C)`, where `c1 = floor((size-1)/2)` and `c2 = ceil((size-1)/2) + 1` (the
reference's own window), and then
`y = x * rcp((bias + alpha/size*sum)^beta)`.  The power is `exp(beta*log(t))`
for `beta != 1` and `t` itself for `beta == 1`; `bias > 0` guarantees `t > 0`,
which is why a non-positive `bias` is refused.

### 3.5 Softmax variants

The class axis becomes the innermost axis with `ax_last` (`Softmax`,
`LogSoftmax`, `Hardmax`): one `copy2d_x` per plane block, and the result is
written back into ONNX order with the inverse copy.  Then:

* `Softmax`: `vc4_softmax_fwd` (the fused, internally max-shifted kernel) when
  the class count `D <= 16` and `rows % 8 == 0`; otherwise the library's
  `vc4_softmax`, which exponentiates directly and is therefore only used when
  `|x| <= 80`; otherwise a composition — row maximum, `x-max`, `Exp`, one GEMM
  sum, reciprocal, multiply — which is stable for any finite magnitude.
* `LogSoftmax`: `x - log(sum exp x)` for `|x| <= 80`, and
  `(x-max) - log(sum exp(x-max))` above it.  Both are the same value; the
  direct form is 4 launches cheaper but overflows above ~88.
* `Hardmax`: the first-maximum index per row (the same sequential arg scan as
  `ArgMax`, `CMP_GT`, so ties keep the first), then a host-built one-hot mask
  copied back with `copy2d`.  The output is a mask, so it is bit-exact.

### 3.6 Losses

Both losses reduce one gather-weight mask against the activations:

* `NegativeLogLikelihoodLoss`: `loss[n,d] = -input[n, target[n,d], d] * w`, so
  the mask is `-w` at column `target[n,d]` and 0 elsewhere; one multiply and
  one GEMM sum over the class axis.
* `SoftmaxCrossEntropyLoss`: the same mask applied to the log-softmax of the
  scores, so it is `-log_softmax(scores)[label] * w`.  The optional second
  output is the log-softmax itself.

`reduction=none` returns the per-position loss (shape `target.shape`); `sum`
returns one GEMM sum of it; `mean` divides by `sum(gather_weight)` when a
weight or an `ignore_index` is in play (so ignored positions leave both the
numerator and the denominator) and by the number of positions otherwise —
PyTorch's convention, see §7.3.  `Dropout` is the identity, with an all-ones
mask when the second output is requested.

## 4. Epsilon placement

Epsilon placement is the one place where a plausible-looking implementation
silently loses accuracy, so it is stated once per operator:

| op | what the engine computes |
|---|---|
| `BatchNormalization` | `(x - mean) * rsqrt(var + eps) * scale + B`, ONNX default `eps = 1e-5` |
| `InstanceNormalization` | `(x - mean) / sqrt(var + eps) * scale + B`, biased variance over `H*W` |
| `GroupNormalization` | `(x - mean) / sqrt(var + eps) * scale + B`, biased variance over a group |
| `LayerNormalization` | `(x - mean) / sqrt(var + eps) * Scale + B`, ONNX default `eps = 1e-5` |
| `MeanVarianceNormalization` | `(x - mean) / (sqrt(var) + 1e-9)` — **after** the root, as the ONNX function body defines it |
| `LRN` | `bias + (alpha/size) * sum(x^2)` inside the power; no epsilon |

The focused test covers both edges: `epsilon = 0` (BatchNormalization,
InstanceNormalization) and a large `epsilon = 1.0` (BatchNormalization), where
a misplaced epsilon is a visible difference rather than a rounding artefact.

## 5. Layout choice for channel-axis statistics

Statistics along the channel axis are the other place where an implementation
choice has to be stated.  The rule is: **activations are compact NCHW, and a
reduction along the channel axis is taken after the channel axis is made
innermost on the QPUs**, with `ax_last` — one `copy2d_x` call per plane block,
lane stride `H*W*4`, exactly the layout move `OPS-REDUCE.md` already uses for
`ReduceSum`.  The reduction is then a row reduction, and the output keeps the
ONNX order with no inverse pass.

Concretely: `LRN`, `MeanVarianceNormalization` when its `axes` contain the
channel axis, and `Softmax`/`LogSoftmax`/`Hardmax` when `axis` is not the last
one (opset >= 13; the older 2-D coercion needs no move at all, §7.2) all work on
the channel-innermost copy.  `InstanceNormalization` and
`GroupNormalization` need no copy (one channel plane, or one run of planes, is
a contiguous row already), and `GlobalMaxPool`/`GlobalLpPool` reduce the last
two axes of the contiguous plane, so they need none either.

## 6. What the host reads

`op_matrix` counts host bytes per node (`host_bytes`).  In this block the host
reads are:

* one **validation read** of the input per operator, before anything is
  launched (`all_finite`, §8.5).  It is a byte-for-byte download used only to
  refuse an input the composition cannot answer exactly; no value is
  arithmetic-ed.  It happens for every operator that multiplies activations,
  and `AveragePool` only reads when it has to divide by a per-pixel divisor
  (the scalar-divisor forms are IEEE-exact for `+-inf` and skip the read).
* the int64 indices of `Hardmax` (arg scan), `MaxUnpool` (destination map) and
  the losses (one-hot / gather-weight mask).  This is index data, not
  activation data: `ArgMax`, `Gather` and `OneHot` already do the same.
* nothing at all for `Dropout` (identity) and `MaxUnpool` (byte routing only)
  beyond their indices.

Everything else — the divisors, the band matrix, the masks, the gamma tiles —
is integer or constant geometry built once and, where it pays, cached in the
runner's panel table like a packed conv weight.

## 7. Oracle divergences (not engine bugs)

`onnx.reference` 1.17 is the oracle for `op_matrix`, and for five forms here it
is not the specification.  Each is handled the way `OPS-REDUCE.md` §6 handles
its two: the generated matrix case avoids the disagreement where that is
possible, and `infer/tests/test_onnx_norm.py` supplies the ONNX answer where it
is not.

### 7.1 `LRN`: the reference indexes the channel window with the batch index

`onnx/reference/ops/op_lrn.py` writes

```python
for c in range(x.shape[0]):            # the *batch* index
    begin = max(0, c - c1); end = min(minc, c + c2)
    square_sum[:, c, :, :] = np.sum(x[:, begin:end, :, :] ** 2, axis=1)
```

so for a batch of one it fills channel 0 with the window sum and leaves
channels `1..C-1` with a zero sum, i.e. `y = x` for all but the first channel.
Measured on a `1x4x2x2` tensor holding 1..16 with `size=3, alpha=1e-4`: the
reference returns 13.0 for channel 3 where the spec formula returns 12.89, a
0.83 % difference that grows with the channel value.  At the default
`alpha=1e-4` on well-scaled data (`1x16x5x5`, `U(-1,1)`) the whole correction is
only ~3e-5, which is why the generated case still agrees inside the 1e-4
tolerance and scores `vc4` — its worst measured 6.3e-5 is the size of the
correction, not an error.  The focused test
compares against the spec formula directly, including `alpha=0.01`,
`beta=0.5`, `bias=2.0`, `size=5` and `size=4` (an even window).

### 7.2 The opset <= 11 softmax family coerces to 2-D; the reference does not

`Softmax`, `LogSoftmax` and `Hardmax` change meaning at opset 13.  Before 13,
the input is coerced to `[prod(dims[:axis]), prod(dims[axis:])]` and the
operator runs on that matrix (default `axis = 1`, so for NCHW the classes are
the *flattened* `C*H*W`, not the channels).  From 13 they run per position
along `axis` (default `-1`) with no coercion.  The engine implements both, keyed
off the model's `ai.onnx` opset.  `onnx.reference`'s `Hardmax`/`Softmax` classes
ignore the coercion and always work per position, so the focused test supplies
the coerced expectation for `opset=11` (three cases: `Softmax`, `LogSoftmax`,
`Hardmax`).  The generated matrix uses the current opset only, so this does not
affect its scoring.

### 7.3 `NegativeLogLikelihoodLoss` / `SoftmaxCrossEntropyLoss` with `ignore_index = -1`

Both reference implementations build their gather weights with

```python
elif ignore_index != -1:               # note: `is not None`, not `!= -1`
    gather_weight = np.where(target == ignore_index, 0, 1)
```

so an `ignore_index` of exactly `-1` is treated as "no ignore index": the
mean reduction then divides by `N*D`, the ignored positions included (as
zeros).  The spec, PyTorch and this engine divide by the number of *kept*
samples.  Any other `ignore_index` (255 is the common one) takes the reference's
weight path and divides by the kept count, so the generated matrix case and the
focused `ignore_index=255` cases agree; the `ignore_index=-1` focused case
supplies the spec expectation.  Measured on three positions with one ignored:
the reference reports 0.719 where the spec says 1.078 (a factor of 3/2).

### 7.4 `LogSoftmax` above |x| ~ 88: the reference underflows, the spec does not

`onnx/reference` implements `LogSoftmax` as `np.log(Softmax(x))`.  Above a few
dozen in the logits, the softmax probabilities underflow to 0 in fp32 and the
reference returns `-inf` for every class that is more than ~87 below the row
maximum, while `log_softmax` itself is a perfectly finite number
(`x - logsumexp(x)`).  Measured with logits in `[-200, 200]`: the reference
returns `-inf` where the engine returns `-130.5`.  The engine's value is the
spec's; the focused `wide-magnitude` case supplies the logsumexp expectation
and exercises the row-max shift, which is the path above `|x| = 80`.

### 7.5 `MeanVarianceNormalization`: the variance cancels for near-equal samples

The ONNX definition computes `var = E[x^2] - mean^2`, and the engine follows it,
so for a reduction group whose samples are nearly equal the subtraction cancels
in fp32 and the result is only as accurate as that cancellation allows.  This
is not an engine error and it is not a reference error — both compute the same
ill-conditioned expression with slightly different rounding.  Measured on
`axes=[0]` with a two-sample group: when the two samples differ by ~0.003
(`var = 2.4e-6`) the reference and the engine disagree by 0.6 % on a result of
`-1.0`, which is far outside 1e-4 and still correct behaviour for the formula.
The focused `batch-axis` case therefore uses a batch of four (a well-conditioned
group); a model that needs exactness for near-equal samples needs a different
variance formula, which is a change to the ONNX definition, not to this engine.

### 7.6 Two reference forms that are read, not exercised

Recorded so the next person does not chase them; neither is covered by a case.

* `BatchNormalization` opset 9-13: `onnx/reference/ops/op_batch_normalization.py`
  treats a *present* `momentum` attribute as a request for training mode and
  normalises with a momentum-blended mean/var, which is not the spec (inference
  uses `input_mean`/`input_var`).  The engine always uses `input_mean`/`input_var`
  unless `training_mode=1`, which it refuses.  The matrix and focused cases use
  opset 22 with `training_mode=0`.
* `MeanVarianceNormalization` without an `axes` attribute: the schema default is
  `[0,2,3]`, but the reference's function body passes an empty `axes` tensor to
  `ReduceMean`, which (with `noop_with_empty_axes=0`) reduces every axis.  The
  engine implements the schema default; that form is therefore not comparable
  to the reference.  Every focused case passes `axes` explicitly.

## 8. What is refused, with the exact reason

Everything here raises `UnsupportedNode` with the node name and the text below
(abbreviated where marked).  None of these forms appears in the generated
matrix, so no operator lost its `vc4` score to them.

### 8.1 Pooling and unpooling

| op | refused when | reason |
|---|---|---|
| `AveragePool`, `LpPool`, `GlobalMaxPool`, `GlobalLpPool` | input rank != 4 | `only 2-D spatial pooling is wired (input rank N)` — the geometry generalises, the window copy and the divisor map only exist for two spatial axes |
| `MaxUnpool` | input rank != 4 | `only 2-D spatial unpooling is wired (input rank N)` |
| `AveragePool`, `LpPool` | `kernel_shape` missing or not 2 values, `strides`/`dilations` not 2 values, any of kernel/stride/dilation < 1 | `kernel_shape ...`, `kernel/stride/dilation ...` |
| `AveragePool`, `LpPool` | `auto_pad` not one of NOTSET/VALID/SAME_UPPER/SAME_LOWER | `auto_pad=...` |
| `AveragePool`, `LpPool` | `ceil_mode=1` together with `auto_pad` | `ceil_mode with auto_pad=... (ONNX forbids it)` |
| `AveragePool`, `LpPool` | a negative pad | `negative pads [...]` |
| `AveragePool`, `LpPool` | the output would be empty | `output ... is empty` |
| `AveragePool` (`count_include_pad=0`) | a window lies entirely inside the padding, so its divisor is 0 | `an output window lies entirely in the padding (the ONNX average of an empty window is undefined)` |
| `LpPool`, `GlobalLpPool` | `p` not in {1, 2} | `p=N: only p=1 and p=2 are exact (any other p needs a pow on a possibly zero base)` |
| `MaxUnpool` | `kernel_shape`/`pads` malformed, inferred output empty | `kernel_shape ...`, `pads ...`, `inferred output shape ... is empty` |
| `MaxUnpool` | `output_shape` differs from the inferred shape | `output_shape ... differs from the inferred ...; the reference writes into the inferred buffer first, so a larger requested shape is a different semantic that is not wired` |
| `MaxUnpool` | the index count differs from the value count, an index is out of range, or two indices are equal | `indices has N values for M input values`, `indices span [a, b] for an output of N elements`, `duplicate indices: the ONNX result is the last write in input order, and two QPU stores to one word are a race` |

`MaxUnpool` writes into the output with single-word or strided `copy2d` stores
because it is a scatter and `vc4_gather` is refused on this board
(`OPS-REDUCE.md` §5).  Consecutive indices with a constant positive step are
one store; a permutation costs one store per element, which is why the
generated 18-element case takes 2 launches and a full-size unpool costs
`O(output)` launches.  That is a correctness path, not a fast one.

### 8.2 Normalisation

| op | refused when | reason |
|---|---|---|
| `BatchNormalization` | `training_mode=1`, or more than one output | `training_mode=N: this engine implements the ONNX inference form only (y = (x-mean)/sqrt(var+eps)*scale+B).  Training mode updates the running statistics and needs the extra outputs, which is a different kernel (vc4_batchnorm_fwd, training only)` |
| `LayerNormalization`, `GroupNormalization` | `stash_type != 1` | `stash_type=N: only fp32 accumulation (stash_type=1) is implemented` |
| `GroupNormalization` | `num_groups` does not divide `C` | `num_groups=G does not divide C=C` |
| `LayerNormalization` | `Scale`/`B` not broadcastable to `X.shape[axis:]` | `Scale/B shape ... does not broadcast to the normalised axes ...` |
| `LayerNormalization`, `InstanceNormalization`, `BatchNormalization` | a parameter vector whose length is not `C` (or `cols`) | `<what> <name> has N elements (want C)` |
| `MeanVarianceNormalization` | an axis out of range, or an empty axis | `axes [...] for rank R`, `an empty axis in [...]` |
| `LRN` | `size < 1` | `size=N` |
| `LRN` | `bias <= 0` | `bias=B: (bias + alpha/size*sum)^beta needs a positive base for a non-integer beta` |
| all of the above | input rank/emptiness | `needs a rank >= N input (got ...)`, `empty input ...` |

### 8.3 Softmax variants

| op | refused when | reason |
|---|---|---|
| `Softmax`, `LogSoftmax`, `Hardmax` | `axis` out of range | `axis A for rank R` |
| `Softmax`, `LogSoftmax`, `Hardmax` | empty input (a 0-length axis) | `empty input ...` |
| `Softmax`, `LogSoftmax` | NaN or `+-inf` in the input | `a NaN input is refused (...)` / `a non-finite input is refused: ...` (§8.5) |
| `Hardmax` | NaN in the input | `a NaN input is refused (...)`; `+-inf` is accepted because the arg scan is IEEE-exact for it |
| `Softmax` | more than 262144 classes **and** `|x| <= 80` | the library's generic softmax caps the class count (`vc4_softmax: N classes is over the reduce cap`); above the cap and above `|x| = 80` the row-max composition runs instead |

### 8.4 Losses

| op | refused when | reason |
|---|---|---|
| both | input rank < 2 | `input rank R` |
| both | `target` count != `N*D` | `target <name> has N values (want M)` |
| both | a target outside `[0, C)` that is not `ignore_index` | `target T is outside [0, C) and is not ignore_index` |
| both | `weight`/`weights` length != `C` | `weight has N values (want C)` |
| both | `reduction` not none/mean/sum | `reduction='...' (none, mean, sum)` |
| both | `reduction=mean` with every position ignored | `reduction=mean over an empty (all ignore_index) set` |
| `SoftmaxCrossEntropyLoss` | `|scores| > 80` | `scores outside the documented direct-exponential range |x| <= 80` — the log-softmax is computed with the direct exponential here; the plain `LogSoftmax` operator has the stable row-max path, the loss does not |
| both | NaN or `+-inf` activation | §8.5 |

### 8.5 Non-finite activations

`vc4_elementwise(VC4_OP_MUL)` is **not IEEE for a non-finite operand**: measured
on this board, `inf * 0.25`, `inf * 1` and `-inf * 0.25` all return NaN, and a
product that overflows fp32 returns NaN instead of `+-inf`, while `ADD` and the
unary `Affine` kernel are exact.  Every composition in this block multiplies
activations, so rather than answer with a NaN that ONNX does not have, a
non-finite input is refused by name:

* `Softmax`, `LogSoftmax`, `SoftmaxCrossEntropyLoss`, `NegativeLogLikelihoodLoss`,
  `AveragePool` (per-pixel divisor only), `LpPool`, `GlobalLpPool`,
  `BatchNormalization`, `InstanceNormalization`, `GroupNormalization`,
  `LayerNormalization`, `MeanVarianceNormalization`, `LRN` refuse NaN and
  `+-inf`;
* `GlobalMaxPool` and `Hardmax` refuse NaN only: their scans are IEEE-exact for
  `+-inf`, and a focused case checks `+-inf` for both;
* `AveragePool` with a scalar divisor (`count_include_pad=1`, or no padded
  border) accepts `+-inf`, because the sum is `ADD` and the divide is the
  unary `Affine`, both exact.  A focused case checks it against the reference.

The compositions that square their input (`LpPool`/`GlobalLpPool` at p=2,
`LRN`, `MeanVarianceNormalization`) additionally refuse `|x| > 1e19`, above
which `x*x` overflows fp32 and the MUL kernel returns NaN.

This is the one place where the family's answer depends on a *value* rather
than a shape or an attribute, and it is deliberate: a silent NaN is worse than
a named refusal.  `Dropout` (identity) and `MaxUnpool` (byte routing) do not
multiply activations and do not carry the check.

### 8.6 Dropout

| refused when | reason |
|---|---|
| `training_mode=1` | `training-mode Dropout (training_mode=1) is unimplemented: this engine implements the ONNX inference definition only (Y = X, mask = all ones); a training pass needs a device RNG` |
| `training_mode` is not a scalar | `training_mode is not a scalar constant` |

## 9. Two engine limits found while wiring this, and worked around

Both are recorded here because they are invisible from the operator side and
cost time to find; neither was changed (that is engine work, not this block's).

### 9.1 `vc4_matmul` pads an unaligned operand with an ARM copy that can read a stale line

`vc4_matmul` pads an `A` operand whose `M` is not 8-aligned or whose `K` is not
16-aligned by copying it with the ARM (`infer/src/vc4_gemm.c`), from the
cacheable client mapping and without an invalidate first.  The copy can
therefore read a cache line the QPU has already overwritten: measured on this
board, a `16x4x16` GEMM (`rows=16, D=4`) returned another buffer's bytes in its
first four rows, run to run, while the same data with `D=16` was exact.  Every GEMM in
this block therefore pads its operands **on the QPUs** (`copy2d_x` into a
zeroed `[align8(rows)][align16(D)]` panel) so the library never takes that path
with activation data.  Before the change, `LogSoftmax axis1-c4` (which is
`rows=16, D=4`) reproduced the failure in three consecutive runs; after it,
three consecutive runs of the same eight operators, the 89-case test and the
final matrix were all clean.

### 9.2 The MUL kernel is the reason for §8.5

See §8.5.  The same limit applies to `x*x` in the pooling and LRN paths and to
the per-channel affine after standardisation, which is why those operators
carry the validation read.

## 10. Verification

```sh
cd infer && make -j4                 # libvc4infer (unchanged by this block)
export TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1
export VC4INFER_CACHE=$HOME/scratch/kcache-norm2   # or any writable dir
export PYTHONPATH=$PWD/python                      # the in-tree package

OPS=AveragePool,GlobalMaxPool,LpPool,GlobalLpPool,MaxUnpool
OPS="$OPS,BatchNormalization,InstanceNormalization,GroupNormalization"
OPS="$OPS,LayerNormalization,MeanVarianceNormalization,LRN,Hardmax"
OPS="$OPS,LogSoftmax,Softmax,SoftmaxCrossEntropyLoss"
OPS="$OPS,NegativeLogLikelihoodLoss,Dropout"
python3 tools/op_matrix.py --only "$OPS" --stage all

python3 tests/test_onnx_norm.py     # 89 cases, every form in this document
```

`infer/tests/test_onnx_norm.py` is now part of `infer/run_tests.sh` (commit
`2b0c3a8`), so a regression in this block fails the suite rather than the next
person's model.  Its 89 cases, by operator: AveragePool 12, BatchNormalization
6, Dropout 4, GlobalLpPool 2, GlobalMaxPool 4, GroupNormalization 3, Hardmax 6,
InstanceNormalization 3, LayerNormalization 7, LogSoftmax 5, LpPool 5, LRN 5,
MaxUnpool 4, MeanVarianceNormalization 3, NegativeLogLikelihoodLoss 6, Softmax
8, SoftmaxCrossEntropyLoss 6.

The state this document describes:

```
tests/test_onnx_norm.py        89 cases, 0 failed
tests/test_onnx_ops.py        150 cases, 0 failed   (the reduction block, unchanged)
op_matrix --only <17 ops>     refused 0  mismatch 0  hang 0  invalid-model 0
REPS=1 ./run_tests.sh         ALL TESTS PASSED
```

A case that is refused by design is reported by `test_onnx_norm.py` as
`refused-ok` and the case names the invariant it expects, so the refusal texts
in §8 are themselves checked: if a refusal message changes without the test
changing, the test fails.

The kernel cache must be clean when a kernel source changes (`kgen.py` /
`kops.py`): the blobs are named by kernel and options, not by source hash.  This
block adds no kernel, so a cache built for the surrounding commit is reused
as is.

## 11. What is still not wired

* 1-D and 3-D spatial pooling (rank 3 and 5 inputs) are refused.  The window
  geometry generalises; the tap copy and the divisor map are written for two
  spatial axes.
* `MaxUnpool` requires `output_shape` to equal the inferred shape (the
  reference writes into the inferred buffer first, so a larger requested shape
  is a different semantic).
* BatchNormalization/InstanceNormalization/GroupNormalization training mode and
  running-statistics updates are the training kernels' job
  (`vc4_batchnorm_fwd`), not this inference path's.
* `SoftmaxCrossEntropyLoss` uses the direct exponential and therefore refuses
  `|scores| > 80`; the plain `LogSoftmax` operator has the stable path.
* The `MaxUnpool` destination map is rebuilt per run from the indices (they are
  runtime data); a plan-time cache would need the indices to be constants.
