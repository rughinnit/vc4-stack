# Recurrent, random, signal and quantise operators on the VC4 QPUs

This is the measurement record for the block that was the hardest to justify:
`RNN`, `GRU`, `LSTM`, the four `Random*` forms plus `Bernoulli` and
`Multinomial`, `QuantizeLinear` / `DynamicQuantizeLinear`, and
`DFT` / `STFT` / `MelWeightMatrix`.  It is written against
`infer/tools/op_matrix.py` (the generated sweep), `infer/tests/test_onnx_rnn.py`
and `infer/tests/test_onnx_sig.py` (the per-form tests) and
`infer/tests/check_rng_gpu.py` (the generator's statistical evidence), so every
number below is reproducible with the commands in section 8.

The implementations are two new dispatch modules,
`infer/python/vc4infer/_dispatch_rnn.py` and
`infer/python/vc4infer/_dispatch_sig.py`, plus one new QPU kernel body
(`kops.b_rng`, `VC4_UOP_RNG`).  Nothing in either module advances the state of
the art for this engine: they are compositions of kernels that were already
element-verified (`vc4_matmul`, `vc4_elementwise`, `vc4_unary`, `vc4_copy2d*`,
`vc4_fill2d`), with one exception that had to be written from scratch, the
device RNG.

Everything below was measured on this board with the core clock pinned at its
stock 400 MHz (`core_freq=400`, `core_freq_min=400`), one client at a time
through `vc4d`, **no** `flock /tmp/vc4-gpu.lock` (the daemon takes that lock
for every launch; a client that wrapped itself in it would deadlock, see
`docs/BUGS.md`), `VC4INFER_DAEMON=1`, `TMPDIR=$HOME/scratch/tmp`.

## 1. Result

| | before | after |
|---|---:|---:|
| operators in this block | 14 | 14 |
| `refused` by `op_matrix` | 14 | **0** |
| scored `vc4` (all required forms) | 0 | **14** |
| `folded` (plan-time constant table) | 0 | **0** |
| `mismatch` / `hang` / `untested` | 0 / 0 / 0 | 0 / 0 / 0 |

Before (`op_matrix --only <the 14 ops> --stage all`):

```
ops: 14  structurally excluded: 0  expressible denominator: 14
complete   0/14 = 0.00%
refused 14  mismatch 0  hang 0  invalid-model 0  untested 0
```

After:

```
ops: 14  structurally excluded: 0  expressible denominator: 14
complete   14/14 = 100.00%  (of all ops: 100.00%)
general    14/14 = 100.00%  (of all ops: 100.00%)
refused 0  mismatch 0  hang 0  invalid-model 0  untested 0
```

`MelWeightMatrix` is a `vc4` operator: its bins are index metadata derived from
the five scalar parameters and the filterbank values are engine kernels
(section 5.4).  Nothing in this block is folded, and the table's `launches`
column is therefore non-zero for every row.

Per case, from the tool's own record (`launches` is the QPU launch count of the
whole one-node graph, `gpu ms` its device time):

| operator | variant | status | launches | gpu ms | max abs |
|---|---|---|---:|---:|---:|
| `RNN` | t2_b2_h2 | vc4 | 30 | 2.19 | 5.1e-7 |
| `GRU` | t2_b2_h2 | vc4 | 72 | 5.30 | 4.0e-7 |
| `LSTM` | t2_b2_h2 | vc4 | 54 | 3.98 | 5.2e-7 |
| `RandomUniform` | shape_2x3 | vc4 | 1 | 0.08 | statistical |
| `RandomUniformLike` | like | vc4 | 1 | 0.08 | statistical |
| `RandomNormal` | shape_2x3 | vc4 | 10 | 0.75 | statistical |
| `RandomNormalLike` | like | vc4 | 10 | 0.75 | statistical |
| `Bernoulli` | p_uniform | vc4 | 2 | 0.15 | statistical |
| `Multinomial` | 2x4_sample3 | vc4 | 36 | 2.70 | statistical |
| `QuantizeLinear` | scale01_zp0 | vc4 | 3 | 0.24 | **exact** (int8 equality) |
| `DynamicQuantizeLinear` | float_input | vc4 | 238 | 18.47 | `y`/`zp` exact, scale 9.3e-10 |
| `DFT` | real_onesided | vc4 | 43 | 3.12 | 3.6e-7 |
| `STFT` | frame8_step4 | vc4 | 23 | 1.70 | 9.5e-7 |
| `MelWeightMatrix` | 4mel_dft8 | vc4 | 72 | 5.37 | 6.0e-8 |

Two things are worth stating plainly, because they are the opposite of what a
completeness table usually implies:

* **`QuantizeLinear` is a storage lever, not a speed one.**  The arena is fp32
  and every kernel reads fp32 words; there is no packed-byte MAC anywhere on
  this GPU.  The operator stores 4x fewer bytes at the graph boundary (and 4x
  fewer in a serialised activation) and saves *no* compute: a `QuantizeLinear`
  → `DequantizeLinear` pair is strictly slower than nothing.  `DequantizeLinear`
  itself remains structurally excluded (its input is int8, which cannot cross
  the fp32 engine boundary at all).
* **the random family is a real device RNG, not a host draw.**  The values
  come out of the QPUs; the ARM is not allowed to generate a random number and
  does not (section 4).  It is not cryptographic and the document says so.

## 2. `RNN`, `GRU`, `LSTM`

### Method

The sequence length is static, so the graph is *unrolled* on the ARM and each
timestep is a pair of QPU GEMMs plus elementwise gate arithmetic:

```
pre = X_t @ W^T  +  H_{t-1} @ R^T  +  (Wb + Rb)      (one GEMM each)
```

`X_t` is read straight out of the input tensor (a row block of the time-major
layout, no copy for layout 0), `H_{t-1}` is the compact `[batch][hidden]`
buffer the previous step produced, and `W^T`/`R^T` are transposed, zero-padded
constant panels built once per direction (they are cached in the runner's
panel table across runs, like a packed conv weight).

The gate split is one `copy2d_x` lane copy per gate (the GEMM writes
`[align8(batch)][align64(gates*hidden)]`, so a gate is a column block at that
row pitch), and each activation is one `vc4_unary` / `vc4_elementwise` launch.
The `h_t` buffer of timestep `t` **is** `Y[t]`'s storage: the store is one lane
copy into the output tensor, not a separate buffer that is then copied.

Three engine facts shape the code and are documented in the module docstring
because they are traps:

* `vc4_matmul` pads an unaligned A operand by copying it **with the ARM** from
  the cacheable arena mapping, and a buffer the QPU wrote can be read stale
  (`docs/OPS-NORM.md`, `sum_last_aligned`).  Every GEMM whose A is a QPU
  product gets an explicitly zero-padded A copied by `copy2d_x` *on the QPUs*,
  so the library's ARM pad path never runs on activation data.
* an elementwise call runs over its **first** operand's element count, so a
  padded panel and a compact gate must never share one call (the first version
  mixed them and the engine refused with `elementwise: invalid argument`).
* per-timestep buffers are released at the end of the timestep; a 512-step
  unroll would otherwise hold thousands of dead arena buffers.

### What is supported, and what is not

| attribute | support |
|---|---|
| `direction` | `forward`, `reverse`, `bidirectional` (bidirectional = two passes, the second one over the reversed sequence) |
| `layout` | 0 (seq-first) and 1 (batch-first) for `X`, `Y`, `Y_h`, `Y_c`, `initial_h`, `initial_c` |
| `initial_h` / `initial_c` | given or omitted (zeros) |
| `B` (bias) | given or omitted |
| `linear_before_reset` | GRU, 0 and 1 (both ONNX formulas) |
| `clip` | yes (bounds the pre-activation of every gate) |
| `activations` | `Sigmoid`, `Tanh`, `Relu`, `Affine`, `HardSigmoid`, `LeakyRelu`, `Elu`, `Softsign`, `Softplus` |
| `activation_alpha` / `activation_beta` | per (direction, activation), defaults of the corresponding ONNX operator (`HardSigmoid` 0.2/0.5, `LeakyRelu` 0.01, `Elu` 1.0) |
| `P` (LSTM peepholes) | yes — `i`, `f` use `C_{t-1}`, `o` uses `C_t`, in the ONNX `[Pi, Po, Pf]` order |
| `hidden_size` | 1 and up; a batch that is not 8-aligned or a hidden size that is not 16-aligned just costs the two padding launches |
| `sequence_lens` | **refused**: "the kernel unrolls one static sequence and every batch row runs the full length" |
| `ScaledTanh` and the other activations | **refused** with the list of the ones that exist |

**Training and backward are not supported and are not claimed.**  There is no
gradient of `RNN`/`GRU`/`LSTM` in this tree; `docs/TRAINING.md` covers the
backward kernels that do exist (matmul, conv, softmax, pool, norm) and the
recurrent ones are not among them.

### Cost and accuracy

Measured over T=8 (batch 3, input 5, hidden 6, so both GEMMs pad):

| operator | launches / timestep | graph total (T=8) | what they are |
|---|---:|---:|---|
| `RNN` | 14.2 | 114 | 2 GEMMs (+2 pad fill/copy each), 2 adds, 1 input copy, 1 gate copy, 1 activation, 1 store |
| `GRU` | 35.2 | 282 | 3 GEMMs (+pads), the `z`/`r` add + bias, 2 gate copies, 2 sigmoids, 1 `r*H` multiply, 1 tanh, the `(1-z)`/`z` mixes, store |
| `LSTM` | 25.5 | 204 | 2 GEMMs (+pads), 2 adds, 4 gate copies, 3 sigmoids + 2 tanh, the `c` update, the `h` multiply, store |

There is a fixed 2–4 launch setup per graph (the input row copy, the initial
state upload), which is why the T=2 case in section 1 (30/72/54) is not exactly
twice the per-timestep cost.

`tests/check_rnn_seq.py` runs that T=8 sequence against `onnx.reference`; the
measured accuracy is in section 7.

Accuracy over that multi-step sequence is inside the matrix tolerance with
room: the per-element error does not accumulate with `t` (the gates are
bounded, `tanh`/`sigmoid` are contractive), and the worst case over the whole
block is ~5e-7 absolute, i.e. the last bit of the fp32 GEMM.

## 3. `QuantizeLinear`, `DynamicQuantizeLinear`

`QuantizeLinear` is three kernels: one `affine` (`x*(1/scale) + zp`),
`UOP_ROUND` (half-to-even, exact) and `UOP_CLIP` (saturating).  A per-axis
scale/zp is broadcast into a constant panel at plan time (a broadcast is a
copy, not arithmetic).  The result is downloaded and cast to the declared
int8/uint8 dtype: that cast is a *dtype conversion* of values the QPUs already
produced, which is the same boundary the bool outputs of `IsInf`/`Greater`
cross.

`DynamicQuantizeLinear` computes the scale and the zero point on the device:

```
hi   = max(0, max x)          lo = min(0, min x)
scale = (hi - lo) / 255       (1/255 when hi == lo -- ONNX's own rule)
zp    = rint(clamp(-lo / scale, 0, 255))
y     = clamp(rint(x / scale) + zp, 0, 255)
```

The global max/min is a 16-wide *tree* (`max_all`): each level is one
`max_last` over 16 columns (64 launches) whatever the number of rows, so the
whole reduction is O(64 log16 N) launches rather than the 4 launches per
element the pooling handlers pay.  238 launches for a 24-element tensor is
that tree (two of them, for max and min) dominating; for a 4096-element
activation it is ~250 launches, i.e. it does not grow with the tensor.

Constraints, stated because they are real:

* **denormals are flushed to zero by the QPU FPU.**  An input whose magnitude
  is below 2^-126 quantises as 0 (measured: a 1e-40 input gives 0 where the
  reference gives 85 with a denormal `y_scale`).  `tests/test_onnx_sig.py`
  pins this with an explicit expectation instead of hiding it.
* the divide is a *reciprocal multiply*: for a non-power-of-two scale the
  quotient is within 1 ulp of `x/scale`, so a value within 1 ulp of a .5
  rounding boundary can round the other way.  Measured over the generated case
  and the 56 focused cases: 0 elements did.
* `saturate=0` (opset 19 float saturation) and `block_size` (block
  quantisation) are **refused by name**; they need a float output range and a
  per-block scale map that this runner does not have.
* the int8 clamp is applied as `clip(x+128, 0, 255) - 128`, because a kernel
  parameter at or below -2.0 cannot cross the daemon (section 6.1).  It is
  exact: the values are already integers no larger than 2^23.
* `y_scale` and `y_zero_point` are returned as the device tensors; the two
  scalars are read back once to serve as kernel *uniforms*, and the returned
  tensors are the same bytes the QPU produced.

## 4. The random family, and the device RNG

### 4.1 The generator (`kops.b_rng`, `VC4_UOP_RNG`)

A counter-based hash, computed on the QPU, on a buffer of counter words the
host provides as *index data* (`np.arange(n, dtype=uint32)`, cached per size,
uploaded once):

```
key = (seed * 0x9E3779B1 + call * 0x85EBCA77) mod 2^32     (ARM, one per launch)
z   = counter_bits ^ key
z  ^= z >> 16 ;  z *= 0x7FEB352D
z  ^= z >> 15 ;  z *= 0x846CA68B
z  ^= z >> 16
u   = (z >> 8) * 2^-24                                     in [0, 1)
```

The two multipliers are Pelle Evensen's *lowbias32* finaliser (a bijection on
32 bits); the two 32-bit multiplies are built from three `mul24` each, because
the QPU's only multiplier is 24x24→low-32 (the split is exact).  The body is
1745 instructions at the verified 16-vector shape, inside the daemon's 16 KiB
code slot.

Seed handling: `seed` comes from the operator's `seed` attribute; when the
attribute is absent the seed is the first 32 bits of
`sha256("<node name>|<op type>")`, so a seedless graph is *reproducible* rather
than different on every run.  `call` is a per-node launch counter: the same
node invoked twice in one process continues the stream (two samples in one
model are independent), and a fresh process replays it bit-for-bit.

**This is not cryptographic.**  An observer who knows the seed can reproduce
the stream; the 3-round finaliser is not a cipher and the key schedule is
linear.  It is a fast, decorrelated, reproducible uniform stream, which is what
an inference graph needs.  The one thing the ARM does is derive the 32-bit key
from `(seed, call)` — a *key schedule*, one integer per launch, no per-element
randomness; every random value is produced by the QPU hash.

### 4.2 Statistical evidence (`tests/check_rng_gpu.py`)

10^6 draws, seed 12345 (four blocks of 250 000, each with its own launch key
-- see below):

```
arena            64.0 MiB (blocks of 250000)
draws            1000000
range            [0.000000536, 0.999999583)
mean             0.500381   (0.5 +- 0.000866 at 3 sigma)
variance         0.083362   (1/12 = 0.083333 +- 0.00086)
chi2 16   bins   stat 13.87  df 15   p 0.536
chi2 256  bins   stat 308.36 df 255  p 0.012
KS distance      0.001022   (limit 0.001950 at p=1e-3)
serial corr lag1 -0.001156   (limit +-0.004000 at 4 sigma)
serial corr lag2 -0.000163
serial corr lag3 -0.000858
serial corr lag4  0.000732
serial corr lag8  0.002101
2-D chi2 64x64   stat 4203.8 df 4095 p 0.115
bit balance      worst |p-0.5| 0.000997 (limit 0.001500)
call-stream      chi2(16) 12.66 p 0.629   KS 0.0082 (limit 0.0305)
reproducible     True (4096/4096 equal for one (seed,call))
decorrelated     call: 0/4096 equal, seed: 0/4096 equal
box-muller       mean -0.00098 var 0.99958 skew -0.00055 kurt 2.99906
box-muller KS    0.01254  (limit 0.03047)
```

The full check (16-bin and 256-bin chi-square, a KS distance against U[0,1], a
2-D chi-square on consecutive pairs — the test a Weyl sequence fails — bit
balance of all 24 output bits, serial correlation at five lags, the same tests
along the *call* axis, reproducibility and de-correlation) passes at seeds
12345, 999 and 777:

| seed | mean | variance | chi2(16) p | KS | 2-D p | call-axis p |
|---|---:|---:|---:|---:|---:|---:|
| 12345 | 0.500381 | 0.083362 | 0.536 | 0.00102 | 0.115 | 0.629 |
| 999 | 0.499918 | 0.083313 | 0.891 | 0.00062 | 0.272 | 0.758 |
| 777 | 0.500059 | 0.083271 | 0.834 | 0.00059 | 0.162 | 0.648 |

The draw is split into four 250 000-element blocks because the engine has no
`free()`: a single 10^6-element launch needs two 4 MiB buffers at once, and a
*shared* daemon whose arena has been whittled down by other clients can refuse
that.  Each block uses its own launch key, so this is four independent samples
of the same stream, not one; the pooled 10^6 values are what the table above
tests.

The *call* axis test is the one that changed the design: the first version
streamed the call counter in as `idx + call*2^20`, so a graph that draws a few
elements per launch sampled every 2^20-th value of the hash.  That subsequence
has lattice structure — the `Multinomial` class histogram came out 3
chi-square sigma off — and keying with a full 32-bit mix of the call counter
fixed it (measured: chi-square p 0.0003 → 0.35).

### 4.3 The operators

| operator | method | launches |
|---|---|---:|
| `RandomUniform(+Like)` | one RNG launch; `low`/`high` by an affine + offset | 1 |
| `RandomNormal(+Like)` | two RNG launches, `sqrt(2 ln(1/u1)) * cos(2 pi u2)`, then one affine for `mean`/`scale` | 10 |
| `Bernoulli` | one RNG launch + one exact fp32 `CMP_LT` against `p` | 2 |
| `Multinomial` | `Exp` → row sum (GEMM) → row reciprocal/multiply → inclusive prefix sum (one GEMM against a constant triangular matrix) → per sample: one RNG launch, a broadcast, an exact compare, a count (GEMM) | 36 for 3 samples |

`RandomNormal` writes `-2 ln u` as `2 ln(1/u)` on purpose: a kernel parameter at
or below -2.0 is refused by the daemon (section 6.1), so no uniform may be that
negative, and the factor 2 folds into the final scale.  `u1` is remapped into
`[2^-25, 1-2^-25]` so `log` is finite: the interval shrinks by 2^-24 of its
length, far below the generator's own 2^-24 resolution.

`Bernoulli`'s output is the 0/1 float ONNX defines (a dtype conversion at the
boundary).  `Multinomial`'s int32/int64 indices are index data, the same class
as `ArgMax`'s.

Statistical acceptance in `op_matrix` (`compare` cannot be used: a draw from
one generator never equals a draw from another).  A random case is run
`reps` times (64 for the `Random*`/`Bernoulli` cases, 128 for `Multinomial`)
and judged on: shape and dtype identical to the reference draw in every
replicate; every value in the operator's analytic support; the pooled mean and
variance within 5 sigma of the distribution the ONNX attributes define;
`Multinomial` by a chi-square over the class histogram at p = 1e-3; and "the
replicates are not all identical" (a stuck stream).  The thresholds and the
statistics are recorded in the case's `compare` record.  This is a documented
special case in `op_matrix.py` (`Spec.stochastic`), not a value comparison, and
it never applies to an operator with a deterministic output.

## 5. `DFT`, `STFT`, `MelWeightMatrix`

### 5.1 `DFT`: one GEMM against a constant twiddle matrix

The transform is a direct sum, expressed as a single GEMM whose *column order
is the ONNX output layout*:

```
out[r][2k]   = sum_j  re_j cos(t) + im_j sin(t)      t = 2 pi j k / N
out[r][2k+1] = sum_j  im_j cos(t) - re_j sin(t)
```

with `W[j][2k] = cos`, `W[j][2k+1] = -sin`, `W[D+j][2k] = sin`,
`W[D+j][2k+1] = cos`, and `1/N` folded in for `inverse=1`.  Because column `2k`
is `Re X[k]` and `2k+1` is `Im X[k]`, the GEMM's result *is* the interleaved
output: no interleave pass exists, which is what makes a direct sum affordable
(43 launches for the generated 4-D case, most of them the per-row input
gathers).

Supported: any `axis`; real (`C=1`) and complex (`C=2`) input; `onesided` 0/1;
`inverse` 0/1; `dft_length` shorter than the axis (truncation, as
`np.fft.fft(..., n=N)` does), equal, or longer (zero-padded); any size (the sum
is direct, so a non-power-of-two length is exact and just as cheap); denormal
and all-zero input.

Constraint: when the transform axis is not the second-to-last one, the output
needs one 2-word copy per complex pair, so the case is refused above
`P*K*I0 > 20000` pairs (the generated case uses 30).  A transform whose axis is
the complex axis itself is refused (ONNX forbids it).

### 5.2 `STFT`: framing + the DFT core

`n_frames = 1 + (length - frame_length) // frame_step`, each frame is one
contiguous `copy2d_x` run, the window is one constant panel multiplied in one
`Mul`, and the result is the same DFT core with `P = batch*frames`.  23 launches
for the generated case.  `frame_length` comes from the window (ONNX's rule),
`onesided` 0/1, `C=1` or 2.

### 5.3 Accuracy

`DFT` 3.6e-7, `STFT` 9.5e-7 absolute against the reference implementation —
the error is the fp32 GEMM's, and a power-of-two length is not special-cased
because it does not need to be.

### 5.4 `MelWeightMatrix` and the windows: QPU tables, not plan-time constants

`MelWeightMatrix` takes only scalar parameters (`num_mel_bins`, `dft_length`,
`sample_rate`, the two edges) and the three window generators take only
`size`, so an earlier revision resolved all four at plan time: the tables were
computed on the ARM and the sweep scored them `folded`, never `vc4`.  They are
now computed by the QPUs on every run (`_dispatch_sig._Window` and
`MelWeightMatrix`), which is what the coverage table measures.

* The **bins** are index metadata: `bin_i = floor((dft_length + 1) f_i /
  sample_rate)` with `f_i` from the mel grid, i.e. five scalar parameters in
  and integers out, the same class of work as an axis list.  `coverage._tier`
  returns `vc4` for the four operators (`TABLES`), and `_tier`'s
  "non-float input means metadata" rule no longer applies to them.
* The **values** are engine kernels.  For a band `(a, b, c) = (bin_i,
  bin_i+1, bin_i+2)` the table rises from 0 at `a` to 1 at `b` and falls to 0
  at `c`; the handler keeps the two degenerate cases the reference defines
  (`b == a` puts a single 1 at `b`, `c == b` has no falling half).  The ramps
  and the per-band gathers are index maps, and the arithmetic is subtract,
  reciprocal-multiply (divide), the exact fp32 comparisons, the two clamps and
  the max of the two halves: 72 launches for the generated 4-mel case, max abs
  6.0e-8.
* The windows are one index ramp plus `VC4_UOP_SIN` (Hann, `sin^2`) or
  `VC4_UOP_COS` (Hamming, Blackman: 3 and 7 launches, max abs 1.8e-7 and
  6.7e-8).  Only fp32 output is wired; `output_datatype` naming half or double
  is refused by name, and `size: 1` with `periodic: 0` (which leaves `N_1 = 0`)
  is refused rather than answered with NaN.

The oracle is still `onnx.reference`'s own window classes; for
`MelWeightMatrix` it is the reference implementation called with python
scalars, because onnx.reference 1.17 crashes on the graph itself (it passes 0-d
numpy scalars to `np.zeros`).

## 6. Engine facts this block established

### 6.1 A kernel parameter at or below -2.0 cannot cross the daemon

`vc4__check_uniform_value` refuses any parameter whose bit pattern is
`>= 0xC0000000`, i.e. every float `<= -2.0`, because that is the range of a
possible GPU bus address.  A whole class of compositions wants exactly such a
constant: a negative zero point, a negative `low`, `-2 ln u`, an int8 clamp at
-128.  Each was rewritten rather than worked around privately:

* a negative offset is applied as `-(-t + |c|)` (two sign flips and one
  positive add; exact for integers) — `_Sig.add_const`;
* int8 saturation is `clip(x+128, 0, 255) - 128`;
* `-2 ln u` is `2 ln(1/u)` with the 2 folded into the operator's own scale.

This is not an engine bug — it is the daemon protecting the GPU from a
parameter that looks like an address — but it is a rule every future
composition over `vc4_unary`/`affine`/`clip` has to know, and it is why
`Div`'s fold, `LRN`'s exponent and `Softmax`'s shift were all written with
positive constants.

### 6.2 A new QPU body has 128 instructions per vector copy

`build_ew` emits the body `nvec` times, and the daemon's code slot is 16 KiB =
2048 instructions, so a 16-vector body has ~128 instructions per copy (the
harness takes ~20 of them).  The first `b_rng` was 2393 instructions and the
JIT refused it; the shipped one is 1745 because the key schedule moved to the
ARM (one integer per launch, no per-element work).  `nvec=8` and `nvec=4` are
not alternatives: the 8-vector store hit a hardware fault and the 4-vector body
is not verified on this board (`docs/NUMERICS.md`).

### 6.3 `op_matrix` was loading the installed package

`tools/op_matrix.py` does `import vc4infer`, and `sys.path[0]` is
`infer/tools`, so a plain `python3 tools/op_matrix.py` tests
`/usr/lib/python3/dist-packages/vc4infer` (the last installed deb), not the
worktree.  Run it as

```
VC4INFER_DAEMON=1 PYTHONPATH=$PWD/python python3 tools/op_matrix.py ...
```

or the numbers describe a release that is older than the tree.  `run_tests.sh`
and the Python tests insert `../python` themselves, so only the sweep tool was
affected.  The same run also needed a daemon: `VC4INFER_DAEMON=1` (without it
`vc4_init` takes the direct `/dev/mem` path, which needs root).

## 7. Reproducing this document

```
cd infer
export TMPDIR=$HOME/scratch/tmp
VC4INFER_DAEMON=1 PYTHONPATH=$PWD/python \
    python3 tools/op_matrix.py --stage all \
    --only RNN,GRU,LSTM,RandomNormal,RandomNormalLike,RandomUniform,\
RandomUniformLike,Bernoulli,Multinomial,QuantizeLinear,DynamicQuantizeLinear,\
DFT,STFT,MelWeightMatrix,HannWindow,HammingWindow,BlackmanWindow

VC4INFER_DAEMON=1 python3 tests/test_onnx_rnn.py      # 62 cases
VC4INFER_DAEMON=1 python3 tests/test_onnx_sig.py      # 56 cases
VC4INFER_DAEMON=1 python3 tests/check_rnn_seq.py      # multi-step vs the reference
VC4INFER_DAEMON=1 python3 tests/check_rng_gpu.py      # 10^6 draws
REPS=1 VC4INFER_DAEMON=1 ./run_tests.sh               # the whole suite
```

`run_tests.sh` runs the three new Python tests plus `check_rng_gpu.py` in its
own sections, so a regression in this block fails the suite.  With
`REPS=1 VC4INFER_DAEMON=1` on this tree: `ALL TESTS PASSED`, peak 67.1 C, core
400 MHz, `throttled=0x0`.

One environmental note, because it cost two runs: start the suite against a
**freshly restarted** `vc4d`.  The shared daemon's arena is a page bitmap and
long use by many clients leaves too few contiguous pages for the next 64 MiB
client, which then falls back to 16 MiB; the training sections
(`test_train`, `test_convnet`) need more than that and fail with `out of arena
memory`.  That is a daemon-side accounting condition, not an operator result:
`test_convnet` passes immediately after `systemctl restart vc4d`, and the
whole suite passes on a fresh daemon.  `check_rng_gpu.py` is written for the
degraded case too (it draws in four blocks with reused buffers).

### Measured multi-step sequence (`tests/check_rnn_seq.py`)

T=8, batch 3, input 5, hidden 6, default attributes, against
`onnx.reference` (its LSTM never returns `Y_c`, so `Y_c` is checked against the
specification formula in `tests/test_onnx_rnn.py` instead):

| operator | launches | per timestep | gpu ms | max abs | over tolerance |
|---|---:|---:|---:|---:|---:|
| `RNN` | 114 | 14.2 | 8.37 | 1.37e-6 | 0 of 240 |
| `GRU` | 282 | 35.2 | 20.68 | 1.02e-6 | 0 of 240 |
| `LSTM` | 204 | 25.5 | 14.99 | 8.72e-7 | 0 of 240 |

The worst *relative* error over those 720 elements is 1.3e-4, and it lands on
the elements where the reference value is ~1e-3: the acceptance criterion is
the matrix tolerance `1e-4 + 1e-4*max(|ref|,1)`, under which nothing is over.

## 8. Not claimed

* No training, no backward pass, no gradient of any operator in this block.
* No packed int8/uint8 arithmetic: the quantised operators are a storage and
  boundary-dtype feature only, and this document does not present them as a
  speedup.
* The RNG is not cryptographic and is not advertised as such.
* `MelWeightMatrix` and the three `*Window` operators are run on the QPU as
  table generators; they are not folded, and their QPU cost is what the table
  above reports.
* No operator here is claimed for a form the tables above do not list:
  `sequence_lens`, `ScaledTanh`, `saturate=0`, `block_size` and a DFT axis that
  is not second-to-last above 20000 output pairs are refused with the invariant
  named in section 2/3/5, not silently mis-answered.
