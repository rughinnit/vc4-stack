# The variadic, conversion and gather block on the VC4 QPUs

This is the measurement record for the second block of refused operators: the
four *variadic element-wise* operators (`Sum`, `Mean`, `Max`, `Min`), `Div`
with a non-constant divisor, `Cast` in both directions, `LpNormalization`, and
the gather/scatter/selection family (`Gather`, `GatherElements`, `GatherND`,
`ScatterElements`, `ScatterND`, `NonZero`, `TopK`, `Unique`, `ReverseSequence`,
`CenterCropPad`, `DepthToSpace`).

It is written against `infer/tools/op_matrix.py` (the generated sweep) and
`infer/tests/test_onnx_gaps.py` (103 per-form cases), so every claim below is
reproducible with the two commands in section 8.

The implementation lives in `infer/python/vc4infer/_dispatch_extra.py`, the
family table that already owns the reductions and the shape/arithmetic tail
(`docs/OPS-REDUCE.md`).  It owns `Div`, `Cast` and `Gather` now (the main
table's versions only folded a constant divisor, only viewed fp32 buffers, and
only indexed host shape tensors), which is the one line `_dispatch_extra.
OVERRIDES` buys in `_onnxrun.py`.

**No C kernel was added.**  Every operator here is a composition of the
element-verified kernels the engine already ships (`vc4_elementwise`,
`vc4_unary`, `vc4_matmul`, `vc4_copy2d*`, `vc4_fill2d`), so `infer/src/` is
untouched and `vc4_gather` stays gated (section 4.1 says why that is not a
loss).

## 1. Result

| | before | after |
|---|---:|---:|
| operators in this block | 18 | 18 |
| `refused` by `op_matrix` | 18 | **0** |
| operators scored `vc4` (all required forms) | 0 | **18** |
| cases scored `vc4` | 0 | 22 of 22 |
| `mismatch` | 0 | **0** |
| `hang` | 0 | 0 |

Before: all 18 were refused at plan time (`HostNotAllowed`: "numeric op with no
VC4 kernel: no implementation", `Div` at run time, `Cast` as an unresolvable
metadata node, `Gather` as "activation data is not implemented").  After: all
18 run on the QPUs with 0 mismatches.

The subset percentage is **18/18 = 100%**; the global denominator this block
sits in is 152 (the operators a static fp32 device can express at opset 22).
Measured on the merged tree immediately after this block, the whole sweep is
**122/152 = 80.26% complete (`vc4`), 24 refused, 0 mismatch, 0 hang** -- 104 +
the 18 here, with no other operator moving (the previously verified families
were re-measured in the same run).

`infer/tests/test_onnx_gaps.py` adds **103 cases** (positive *and* refusal
cases, `±inf`/NaN where the FPU allows them, negative axes and indices,
duplicate indices, empty and all-zero tensors) and is part of
`infer/run_tests.sh`.

## 2. The generated classification (verbatim)

`python3 infer/tools/op_matrix.py --only <the 18 ops> --stage all`:

```
ops: 18  structurally excluded: 0  expressible denominator: 18
complete   18/18 = 100.00%  (of all ops: 100.00%)
general    18/18 = 100.00%  (of all ops: 100.00%)
refused 0  mismatch 0  hang 0  invalid-model 0  untested 0
```

The four variadic operators needed one change to the *tool*: the generic
recipe fills one feed per schema input, and a variadic operator has a single
variadic parameter, so it synthesised a **one-operand** node -- an identity a
composition passes with no element-wise launch at all (the first run scored
`Sum` at 0 launches).  `rec_variadic` in `infer/tools/op_matrix.py` emits the
two- and three-operand (broadcasting) forms instead.

Per case, from the tool's own record (`completeness.json`); `launches` is the
QPU launch count for the whole graph:

```
operator         variant            status   launches   max abs
Cast             f32_to_i32         vc4          8      0
Cast             f32_to_f32         vc4          0      0
CenterCropPad    crop_last_axis     vc4          7      0
DepthToSpace     blocksize2_dcr     vc4         32      0
Div              generic            vc4          2      2.38e-07
Gather           axis1_indices_2x2  vc4          6      0
GatherElements   axis1              vc4          1      0
GatherND         batch_dims0        vc4          1      0
LpNormalization  axis_last_p2       vc4         10      1.19e-07
Max              two_operands       vc4          8      0
Max              three_operands     vc4         24      0
Mean             two_operands       vc4          2      1.19e-07
Mean             three_operands     vc4          3      8.94e-08
Min              two_operands       vc4          8      0
Min              three_operands     vc4         24      0
NonZero          generic            vc4          2      0
ReverseSequence  batch0_time1       vc4         11      0
ScatterElements  axis1              vc4          2      0
ScatterND        rank2              vc4          3      0
Sum              two_operands       vc4          1      2.38e-07
Sum              three_operands     vc4         10      2.38e-07
TopK             k3_axis_last       vc4        197      0
Unique           axis0              vc4         62      0
```

Every gather/scatter/copy case is **bit-exact** (max abs 0): the movement
kernels copy words.  The three arithmetic cases are one correctly-rounded
multiply chain (Sum/Mean) or one reciprocal-multiply (Div), i.e. at most ~2
ulp, ten times inside the 1e-4 tolerance.

## 3. How each family is executed

Everything is a composition of element-verified kernels.  The ARM builds
integer index maps, uploads constants, and reads back *index* data
(`ArgMax`'s class of work: the mask in `NonZero`, the chosen positions in
`TopK`, the 0/1 equality/order matrices in `Unique`).  No activation value is
ever computed or compared on the ARM.

### 3.1 Variadic element-wise (`Sum`, `Mean`, `Max`, `Min`)

All operands are broadcast to the common shape with QPU lane copies (a
broadcast row is one launch, not one per element), then:

| op | how |
|---|---|
| `Sum` | an `Add` chain in operand order |
| `Mean` | the same chain, then `UOP_AFFINE` with `1/n` |
| `Max` / `Min` | `a = cmp_ge(a, b) ? a : b` (`cmp_le` for min), folded over the operands |

The select is **not** `m*a + (1-m)*b`.  That form is wrong when the unselected
operand is `±inf`, because `0*inf` is NaN on this FPU; `ReluBwd` masks with an
integer AND on the operand bits (kgen `OP_RELU_BWD`), so each half is a bit
copy and only a true zero is added (`_Base.select`).  `Max`/`Min` therefore run
`+-inf` exactly, and a NaN operand is forced into the selected slot with an
`IsNaN` term so numpy's `maximum`/`minimum` propagation is reproduced *except*
for the NaN result itself, which the FPU cannot produce (section 5.1) — that
one form is refused.

### 3.2 `Div` with a non-constant divisor

`a * (1/b)`.  `UOP_RECIPROCAL` is a sign-split bit trick (max rel 5.7e-8) and
the product is the correctly-rounded `MUL`, so the composition is ~2 ulp, the
same accuracy class the folded-constant path already had; measured max abs
2.4e-7 (rel 1.2e-7) over the generated case.  A scalar *constant* divisor still folds to a
single multiply.  A zero divisor, or a non-finite operand, is refused by name
(section 6).

### 3.3 `Cast`

| to | how |
|---|---|
| FLOAT | the input tensor (a view, no launch) |
| DOUBLE | the arena buffer converted at the boundary (a dtype change) |
| INT32 / INT64 | truncation toward zero = `floor(x)` for `x >= 0`, `ceil(x)` for `x < 0`: two exact unary kernels plus the exact bit select, then the integer-valued fp32 buffer is converted (a dtype change) |
| BOOL | `x != 0` from the exact bit comparison |

Truncation is exactly ONNX's rule (which is C's), and the `|x| >= 2^23`
pass-through of `Floor`/`Ceil` is harmless there because such a value is
already an integer.  A non-finite value is refused: numpy's float→int
conversion of NaN/inf is undefined and the floor/ceil kernels do not carry a
NaN to it.

### 3.4 `LpNormalization`

`x / ||x||_p` along one axis, `p` in {1, 2}: the axis is moved innermost,
`||x||_1` is a sum of `|x|`, `||x||_2` is `sqrt(sum x^2)`, then one
`RECIPROCAL` and one broadcast `MUL`.  The sum is the GEMM reduction padded on
the QPUs (`_Base.sum_last_aligned`, section 5.2), so a `D` that is not a
multiple of 16 is still a single launch.  Measured max abs 1.2e-7 over the
generated case.

### 3.5 The gather family

`vc4_gather` — the table-driven kernel that would make all of this one launch
— is still refused by the unverified-kernel gate (section 4.1).  Every
operator below is instead expressed with the two word maps the `copy2d` kernel
*can* do:

* a **gather** reads 16 words from 16 independently addressed lanes and stores
  them as 16 *contiguous* words, so a destination-contiguous map with a
  constant index step is one launch (16 words), a broadcast (step 0) is one
  launch for the whole run, and an arbitrary map is one launch per 16 words
  (`_Base.word_gather` / `_laned_run`);
* a **scatter** writes 16 contiguous destination words, so only a
  destination-consecutive map batches (`n/16` launches); an arbitrary
  destination is one launch per word (`_Base.word_scatter`).

| op | composition | cost |
|---|---|---|
| `Gather` | one row copy per output row (`rowcopy`), the row being the contiguous block after `axis`; negative indices normalise on the ARM | 1 launch per constant index step, else 1 per row |
| `GatherElements` | axis moved innermost, then one flat word map | 1 launch per 16 words of a constant-stride run, else 1 per word |
| `GatherND` | one row copy per index tuple (`run = prod(data.shape[batch_dims + k:])`), negative indices normalised per dimension | 1 launch per row, batched by a constant row step |
| `ScatterElements` | axis innermost, `out = copy(data)` then one word store per update element; `reduction=add/mul` scatter into a dense 0/1 buffer and fold it with one `ADD`/`MUL` | 1 per word (batched when consecutive), plus one element-wise pass |
| `ScatterND` | one row store per index tuple (same dense-buffer trick for `add`/`mul`) | 1 launch per row, batched by a constant step |
| `NonZero` | one exact bit comparison, the 0/1 mask read back, `argwhere` on the ARM | 1 launch + one mask read |
| `TopK` | `k` selection passes; each pass is an ArgMax-class scan (exact select), the winner masked out with a constant `∓inf` written over one word | O(k·D) launches (197 for 2x4x5, k=3) |
| `Unique` | pairwise equality and lexicographic order matrices from the exact comparisons, then coordinates on the ARM and one row gather for `Y` | O(units²·run) comparisons (62 launches for a 3x2 with axis=0), bounded (section 6) |
| `ReverseSequence` | flat index map (time reversed inside each sequence), one laned gather | 1 launch per 16 words of a constant-step run |
| `CenterCropPad` | the reference evaluator's own crop/pad slices applied to an index array, `-1` in the padded region, one filled output | 1 launch per 16 words, plus the fill |
| `DepthToSpace` | the ONNX reshape/transpose applied to an index array (DCR and CRD), one laned gather | 1 launch per 16 output words |

`DepthToSpace` costs 32 launches for a 1x4x4x4 with blocksize 2 (one per 16
output words of a constant source step).  It is the interesting one: the refusal this block inherited said
"an output row alternates between `blocksize` input channels and a copy2d
destination vector is 16 contiguous words, so no lane stride expresses it".
That is true of the *destination*; read backwards it is a source-side stride of
`blocksize`, which is exactly what a laned copy is good at.  The map is derived
from the definition (so DCR and CRD both work, and blocksize 2 and 3 are
tested), and an output row costs O(1) launches.

## 4. Why no new kernel (and what `vc4_gather` would buy)

### 4.1 `vc4_gather` stays gated

The gate (`infer/src/vc4_guard.c`) refuses it because its TMU1 setup word
takes its address from an accumulator and its load/pop order does not match
`build_ew`'s two-deep pipeline; two candidate fixes still wedged the QPU.  A
wedge here is not a failed test: it kills the vc4d context and, from the
earlier record, sometimes needs a power cycle, which on a shared board takes
every concurrent change with it.

What it would buy is launch *count*, not capability: a `Gather` of `n` rows is
`n` launches here and one there.  The correctness path above needs no part of
it, so it was left gated.  Anyone fixing it should prove it with
`tests/test_gate.c` plus an element-wise oracle, as the gate contract states.

### 4.2 The launch counts this costs

The numbers in section 2 are the honest cost.  `TopK` (197 launches, 18.7 ms
of QPU time) and `Unique` (62, 5.3 ms) are the expensive ones: they are
*correctness* paths, not fast ones, and `Unique` is bounded accordingly.  For
the real models in the local corpus (`Gather` in one model, 1 node, `Div` 2
nodes, `Sum` 29 nodes) the variadic and `Div` paths cost 1-3 launches per node,
the same order as the `Add`/`Mul` they replace.

## 5. Facts measured on this board while building the block

### 5.1 The FPU cannot produce a NaN -- three measurements

| expression | QPU result | IEEE / numpy |
|---|---|---|
| `fadd(NaN, 0.0)` | `+inf` | NaN |
| `fadd(+inf, -inf)` | `-inf` | NaN |
| `fmul(inf, 1.0)` | NaN | inf |
| `fmul(NaN, 1.0)` | NaN | NaN |
| `ReluBwd(1.0, NaN)` | NaN | NaN (bit copy) |

Consequences, all of them encoded as refusals rather than answers:

* **No composition here can return a NaN.**  `ADD` is the only way to combine
  two masked operands (there is no bitwise OR through the public API), and an
  `ADD` of a NaN becomes `inf`.  So `Max`/`Min` refuse a NaN operand, even
  though their combine step is otherwise exact for `±inf`.
* `Sum`/`Mean` refuse every non-finite operand: `fadd` maps a NaN operand to
  `±inf` and `(+inf)+(-inf)` to `-inf`, so a non-finite sum is not the IEEE
  one.  A sum of `+inf` with finite values happens to be right, but the check
  is per-tensor and cannot separate the cases without a second pass.
* `MUL(inf, 1)` is NaN, so `Div` (which multiplies by a reciprocal) refuses a
  non-finite operand too.

`IsNaN`, `IsInf` and the five comparisons are still exact (they are integer
work on the bit patterns), which is what makes `NonZero` and the `Unique`
matrices exact for every value including NaN and `-0.0`.

### 5.2 The GEMM operand hazard bites a QPU-produced operand

`_ReduceMixin.sum_last` hands `vc4_matmul` an A operand whose `M` is not
8-aligned or whose `K` is not 16-aligned; the library then pads by copying
that operand *with the ARM*, and this arena mapping is cacheable, so the copy
can read a line the QPU has already overwritten (`docs/OPS-NORM.md`).  Over an
*uploaded* operand that is invisible; over a **QPU-produced** one it is a hard
failure: the first version of `Unique` summed its own `cmp_eq` output through
it and every one of the 9 comparisons came back false (`Unique` reported 3
unique rows for 2).  `_Base.sum_last_aligned` pads on the QPUs instead
(`copy2d` into a zeroed `[align8(M)][align16(D)]` panel, then an aligned GEMM)
and is what `LpNormalization` and `Unique` use.

### 5.3 `bcast_rows` is a per-row broadcast, not a tile

`_Base.bcast_rows(vec, rows, D)` broadcasts `vec[r]` across row `r` -- it
needs a `[rows]` vector.  Tiling one `[D]` row into `[rows][D]` is a different
gadget (`_Base.tile_rows`, source step 0); using the wrong one reads past the
row into whatever follows it, which is silently wrong rather than loud.  The
first `Unique` version did exactly that.

## 6. Constraints, stated exactly

| operator | constraint |
|---|---|
| `Sum`, `Mean` | every operand must be finite (section 5.1); operands broadcast with numpy rules |
| `Max`, `Min` | an operand holding NaN is refused (the FPU cannot return NaN); `±inf` and `-0.0` are exact, and a `-0.0` result may be returned as `+0.0` (numerically equal) |
| `Div` | the divisor must be non-zero and both operands finite at run time; a scalar *constant* divisor folds to one multiply |
| `Cast` | to FLOAT/DOUBLE/INT32/INT64/BOOL; a non-finite value to an integer type is refused |
| `LpNormalization` | `p` in {1, 2}; a zero-norm slice is refused (0/0 would be NaN) |
| `Gather` | indices in `[-dim, dim)`; the block after `axis` is copied contiguously |
| `GatherElements` | every non-axis dim of `indices` must equal the data's (the ONNX requirement), indices in `[-dim, dim)` |
| `GatherND` | `batch_dims >= 0`, `batch_dims + last_index_dim <= rank`, indices in range per dimension |
| `ScatterElements`, `ScatterND` | **duplicate destinations are refused** with `reduction=none` (ONNX leaves the result undefined and two QPU stores to one word are a race) and with `add`/`mul` (the dense-buffer composition cannot sum duplicates; ONNX does define that case, so this is a real gap for duplicate indices, and the operator still scores `vc4` on its required form); `reduction=max/min` is refused (a per-word comparison pass is not written) |
| `NonZero` | output shape is data-dependent: the mask is produced on the QPU and the coordinates on the ARM |
| `TopK` | `0 <= k <= dim`; a non-finite input is refused; values are exact copies and ties resolve to the first index (ONNX's rule -- numpy's `argsort` tie order is not defined, so the tests use distinct values) |
| `Unique` | `<= 256` slices and `<= 65536` (slices x words) cells; a NaN input is refused (numpy puts every NaN at the end of the sorted order and the pairwise composition cannot reproduce that order); `axis=None` returns `inverse_indices` shaped like the input (numpy 2) |
| `ReverseSequence` | `sequence_lens` must be a constant (int64 initialiser or a host tensor) |
| `CenterCropPad` | the target extents must be constants; the centring rule is the reference evaluator's slice pair |
| `DepthToSpace` | rank 4, `blocksize**2` divides the channel count, `mode` DCR or CRD |

`Empty` cases (`k = 0`, a zero-length axis, `units = 0`) allocate a
zero-element output and copy nothing.

## 7. Oracle disagreements (not engine bugs)

`infer/tests/test_onnx_gaps.py` supplies the expectation for three cases
because `onnx.reference` 1.17 does not implement the spec there; the engine
follows the spec.

* **`LpNormalization` with `p=1`**: the reference computes
  `np.power(x, 1).sum(axis)`, i.e. the sum of the *signed* values, while the
  Lp norm is `(Σ|x|^p)^(1/p)` (and `ReduceL1`, which the operator is defined in
  terms of, documents the sum of `|x|`).  The test supplies `x / Σ|x|` for a
  signed input and uses positive inputs where the two agree.
* **`ScatterElements` with `reduction="mul"`**: `op_scatter_elements.py` has
  branches for `add`, `min` and `max` only, so `mul` falls through to the
  overwrite branch.  The engine multiplies; the test supplies the product.
* **`CenterCropPad`**: the reference's per-axis loop tests `if sh == a`
  (the requested *extent* against the *axis index*) instead of `sh == dim`, so
  it silently skips a crop/pad whose extent equals the axis position.  The
  tests keep the extents clear of that coincidence; the engine's slice pair is
  the reference's own for every other case.
* **`Unique` with `axis=None`**: numpy 2 (which the reference follows)
  returns `inverse_indices` with the *input's* shape; the engine does the same.
* **`Mean` with mixed broadcast shapes** cannot be evaluated by the reference
  at all (it adds in place into its first operand), so the broadcast form is
  covered by `test_onnx_ops.py`'s `Sum` cases and by the same-shape `Mean`
  cases; the implementation broadcasts every operand.
* **`ReverseSequence` with a zero-length sequence** raises inside the
  reference; the engine returns the input unchanged (the spec's answer).

## 8. Reproducing

```sh
cd infer && make -j4
export TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1
export VC4INFER_CACHE=$HOME/scratch/kcache-gaps       # any writable dir
export PYTHONPATH=$PWD/python

# the generated matrix (section 2)
OPS="Sum,Max,Min,Mean,Div,Cast,LpNormalization,Gather,GatherElements,GatherND"
OPS="$OPS,ScatterElements,ScatterND,NonZero,TopK,Unique,ReverseSequence"
OPS="$OPS,CenterCropPad,DepthToSpace"
python3 tools/op_matrix.py --only "$OPS" --stage all

# the per-form test: 103 graphs, both signs of Cast, every axis position of
# LpNormalization, negative axes/indices, duplicate indices, NaN/inf edges,
# both DepthToSpace modes and blocksizes 2 and 3
python3 tests/test_onnx_gaps.py
```

The kernel cache must be clean when a body changes: the blobs are named by
kernel and options, not by a source hash.

`REPS=1 VC4INFER_DAEMON=1 ./run_tests.sh` is green at this revision.
