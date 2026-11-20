# Reductions, comparisons and the shape/arithmetic tail on the VC4 QPUs

This is the measurement record for one block of the ONNX operator set: the
nine `Reduce*` operators, the five comparisons plus `Where`/`IsInf`/`IsNaN`,
`ArgMax`/`ArgMin`, `CumSum`, and sixteen shape/arithmetic operators.  It is
written against `infer/tools/op_matrix.py` (the generated sweep) and
`infer/tests/test_onnx_ops.py` (the per-form test), so every claim below is
reproducible with the two commands at the end.

The implementation lives in `infer/python/vc4infer/_dispatch_extra.py`, a
second dispatch table the runner consults when an operator is not in its own
table.  One kernel family is new: `kgen._ew_cmp`, the exact fp32
comparison/predicate body (see section 3).

## 1. Result

| | before | after |
|---|---:|---:|
| operators in this block | 39 | 39 |
| `refused` by `op_matrix` | 39 | 1 |
| operators scored `vc4` (all required forms) | 0 | 38 |
| cases scored `vc4` | 0 | 64 of 65 |
| `mismatch` | 0 | 0 |
| `hang` | 0 | 0 |

Before: every one of the 39 operators was `refused` ("numeric op with no VC4
kernel").  The whole sweep at that revision reported 111 refused operators.
After: 38 of 39 run on the QPUs with 0 mismatches; `DepthToSpace` is refused
with an exact constraint (section 4).

## 2. The generated classification (verbatim)

`python3 infer/tools/op_matrix.py --only <the 39 ops> --stage all`:

```
ops: 39  structurally excluded: 0  expressible denominator: 39
complete   38/39 = 97.44%  (of all ops: 97.44%)
general    38/39 = 97.44%  (of all ops: 97.44%)
refused 1  mismatch 0  hang 0  invalid-model 0  untested 0
wrote /home/raspi/scratch/tmp/opm-final/completeness.json
wrote /home/raspi/scratch/tmp/opm-final/COMPLETENESS.md
```

Per case, from the tool's own record (`completeness.json`); `launches` is
the number of QPU launches the runner reported for that graph:

```
operator         variant                          status      QPU
ArgMax           axis1_keepdims                   vc4          23
ArgMax           axis_last_no_keepdims            vc4          31
ArgMin           axis1_keepdims                   vc4          23
ArgMin           axis_last_no_keepdims            vc4          31
CastLike         f32_to_f32                       vc4           0
Celu             generic                          vc4          10
Compress         axis0                            vc4           1
CumSum           axis1                            vc4           6
DepthToSpace     blocksize2_dcr                   refused        
Equal            generic                          vc4           1
Expand           broadcast                        vc4           5
EyeLike          4x4                              vc4           0
Greater          generic                          vc4           1
GreaterOrEqual   generic                          vc4           1
Hardmax          last_axis_2d                     vc4          41
Hardmax          axis1_c16_4d                     vc4         153
Hardmax          axis1_c4_4d                      vc4          33
IsInf            default                          vc4           1
IsNaN            default                          vc4           1
Less             generic                          vc4           1
LessOrEqual      generic                          vc4           1
LogSoftmax       last_axis_2d                     vc4          11
LogSoftmax       axis1_c16_4d                     vc4          24
LogSoftmax       axis1_c4_4d                      vc4          12
Mod              generic                          vc4          10
OneHot           depth4                           vc4           2
PRelu            generic                          vc4           5
Pad              constant_pads                    vc4           7
Pad              reflect                          vc4          25
Pad              axes                             vc4           7
Pow              generic                          vc4           3
Range            float_0_1_0p2                    vc4           0
ReduceL1         all_axes                         vc4          10
ReduceL1         axis1_keepdims                   vc4           6
ReduceL1         axis_last_no_keepdims            vc4           4
ReduceL2         all_axes                         vc4          11
ReduceL2         axis1_keepdims                   vc4           7
ReduceL2         axis_last_no_keepdims            vc4           5
ReduceLogSum     all_axes                         vc4          10
ReduceLogSum     axis1_keepdims                   vc4           6
ReduceLogSum     axis_last_no_keepdims            vc4           4
ReduceLogSumExp  all_axes                         vc4          11
ReduceLogSumExp  axis1_keepdims                   vc4           7
ReduceLogSumExp  axis_last_no_keepdims            vc4           5
ReduceMax        all_axes                         vc4          39
ReduceMax        axis1_keepdims                   vc4          15
ReduceMax        axis_last_no_keepdims            vc4          19
ReduceMean       all_axes                         vc4          10
ReduceMean       axis1_keepdims                   vc4           6
ReduceMean       axis_last_no_keepdims            vc4           4
ReduceMin        all_axes                         vc4          39
ReduceMin        axis1_keepdims                   vc4          15
ReduceMin        axis_last_no_keepdims            vc4          19
ReduceProd       all_axes                         vc4          15
ReduceProd       axis1_keepdims                   vc4           7
ReduceProd       axis_last_no_keepdims            vc4           7
ReduceSum        all_axes                         vc4           9
ReduceSum        axis1_keepdims                   vc4           5
ReduceSum        axis_last_no_keepdims            vc4           3
Shrink           generic                          vc4           8
SpaceToDepth     blocksize2                       vc4          64
ThresholdedRelu  generic                          vc4           3
Tile             2x2                              vc4           6
Trilu            upper_k0                         vc4           1
Where            cond_init                        vc4           4
```

The one refusal, verbatim:

```
node 'DepthToSpace_blocksize2_dcr' (DepthToSpace) may not run on the host under host_policy='device': numeric op with no VC4 kernel: DepthToSpace needs an element-level scatter: an output row [ow*blocksize + j] alternates between blocksize input channels, and a copy2d destination vector is 16 contiguous words, so no lane stride expresses it.  vc4_gather, which would, is refused on this board (it wedges the QPU).  SpaceToDepth, its inverse, runs.
(host_policy='any' runs it in the numpy reference evaluator; the operator still needs a VC4 kernel)
```

## 3. How each family is executed

Everything below is a composition of element-verified kernels: no kernel in
this block is new except the comparison body, and the ARM never touches an
activation value (it builds integer index maps, uploads constants, and reads
back the int32 argmax indices).

### 3.1 Reductions

`vc4_reduce` exists in the library but **hangs the QPU** on this board
(section 5), so reductions are composed:

- **axis to last.** `copy2d_x` reads 16 destination words from 16 source
  lanes with a constant lane stride, so the map `T[p*I+i][d] = x[p][d][i]`
  (the reduced axis innermost, the destination row index `(p,i)` exactly the
  ONNX output order) is one lane copy per outer block.  An axis that is
  already innermost needs no copy at all.  The inverse map is the same
  gadget with the roles swapped, which `CumSum`/`Hardmax` use.
- **sum/mean.** One GEMM: `C[outer][1] = T[outer][D] x ones[D][16]`
  (`vc4_matmul` pads M/N/K internally), then a lane copy extracts column 0.
  `ReduceMean` scales by `1/count` with `UOP_AFFINE`.
- **max/min.** An exact masked chain over the slices: `cmp_ge` (bit-exact)
  selects `a` or `b` through two multiplies and an add, so the result is a
  bit copy of one input -- no rounding, no `a-b` difference.
- **prod.** A multiply chain over the slices (`vc4_elementwise MUL` is the
  correctly-rounded product).
- **L1/L2/SumSquare/LogSum/LogSumExp.** `Abs` / `MUL` / `Exp` on the whole
  tensor, then the sum, then `Sqrt`/`Log`.  `ReduceLogSumExp` refuses an
  input outside the `Exp` kernel's documented `|x| <= 88`, and both
  log-forms refuse a non-positive sum (the `Log` kernel's domain).

Cost: sum/mean are one GEMM per reduced axis; max/min/prod are O(D) QPU
launches (two layout copies plus two or three elementwise kernels per
element of the reduced axis).  For small axes that is tens of microseconds
(measured: 13-24 ms per graph, including graph construction and the
oracle).  A very wide reduction (D in the thousands) is legal but slow; the
tiled kernel that would fix it is not written.

### 3.2 ArgMax / ArgMin / Hardmax

A sequential scan over the reduced axis: `cmp_gt` (or `cmp_ge` for
`select_last_index=1`) tests `v > best`, the best value is updated by the
same masked select, and the index by `idx += m*(k - idx)` -- every value is a
small integer, so that arithmetic is exact in fp32.  Ties therefore resolve
exactly as ONNX specifies: the first maximum by default, the last with
`select_last_index=1`.  `Hardmax` is the same scan plus a host-built 0/1
pattern (integer index bookkeeping) and the inverse layout copy.

`ArgMax`/`ArgMin` return int64 index tensors, like `Shape`/`Gather` return
integer tensors: the runner's boundary converts, the QPU found the index.

### 3.3 Comparisons, Where, IsNaN, IsInf

`Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `IsNaN` and
`IsInf` are one new elementwise body each (`kgen._ew_cmp`, ops 10..16 of the
`ew` harness, mirrored by `VC4_OP_CMP_*`/`VC4_OP_ISNAN`/`VC4_OP_ISINF`).
The body is **integer work on IEEE-754 bit patterns only**:

- ordering uses the monotonic key `key(x) = bits(x) ^ ((bits(x)>>31) &
  0x7FFFFFFF)`, which maps `-inf..+inf` onto a strictly increasing *signed*
  int32 (injective, `key >= 0` iff `x >= 0`).  Two keys of the same sign
  cannot overflow when subtracted, and across signs the answer is the sign
  of the first operand, so no overflow correction is needed;
- equality is bit equality of the two operands, which is exact for
  `inf == inf`, plus a both-zero term so that `-0.0 == +0.0`;
- a NaN mask (`|bits| > 0x7F800000`) kills every comparison result, so NaN
  compares false everywhere, `Equal(NaN,NaN)` included.

No subtraction of the operands and no float operation is involved, which
matters twice over: `a - b` would be wrong for `-0.0` vs `+0.0`, and the
QPU FPU flushes denormals (ONNX.md 11.1), so an FPU-based compare would
report `1e-45 > 0.0` as false and `1e-45 == 2e-45` as true.  The kernel
compares those by value.  Verified element-by-element over 512 values
including NaN, +-inf, +-0.0, denormals and equal pairs (six shapes), then
through ONNX graphs in `test_onnx_ops.py`.

The result is a 0.0/1.0 arena tensor; the handler converts it to the bool
array ONNX declares (a dtype conversion, exactly like `Cast`).  `Where` uses
the mask as `c*x + (1-c)*y` (exact for a 0/1 mask, and it broadcasts both
branches with lane copies); `IsInf`'s `detect_negative`/`detect_positive`
flags are applied by the verified `ReluBwd` mask, so all four flag
combinations are exact.

### 3.4 CumSum

`vc4_cumsum` also hangs (section 5), so a prefix sum is one GEMM against a
constant triangular matrix: `out[o][k] = sum_j T[o][j] * M[j][k]` with
`M = triu(ones)` (inclusive forward) or `tril(ones)` (inclusive reverse).
`exclusive` is one contiguous shifted copy per row into a zero-filled
buffer (a right shift for the forward scan, a left shift for the reverse
and `reverse` comes from the matrix, so all four flag combinations are one
GEMM plus at most one copy.  Any axis works (the axis is moved to the last
position first and moved back afterwards).  The accumulation is fp32 like
the reference's, and the scan axis no longer needs a multiple of 256.

### 3.5 Shape / arithmetic

| op | how |
|---|---|
| `Expand`, `Tile` | one integer index map, one `copy2d` lane copy per
  constant-stride run (a broadcast row is a lane stride of 0, so it is one
  launch, not one per element) |
| `Pad` | `fill2d` for the constant, then the same run batcher: constant,
  edge, wrap and reflect (reflect is the one mode whose runs can run
  backwards, and those fall back to one word per copy), with or without the
  `axes` input |
| `Compress` | row select along an axis (`copy2d`); `axis=None` is refused
| `OneHot` | the 0/1 pattern from the integer indices (host bookkeeping),
  then `Mul`/`Add` with the two constant `values` |
| `Trilu` | a triangle mask (a constant) times the input |
| `Range` | a constant sequence |
| `EyeLike` | a constant identity (any `k`, and the int dtypes ONNX allows) |
| `CastLike` | a view for fp32, a dtype conversion otherwise |
| `SpaceToDepth` | one `copy2d` laned gather per block position |
| `Pow` | a small integer exponent is an exact multiply chain (negative
  bases allowed), `0.5`/`-1` are `Sqrt`/`Reciprocal`, anything else is
  `exp(b*log a)` and refuses a non-positive base |
| `Mod` | `q = a*rcp(b)`, `n = round(q)`, `r = a - b*n`, then the exact
  remainder is `r + c*b` where the sign comparison `c` picks floor or trunc
  semantics -- rounding first keeps the decision away from the reciprocal's
  1e-7 error |
| `Shrink` | two comparisons plus masked multiplies |
| `ThresholdedRelu` | `ReluBwd(x - alpha, x)` |
| `Celu` | `Relu` + `Exp` + `Affine` composition |
| `PRelu` | `x * (x>0 ? 1 : slope)`: one comparison and two multiplies |

## 4. Constraints, stated exactly

| operator | constraint |
|---|---|
| `ReduceLogSum`, `ReduceLogSumExp` | the sum (of `exp`) must be positive
  and within the `Log` kernel's domain; `ReduceLogSumExp` also needs
  `|x| <= 88` for the `Exp` kernel |
| `ReduceMax/Min/Prod/ArgMax/ArgMin` | O(D) QPU launches in the reduced
  axis: legal for any D, but a reduction with D in the thousands costs
  thousands of launches (13-24 ms for the small shapes measured here); a
  tiled max kernel would be the fix and is not written |
| `Reduce*` with an empty axis (a 0-length dimension) | refused: the
  identity element is not defined for max/min/prod over an empty set |
| comparisons | exact for every fp32 including NaN/+-inf/+-0/denormals;
  `Where`'s condition must be 0/1 (a bool or a comparison result) |
| `CumSum` | any axis; the scan is a GEMM so a very long axis accumulates
  fp32 rounding like the reference does |
| `ArgMax`/`ArgMin`/`Hardmax` | `select_last_index` supported; the axis is
  scanned sequentially |
| `Pow` | a non-constant exponent needs `base > 0` (`exp(b*log a)`); a
  constant integer exponent in [-4,4], `0.5` and `-1` are exact and accept
  negative bases |
| `Mod` | needs `|a/b| < 2^20` for the rounded-quotient correction to be
  exact; a constant zero divisor is refused |
| `Compress` | an `axis` is required (`axis=None` selects single elements
  from the flattened input, which is a scatter) |
| `SpaceToDepth` | O(C*blocksize^2*Ho) QPU launches (one laned copy per output
  row of a block position); exact, but a 64x64 blocksize-2 tensor is ~16k
  launches, so it is a correctness path rather than a fast one |
| `Pad` | `reflect` needs each pad smaller than the padded dimension (an
  ONNX requirement); negative pads are refused |
| **`DepthToSpace`** | **refused**: an output row `[ow*blocksize + j]`
  alternates between `blocksize` input channels, and a `copy2d`
  destination vector is 16 *contiguous* words, so no lane stride expresses
  it.  The gather kernel that would (`vc4_gather`) is refused on this board
  because it wedges the QPU.  `SpaceToDepth`, its inverse, does run |

## 5. Three library kernels that look like the primitives, and are not

The obvious way to write this block is `vc4_reduce`, `vc4_cumsum` and
`vc4_select3`: they are in the public API, they are not behind the
unverified-kernel guard, and they are exactly the right shape.  Measured on
this board (2026-11-20, stock clocks, clean kernel cache, one launch per
process):

| call | result |
|---|---|
| `vc4_reduce(outer=8, red=3, VC4_RED_SUM)` | `GPU timeout (context is dead)` |
| `vc4_reduce(outer=256, red=16)` | `GPU timeout (context is dead)` |
| `vc4_reduce(outer=16, red=16)` | `GPU timeout (context is dead)` |
| `vc4_reduce(outer=256, red=1)` | `GPU timeout (context is dead)` |
| `vc4_cumsum(rows=2, cols=256)` | `GPU timeout (context is dead)` |
| `vc4_select3(SEL_WHERE, ...)` | returns, but every element is wrong |

The board survives each one (`STATE: LIVE`) because vc4d kills the launch,
but the daemon's context is then dead and needs
`sudo systemctl restart vc4d` before the next client can launch.  All three
share the `kops.py` TMU pattern that `vc4_gather` is already guarded
against, so the same fix applies; until it lands, this block uses
`vc4_matmul`, `vc4_elementwise`, `vc4_unary` and `vc4_copy2d` only, which
are element-verified and whose blobs are byte-identical to the ones the
existing suite exercises.

## 6. Two oracle disagreements (not engine bugs)

`infer/tests/test_onnx_ops.py` supplies the expectation for two cases
because `onnx.reference` 1.17 does not implement the spec there; the engine
follows the spec.  Both are one-line notes, not tolerance relaxations:

- `Mod` with `fmod=0` on floats: `op_mod.py` returns `np.fmod` for *both*
  `fmod` values, while the spec says `fmod=0` is Python's modulus (the sign
  of the divisor).  The test compares against `np.mod`.  The `op_matrix`
  case uses positive operands, where the two agree, so it scores `vc4`.
- `Reduce*` with an explicit empty `axes` tensor and `noop_with_empty_axes=1`:
  `OpRunReduceNumpy.is_axes_empty` only recognises `axes=None`, so the
  reference reduces every axis while the spec (and this engine) returns the
  input unchanged.  The test expects the input.

## 7. Reproducing

```sh
cd infer && make -j4            # libvc4infer + the suite
export TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1
export VC4INFER_CACHE=$HOME/scratch/kcache-ops   # or any writable dir
export PYTHONPATH=$PWD/python                     # the in-tree package

# the generated matrix (section 2)
OPS="ReduceSum,ReduceMean,ReduceMax,ReduceMin,ReduceProd,ReduceL1,ReduceL2"
OPS="$OPS,ReduceLogSum,ReduceLogSumExp,Equal,Greater,GreaterOrEqual,Less"
OPS="$OPS,LessOrEqual,Where,IsInf,IsNaN,ArgMax,ArgMin,CumSum,Pow,Mod,Expand"
OPS="$OPS,Tile,Pad,Compress,OneHot,Trilu,Range,EyeLike,CastLike"
OPS="$OPS,DepthToSpace,SpaceToDepth,Shrink,ThresholdedRelu,Celu,PRelu"
OPS="$OPS,Hardmax,LogSoftmax"
python3 tools/op_matrix.py --only "$OPS" --stage all

# the per-form test: 150 graphs, every axes form, all four CumSum flag
# combinations, the comparison edges, non-last axes
python3 tests/test_onnx_ops.py
```

The kernel cache must be clean when the comparison body changes: the blobs are
named by kernel and options, not by source hash, so `rm -rf $VC4INFER_CACHE`
before a run that changed `kgen.py`/`kops.py`.  (`run_tests.sh` checks that
`infer/tools` and the packaged `infer/python/vc4infer/kgen` copy agree.)

The test suite at this revision:

```
ok   SpaceToDepth     b2-3d-batch              vc4
ok   ThresholdedRelu  alpha1                   vc4
ok   Tile             2x2                      vc4
ok   Tile             1x3x2                    vc4
ok   Tile             size1                    vc4
ok   Trilu            upper-k0                 vc4
ok   Trilu            lower-k1                 vc4
ok   Trilu            batched                  vc4
ok   Where            cond-from-init           vc4
ok   Where            broadcast                vc4
ok   Where            all-true-cond            vc4
150 cases, 0 failed
```

`REPS=1 ./run_tests.sh` is green at the same revision (`SUITE_EXIT=0`,
`ALL TESTS PASSED`, arena 64.0 MiB, no other GPU client running).  Run it with
`sudo systemctl stop vc4d` first: with the daemon holding its own arena the
engine's dma-buf falls back to 16-32 MiB (observed both values on this board
within one hour) and `test_nhwc`'s 64->32 64x64 conv then reports "out of arena
memory".  That is the CMA interaction in docs/TROUBLESHOOTING.md section 7, not
an operator result.

