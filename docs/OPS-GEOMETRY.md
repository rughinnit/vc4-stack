# Geometry, detection and contraction on the VC4 QPUs

This is the measurement record for the third block of refused operators: the
ten that a detection or segmentation graph needs beyond convolution --
`ConvTranspose`, `Col2Im`, `AffineGrid`, `GridSample`, `RoiAlign`,
`MaxRoiPool`, `NonMaxSuppression`, `Det`, `Einsum` and `DeformConv`.

It is written against `infer/tools/op_matrix.py` (the generated sweep) and
`infer/tests/test_onnx_geom.py` (82 per-form cases), so every claim below is
reproducible with the two commands in section 9.

The implementation lives in `infer/python/vc4infer/_dispatch_geom.py`, a third
family table beside `_dispatch_extra.py` (`docs/OPS-GAPS.md`) and
`_dispatch_norm.py` (`docs/OPS-NORM.md`).  It is registered in
`_onnxrun._extra_handlers()` and described in `coverage.OPS`.

**No C kernel was added.**  Every operator is a composition of kernels that
are already element-verified:

| kernel | what it does here |
|---|---|
| `vc4_copy2d_x` | every layout move, every indexed gather, every scatter store, every interleave |
| `vc4_elementwise` | add / mul and the exact comparisons and selects (padding masks, IoU, the suppression predicate) |
| `vc4_unary` | floor / ceil / round / abs / sign / reciprocal and the affine combination (coordinates, ROI edges, ranks) |
| `vc4_matmul` | the contraction (Einsum, DeformConv, ConvTranspose), the reduction (RoiAlign avg, NMS ranks) and the boolean closure (NMS) |
| `vc4_fill2d` | every zeroed scatter buffer and every padded panel |

## 1. Result

| | before | after |
|---|---:|---:|
| operators in this block | 10 | 10 |
| `refused` by `op_matrix` | 10 | **0** |
| operators scored `vc4` (every required form) | 0 | **10** |
| `vc4-unverified` (ran, no usable oracle) | 0 | **0** |
| `mismatch` / `hang` | 0 / 0 | 0 / 0 |
| focused cases in `test_onnx_geom.py` | 0 | **82, 0 failed** |

Before (tree `f8a3b31`, `op_matrix --only <the 10 ops> --stage all`):

```
ops: 10  structurally excluded: 0  expressible denominator: 10
complete   0/10 = 0.00%  (of all ops: 0.00%)
refused 10  mismatch 0  hang 0  invalid-model 0  untested 0
```

After (the tree this document describes):

```
ops: 10  structurally excluded: 0  expressible denominator: 10
complete   10/10 = 100.00%  (of all ops: 100.00%)
general    10/10 = 100.00%  (of all ops: 100.00%)
refused 0  mismatch 0  hang 0  invalid-model 0  untested 0
```

`Col2Im` and `MaxRoiPool` are scored `vc4` because `op_matrix.py` now carries an
**independent numpy oracle** for them (`_spec_oracle`, written from the ONNX
definition: the accumulation rule of `Col2Im-18` and the Fast R-CNN bin geometry
of `MaxRoiPool-1`).  It is used only where `onnx.reference` 1.17 cannot
evaluate the graph: its `Col2Im` passes the flat `pads` list where a 2-D array
is expected (`IndexError: index 2 is out of bounds`), and its
based-on-torchvision `MaxRoiPool` is not registered at all.  The oracle reads
only the graph inputs and attributes, never the engine's output.

Both operators run entirely on the QPU kernels: 17 launches, max abs 2.4e-7 for
`Col2Im`; 238 launches, max abs 0.0 for `MaxRoiPool` (a max is exact).  The
same formulas are covered form by form in `test_onnx_geom.py`: five `Col2Im`
cases (blocks 2x2 and 3x3, pads, stride 2, dilation 2, non-square blocks) and
four `MaxRoiPool` cases (2x2 and 3x3 pooling, `spatial_scale`, two batches).

## 2. Per-operator record

*Worst measured* is the largest `|got - ref|` over the generated case, against
`onnx.reference` (or the numpy expectation for the two above).
*Launches* and *gpu ms* are what the runner counted for the whole one-node
graph on the board; this engine is launch-bound, so both are part of the cost
of the method.  Clock pinned at the stock 400 MHz (`throttled=0x0`), one
client at a time through `vc4d`, no `flock`.

Thermal conditions (the board is fanless): the matrix and focused runs started
between 55 and 66 C and peaked at 72 C -- every run longer than a minute was
wrapped in `tools/thermal-guard.sh --max 72`.  The `REPS=1 ./run_tests.sh`
that closes this block started at 61.2 C, peaked at 71.4 C and ended at
70.4 C, and was not cut (exit 0).  One earlier attempt was cut at 72 C in a
warm window (exit 99), which is the guard doing its job, not a test failure.

| op | method | worst measured | launches | gpu ms |
|---|---|---|---:|---:|
| `ConvTranspose` | `cols = W^T @ X` (one GEMM per batch and group), then the same k^2-tap scatter Col2Im uses; bias by QPU broadcast | abs 2.4e-7, rel 1.8e-5 | 41 | 3.3 |
| `Col2Im` | k^2 zeroed tap buffers, one scatter copy per (plane, tap), one vector add per tap | abs 2.4e-7 (spec-numpy oracle) | 17 | 1.3 |
| `AffineGrid` | two QPU affine planes per batch + an interleaved one-word store | abs 6.0e-8, rel 1.1e-7 | 18 | 1.4 |
| `GridSample` | QPU denormalise / reflect / floor, index read-back, four `copy2d` tap gathers, QPU bilinear weights | abs 5.4e-7, rel 9.2e-6 | 238 | 17.0 |
| `RoiAlign` | QPU ROI scalars and sample coordinates, four tap gathers per sample, GEMM sum (avg) or exact max chain (max) | abs 2.4e-7, rel 7.3e-6 | 297 | 21.4 |
| `MaxRoiPool` | QPU-rounded ROI edges and bin bounds, one window copy per channel plane, exact max chain | exact (spec-numpy oracle) | 238 | 17.5 |
| `NonMaxSuppression` | QPU IoU matrix, pairwise score ranks (2 GEMMs), boolean suppression closure (`log2 N` squarings), QPU cap | **exact** | 173 | 12.6 |
| `Det` | Leibniz expansion: one constant-stride gather per factor, one mul/add chain per permutation | abs 1.2e-7, rel 1.6e-7 | 36 | 2.6 |
| `Einsum` | lowered to permutations + one GEMM per batch (not a kernel) | abs 1.2e-7, rel 4.5e-7 | 8 | 0.6 |
| `DeformConv` | QPU offset coordinates, four-tap bilinear im2col per (channel, tap), QPU mask multiply, one GEMM per (batch, group) | abs 6.6e-7, rel 3.1e-5 | 3217 | 230.5 |

The launch counts are the honest cost of a launch-bound engine and a gather
instruction that is still gated: `GridSample` is 238 launches for a
`1x3x6x6` sampled at `5x5`, and `DeformConv` is 3217 for `1x4x6x6` with a
`3x3` kernel (four tap gathers per (channel, tap, batch)).  Section 4 says
exactly which part of each count would collapse if `vc4_gather` were fixed.

## 3. How each family is executed

Everything is a composition of element-verified kernels.  The ARM builds
integer index maps and geometry, uploads constants, and reads back *index*
data -- the same class of work `ArgMax`, `TopK`, `NonZero` and `Unique`
already do.  No activation value is ever computed or compared on the ARM, and
this block adds no new exemption: `RoiAlign`'s out-of-range sample mask, for
instance, is a QPU comparison that multiplies the four bilinear weights (an
earlier draft read the coordinates back and masked on the ARM, which was
tighter than it needed to be).

### 3.1 `ConvTranspose` and `Col2Im`: the scatter family

`ConvTranspose` is **col2im + GEMM**, not a zero-insertion forward conv.  The
brief asks for the choice and the MACs:

| method | MACs |
|---|---|
| col2im + GEMM (used) | `N*Cin*Hin*Win*Cout/g*kh*kw` = the operator's own count |
| zero-insertion forward conv | `N*Cout*Hout*Wout*Cin*kh*kw`, i.e. `≈ sh*sw` times more |

For the generated case (`1x4x8x8 -> 1x2x15x15`, 3x3, stride 2) that is 4608
against 16200 MACs: 3.5x, which is `sh*sw = 4` minus the boundary.  The GEMM
also means `W^T` is a *constant* layout change built once per node, and the
same tap-scatter routine serves both operators.

The scatter itself is the interesting constraint.  A `copy2d` store is 16
*contiguous* words, so an interleaved destination (stride `sw` > 1) cannot be
a vector store at all; it is written with `nlane=1, depth=1` one-word stores,
one launch per (plane, tap) with the tap's block rows spread over the QPUs.
Accumulation of overlapping blocks cannot be a QPU store either (two stores to
one word are a race), so each tap is a **fresh zeroed buffer plus one vector
add**, in fixed `(ky, kx)` order -- the same fixed order `docs/NUMERICS.md`
records for the engine's C col2im.  A tap's destination positions form a
sublattice (`y = iy*sh - pad + ky*dil`), which is exactly a row rectangle.

### 3.2 `AffineGrid`

`x = t00*xs + t01*ys + t02` (and the y row) is a theta broadcast, two
multiplies and two adds per coordinate plane.  Theta is read with lane copies:
the ARM never sees a theta value.  The only unusual step is the output layout
`[N][H][W][2]`, which is written directly by a one-word strided store
(`stride_store`), so the grid is never interleaved on the ARM.

### 3.3 `GridSample`: a gather with computed addresses

`vc4_gather` -- the table-driven kernel that would make this one launch -- is
still refused by the unverified-kernel gate, for the reasons
`docs/OPS-GAPS.md` section 4.1 records.  This is therefore **four predicated
`copy2d` taps**, and the launch count is reported as measured, not estimated:

1. deinterleave the grid into x and y planes (a laned gather, `slan=8`);
2. denormalise (`(g+1)/2*(d-1)` or `((g+1)*d-1)/2`) and, per `padding_mode`,
   clamp (`border`) or fold (`reflection`: `n = floor(dx/rng)`, parity selects
   `lo + r` or `hi - r`, all QPU including the parity test);
3. `floor` for the base index and the fractional part for the weights (QPU);
4. read the *floor* back as integer indices -- index data, the `ArgMax` class
   -- and build four word maps (`-1` = leave the zeroed tap buffer alone);
5. one `word_gather` per tap: a constant source step is one `copy2d` per 16
   words, so a locally regular grid batches and a random one does not;
6. combine with the QPU weights `(1-fx)(1-fy)`, `fx(1-fy)`, ... (nine
   elementwise launches over `[N][C][Ho][Wo]`).

`nearest` is one tap and no arithmetic, so it is bit-exact.  `cubic` is refused
(section 8): it needs a 4x4 Keys polynomial per axis and 16 taps.

### 3.4 `RoiAlign` and `MaxRoiPool`

`RoiAlign`'s sampling coordinates are QPU arithmetic: per ROI the four scalars
`(start_w, start_h, bin_w, bin_h)` come from the ROI tensor with QPU
multiplies and a `max(.,1)` for `output_half_pixel`, and every sample position
is `start + bin * (p + (s + 0.5)/sr)` with a constant inner factor, so one
broadcast plus one multiply and one add place all `oh*ow*sr*sr` samples.  The
bilinear taps are the same four gathers, the weights are QPU products, and the
bin reduction is one GEMM (`avg`) or the exact max chain (`max`).  The
reference's out-of-range rule (`y < -1 or y > H`) is a QPU comparison whose
0/1 result multiplies the weights.

`MaxRoiPool` is torchvision's integer geometry: `round(x*scale)` with ties
away from zero (`sign(x)*floor(|x|+0.5)` on the QPU), then
`floor(i*bin + start)` / `ceil((i+1)*bin + start)` on the QPU; the ARM only
turns the integral edges into the window map (the `Col2Im` class of work).  A
window is one `copy2d` per channel plane and the max is the exact comparison
chain, so a NaN operand is carried the way numpy's `max` carries it.

### 3.5 `NonMaxSuppression`: sort + bounded predicated suppression

The greedy loop is replaced by its **closed form**, which is what makes this
operator affordable without a sequential pass:

* **rank** every box by score with two exact comparison matrices and two GEMM
  row sums: `rank[i] = #{s[j] > s[i]} + #{j < i: s[j] == s[i]}` -- the
  reference's own tie order (equal scores keep the lower index first);
* the **IoU matrix** is built once per batch (boxes are shared across classes):
  `max`/`min` of the corner planes pairwise, `inter = max(0, ...)`, `union`,
  one reciprocal.  Degenerate boxes come out as IoU 0, which is what
  `suppress_by_iou` returns for them;
* the **suppression graph** `A[i][j] = (iou[i][j] > thr) and rank[j] <
  rank[i] and cand[j]` is a DAG (every edge goes to a strictly earlier box),
  so greedy suppression is exactly *"i has no outgoing path"*: every path ends
  at a box nothing earlier suppresses.  That is `rowsum((I + A)^N) > 1`,
  computed with `ceil(log2 N)` boolean matrix squarings (exact: 0/1 entries,
  multiply/add only, thresholded at the end);
* the `max_output_boxes_per_class` cap is one more GEMM row sum over the
  already-decided keep vector.

So there is **no bitonic sort on the QPU**: ranking by pairwise comparison is
O(N^2) elementwise work in a fixed number of launches, where a sort network
would be O(log^2 N) launches of the same O(N^2) work.  Host versus QPU work is
explicit:

| step | where |
|---|---|
| candidate mask, IoU, ranks, closure, keep vector | QPU |
| select the output rows from the ranks and the 0/1 keep vector | ARM (index data) |

The read-back is two vectors of `N` (ranks, keep), not a decision per box, and
the suppression decision itself never leaves the QPU.  The bound is
`N <= 512` boxes (section 8).

### 3.6 `Det`

`det = sum_pi sign(pi) prod_i a[i][pi(i)]` over the `M!` permutations.  Each
permutation's diagonal is `M` laned gathers with a constant batch stride, the
product is `M-1` multiplies on a `[batch]` vector, and the terms accumulate in
two exact chains (positive, negative) so the result is a deterministic
fixed-order sum.  `M! * 2M` launches: 36 for the generated 3x3, 192 for a 4x4.
`M > 4` is refused by name (120 permutation terms for a 5x5), and the accuracy
against `np.linalg.det`'s LU is 1.2e-7 on random 3x3s.

### 3.7 `Einsum`: a lowering, not a kernel

An equation with at most two operands and no repeated label inside one operand
is exactly a batched contraction:
`out[batch, freeA, freeB] = A[batch, freeA, K] @ B[batch, K, freeB]`.
Labels that appear in one operand and nowhere else are summed out first (one
GEMM against a ones column); permutations are flat index maps; the contraction
is `vc4_matmul` with both operands padded on the QPUs (the library pads with
the ARM otherwise, which `docs/OPS-NORM.md` measures as a hard failure over a
QPU-produced operand).  No contracted label at all is an outer product, which
is the same GEMM with `K = 1`.

Supported: `ij,jk->ik`, `bij,bjk->bik`, `ijk,ikl->jl`, `ij,jk->ki`,
`ij,ji`, `ij->i`, `ij->`, `ij,ij->`, `i,j->ij`, and the implicit form (no
`->`).  Refused **by equation**: an ellipsis, three or more operands, and a
repeated label inside one operand (a diagonal or a trace).

### 3.8 `DeformConv`

The offset sampling is a bilinear im2col: the constant base grid
`-pad + i*stride + ky*dil` plus the offset plane is QPU arithmetic, `floor`
gives the base index and the fractional weights, and the four taps are the
same gathers GridSample uses (zeros padding, so an out-of-range tap is a `-1`
map entry).  The mask multiplies the sampled value (per `(offset group, tap)`,
broadcast over the group's channels), and the `(Cin*kh*kw) x (Cout)`
contraction is one `vc4_matmul` per (batch, group).  MACs are the operator's
own `N*Cout*Ho*Wo*Cin*kh*kw` plus `4x` that in tap *reads* for the
interpolation.  `offset_group` splits the channel axis into contiguous blocks
and `group` splits the GEMM; both were measured (section 6).

## 4. What a working `vc4_gather` would buy

Launch *count*, not capability.  Each of the four taps of a sample is already
one `copy2d` per constant-step run of its index map, so a fully random grid is
one launch per 16 words and a regular one batches:

| operator | launches now (measured) | words gathered | with a table-driven gather (estimated) |
|---|---:|---:|---:|
| `GridSample` (`1x3x6x6`, `5x5` grid) | 238 | 300 | ~25 |
| `RoiAlign` (`2` ROIs, `2x2`, `sr=2`) | 297 | 384 | ~30 |
| `DeformConv` (`1x4x6x6`, 3x3) | 3217 | 5184 | ~350 |
| `MaxRoiPool` (`2` ROIs, `2x2`) | 238 | 96 | ~10 |

The third column counts the words each operator actually gathers (four taps
per output sample); the fourth is `words/16` plus the arithmetic launches --
what an element-at-a-time gather instruction would collapse the gather half of
each count to.  Those are the numbers an unverified-kernel gate is buying.
Anyone fixing `vc4_gather` should prove it with `tests/test_gate.c` plus an
element-wise oracle, and these four operators are the regression suite to
re-measure.

## 5. Facts measured on this board while building the block

### 5.1 A float uniform that looks like a device address is *denied*

`vc4d` validates every literal uniform word against the windows a device
address can name: `>= 0xC0000000` (the GPU bus), the V3D register block at
`0x3FC00000`, and the arena's physical range.  A unary parameter is passed as
its bit pattern, and **`1.5f` is `0x3FC00000`, exactly `V3D_PHYS_BASE`**.

Every GridSample case whose dimension gave `(dim-1)/2 == 1.5` failed with
`unary: device I/O error (LAUNCH: denied by daemon policy)` -- and the daemon
journal said `DENY ... op=LAUNCH: arena uniform word is outside the arena`.
That is why `_dispatch_geom.affine_t` checks a parameter's bit pattern before
using the uniform form and falls back to an uploaded constant plus a separate
add when it lands in a forbidden window.  It cost a launch, not a wrong
answer: the fallback is exact.

This also means the *shape-dependent* floats in a handler are a hazard class,
not just 1.5: `spatial_scale` or a normalisation constant can hit the same
window.  Anything that passes a model-derived float as a uniform should use
`affine_t`.

### 5.2 A `bcast_rows` / `tile_rows` mix-up is silently wrong, not loud

`bcast_rows(v, N, N)` repeats `v[i]` *across* row `i`; `tile_rows(v, N, N)`
makes every row the whole vector (so element `[i][j] = v[j]`).  The IoU matrix
wants the first for a row-indexed factor and the second for a column-indexed
one.  Using the wrong one gave a *plausible* matrix with the right shape -- the
first NMS run produced 1 selected box where the reference produced 2, and only
the pairwise counts made the difference visible.  Every pairwise composition
here now says which axis it needs in a comment.

### 5.3 A `Pool` view registered as garbage frees live memory

`Pool.release` frees whatever arena offset it is handed, so registering a
*view* (`Buf(off + k, n)`) as per-node garbage frees a block in the middle of
its parent; the next allocation then hands out memory that is still live.  The
first DeformConv run (four tap buffers per channel, each a sub-view of the
input plane) returned the first channel's values in every channel.  Views are
now never registered as garbage; the parent owns the storage.

### 5.4 The `-1` sentinel has to be applied *after* the plane offset

In the same DeformConv run, `word_gather`'s `-1` ("leave this word alone") was
added to the channel's plane offset first, turning it into a valid index into
the *previous* channel.  A masked tap then read the neighbour's pixel.  It
only showed with `Cin > 1`: the 1-channel form was exact, which is exactly how
this class of bug survives a quick test.

## 6. Fidelity against the oracle

Worst errors are in section 2.  Per family:

* the pure movement parts -- `AffineGrid`'s interleave, the tap copies of
  `GridSample`/`MaxRoiPool`, `Col2Im`'s scatter -- are **bit-exact**: they copy
  words.  The `Col2Im`/`ConvTranspose` accumulation is a chain of adds of
  zero-padded tap buffers (adding a true zero is exact), so it differs from
  the oracle only where the tap sum itself rounds.
* `GridSample`, `RoiAlign`, `ConvTranspose`, `Det`, `Einsum` and `DeformConv`
  are within 2-7e-7 absolute, i.e. a handful of ulp: the only arithmetic is
  the correctly-rounded multiply, truncating adds in a fixed order, and the
  GEMM's accumulation order.
* `NonMaxSuppression` is **exact** on the generated case (it is integer and
  comparison work all the way through).  Its one approximation is the IoU's
  reciprocal: a pair whose IoU is within ~1e-7 of `iou_threshold` could flip.
* `Det`'s worst case is 1.2e-7 against LU; a badly conditioned matrix loses
  relative accuracy in the cofactor sum the same way it does in `numpy`, and
  the tolerance is the matrix's.

## 7. Oracle disagreements (not engine bugs)

`test_onnx_geom.py` supplies the expectation for five forms because
`onnx.reference` 1.17 does not implement the operator (or implements it
wrongly); the engine follows the spec.

* **`Col2Im`**: the reference passes the 4-D `image_shape` to
  `col2im_naive_implementation`, which indexes `pads[i, :]` with `i` up to 3
  on a `(2, 2)` array (`IndexError: index 2 is out of bounds`).  Every form is
  checked against the spec's accumulation instead.
* **`MaxRoiPool`**: there is no reference class at all; the tests check the
  torchvision geometry (round-half-away-from-zero edges, floor/ceil bins).
* **`RoiAlign` with `mode="max"`**: the reference computes
  `max(w1*p1, w2*p2, w3*p3, w4*p4)` per sample -- the maximum of the four
  *weighted tap products* -- where torchvision (and the operator) maxes the
  interpolated values across the samples.  The engine follows torchvision; the
  test supplies `roialign_ref`'s expectation.
* **grouped / depthwise `ConvTranspose`**: the reference's recursive grouped
  branch writes the whole group into one channel
  (`final[image, g:g+1] = group_output[image]`) and applies the group's first
  bias to every channel.  For `depthwise` (one output channel per group) the
  shape happens to fit and only the bias is wrong, which is why the first
  focused run disagreed by exactly `B[1] - B[0]`; for `group=2` it does not
  even fit.  Both are checked against `convtranspose_ref` (the spec scatter).
* **`NonMaxSuppression` with an empty `max_output_boxes_per_class`**: the
  reference raises `TypeError: Issues with types ...` on the empty tensor,
  where ONNX says the default is 0 (no output).  The test supplies the empty
  result.

## 8. Constraints, stated exactly

| operator | constraint |
|---|---|
| `ConvTranspose` | 4-D NCHW; the weight must be a **graph constant** (its transpose is the GEMM operand); `group >= 1`, `output_padding` and `auto_pad` NOTSET/SAME_UPPER/SAME_LOWER/VALID, explicit `output_shape`, any `dilations` |
| `Col2Im` | 4-D `image_shape`; `block_shape` 2-D; `L` must equal the sliding-block count; `dilations`/`pads`/`strides` honoured |
| `AffineGrid` | 4-D `size` (`[N, C, H, W]`), `theta` `[N, 2, 3]`; `align_corners` 0/1 |
| `GridSample` | 4-D input, grid `[N, Ho, Wo, 2]`; `mode` `linear` (a.k.a. `bilinear`) or `nearest`; `padding_mode` `zeros`/`border`/`reflection`; `align_corners` 0/1.  **`cubic` is refused**: a 4x4 Keys polynomial per axis and 16 taps, not composed |
| `RoiAlign` | `sampling_ratio > 0`.  **`sampling_ratio = 0` (adaptive) is refused**: the per-bin sample count `ceil(roi_height/oh)` is a per-ROI data-dependent count, so the gather map is ragged; the reference is also its only source for it.  `mode` avg/max, `spatial_scale`, `coordinate_transformation_mode` half_pixel/output_half_pixel all supported |
| `MaxRoiPool` | `rois` `[R, 5]` (batch index first), `pooled_shape` 2-D, `spatial_scale`; in-range batch indices |
| `NonMaxSuppression` | `num_boxes <= 512` (the formulation keeps ~10 `N^2` matrices live); `center_point_box` 0/1; `max_output_boxes_per_class` absent/empty means 0 (no output); `iou_threshold`/`score_threshold` absent means 0 |
| `Det` | `M x M` with `M <= 4`; **`M > 4` is refused** (`5! = 120` permutation terms, `1200` launches) |
| `Einsum` | at most two operands, no ellipsis, no repeated label inside one operand (a diagonal); every other equation in that class is lowered |
| `DeformConv` | 4-D NCHW; the weight must be a **graph constant**; `group`, `offset_group`, `strides`, `pads`, `dilations`; mask optional |

## 9. Reproducing

```sh
cd infer && make -j4
export TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1
export VC4INFER_CACHE=$HOME/scratch/kcache-geom       # any writable dir
export PYTHONPATH=$PWD/python

# the generated matrix (section 1)
python3 tools/op_matrix.py --stage all --only \
  ConvTranspose,Col2Im,AffineGrid,GridSample,RoiAlign,MaxRoiPool,\
NonMaxSuppression,Det,Einsum,DeformConv

# the per-form test: 82 graphs
python3 tests/test_onnx_geom.py
python3 tests/test_onnx_geom.py --only GridSample,NonMaxSuppression -v
```

The kernel cache must be clean when a body changes: the blobs are named by
kernel and options, not by a source hash.

`REPS=1 VC4INFER_DAEMON=1 ./run_tests.sh` is green at this revision (27
sections, `ALL TESTS PASSED`; the geometry section is `test_onnx_geom`, 82
cases, 0 failed).  The board is shared: earlier attempts were interrupted by
a concurrent client restarting `vc4d` under the run (random `UPLOAD/LAUNCH:
I/O error` in whichever section was in flight) or by a kernel-blob race in
the shared cache; the green run was taken in a quiet window and used a
private `VC4INFER_CACHE`.  Nothing in the geometry tests depends on the
daemon staying up beyond the run itself.
