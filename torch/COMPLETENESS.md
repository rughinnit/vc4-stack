# torch_vc4 0.3.0 — completeness, accuracy and measured cost

This is the authoritative op table for the backend.  Every aten op that
`torch_vc4` intercepts is listed with

* its **backend**: `device` (all arithmetic is a QPU kernel), `composite`
  (a documented composition of QPU kernels plus byte-level layout work on the
  ARM), or `raise` (unsupported, with the exact reason), and
* its **measured accuracy** against PyTorch's CPU fp32 implementation on at
  least three shapes (max abs and max rel with the engine's `|ref| >= 1`
  denominator), and
* its **measured wall time at three sizes**: vc4 end-to-end (dispatch + vc4d
  round trip) vs the same op on the CPU.

## Measurement protocol (applies to every table below)

Pi 3B, stock clocks (ARM 1200 MHz, V3D 300 MHz), 905 MB, torch 2.15.0+cpu,
**`torch.set_num_threads(4)`**, `vc4d` active, **no other GPU client**,
**median of 5** timed calls after 2 warm-ups.  The vc4 wall time is measured
around the Python call and therefore includes the synchronous `LAUNCH_WAIT`
round trip to `vc4d`; the CPU time is the same PyTorch op on the same data.
The engine splits elementwise work into ~3K-element launches (12 QPUs x 256
lanes), so a 64 Ki-element elementwise op costs ~21 launches; that is the
main reason the eager path loses on large maps too, not only on small ones.

## The honest headline

**The eager path is launch-bound and slower than the CPU at every size
measured.**  At `mm 256x256x256` the `torch_vc4` call takes **5.09 ms vs
3.632 ms on the CPU** (9.24 GFLOP/s CPU; **1.40x slower**), of which the raw
engine accounts for 3.996 ms (8.40 GFLOP/s, **1.10x slower than the CPU**).
A 64-float `relu` costs 0.244 ms vs 0.018 ms (13x), a 64-float `add` 1.23 ms
vs 0.018 ms (67x), and `mm 32x32x32` 0.42 ms vs 0.040 ms (10x).  An earlier
"1.7x slower" figure was **load-contaminated** (a concurrent job was
running), not a cold-cache effect; a "1.2x faster" reading taken in the same
window was contaminated too.  The numbers above are the quiet-board
re-measurement.

This backend is useful because the ops are real GPU kernels with a verified
numerical contract, not because it beats a 1.2 GHz Cortex-A53.

## Op table

Sizes are the three listed per row.  Accuracy is the worst over the three
shapes in `tests/completeness.py`; `tests/test_complete.py` independently
checks the same ops (222 checks, all passing); `tests/test_cpu_registration.py`
adds 46 device-free checks for registration, the broadcast and reduction
plans, and the exact-GELU derivative.

| aten op | backend | max abs / max rel | vc4 ms (3 sizes) | CPU ms (3 sizes) |
|---|---|---|---|---|
| `aten::mm` (32³ / 128³ / 256³) | device | 4.31e-04 / 2.00e-04 | 0.421 / 1.042 / 5.090 | 0.040 / 0.345 / 3.632 |
| `aten::linear` (1x256x64 / 32x256x64 / 128x512x128) | device | 1.29e-03 / 5.46e-04 | 2.734 / 1.994 / 5.612 | 0.609 / 0.889 / 3.130 |
| `conv2d` dense stride 1 | device (native NHWC for batches) | 5.11e-04 / 2.62e-04 | 1.565 / 10.066 / 9.486 | 0.077 / 2.450 / 3.209 |
| `conv2d` dense stride 2 | device (native NHWC, stride 2) | 4.69e-04 / 1.85e-04 | 1.781 / 5.680 / 6.122 | 0.085 / 1.589 / 1.899 |
| `conv2d` depthwise 3x3 | device (native NHWC depthwise) | 5.72e-06 / 1.80e-06 | 1.970 / 3.376 / 7.342 | 0.893 / 1.138 / 1.405 |
| `max_pool2d 2x2` | device | 0 / 0 (exact) | 0.898 / 3.610 / 5.121 | 0.053 / 0.302 / 0.633 |
| `adaptive_avg_pool2d 1x1` | composite (GEMM) | 2.60e-06 / 2.60e-06 | 1.352 / 2.624 / 4.226 | 0.073 / 0.098 / 0.152 |
| `upsample_nearest2d 2x` | composite | 0 / 0 (exact) | 0.489 / 5.304 / 14.616 | 0.107 / 1.360 / 4.497 |
| `aten::cat` (channel) | composite (host byte copy) | 0 / 0 (exact) | 0.711 / 5.357 / 9.376 | 0.046 / 0.240 / 1.237 |
| `softmax` <= 16 classes | device | 4.47e-07 / 4.47e-07 | 0.660 / 0.784 / 2.301 | 0.027 / 0.035 / 0.098 |
| `softmax` > 16 classes | composite (sigmoid-exp + max/sum) | 8.49e-07 / 8.49e-07 | 10.272 / 37.214 / 503.317 | 0.027 / 0.073 / 1.195 |
| `aten::relu` (64 / 4096 / 65536) | device | 0 / 0 (exact) | 0.244 / 0.408 / 3.797 | 0.018 / 0.026 / 2.357 |
| `aten::add.Tensor` same shape | device | 4.77e-07 / 1.19e-07 | 1.226 / 1.349 / 4.386 | 0.018 / 0.033 / 2.238 |
| `aten::add.Tensor` broadcast | composite (expand copy + device) | 4.77e-07 / 1.19e-07 | 1.650 / 4.625 / 49.262 | 0.022 / 0.037 / 2.187 |
| `aten::abs` | device | 0 / 0 (exact) | 0.583 / 0.912 / 4.226 | 0.025 / 0.033 / 2.051 |
| `aten::tanh` | composite (2*sigmoid(2x)-1) | 1.62e-06 / 1.62e-06 | 1.492 / 2.609 / 20.374 | 0.022 / 2.208 / 4.172 |
| `aten::sqrt` | composite (x*rsqrt(x)) | 1.19e-07 / 1.19e-07 | 0.747 / 1.340 / 8.854 | 0.018 / 2.160 / 2.382 |
| `aten::sum(dim)` | composite (GEMM) | 2.86e-04 / 7.50e-05 | 1.307 / 1.097 / 2.454 | 0.034 / 0.046 / 0.072 |
| `aten::amax(dim)` | composite (relu max tree) | 7.15e-07 / 2.53e-07 | 5.363 / 10.687 / 30.274 | 0.032 / 0.037 / 0.046 |
| `aten::argmax(dim)` | composite (host index scan) | 0 / 0 (exact) | 0.211 / 0.657 / 1.685 | 0.032 / 0.052 / 0.109 |
| `aten::index_select` | composite (host word gather) | 0 / 0 (exact) | 1.562 / 7.013 / 49.379 | 0.048 / 0.189 / 3.273 |
| `batch_norm` eval | composite (device elementwise) | 4.77e-07 / 1.18e-07 | 16.354 / 270.786 / 535.504 | 2.101 / 2.473 / 2.367 |
| `batch_norm` train | composite (device reductions) | 8.39e-05 / 2.36e-05 | 25.285 / 368.205 / 700.309 | 6.259 / 6.700 / 0.775 |

The `batch_norm` rows are the worst case of the launch model: each
per-channel reduction plus each full-map elementwise op is many ~3K-element
launches, so a 32 Ki-element BN costs ~126 launches (~12 ms of GPU kernel
time, the rest is round-trip latency).  A fused device BN kernel would fix
this; it is on the engine's side, not hidden here.

## Newly routed unary bodies (measured 2026-11-21)

These six aten entries used to raise.  The engine's bodies were reworked and
element-verified on 2026-11-20 (`vc4-stack` `docs/ONNX.md` §12.3: 0
mismatches over 105,264 elements / 120 runs); this backend now routes them
and the numbers below are the aten calls re-measured through `vc4d` on the
same protocol as the main table.

| aten op | backend | max abs / max rel | vc4 ms (64/4096/65536) | CPU ms (same sizes) |
|---|---|---|---|---|
| `aten::floor` | device | 0 / 0 (exact) | 0.258 / 0.421 / 3.399 | 0.0170 / 0.0292 / 0.1528 |
| `aten::ceil` | device | 0 / 0 (exact) | 0.263 / 0.416 / 3.373 | 0.0162 / 0.0299 / 0.1537 |
| `aten::round` | device | 0 / 0 (exact) | 0.259 / 1.046 / 12.658 | 0.0168 / 0.0298 / 0.1505 |
| `aten::exp` | device | 1.15e-02 / 3.96e-06 | 0.254 / 1.025 / 12.690 | 0.0179 / 0.0729 / 0.4689 |
| `aten::log` | device | 1.91e-06 / 2.08e-07 | 0.253 / 1.022 / 12.854 | 0.0178 / 0.0806 / 0.5271 |
| `aten::erf` | device | 1.43e-06 / 1.43e-06 | 0.254 / 1.029 / 12.732 | 0.0167 / 0.0635 / 0.4024 |
| `gelu(approximate='none')` | composite (erf) | 6.56e-07 / 6.56e-07 | 1.323 / 2.989 / 30.338 | 0.1104 / 0.1292 / 0.5938 |

`floor`/`ceil`/`round` are exact over the finite range tested (±3e7 and
±1e7, so both sides of the |x| >= 2^23 pass-through), and pass `±inf` and
`NaN` through unchanged (measured: `floor(inf) = inf`, `floor(nan) = nan`).
`round(2.5) = 2` (ties to even).

Two domain limits are measured and stated rather than hidden:

* `exp` flushes a denormal result to the smallest normal: `exp(-100)` returns
  2^-126 = 1.175e-38 where the CPU returns 3.78e-44.  Finite overflow is
  correct (`exp(100) = inf` on both).
* `log` is measured for `x > 0`.  The body clamps the argument at 2^-126, so
  `log(0)` and `log(-1)` return about -87.34 instead of `-inf`/`NaN`.

The exact GELU gradient (composed from the erf and exp bodies) was checked
against CPU autograd at 4096 elements: forward 5.96e-07 abs, gradient
1.43e-06 abs / 1.43e-06 rel.

The broadcast and dim-reduction paths were re-verified on the same pass.
Broadcast `add` matched the CPU to 4.77e-07 abs for scalar, trailing, leading
and middle broadcast (`(64,)+()`, `(8,8)+(8,)`, `(2,1)+(1,3)`,
`(2,1,4)+(3,1)`).  `sum`/`mean` over one dim and over two dims, 2-D and 3-D,
with and without keepdim, matched to 2.04e-04 abs (sum, 3-D) and 2.6e-06 abs
(mean); a 512-element reduced axis reached 8.3e-04 abs for `sum`, the same
truncating-accumulation floor the main table records.

## The rest of the intercepted surface (verified; same backend pattern)

These share the rows above (one launch for a native unary body, a handful for
a composite, or a host word copy for indexing); they are listed separately so
the table stays readable.

| aten op | backend | measured accuracy (worst of 3 shapes) |
|---|---|---|
| `aten::neg`, `aten::sign`, `aten::abs` | device (bit ops) | exact (sign: +0 for -0, as ONNX) |
| `aten::clip` (any finite bounds) | composite (relu + constants) | 1.2e-07 abs / 1.1e-07 rel |
| `aten::leaky_relu` | composite (relu + constants) | 0 / 0 (exact in fp32) |
| `aten::hardsigmoid` | device | 5.96e-08 / 5.96e-08 |
| `aten::hardswish` | device | 2.38e-07 / 1.80e-07 |
| `aten::reciprocal` | composite (rsqrt) | 1.91e-06 / 1.70e-07 |
| `aten::gelu` (both forms) | composite (sigmoid for 'tanh', erf for 'none') | 4.77e-07 / 3.13e-07 ('tanh'); 6.56e-07 / 6.56e-07 ('none') |
| `aten::sum` / `mean` full and dim, keepdim | composite (GEMM) | 1.6e-05 abs / 1.9e-06 rel |
| `aten::amin`, `aten::min`, `aten::max` | composite (relu tree) | 7.15e-07 / 3.05e-07 |
| `aten::argmin`, `aten::argmax` (full/dim) | composite (host scan) | exact |
| `aten::max.dim` / `min.dim` | composite (values + host indices) | values 2e-5, indices exact |
| `aten::gather`, `aten::index_select` | composite (host word gather) | exact |
| `aten::sigmoid`, `aten::silu`, `aten::relu` | device | 2e-6 / 2e-6 (sigmoid/silu), exact (relu) |
| `aten::convolution` any batch, stride 1/2, depthwise | device | 5.6e-04 / 2.6e-04 |
| `aten::max_pool2d` generic NxN/stride/pad, any batch | device | exact |
| `aten::_adaptive_avg_pool2d`, `upsample_nearest2d`, any batch | composite | 2.6e-06 / 2.6e-06; resize exact |
| `aten::native_batch_norm` / `batch_norm` train+eval | composite | 8.4e-05 / 2.4e-05 |
| `aten::view` / `flatten` / `reshape` / `as_strided` | metadata only | exact |
| autograd | see README | gradients <= 2e-3 relative vs CPU (measured) |

## Unsupported ops (raise, never a CPU fallback)

| op | reason |
|---|---|
| `aten::scatter*` | No device scatter kernel.  The table-driven `vc4_gather` only reads, and the engine's reduce/gather slice is not element-verified; a host scatter would be a CPU fallback. |
| `aten::sort`, `aten::topk`, `aten::kthvalue`, `aten::median`, `aten::unique*` | No QPU sort/selection kernel; a host sort would be a full CPU fallback. |
| `amax`/`amin` **backward**, `gather`/`index_select` **backward** | Their gradients need an additive scatter (equality mask for ties); no such kernel exists.  Forward is supported. |
| float64 (`.double()`, `dtype=torch.float64`) | Unsupported **by design**: the VC4 QPU and every engine kernel are float32-only.  Emulating fp64 would double storage and add conversion launches while still computing in fp32 — rejected instead of silently downgraded. |
| conv `dilation != 1` | The engine's NHWC path has an explicit-dilation entry point, but it is not wired into this backend; the NCHW/NHWC paths used here are dilation 1. |
| transposed conv, `output_padding`, `ceil_mode` pooling, non-square stride/padding, kernels other than 1x1/3x3, `groups` other than 1 or depthwise | Not implemented by the kernels this backend wires; the message names the constraint. |
| `torch.device("vc4")` string | torch 2.15's device-string parser was initialised before the PrivateUse1 rename; use `torch_vc4.device()` (documented in the README). |
| host RNG (`normal_`, `uniform_`) | The GPU has no RNG kernel; parameters/data are generated on the host and uploaded.  Stated, not hidden: data generation, never compute. |

## Engine findings that shaped the implementation (measured)

* **`vc4_reduce` hangs the GPU in this engine build.**  A single call on a
  1x37 tensor returned `VC4_ERR_TIMEOUT` (-4) and killed the context.  Reductions
  therefore use a GEMM (sum/mean) or a relu max tree (amax/amin), all on the QPU.
* **Native NHWC conv: re-verified and enabled.**  The earlier claim in this
  file that the engine's NHWC path hangs was **wrong and is withdrawn**:
  `infer/build/test_nhwc 5` passes all 20 cases on this board (stride 1/2,
  batch 2, 5x5, dilated d2/d3, grouped g2/g3/g4, depthwise, pool, up2,
  concat; 5/5 clean each).  The exact binding call that once returned
  `VC4_ERR_TIMEOUT` — `N=2, C=4->16, H=W=8, Cout=5->16, k=3, s=1, p=1`,
  NHWC input `[2][8][8][16]`, weight `[16][16][3][3]`, bias `[16]`, panel
  `[16][16][3][3]`, `in_pitch=8` — now succeeds and matches the CPU to
  1.9e-05.  The timeout happened while the engine archive and its
  name-keyed kernel cache were being rebuilt concurrently (the
  stale-blob trap the engine documents), not in a steady build.  The backend
  now uses `vc4_conv_nhwc` for `batch > 1` and `stride 2`, and
  `vc4_depthwise_nhwc` for 3x3/stride-1/pad-1 depthwise (that kernel has no
  bias argument, so the bias is added on the QPU after the conv); other
  depthwise cases use the exact block-diagonal decomposition.
* **The transcendental unary bodies were wrong in the engine build this
  backend first measured; that is fixed.**  `vc4-stack` reworked the bodies
  and element-verified the whole family on 2026-11-20 (`docs/ONNX.md` §12.3),
  and this backend now routes `exp`, `log`, `erf`, `floor`, `ceil` and
  `round` natively (numbers above).  `tanh`/`sqrt`/`reciprocal` keep the
  earlier composites built from `vc4_sigmoid`/`vc4_rsqrt`, which are verified
  (batch norm has used them since 0.1.0); `gelu(approximate='none')` composes
  the erf body.
* **vc4d's launch policy rejects uniform words that alias the GPU bus
  window.**  A float with |v| >= 2 and a negative sign (including -inf) has
  that bit pattern, so `clip` is composed from `relu` + host-materialised
  constants instead of passing the bounds as unary uniforms.

## Install

```sh
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ torch_vc4
```

The wheel is a cp313/aarch64 binary (the backend is a compiled extension
statically linked against `libvc4infer.a`), published next to `vc4infer` in
the signed `vc4-apt` repository:

* index: <https://rughinnit.github.io/vc4-apt/pip/simple/>
* wheel: `pip/torch_vc4-0.3.0-cp313-cp313-linux_aarch64.whl`
* sha256: `9fdc000e215f6192d5de7164d7caee84433fa52876515a92a33b7b9f8c9e3759`

## Reproduce

```sh
.venv/bin/python tests/test_cpu_registration.py  # device-free: 46 checks
.venv/bin/python tests/test_complete.py   # this table's accuracy checks
.venv/bin/python tests/completeness.py    # this table's timings
sh tests/run_all.sh                       # all five suites
# against an installed wheel (a clean venv):
PYTHON=/path/to/cleanvenv/bin/python sh tests/run_all.sh
```

The unary rows above were measured with the worktree engine and kgen:

```sh
cd ~/work/vc4-kops && make -C infer -j2
cd ~/work/torch-vc4-gaps
export PYTHONPATH=$PWD TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1
export VC4INFER_KGEN=$PWD/../vc4-kops/infer/tools/kgen.py
export ASM3_DIR=$PWD/../vc4-kops/infer/tools
tools=../vc4-kops/tools/thermal-guard.sh
$tools --interval 10 --max 71 -- ../torch-vc4/.venv/bin/python \
    tests/test_cpu_registration.py
$tools --interval 10 --max 71 -- ../torch-vc4/.venv/bin/python \
    tests/test_ops.py
$tools --interval 10 --max 71 -- ../torch-vc4/.venv/bin/python \
    tests/test_autograd.py
```
