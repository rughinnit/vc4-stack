# torch-vc4 — what was measured

One boot of the Raspberry Pi 3B, stock clocks (V3D 300 MHz, ARM 1200 MHz,
`get_throttled=0x0` after the runs), user `raspi`, GPU work through `vc4d`
(the daemon holds `/tmp/vc4-gpu.lock` per launch; the client never does).

Date of the run: 2026-11-19, Debian 13 (trixie) arm64, Python 3.13.5,
glibc 2.41, torch 2.15.0+cpu.

> **Superseded for the op table and for all timings by `COMPLETENESS.md`
> (0.3.0)**, which records the quiet-board re-measurement, the measurement
> protocol (threads pinned, `vc4d` active, no other GPU client, median of N)
> and the ops added since this 2026-11-19 run.  Keep this file as the dated
> log of that boot.

## Step 0 — is there a wheel?

| item | result |
|---|---|
| PyPI aarch64 wheel for cp313 | **yes**: `torch-2.15.0-cp313-cp313-manylinux_2_28_aarch64.whl`, 483 665 057 B, sha256 matches PyPI |
| does it import? | **no.** It is the CUDA build: `OSError: libcudart.so.13: cannot open shared object file`, then `ValueError: libcublasLt.so.*[0-9] not found`. Its dependency closure is `nvidia-cudnn-cu13` (595 MB) plus cuBLAS/cuSPARSELt/NCCL/cuDNN etc. — ~3 GB of downloads at the Pi's ~0.45 MB/s |
| CPU-only wheel | `torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl`, 170 747 118 B, sha256 `ab35875b…f0a5e78` (verified), deps are pure Python only |
| installed | yes: `torch.__version__ == "2.15.0+cpu"` |
| download path | PyPI direct to the Pi was ~0.45 MB/s (first attempt died at 30 MB in `/tmp` — `/tmp` is a 453 MB tmpfs). The wheel was fetched at 94 MB/s on the VPS and copied; the sha256 was re-verified on the Pi |
| venv size | **816 MB** (`.venv`), of which `site-packages/torch` is 688 MB. Wheel kept at `.tmp/` (163 MiB), `pip --no-cache-dir` used |
| ABI | manylinux_2_28 on glibc 2.41 — fine. Nothing built from source except this extension |

Import cost: `import torch` 11.9 s with a warm page cache, 42 s cold from the
SD card; `import torch_vc4` +0.15–0.55 s; peak RSS 135–220 MB.

## Step 1 — the extension

- `PrivateUse1` renamed to `vc4`; allocator, device guard, `PrivateUse1Hooks`
  interface (torch refuses to create a privateuse1 tensor without the hooks),
  nine aten kernels, catch-all `PrivateUse1` fallback that throws.
- All tensors come from one 24 MiB `vc4_alloc()` region inside the engine's
  64 MiB client arena; 64-byte alignment + 64-byte slack per block, free list
  with coalescing. The engine's scratch is the rest of the arena.
- The native part is loaded with `ctypes`, not pybind11 (see "build" below).

### Build on a 1 GB board (and the two reboots)

| attempt | result |
|---|---|
| `torch/extension.h` + pybind11, `-O2`, no swap beyond the 926 MB zram | the **board reset** (hard reboot) twice, once in `cc1plus`; no OOM line in the previous boot's journal, last messages were normal |
| same, after adding a 2 GB swapfile | compiled for 8+ min, 849 MB RAM + 1.35 GB swap in use, then the board reset again |
| final: ATen + `torch/library.h` only (no C++ frontend, no pybind11), `-O0 -g0 -std=c++20`, one TU, `MAX_JOBS=1`, `ulimit -v 1200000` | **builds**, ~4 min, no reset |

So: `torch 2.15 requires -std=c++20`, and the compile needs roughly 1 GB of
address space. A 2 GB swapfile was added and persisted (`/swapfile` in
`/etc/fstab`; total swap 2.9 GB with the existing zram). It is only needed for
rebuilds, not for running.

## Step 2 — acceptance run

`tests/test_backend.py`, all checks pass:

```
PASS  backend renamed to vc4  vc4
PASS  torch_vc4.device()  vc4:0
      torch.device('vc4') string: rejected by torch's parser
      (Invalid device string: 'vc4'); torch_vc4.device() is the supported spelling
      arena: {'region_off': 1048576, 'region_bytes': 25165824,
              'arena_bytes': 67108864, 'arena_used': 26214400}
PASS  engine arena up (vc4d)
PASS  randn on vc4 (host RNG + upload)  torch.Size([256, 256])
PASS  vc4->cpu copy is bit exact
PASS  x @ x returns a vc4 tensor
      256^3 wall 5.384 ms, engine gpu 3.974 ms (9.46 GFLOP/s kernel), 2 launches
PASS  matmul matches CPU (allclose rtol 1e-4, atol 1e-3)
      max rel 1.78e-04, max abs 5.04e-04 (fp64 ref: 4.54e-04)
PASS  matmul max relative error < 1e-3 (engine-style denominator)  1.78e-04
      256^3 best wall 4.201 ms -> 7.99 GFLOP/s wall; engine GPU 3.980 ms
PASS  add on vc4 == CPU
PASS  mul on vc4 == CPU
PASS  relu on vc4 == CPU
PASS  odd-shaped mm 20x36x44 == CPU  max rel 7.05e-06, max abs 1.91e-05
PASS  torch.sin(vc4) raises    ('aten::sin' is not implemented for device vc4)
PASS  torch.mean(vc4) raises   ('aten::mean' is not implemented for device vc4)
PASS  broadcasting add raises  (add requires identical shapes, [256,256] and [256,1])
PASS  slicing a vc4 tensor raises (as_strided missing)  (as_strided not implemented)
PASS  float64 on vc4 raises    (only float32 tensors are supported, got Double)
PASS  mm cpu x vc4 raises      (mm(A): expected a vc4 tensor, got cpu)
PASS  client process has no /dev/mem fd
PASS  client holds a vc4d socket
      CapEff: 0000000000000000  uid=1000
```

### Numerics, stated precisely

256³, `x = x @ x` with `x ~ N(0,1)`:

- vs CPU **fp64**: max |err| 4.54e-04 on entries with |y| up to ~60 (typical
  |y| ≈ 16). This is the expected fp32 accumulation error for 256 terms
  (`n·eps·Σ|terms|` ≈ 5e-4).
- vs CPU **fp32**: max |err| 5.04e-04, `torch.allclose(rtol=1e-4, atol=1e-3)`
  passes.
- engine-style relative metric `|err| / max(|ref|, 1)`: 1.78e-04. The 1.0
  denominator floor makes near-zero output entries dominate, so it is slightly
  above the engine's own `TOL 1e-4`; the absolute error is what the fp32 maths
  predicts. Nothing is rounded in the report: both numbers are above.
- 20×36×44 (non-multiple-of-64 N, exercises the engine's padded scratch):
  max rel 7.05e-06, max abs 1.91e-05.
- `add`, `mul`, `relu` match the CPU to 1e-6 (`allclose`).

### Timing (best of 5, same boot, stock clocks)

| shape | wall ms | wall GFLOP/s | engine GPU ms | GPU GFLOP/s |
|---|---:|---:|---:|---:|
| 64³ | 0.390 | 1.34 | 0.211 | 2.48 |
| 128³ | 0.962 | 4.36 | 0.772 | 5.44 |
| 256³ | 4.182 | 8.02 | 3.960 | 8.47 |

A second run of the same binary on the same boot, ~25 minutes later with the
board at load average 2.4 (other users; core still 400 MHz, V3D 300 MHz,
`get_throttled=0x0`), gave 64³ 0.424 ms, 128³ 1.254 ms, 256³ 4.990 ms / 7.61
GFLOP/s GPU — i.e. the spread between runs on this shared board is ~15–25 %,
which is why both are quoted. See `docs/CLOCK-VARIANCE.md` for why wall time
moves with unrelated load while the kernel cycle counts do not.

"engine GPU" is `vc4_get_stats()`: pack + kernel time as reported by the
engine; "wall" is `time.perf_counter()` around the Python call (two launch
round trips plus the copy of the inputs, no host copies for the result).

Elementwise is launch-bound: 256 floats per QPU group, 8 groups per launch, so
65 536 floats = 32 `LAUNCH_WAIT` round trips ≈ 3.2 ms (relu 3.22 ms, add
3.20 ms, mul 4.53 ms). The 64×128 elementwise checks in the acceptance test
are unaffected by this.

### Unprivileged operation

`/proc/self/fd` during the run: 1 socket to `vc4d`, 2 pipes, nothing else;
`CapEff=0000000000000000`, uid 1000.

`strace -f -e trace=openat,connect,socket,mmap`:

```
/dev/mem opens:                 0
unix socket connects:           1  sun_path="/run/vc4d.sock"
mmap(NULL, 67108864, ...):      1  (the dma-buf arena)
```

### GPU lock

`flock` is taken by `vc4d` per launch. Evidence: with another process holding
`/tmp/vc4-gpu.lock` for 6 s, the engine's `bench_matmul 64` took **5.70 s**
wall for 0.25 ms of GPU time (`vc4d/src/vc4d.c:133`, `VC4D_GPU_LOCK=1`).
The torch client process itself never calls `flock`; holding it there would
make the daemon answer `E_BUSY` and the engine retry for ~60 s.

## What is not there (and is reported as such)

- `torch.device("vc4")`: torch's device-string parser is populated during
  `import torch`, before the rename (it still accepts the built-in spelling
  `"privateuseone"` and rejects `"vc4"` afterwards; consistent with a static
  name table, though the C++ source is not shipped in the wheel).
  `torch_vc4.device()` is the supported entry point.
- No autograd, no `torch.compile`, no conv/pool/softmax aten ops, no
  broadcasting, no `as_strided` (hence no slicing/reshape/view), no device-side
  dtype conversion, no multi-device, no streams.
- Printing a vc4 tensor fails (the formatter needs unimplemented ops).
- `torch.randn` is host-generated.

---

# Part 2 — subset backend with autograd (2026-11-19/20)

Same board, same stock clocks (V3D 300 MHz, ARM 1200 MHz), engine tree
`~/work/vc4-stack/engine` (the checkout that also carries the ONNX/training
kernels), one GPU client at a time through `vc4d`.

What changed: the 9-op prototype became a subset backend a small CNN can
train on.  Forward: `conv2d` (3x3/1x1, stride 1/2, padding, groups=1 and
depthwise), `max_pool2d` (2x2 engine kernel + a generic NxN/stride/pad
kernel), `adaptive_avg_pool2d`, `softmax` (last dim, ≤16), `flatten`/`view`
(metadata aliases), `cat` (channel), `upsample_nearest2d`, `silu`, `sigmoid`,
`linear`, `relu`, `sum`/`mean`, scalar arithmetic, `contiguous`/`clone`,
inference-mode `batch_norm` (+ an offline `fold_bn_into_conv`).  Autograd:
`mm`/`matmul`, `linear`, `add`, `sub`, `mul`, scalar ops, `pow(2)`, `relu`,
`silu`, `sigmoid`, `softmax`, `conv2d` (stride 1 and 2, depthwise),
`max_pool2d`, `adaptive_avg_pool2d`, `sum`/`mean`, `view`.  Strictness is
unchanged: anything else raises, including shapes/attributes the engine
cannot do.

`tests/run_all.sh` runs the three suites (each with a fresh `vc4d`):

```
=== ops: PASS          (50 forward parity checks, 12 rejections, 1 allocator)
=== autograd: PASS     (46 gradient checks, 3 module checks, CNN training)
=== strict: PASS       (original acceptance test)
```

## Forward parity vs the CPU

`tests/test_ops.py`, fp32 CPU reference (same maths, different summation
order).  "exact" is `torch.equal`.  `maxrel` uses the engine's
`|err| / max(|ref|, 1)` denominator.

| op | shape | max abs | max rel |
|---|---|---:|---:|
| conv 3→5 k3 s1 p1 | 1x5x8x8 | 1.05e-05 | 7.39e-06 |
| conv 16→16 k3 s1 p1 | 1x16x12x12 | 2.02e-04 | 5.28e-05 |
| conv 5→7 k3 **s2** p1 | 1x7x5x5 | 1.53e-05 | 5.19e-06 |
| conv 4→6 k3 s2 p0 | 1x6x4x4 | 1.53e-05 | 5.48e-06 |
| conv 3→8 **k1** s1 | 1x8x8x8 | 9.54e-07 | 4.77e-07 |
| conv 8→8 k1 **s2** | 1x8x5x5 | 2.38e-06 | 9.54e-07 |
| conv 6→6 k3 s1 p1 **depthwise** | 1x6x8x8 | 3.82e-06 | 1.22e-06 |
| conv 6→6 k3 **s2** p1 depthwise | 1x6x5x5 | 1.67e-06 | 7.61e-07 |
| max_pool2d 2x2/2, 3x3/2p1, 5x5/1p2, 9x11 input | — | **exact** | 0 |
| adaptive_avg_pool2d 1x1, (3,5), (7,7) | — | ≤8.05e-07 | ≤8.05e-07 |
| softmax 4/10/16 classes, 3-D | — | ≤2.09e-07 | ≤2.09e-07 |
| sigmoid, silu | 6x5 | 5.66e-07, 3.58e-07 | same |
| relu, add, mul, neg, pow(2) | 6x5 | **exact** | 0 |
| sub, mul/add/div scalar | 6x5 | ≤2.38e-07 | ≤1.16e-07 |
| sum, mean (full) | 30 elements | 1.91e-06, 6.15e-08 | same |
| cat (channel) | 1x7x5x5 | **exact** | 0 |
| linear 2-D / 3-D | 1x9→4, 2x3x5→7 | 3.58e-07 | 3.12e-07 |
| upsample nearest 2x/3x/(7,5) | — | **exact** | 0 |
| batch_norm eval | 1x5x6x6 | 2.38e-07 | 1.16e-07 |
| fold_bn_into_conv (vs BN∘conv) | 1x3x6x6 | 1.14e-05 | 1.46e-06 |

The 16→16 conv's 2.0e-04 abs error is the fp32 accumulation error over
K=144 (`n·eps·Σ|terms|`); its relative error is 5.3e-05.  Bit-exact where the
op is a selection/copy (pooling, cat, resize, views); bit-exact for
saturating/identity elementwise ops.

## Gradient parity vs the CPU

`tests/test_autograd.py`: the same fp32 model on CPU and on vc4, same data,
`loss = sum(out * fixed_random_weights)` (a plain `.sum()` would make the
softmax gradient identically zero), compare `input.grad`.  The pass
criterion is the norm-relative error `||g - ref|| / max(||ref||, 1e-3)`; the
per-element `maxrel` (denominator `max(|ref|, 1e-3)`) is printed as well
because it is the stricter, near-zero-dominated view.

| gradient | max abs | maxrel | relFro |
|---|---:|---:|---:|
| mm 12x20x9 dA / dB | 2.86e-06 / 4.77e-06 | 1.74e-04 / 1.11e-04 | 2.40e-07 / 3.57e-07 |
| matmul dA / dB | 1.91e-06 / 1.91e-06 | 1.14e-06 / 2.71e-06 | 1.63e-07 / 2.51e-07 |
| add, mul, sub, mul/add scalar, pow2, relu | 0 | 0 | 0 |
| sigmoid | 1.94e-07 | 8.82e-07 | 2.71e-07 |
| silu | 5.36e-07 | 1.13e-06 | 4.68e-07 |
| softmax (non-degenerate upstream) | 1.34e-07 | 2.62e-06 | 3.23e-07 |
| linear dx / dw / db | 9.54e-07 | 1.89e-06 | 1.49e-07 |
| max_pool2d 2x2 | 0 | 0 | 0 |
| adaptive_avg_pool2d 1x1 | 0 | 0 | 0 |
| conv 3→5 s1 dX / dW / db | 7.63e-06 / 2.48e-05 / 7.15e-06 | 1.65e-05 / 1.25e-04 / 1.66e-06 | 3.14e-07 / 1.29e-06 / 8.55e-07 |
| conv 4→6 s2 p1 dX / dW / db | 3.81e-06 / 7.63e-06 / 5.25e-06 | 1.71e-05 / 1.91e-05 / 1.25e-06 | 2.17e-07 / 4.44e-07 / 7.94e-07 |
| conv 3→4 s2 p0 dX / dW / db | 1.91e-06 | 1.37e-05 | 1.72e-07 |
| conv 1x1 s1 / s2 dX / dW | ≤3.72e-05 | ≤4.83e-05 | ≤1.60e-06 |
| conv 16→16 k3 dX / dW | 3.05e-05 / 4.96e-05 | 8.12e-05 / 6.35e-03 | 5.29e-07 / 1.31e-06 |
| depthwise 6→6 s1 / s2 dX / dW | ≤4.39e-05 | ≤4.47e-05 | ≤1.67e-06 |
| CNN w1 / b1 / w2 / b2 / fc | 2.41e-06 … 2.48e-05 | 1.95e-04 / 4.11e-05 / 9.66e-04 / 2.52e-05 / 1.20e-03 | ≤5.39e-06 |

**All 46 gradient checks are within 5.4e-06 norm-relative**, well inside the
1e-3 acceptance bar.  The per-element `maxrel` peaks at 6.35e-03 for the
16→16 conv weight gradient — an accumulation over 144 products whose
smallest entries are ~1e-6, where the `max(|ref|, 1e-3)` denominator is pure
noise amplification (absolute error 5e-05 on a gradient whose max is 0.35).

The same test then checks the module path: `nn.Sequential(nn.Conv2d,
nn.ReLU, nn.MaxPool2d, nn.Flatten, nn.Linear)` built directly on the device
(4/4 parameters get gradients), `nn.BatchNorm2d(...).eval()` on device, and a
CPU-built module moved with `.to(vc4)`.

## Tiny CNN, trained through the backend

Model: `conv(1→16, 3x3, p1) → relu → max_pool2d 2x2 → conv(16→16, 3x3, p1)
→ relu → flatten → linear(1024→4)`, input 1x1x16x16, MSE to a fixed
1x4 target, SGD by hand (`p.copy_(p - lr * p.grad)`), lr 5e-4.

```
initial loss   vc4 1.335411   cpu 1.335415
40 steps in 1.58 s (39 ms/step), 3080 launches, 77 launches/step
loss: first 1.335411  last 0.000000   |pred - target|max 0.0000
```

The whole training loop is on the device: forward (im2col+GEMM conv, relu,
argmax pool, flatten view, GEMM linear), backward (`vc4_conv2d_backward`,
`vc4_relu_backward`, `vc4_maxpool2x2_backward`, `mm_bt`/`mm_at`), and the MSE
(`ones`-GEMM sum).  The only host work is the parameter update's scalar
multiply-add, which is itself a QPU kernel plus a byte `copy_`.

## Per-op cost vs the CPU (where this backend is slower, and why)

`tests/bench_ops.py`, best of 5, stock clocks, CPU = the same torch build on
4x Cortex-A53.  Every vc4 launch is a synchronous `LAUNCH_WAIT` round trip to
`vc4d`.

| op | vc4 | CPU | ratio | launches/call |
|---|---:|---:|---:|---:|
| mm 32³ | 0.41 ms | 0.040 ms | 10.2x | 2 |
| mm 128³ | 1.05 ms | 0.342 ms | 3.1x | 2 |
| mm 256³ | 4.32 ms | 2.98 ms | **1.5x** | 2 |
| linear 1x1024x4 | 2.77 ms | 0.405 ms | 6.8x | 3 |
| conv 3→16 k3 8x8 | 1.62 ms | 0.082 ms | 19.7x | 3 |
| conv 16→16 k3 32x32 | 4.69 ms | 2.32 ms | 2.0x | 3 |
| conv 16→16 k1 32x32 | 2.08 ms | 0.230 ms | 9.0x | 3 |
| conv depthwise 16 k3 32x32 | 4.78 ms | 1.45 ms | 3.3x | 3 |
| conv 16→16 k3 **s2** 32x32 | 8.67 ms | 0.71 ms | 12.3x | 3 |
| max_pool2d 2x2 16x16 | 1.44 ms | 0.076 ms | 19.0x | 3 |
| max_pool2d 3x3 s2 p1 | 1.53 ms | 0.087 ms | 17.5x | 3 |
| adaptive_avg_pool2d 1x1 | 1.65 ms | 0.078 ms | 21.2x | 2 |
| relu 1024 | 0.24 ms | 0.019 ms | 12.2x | 1 |
| relu 65536 | 3.32 ms | 0.135 ms | 24.6x | 22 |
| sigmoid 1024 / 65536 | 0.28 / 3.43 ms | 0.045 / 0.955 ms | 6.3x / 3.6x | 1 / 22 |
| softmax 64x10 | 0.91 ms | 0.034 ms | 26.6x | 1 |
| upsample 2x 16x8x8 | 1.00 ms | 0.141 ms | 7.1x | 0 |
| cat 16+16 x 8x8 | 0.52 ms | 0.027 ms | 19.2x | 0 |
| batch_norm eval 16x16x16 | 3.87 ms | 0.076 ms | 51.3x | 10 |
| sum 1024 | 2.04 ms | 0.026 ms | 78.7x | 2 |
| sum 4096 | 5.49 ms | 0.029 ms | 187x | 2 |

Floor cost: `torch.zeros(4)` (no launch) 0.033 ms, `view` (metadata, no
launch) 0.146 ms, `relu` on 64 floats (one launch) 0.232 ms.

Where it is slower than the CPU, and why:

1. **Everything small is launch- and round-trip-bound.**  One launch costs
   ~0.15–0.25 ms wall (socket write, daemon validation, GPU lock, kick,
   `LAUNCH_WAIT`, cache maintenance); a 16x16 max pool is three launches
   because the engine compacts the padded output and copies the result, so
   1.4 ms.  The QPU itself is idle most of that time.
2. **Host byte work is not free.**  `conv2d` pads channels to a multiple of
   16, permutes the weights, compacts the output and (for stride 2) builds
   four phase images: several hundred KB of ARM copies per call.  That is
   why a 3→16 conv on 8x8 takes 1.6 ms while the CPU does it in 0.08 ms.
3. **The GEMM's 16x64 padding wastes MACs** on narrow shapes: `sum` of 1024
   floats is a 16x64x1024 GEMM (1 M MACs for a 1 Ki reduction) and costs
   2 ms; `linear 1x1024x4` pads the 4 outputs to 64 columns (16x waste).
4. **Scalar-constant kernels are two operations** (a host fill + the QPU
   kernel) and `batch_norm` is ten launches (rsqrt, two multiplies, an add,
   two broadcasts, mul, add) because the engine has no fused BN.
5. **Views are metadata-only but not free**: they go through the
   `AutogradPrivateUse1` Python `Function` (0.15 ms) because the Autograd key
   is in every tensor's dispatch keyset.
6. **A whole-tensor Python dispatch** sits under every op (a `Function.apply`
   plus the ops.py decomposition), ~50–100 µs.

Where it wins: **nothing, at these sizes.**  At 256³ the QPU GEMM is 1.5x
*slower* than torch's 4-core CPU GEMM (4.32 vs 2.98 ms = 7.8 vs 11.3
GFLOP/s), even though the engine's own kernel rate (8.0–8.5 GFLOP/s) is what
Part 1 measured.  The honest summary: on this board the backend's value is
that the work happens on the GPU at all (and that the same code trains), not
that it is faster than four Cortex-A53 cores for tensors that fit in cache.

## Bugs found (and fixed) while wiring this up

1. **Every tensor allocation leaked.**  `Vc4Allocator::allocate` built
   `c10::DataPtr(p, nullptr, &arena_deleter, …)`; c10 only calls a DataPtr's
   deleter when its *context* is non-null (`c10/util/UniqueVoidPtr.h`:
   "the deleter is guaranteed to be called when the unique pointer is
   destructed and the context is non-null"), so `arena_free` never ran.
   Invisible with a few dozen tensors, fatal for a training loop:
   `torch.zeros(4096)` in a loop died after **1531** iterations with the
   24 MiB region full.  Fixed by passing the data pointer as the context
   (the deleter recovers the offset by subtraction); `alloc_info()` and a
   3000-tensor churn check in `test_ops.py` now guard it.
2. **`Tensor::set_` is an aten op.**  Building a view with `t.set_(storage,
   …)` dispatched into the backend's own fallback (`aten::set_` not
   implemented).  Views now use the TensorImpl-level
   `set_storage_keep_dtype` / `set_storage_offset` /
   `set_sizes_and_strides`, which is also the correct alias semantics.
3. **`empty_strided` was stricter than torch.**  It required
   `stride[i] == expect` for every dim, so a contiguous `[256][1]` tensor
   (strides `(1,1)`) was rejected and `to(device)` of an averaging matrix
   failed.  torch's rule skips unit dims; so does the backend now.
4. **`view(-1, n)` had no inference**; `x.view(-1, cols)` in the softmax path
   failed.  Implemented (one `-1`, PyTorch semantics).
5. **`max_pool2d(ceil_mode=True)` silently returned floor-mode results.**
   The `AutogradPrivateUse1` wrapper dropped `ceil_mode` before the check
   could see it (the autograd key is dispatched even when no input requires
   grad).  It now raises; every op wrapper is audited for dropped
   attributes.
6. **Kernel JIT needs `$ASM3_DIR`.**  The engine's `kgen.py` imports
   `qenc.py`/`gen.py`/`vpmwords.py` from `$ASM3_DIR` and does not default it
   to the engine tree, so any shape not already in `kcache` failed with
   "kernel blob missing or generation failed".  The backend now points
   `ASM3_DIR` at the first tree that has `qenc.py`.
7. **The engine's shared object became unloadable.**  `libvc4infer.so` gained
   `-Wl,-soname,libvc4infer.so.1` but no `.so.1` is installed, so `dlopen`
   of the plain `.so` fails.  The extension now links the static
   `libvc4infer.a` + `libvc4client.a`, which also makes an already-built
   backend immune to engine rebuilds.
8. **Python `torch.library.Library` objects must stay alive.**  A `Library`
   returned from a function and then garbage-collected deregisters its
   kernels; `view` appeared to work only because the native `.so` also
   registers it.  All registrations are kept in module-level lists.
9. **Scalar operands arrive as CPU 0-dim tensors** (`0.5 * x`,
   `x - 1.0`), and `div.Tensor` can arrive as a bare Python float.  The aten
   kernels and the autograd wrappers accept both and turn them into QPU
   scalar kernels.
10. **Gradients arrive as expanded (0-stride) views**, e.g. from `sum`; the
    engine needs contiguous buffers.  `contiguous`/`clone` were implemented
    as a host strided gather (bytes only) and every backward normalises its
    incoming gradient.

## What is not there, and why (explicit)

| missing | reason |
|---|---|
| batch > 1 | the engine processes one image per launch; the backend refuses instead of hiding an ARM loop over the batch |
| `Cin`/`Cout` not multiples of 16 | engine im2col/GEMM tiling; handled by zero-padding channels on the host (byte copy), so it *works* but pays `align16(C)` MACs |
| conv stride 2 with 3x3 | not a kernel: decomposed into four parity phase images + one stride-1 conv, i.e. **4x the MACs** (the engine's `docs/ONNX.md` §6 approach).  1x1 stride 2 is an exact subsample |
| depthwise conv | not a kernel: expanded to a dense block-diagonal weight, i.e. `Cin` times the MACs; multiplier must be 1 |
| max_pool2d backward for NxN | only the 2x2/2 argmax-mask kernel exists; the generic NxN pool is forward-only and its backward raises |
| batch_norm training mode | needs a per-channel reduction kernel the engine does not have; inference mode is a chain of elementwise kernels and `fold_bn_into_conv` folds it into conv weights offline |
| softmax > 16 classes / non-last dim | engine lane width (16) and layout |
| `sum`/`mean` over a dimension | only full reductions (ones-vector GEMM); a per-dim reduction needs a kernel |
| broadcasting | only scalars and the linear bias (host-replicated) |
| div of two device tensors | no reciprocal-divide kernel; `x / s` is a GPU multiply by a host-computed `1/s` |
| strided compute | all kernels are contiguous; views are metadata aliases and compute on a non-contiguous view raises |
| float64 tensors | f32 only.  int64 *storage* exists (BatchNorm's counters, `item()`), but no compute kernel accepts it, so `sort`/`topk` still fail |
| `torch.compile`, streams, multi-device | out of scope; every launch is synchronous |

## Environment hazards hit during this work (for the next person)

- Two V3D wedges (`vcd4: TIMEOUT … launch=1; daemon marked dead`, `errstat`
  `0x1040` afterwards): both followed a client being killed mid-launch — a
  `SIGPIPE` from `… | head`, and a `systemctl restart vc4d` issued while a
  launch was in flight.  A daemon restart does **not** reset the V3D; only a
  reboot does.  Never truncate GPU test output with `head`, and do not
  restart the daemon while another process is running a kernel.
- Trying to reset the V3D by unbinding `vc4_v3d`
  (`echo 3fc00000.v3d > .../unbind`) segfaulted, killed the SSH session and
  hung the board until the watchdog rebooted it.  Do not do that.
- `vc4d` (this build) does not reclaim a departed client's arena region
  promptly: the second GPU process in a row is refused with "arena
  exhausted" (24 MiB requested, 4 MiB refused, on a fresh boot).  The
  backend retries the allocation for ~10 s and the test runner restarts the
  daemon per file.
- The board is shared: GPU work ran concurrently in `~/work/vc4-stack-sub`
  during this run, and the engine tree this backend links against was rebuilt
  (hence the static link), and its killed clients contributed to the wedges.
  Run one GPU client at a time.
