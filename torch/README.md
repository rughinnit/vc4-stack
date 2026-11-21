# torch-vc4 — a subset PyTorch backend for the Raspberry Pi VideoCore IV

> Part of the **VC4 stack**. The main repository — engine, kernels,
> measurements, documentation and issue tracker — is
> [`rughinnit/vc4-stack`](https://github.com/rughinnit/vc4-stack); see
> `docs/ECOSYSTEM.md` there for how the repositories fit together.


Status: **feature-complete for the model families it declares** (version
0.3.0). `torch_vc4.device()` is a real PyTorch device (`PrivateUse1`,
renamed `vc4`).  Batch > 1, arbitrary-dim reductions, softmax for any class
count, N-D broadcasting, training-mode batch norm, batched/stride-2/depthwise
conv, the verified unary set, gather/index_select/argmax and autograd for all
of it are measured against the CPU in `tests/test_complete.py`.
**`COMPLETENESS.md` is the authoritative op table**: every intercepted aten op
with its backend (device / composite / raise), measured accuracy and vc4-vs-CPU
time at three sizes, plus the precise reason for every raise.  Everything
outside that table **raises**; there is no CPU fallback and no pretence of one.

Tensors live in the engine's dma-buf arena (`libvc4infer`), and every GPU
launch goes through **vc4d**, so the Python process is unprivileged: one
AF_UNIX socket, one dma-buf mapping, no `/dev/mem`, `CapEff=0`.

Everything that computes runs on the QPU.  The ARM only moves bytes (channel
padding, phase images, weight permutes, concat, nearest resize, compaction)
and materialises constants (`fill_`, the ones vector for `sum`, the
adaptive-pool averaging matrix); there is no multiply-accumulate on the CPU
anywhere in this backend.

## Install (on the Pi)

```sh
cd /home/raspi/work/torch-vc4
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir --no-deps \
    /path/to/torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl
.venv/bin/pip install --no-cache-dir typing_extensions filelock sympy \
    networkx jinja2 fsspec setuptools
ulimit -v 1800000                      # the one-TU compile peaks near 1 GB
MAX_JOBS=1 .venv/bin/pip install --no-build-isolation -e .
```

The wheel must be the **`+cpu`** one (the PyPI aarch64 wheel is the CUDA
build and its dependency closure is ~3 GB).

Prerequisites:
- `vc4d.service` active and the user in group `vc4`;
- `~/work/vc4-stack/engine` built (`make`), which provides
  `build/libvc4infer.a` and `tools/kgen.py`; override the tree with
  `VC4_ENGINE_DIR=...` (and `VC4D_DIR=` for `libvc4client.a`);
- a 2 GB swapfile or `ulimit -v 1800000` if you rebuild.

The extension is linked **statically** against `libvc4infer.a` +
`libvc4client.a` (`setup.py` explains why: the engine's shared object gained
an SONAME that is not installed next to it, so `dlopen` of the plain `.so`
fails).  A side effect is that a rebuild of the engine tree cannot break an
already-built `torch_vc4` at import time.

Kernel blobs are JIT-compiled by the engine's `kgen.py` on first use and
cached in `<engine>/kcache` (override with `VC4INFER_CACHE`).  The engine's
`kgen.py` imports its QPU encoder modules from `$ASM3_DIR`; if that variable
is unset the backend points it at the first tree that has `qenc.py`
(`<engine>/tools`, `<engine>/../asm3`, `$HOME/work/asm3`,
`/home/raspi/work/asm3`).  Without it, JIT of any shape not already in the
cache fails.

## Install from the signed package archive

The 0.3.0 wheel is published next to `vc4infer`:

```sh
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ \
    torch_vc4
# or the wheel directly
pip install https://rughinnit.github.io/vc4-apt/pip/torch_vc4-0.3.0-cp313-cp313-linux_aarch64.whl
```

The wheel is a **cp313 / aarch64 binary** because the backend is a compiled
extension statically linked against `libvc4infer.a`; install it on the Pi
(with the `+cpu` torch wheel), not on a host.

## Run

```sh
cd /home/raspi/work/torch-vc4
.venv/bin/python tests/test_ops.py        # forward parity + strict rejection
.venv/bin/python tests/test_complete.py   # batches, reductions, softmax, ...
.venv/bin/python tests/test_autograd.py   # gradient parity + tiny CNN training
.venv/bin/python tests/test_backend.py    # the original acceptance test
.venv/bin/python tests/bench_ops.py       # per-op cost vs the CPU
sh tests/run_all.sh                       # all four suites, one daemon each
# run the same suite against an installed wheel:
PYTHON=/path/to/cleanvenv/bin/python sh tests/run_all.sh
```

`run_all.sh` restarts `vc4d` before each test file on purpose; see
"Operational notes".

```python
import torch, torch.nn.functional as F, torch_vc4

d = torch_vc4.device()
x = torch.randn(1, 3, 8, 8, device=d)
w = torch.randn(8, 3, 3, 3, device=d).requires_grad_()
y = F.conv2d(x, w, padding=1).relu()
y.sum().backward()
print(w.grad.shape)                  # torch.Size([8, 3, 3, 3]), on vc4
```

### Device string

`torch.utils.rename_privateuse1_backend("vc4")` is applied at import, and
tensors report `device(type='vc4')`.  torch's device-string parser is
populated during `import torch`, before the rename, so `torch.device("vc4")`
raises `RuntimeError: Invalid device string: 'vc4'` (measured).  Use
`torch_vc4.device()`, or `torch.device("privateuseone")`.

## Op table (the whole surface)

Forward, all registered at `PrivateUse1` (Python in `torch_vc4/ops.py`,
engine bindings in `csrc/torch_vc4.cpp`):

| aten op | how implemented | constraints |
|---|---|---|
| `empty.memory_format`, `empty_strided` | arena sub-allocator | f32 (plus int64 storage for bookkeeping buffers, e.g. batch-norm counters); contiguous strides, unit dims may have any stride |
| `copy_` (`.to`, `.cpu`, `.clone`) | `vc4_upload` / `vc4_download` | f32, contiguous |
| `fill_`, `zero_` | host constant + upload | f32 |
| `view`, `flatten`, `reshape`, `as_strided` | **metadata alias** (`set_storage`), no copy | compute ops require contiguity; a non-contiguous view is rejected, not silently copied |
| `contiguous`, `clone` | host strided gather | needed for autograd gradients that arrive as `expand`ed views |
| `mm`, `matmul` | `vc4_matmul` → QPU pack + asm10 GEMM | f32, 2-D, contiguous |
| `linear` | `mm_bt` (x @ W^T) + broadcast bias | 2-D or batched-2-D, f32 |
| `convolution_overrideable`, `convolution_backward_overrideable` | NCHW im2col+GEMM, or native NHWC for batch > 1 | f32; any batch; kernel 3x3 or 1x1; stride 1 or 2; symmetric `pad ≤ k/2`; `groups=1` or depthwise (multiplier 1); dilation 1; NCHW `(H+2p)·align16(W+2p)·4 ≤ 8191` |
| `max_pool2d` | `vc4_maxpool2x2` (2x2/2) or `vc4_maxpooln` (any NxN/stride/pad) | square kernel/stride/pad, dilation 1, `ceil_mode=False`; 2x2 engine kernel needs `H%8==0, W%16==0` |
| `_adaptive_avg_pool2d` | hosting-built averaging matrix + one QPU GEMM | any output size, batch 1 |
| `_softmax` | `vc4_softmax_fwd` ≤ 16 classes, two-pass above | any dim, any class count |
| `sigmoid`, `silu`, `relu` | `vc4_elementwise` / `vc4_sigmoid` / `vc4_silu` | f32, contiguous |
| `add.Tensor`, `mul.Tensor`, `sub.Tensor` | `vc4_elementwise` ADD/MUL (+ `mul(-1)`) | N-D broadcasting (scalar, trailing, leading, middle) materialised as 0-stride copies, per-channel, scalars |
| `add/mul/div/sub.Scalar`, `neg`, `pow.Tensor_Scalar(2)` | GPU scalar kernel with a host-materialised constant | — |
| `sum`, `mean` | permute reduced axes last, then ones-vector GEMM | full and dim/dims, keepdim |
| `cat` | host byte concat | channel axis of NCHW (or dim 0 of a 3-D tensor) |
| `upsample_nearest2d` | host byte gather (PyTorch's nearest index rule) | nearest, any scale; not bilinear |
| `native_batch_norm`, `batch_norm` | per-channel scale/shift as QPU elementwise ops | eval and training mode; `fold_bn_into_conv()` folds BN into conv parameters offline |
| `_local_scalar_dense` | download 1 element | `float(t)`, `t.item()` |
| `normal_`, `uniform_` | host `std::mt19937`, uploaded | f32; the RNG is on the host (`uniform_` is what module constructors call) |
| everything else | catch-all `PrivateUse1` fallback | raises `RuntimeError: torch_vc4: 'aten::…' is not implemented for device vc4` |

Autograd (`torch_vc4/autograd.py`, registered at `AutogradPrivateUse1`):
`mm`/`matmul`, `linear`, `add`, `sub`, `mul`, scalar ops, `pow(2)`, `relu`,
`silu`, `sigmoid`, `softmax`, `conv2d` (stride 1 and 2, depthwise),
`max_pool2d` (2x2/2 only — the engine's argmax-mask kernel),
`adaptive_avg_pool2d`, `sum`, `mean` (full and dim), `view`/`as_strided`
(contiguous only), `gelu` (both forms), `tanh`, `sqrt`, `reciprocal`, `abs`,
`sign`, `clip`, `leaky_relu`, `hardsigmoid`, `hardswish`.
Backward kernels used: `vc4_matmul_bt`, `vc4_transpose` + `vc4_matmul`,
`vc4_relu_backward`, `vc4_conv2d_backward`, `vc4_maxpool2x2_backward`,
`vc4_sigmoid`/`vc4_elementwise` for the composed ones.
`torch_vc4.mse_loss(pred, target)` is an MSE that stays on the device
(`F.mse_loss` decomposes into ops this backend does not have).

`nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.Flatten`, `nn.BatchNorm2d`
(eval) work through the standard modules; use `torch_vc4.mse_loss` or
`(a - b).pow(2).sum()` for the loss and a hand-written update
(`p.copy_(p - lr * p.grad)`) instead of `torch.optim`.

Deliberately absent, therefore raising: `sin`/`cos` (bodies exist but have
no aten route here), `var`/`std`, `topk`/`sort`/`scatter`, `bmm`, `conv3d`,
`conv_transpose`, dilated convs, grouped conv beyond depthwise, `ceil_mode`
pooling, `nn.LSTM`, `torch.compile`, int64/float64 tensors, bilinear resize,
autograd for `floor`/`ceil`/`round`/`exp`/`log`/`erf`, and image
`print()`/`repr()`.  `COMPLETENESS.md` is the authoritative per-op table.

## Operational notes (measured, not folklore)

- **One GPU process at a time, and restart `vc4d` between processes.**  On
  this `vc4d` build the arena region of a client that exits is not promptly
  reclaimed: a second GPU process (torch *or* the engine's own tests) is
  refused with `out of arena memory` although the first has gone.  The
  backend retries a failing `vc4_alloc` for ~10 s, which covers the common
  case, but the test runner restarts the daemon per file
  (`sudo systemctl restart vc4d`).
- **Killing a client mid-launch wedges the V3D.**  A `SIGPIPE` from
  `… | head` is enough: the launch in flight is never reaped and every later
  launch times out (`vcd4: TIMEOUT … daemon marked dead`, `errstat` gains
  `0x40`).  Recovery is a reboot; do not pipe GPU tests through `head`, and
  do not restart `vc4d` while another process is running a kernel.
- **Never unbind `vc4_v3d`** (`echo … > /sys/bus/platform/drivers/vc4_v3d/
  unbind`): it took the whole board down and needed the watchdog reboot.
- `vc4d` takes `flock(LOCK_EX)` on `/tmp/vc4-gpu.lock` per launch, so all GPU
  work from all clients is serialised; a client must not hold that lock
  itself (the daemon would answer `E_BUSY` and the engine retries for ~60 s).
- The arena is a fixed region (default 24 MiB of the 64 MiB client arena;
  `TORCH_VC4_ARENA_MB` overrides).  Running out raises
  `torch_vc4: out of arena …` rather than silently spilling.
- Importing prints torch's one-time "Overriding a previously registered
  kernel" `UserWarning`.  It refers to the `PrivateUse1` stub/batching
  kernels that this backend deliberately replaces (`aten::add`,
  `convolution_overrideable`, `max_pool2d`, …); it is expected, not a bug.

## Files

```
csrc/torch_vc4.cpp      allocator (24 MiB arena, sub-allocator), guard,
                        aten kernels (views, fill, mm, conv, pool, softmax),
                        engine bindings and byte-level layout primitives,
                        catch-all fallback, diagnostics ops
torch_vc4/__init__.py   rename + device module + helpers (device(), stats())
torch_vc4/ops.py        aten forward implementations (subset backend)
torch_vc4/autograd.py   autograd.Function graph + mse_loss
setup.py                one-TU CppExtension, static engine link
tests/test_ops.py       forward parity vs CPU + strict rejection
tests/test_autograd.py  gradient parity vs CPU + tiny CNN training
tests/test_backend.py   original acceptance test (matmul, hygiene)
tests/bench_ops.py      per-op cost vs CPU
tests/run_all.sh        all suites, one fresh vc4d each
RESULTS.md              measurements, bugs found, limits, why
```
