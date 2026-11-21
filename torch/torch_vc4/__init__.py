"""torch_vc4 - a minimal, honest PyTorch backend for the Raspberry Pi VC4 GPU.

Usage:

    import torch, torch_vc4
    x = torch.randn(256, 256, device=torch_vc4.device())
    y = x @ x

A usable subset exists on the device: gemm/linear, conv2d (batched, stride
1/2, depthwise), pools, softmax for any class count, nearest resize,
elementwise/unary ops, broadcasting, arbitrary-dim reductions, indexing,
batch norm (eval and training), views, and autograd for the training path.
Anything else raises instead of falling back to the CPU; COMPLETENESS.md is
the op table with measured accuracy and timings, and the exact reason for
every raise.

Note on the device string: this backend renames PrivateUse1 to "vc4", and
tensors report `device('vc4')`, but torch 2.15's device-string *parser* was
initialised before the rename and still only accepts the built-in spelling
"privateuseone" (measured; see RESULTS.md).  Use `torch_vc4.device()`, or
`torch.device("privateuseone")`, in the few places a string/device object is
needed.
"""

import ctypes as _ctypes
import glob as _glob
import os as _os
import sys as _sys
import types as _types

import torch as _torch

# The native part is a plain shared object (no pybind11: it would not fit in
# this board's compile budget).  Loading it runs the static registrations:
# the PrivateUse1 allocator, the device guard, the aten kernels and the
# torch_vc4.* diagnostic ops.  It also forces VC4INFER_DAEMON=1 (vc4d).
_here = _os.path.dirname(_os.path.abspath(__file__))
_native = sorted(_glob.glob(_os.path.join(_here, "_C*.so")))
if not _native:
    raise ImportError(
        "torch_vc4: the native backend is not built. Run:\n"
        "  MAX_JOBS=1 .venv/bin/pip install --no-build-isolation -e .")
_ctypes.CDLL(_native[0], mode=_ctypes.RTLD_GLOBAL)

if _torch._C._get_privateuse1_backend_name() != "vc4":
    _torch.utils.rename_privateuse1_backend("vc4")

# torch looks up torch.<backend> for RNG/seed bookkeeping; without it
# torch.device("privateuseone") raises ModuleNotFoundError: No module named
# 'torch.vc4'.
_vc4_mod = _types.ModuleType("torch.vc4")


def _manual_seed_all(seed: int) -> None:
    _torch.ops.torch_vc4.seed(int(seed))


_rng_state = _torch.zeros(4, dtype=_torch.uint8)


def _get_rng_state(device=None):
    return _rng_state.clone()


def _set_rng_state(new_state, device=None):
    global _rng_state
    _rng_state = new_state.clone()


_vc4_mod.manual_seed_all = _manual_seed_all
_vc4_mod.is_available = lambda: True
_vc4_mod.device_count = lambda: 1
_vc4_mod.current_device = lambda: 0
_vc4_mod._is_in_bad_fork = lambda: False
_vc4_mod.get_rng_state = _get_rng_state
_vc4_mod.set_rng_state = _set_rng_state
_sys.modules["torch.vc4"] = _vc4_mod
setattr(_torch, "vc4", _vc4_mod)
try:
    # torch insists on parsing the name; "privateuseone" is the only spelling
    # its device-string parser accepts in this build.
    _torch._register_device_module("privateuseone", _vc4_mod)
except Exception:  # pragma: no cover - informational only
    pass

try:  # adds Tensor.vc4() / Module.vc4() convenience methods
    _torch.utils.generate_methods_for_privateuse1_backend()
except Exception as _e:  # pragma: no cover - informational only
    import warnings as _w

    _w.warn(f"torch_vc4: generate_methods_for_privateuse1_backend failed: {_e}")


def device(index: int = 0) -> "_torch.device":
    """The vc4 device (always index 0).

    Uses the built-in "privateuseone" spelling because torch 2.15's
    device-string parser does not accept the renamed name; the resulting
    object still reports `type == "vc4"`.
    """
    return _torch.device("privateuseone", index)


def is_available() -> bool:
    """True if the engine arena came up (vc4d reachable)."""
    try:
        arena()
        return True
    except Exception:
        return False


def stats() -> dict:
    """Engine counters: launches, GPU kernel ms, FLOPs issued on the QPU."""
    keys = ("launches", "pack_ms", "kernel_ms", "gpu_cycles", "flops",
            "gflops_kernel")
    return dict(zip(keys, _torch.ops.torch_vc4.stats()))


def reset_stats() -> None:
    _torch.ops.torch_vc4.reset_stats()


def arena() -> dict:
    """Arena geometry: torch region bytes, engine arena bytes/used."""
    keys = ("region_off", "region_bytes", "arena_bytes", "arena_used")
    return dict(zip(keys, _torch.ops.torch_vc4.arena()))


def synchronize() -> None:
    """No-op: every vc4 launch is a synchronous LAUNCH_WAIT round trip."""
    return None


# The subset op surface is registered from Python: the native part supplies
# the engine bindings and the byte-level layout primitives, ops.py the aten
# forward kernels, autograd.py the torch.autograd.Function graph.  Importing
# this module therefore installs *all* of it, and any op not listed in
# SUPPORTED_OPS still hits the catch-all that raises.
from . import ops as _ops            # noqa: E402
from . import autograd as _autograd  # noqa: E402

_ops.register()
_autograd.register()

# torch 2.15 decomposes aten::cat at CompositeImplicitAutograd before the
# backend's Autograd kernel is consulted, which silently drops the graph node
# for a PrivateUse1 tensor.  Serve the differentiable path explicitly for vc4
# inputs; everything else goes to the original torch.cat unchanged.
_cat_orig = _torch.cat


def _cat(tensors, dim=0, *, out=None):
    ts = list(tensors)
    if (out is None and ts and all(t.device.type == "vc4" for t in ts)
            and any(t.requires_grad for t in ts)):
        return _autograd.CatFn.apply(int(dim), *ts)
    if out is None:
        return _cat_orig(tensors, dim)
    return _cat_orig(tensors, dim, out=out)


_torch.cat = _cat

mse_loss = _autograd.mse_loss
fold_bn_into_conv = _ops.fold_bn_into_conv
adaptive_matrix = _ops.adaptive_matrix

SUPPORTED_OPS = (
    # memory / transfer
    "aten::empty.memory_format",
    "aten::empty_strided",
    "aten::copy_ (and Tensor.to/.cpu for f32/int64, same shape)",
    "aten::fill_.Scalar, aten::zero_",
    "aten::normal_ / uniform_ (host RNG, uploaded)",
    "aten::_local_scalar_dense (Tensor.item, float(t))",
    # views (metadata; non-contiguous views are rejected by compute ops)
    "aten::view, aten::flatten, aten::reshape",
    "aten::as_strided (alias; compute ops need contiguity)",
    # gemm
    "aten::mm",
    "aten::matmul (2-D x 2-D)",
    "aten::linear (F.linear / nn.Linear; x @ W^T + bias, all on the QPU)",
    # conv / pool (batch > 1, stride 1/2, 1x1/3x3, depthwise)
    "aten::convolution_overrideable (conv2d, any batch, stride 1/2, "
    "pad, groups=1 and depthwise)",
    "aten::max_pool2d (2x2/2 engine kernel, any NxN/stride/pad generic, "
    "any batch)",
    "aten::_adaptive_avg_pool2d (any output size, GEMM reduction, any batch)",
    "aten::upsample_nearest2d (nearest, any scale, any batch)",
    # elementwise / unary (verified engine bodies)
    "aten::add.Tensor / sub.Tensor / mul.Tensor / div.Tensor "
    "(N-D broadcasting, per-channel, scalars)",
    "aten::add.Scalar, mul.Scalar, div.Scalar, neg, pow.Tensor_Scalar(2)",
    "aten::relu, silu, sigmoid",
    "aten::abs, sign, clip, leaky_relu, hardsigmoid, hardswish",
    "aten::floor, ceil, round (exact; engine-body unaries)",
    "aten::exp, log, erf (engine-body unaries)",
    "aten::tanh, sqrt, reciprocal (device composites: sigmoid / rsqrt)",
    "aten::gelu(approximate='none' | 'tanh') (erf body / device composite)",
    # reductions over arbitrary dims
    "aten::sum / mean (full and dim, keepdim)",
    "aten::amax / amin (full and dim, keepdim; forward only)",
    "aten::max / min / max.dim / min.dim (forward only)",
    "aten::argmax / argmin (full and dim; host integer scan)",
    # indexing
    "aten::gather, aten::index_select (host word gather; forward only)",
    "aten::scatter/sort/topk and friends raise with the precise reason",
    # softmax
    "aten::_softmax (any dim, any class count: engine <=16, two-pass above)",
    # shape / norm
    "aten::cat (channel axis, any batch)",
    "aten::native_batch_norm / batch_norm (eval and training mode)",
    # autograd
    "autograd for mm/matmul, linear, broadcast add/sub/mul/div, scalar ops, "
    "relu, silu, sigmoid, abs, sign, clip, leaky_relu, hardsigmoid, "
    "hardswish, tanh, sqrt, reciprocal, gelu, softmax, conv2d, "
    "max_pool2d, adaptive_avg_pool2d, batch_norm (train+eval), sum/mean "
    "(full and dim), view/as_strided(contiguous); amax/amin, gather and "
    "scatter-like backwards raise with the precise reason",
)

__all__ = [
    "device",
    "is_available",
    "stats",
    "reset_stats",
    "arena",
    "synchronize",
    "mse_loss",
    "fold_bn_into_conv",
    "adaptive_matrix",
    "SUPPORTED_OPS",
]
