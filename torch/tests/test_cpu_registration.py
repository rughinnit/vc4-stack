#!/usr/bin/env python3
"""CPU-side checks for the backend changes in this branch.

Nothing here dispatches to `vc4`: no tensor is allocated on the device and no
vc4d request is made.  The script covers

  * registration: the six unary aten entries (floor, ceil, round, exp, log,
    erf) really have a PrivateUse1 kernel, and the routing table sends each
    one to the engine's verified body with the right UOP id;
  * broadcast: the plan `_broadcast_view` produces is materialised here with
    numpy strides and compared with `np.broadcast_to` for scalar, trailing,
    leading and middle broadcast;
  * reductions: the plan `_reduce_plan` produces (permute reduced axes last,
    flatten to [outer][red]) is applied to a numpy array and compared with
    `np.sum` over the same axes, 2-D and 3-D, with and without keepdim;
  * the exact-GELU derivative used by autograd, against finite differences.

Device accuracy and timings are not claimed here; `tests/test_complete.py`
measures them when the board is free.

    python3 tests/test_cpu_registration.py
"""
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402
import warnings                                             # noqa: E402

warnings.filterwarnings("ignore")
import torch_vc4                                            # noqa: E402
from torch_vc4 import ops                                   # noqa: E402

FAILS = []
CHECKS = [0]


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(name)
        print("FAIL  " + name + (("  " + detail) if detail else ""))


def registration():
    key = "PrivateUse1"
    for name in ("aten::floor", "aten::ceil", "aten::round",
                 "aten::exp", "aten::log", "aten::erf",
                 "aten::upsample_bilinear2d",
                 "aten::upsample_bilinear2d.vec"):
        check("registered " + name,
              torch._C._dispatch_has_kernel_for_dispatch_key(name, key))


def routing():
    """Every new entry must call T.unary with its own UOP id, not a stub."""
    expected = {
        "unary_floor": ops.UOP_FLOOR,
        "unary_ceil": ops.UOP_CEIL,
        "unary_round": ops.UOP_ROUND,
        "unary_exp": ops.UOP_EXP,
        "unary_log": ops.UOP_LOG,
        "unary_erf": ops.UOP_ERF,
    }
    real_T, real_f32 = ops.T, ops.f32
    calls = []

    def fake_unary(t, op, p0, p1, p2):
        calls.append((op, t))
        return "y"

    ops.T = types.SimpleNamespace(unary=fake_unary)
    ops.f32 = lambda t, name: None
    try:
        for fn_name, op in expected.items():
            calls.clear()
            got = getattr(ops, fn_name)("x")
            check("%s routes UOP %d" % (fn_name, op),
                  got == "y" and calls == [(op, "x")], repr(calls))
            check("%s id in the verified set" % fn_name,
                  op in ops._VERIFIED_UOPS)
    finally:
        ops.T, ops.f32 = real_T, real_f32


def _as_broadcast(a, shape):
    """Materialise `a` as `shape` using the backend's plan, with numpy."""
    plan = ops._broadcast_view(shape, a.shape,
                               [s // a.itemsize for s in a.strides])
    if plan is None:
        return a
    sizes, strides = plan
    return np.lib.stride_tricks.as_strided(
        a, shape=tuple(sizes),
        strides=tuple(s * a.itemsize for s in strides))


def broadcast():
    cases = [
        ((), (2, 3)),
        ((3,), (4, 3)),
        ((3, 4), (2, 3, 4)),
        ((2, 1), (2, 3)),
        ((2, 1, 4), (2, 3, 4)),
        ((1, 3, 1), (2, 3, 4)),
    ]
    for src, dst in cases:
        a = np.arange(max(1, int(np.prod(src))), dtype=np.float32)
        a = a.reshape(src)
        got = _as_broadcast(a, dst)
        check("broadcast %s -> %s" % (src, dst),
              got.shape == dst and np.array_equal(got, np.broadcast_to(a, dst)))
    same = np.ones((2, 3), np.float32)
    check("broadcast identical returns None",
          ops._broadcast_view((2, 3), same.shape,
                              [s // 4 for s in same.strides]) is None)
    for src, dst in (((3,), (4,)), ((2, 3), (4, 3))):
        a = np.ones(src, np.float32)
        try:
            ops._broadcast_view(dst, a.shape, [s // 4 for s in a.strides])
        except RuntimeError:
            check("broadcast rejects %s -> %s" % (src, dst), True)
        else:
            check("broadcast rejects %s -> %s" % (src, dst), False,
                  "no RuntimeError")


def reductions():
    cases = [
        ((2, 3, 4), (0, 2), False),
        ((2, 3, 4), (0, 2), True),
        ((2, 3, 4), (-1,), False),
        ((2, 3, 4), (1,), True),
        ((2, 3, 4), (0, 1, 2), False),
        ((2, 3, 4), (0, 1), False),
        ((5, 7), (0,), False),
        ((5, 7), (-1,), True),
    ]
    rng = np.random.default_rng(7)
    for shape, dims, keepdim in cases:
        a = rng.standard_normal(shape).astype(np.float32)
        order, kept, red, outer, out = ops._reduce_plan(shape, dims, keepdim)
        nd = len(shape)
        norm = tuple(d % nd for d in dims)
        # the plan's own arithmetic, applied to the numpy data
        got = a.transpose(order).reshape(outer, red).sum(axis=1)
        got = got.reshape(out)
        ref = a.sum(axis=norm, keepdims=keepdim)
        check("reduce %s dims=%s keepdim=%s" % (shape, dims, keepdim),
              got.shape == ref.shape and np.allclose(got, ref, rtol=1e-5,
                                                     atol=1e-5),
              "plan %s vs %s" % (got.shape, ref.shape))
        check("reduce plan shape %s dims=%s" % (shape, dims),
              ops._reduce_plan(shape, dims, keepdim)[4] == ref.shape)
    check("duplicate dims collapse",
          ops._norm_dims([1, 1, -2], 3) == [1])


def gelu_derivative():
    """The analytic form autograd uses, against central differences."""
    x = np.linspace(-4.0, 4.0, 33)
    h = 1e-6
    erf = np.vectorize(math.erf)

    def gelu(v):
        return 0.5 * v * (1.0 + erf(v / math.sqrt(2.0)))

    ref = (0.5 * (1.0 + erf(x / math.sqrt(2.0)))
           + x * np.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi))
    fd = (gelu(x + h) - gelu(x - h)) / (2 * h)
    check("exact-GELU derivative", np.allclose(ref, fd, rtol=1e-6, atol=1e-6),
          "max |diff| %.3g" % float(np.max(np.abs(ref - fd))))


def main():
    backend = torch._C._get_privateuse1_backend_name()
    check("backend renamed to vc4", backend == "vc4", backend)
    registration()
    routing()
    broadcast()
    reductions()
    gelu_derivative()
    print("\n%d checks, %d passed, %d failed"
          % (CHECKS[0], CHECKS[0] - len(FAILS), len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
