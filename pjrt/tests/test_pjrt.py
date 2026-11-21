"""End-to-end checks for the vc4 PJRT plugin.

Runs every case on the QPU device and on jax's CPU backend in the same
process, compares element by element, and reports the launch count the plugin
logged for the execution.  Exits non-zero on any mismatch, on a missing
device, or on a case that should have been refused but was not.

This file launches kernels.  Run it only when the board is otherwise idle; the
lowering-only checks in test_lower.py need no QPU.

    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4 VC4PJRT_LOG=$HOME/scratch/logs/pjrt.log \
        python3 pjrt/tests/test_pjrt.py
"""
import os
import re
import statistics
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

TOL = 5e-4
FAILED = []


def cpu(fn, *args):
    return jax.jit(fn, backend="cpu")(*args)


def on_vc4(fn, *args):
    return jax.jit(fn, backend="vc4")(*args)


def compare(name, got, want, info=None):
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        FAILED.append(name)
        print(f"  {name:20s} SHAPE {got.shape} != {want.shape}")
        return
    diff = np.abs(got - want)
    over = int(np.count_nonzero(diff > TOL))
    exact = int(np.count_nonzero(got != want))
    status = "ok" if over == 0 else "MISMATCH"
    if over:
        FAILED.append(name)
    extra = f" launches={info[1]}" if info else ""
    print(f"  {name:20s} {status:9s} max|d|={diff.max():.3e} "
          f"over={over}/{got.size} bit-neq={exact}{extra}")


def launches_from_log():
    path = os.environ.get("VC4PJRT_LOG")
    if not path or not os.path.exists(path):
        return None
    last = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = re.match(r"execute: (\d+) insns, (\d+) launches, (\d+) ns", line)
            if m:
                last = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return last


def timed(fn, *args, reps=20):
    fn(*args)  # compile and warm up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    devices = jax.devices()
    print("jax devices:", devices)
    if not any("vc4" in str(d).lower() for d in devices):
        print("FAIL: no vc4 device")
        return 1

    x = (np.arange(6, dtype=np.float32).reshape(2, 3) / 10).astype(np.float32)
    w = (np.arange(12, dtype=np.float32).reshape(3, 4) / 10).astype(np.float32)
    v = np.array([-1.5, -0.25, 0.0, 0.75, 2.0, 3.5], dtype=np.float32)
    c = (np.arange(12, dtype=np.float32).reshape(4, 3) / 10).astype(np.float32)
    xt = np.ascontiguousarray(x.T)
    x34 = (np.arange(12, dtype=np.float32).reshape(3, 4) / 10).astype(np.float32)
    y3 = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
    m44 = np.arange(16, dtype=np.float32).reshape(4, 4) / 10
    idx = np.array([2, 0], dtype=np.int32)
    rows = np.array([2, 0], dtype=np.int32)
    cols = np.array([1, 3], dtype=np.int32)
    k16 = np.ones((16, 16, 3, 3), np.float32) * 0.05
    x16 = (np.arange(400, dtype=np.float32).reshape(1, 16, 5, 5) / 100)

    cases = [
        ("add", lambda a, b: a + b, (x, w[:2, :3] + 1)),
        ("multiply", lambda a, b: a * a, (x, w)),
        ("subtract", lambda a, b: a - b, (x, x + 1)),
        ("negate", lambda a, b: -a, (x, w)),
        ("relu", lambda a, b: jax.nn.relu(a), (v.reshape(2, 3), w)),
        ("maximum", lambda a, b: jnp.maximum(a, b), (x, x[::-1] * 0.5)),
        ("dot", lambda a, b: a @ b, (x, w)),
        ("dot_lhsT", lambda p, q: jnp.einsum("km,kn->mn", p, q), (xt, w)),
        ("dot_rhsT", lambda p, q: jnp.einsum("mk,nk->mn", p, q), (x, c)),
        ("reshape", lambda a, b: a.reshape(3, 2), (x, w)),
        ("broadcast", lambda a, b: jnp.broadcast_to(
            a.reshape(2, 3, 1), (2, 3, 4)), (x, w)),
        ("sum", lambda a, b: jnp.sum(a, axis=None), (x, w)),
        ("sum_rows", lambda a, b: jnp.sum(a, axis=1), (x, w)),
        ("mlp_forward", lambda a, b: jnp.sum(jax.nn.relu(a @ b)), (x, w)),
        # slice, reverse and pad: one index-table gather each
        ("slice", lambda a, b: a[1:3, 0:2], (x34, w)),
        ("slice_step", lambda a, b: a[::2, ::3], (x34, w)),
        ("slice_rank3", lambda a, b: a[:, 1:3, ::2], (y3, w)),
        ("reverse", lambda a, b: a[::-1, ::-1], (x34, w)),
        ("pad", lambda a, b: jnp.pad(a, ((1, 2), (0, 3))), (x34, w)),
        ("pad_interior", lambda a, b: jax.lax.pad(
            a, 0.0, [(1, 2, 1), (0, 0, 2)]), (x34, w)),
        # iota is a compile-time constant: no kernel
        ("iota", lambda a, b: jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
         (x, w)),
        # gather with a static index tensor
        ("gather_axis0", lambda a, b: a[idx], (x34, w)),
        ("gather_2d", lambda a, b: a[rows, cols], (x34, w)),
        ("take", lambda a, b: jnp.take(a, jnp.arange(2), axis=0), (x34, w)),
        # reductions over leading and interior axes, and max/min bodies
        ("sum_leading", lambda a, b: jnp.sum(a, axis=0), (x34, w)),
        ("max_leading", lambda a, b: jnp.max(a, axis=0), (x34, w)),
        ("min_trailing", lambda a, b: jnp.min(a, axis=1), (x34, w)),
        ("sum_nontrailing", lambda a, b: jnp.sum(a, axis=(0, 2)), (y3, w)),
        ("max_last", lambda a, b: jnp.max(a, axis=-1), (x34, w)),
        # dot_general beyond rank 2 and with batching dimensions
        ("dot_batched", lambda a, b: jnp.einsum("bij,bkj->bik", a, a), (y3, w)),
        ("dot_rank3", lambda a, b: jnp.einsum("ijk,kl->ijl", a, b), (y3, m44)),
        # convolution: the engine's 3x3/1x1 NCHW path
        ("conv_3x3", lambda a, b: jax.lax.conv_general_dilated(
            a, b, (1, 1), "SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW")), (x16, k16)),
        ("conv_1x1", lambda a, b: jax.lax.conv_general_dilated(
            a, b, (1, 1), [(0, 0), (0, 0)],
            dimension_numbers=("NCHW", "OIHW", "NCHW")),
         (x16, np.ones((16, 16, 1, 1), np.float32) * 0.25)),
    ]
    print("\nop subset (vc4 vs cpu):")
    for name, fn, args in cases:
        try:
            got = on_vc4(fn, *args)
            info = launches_from_log()
            want = cpu(fn, *args)
        except Exception as e:  # noqa: BLE001 - the message is the result
            print(f"  {name:20s} ERROR     {str(e).splitlines()[0][:110]}")
            FAILED.append(name)
            continue
        compare(name, got, want, info)

    # The payoff: jax.grad emits dot_general/compare/select/reduce/broadcast.
    print("\ngradient (jax.grad of sum(relu(x @ w))):")
    def loss(a, b):
        return jnp.sum(jax.nn.relu(a @ b))

    grad = jax.grad(loss, argnums=0)
    g_vc4 = jax.jit(grad, backend="vc4")(x, w)
    g_cpu = jax.jit(grad, backend="cpu")(x, w)
    compare("grad_x", g_vc4, g_cpu)
    gw_vc4 = jax.jit(jax.grad(loss, argnums=1), backend="vc4")(x, w)
    gw_cpu = jax.jit(jax.grad(loss, argnums=1), backend="cpu")(x, w)
    compare("grad_w", gw_vc4, gw_cpu)
    info = launches_from_log()
    if info:
        print(f"  last execution: {info[0]} instructions, {info[1]} launches, "
              f"{info[2] / 1e6:.2f} ms QPU")

    # Refusals must name the op and the attribute or shape at fault.
    print("\nunsupported ops must raise, not fall back:")

    def expect_refusal(name, fn, args, want_op, want_word=""):
        try:
            jax.jit(fn, backend="vc4")(*args)
            print(f"  {name:20s} NOT REFUSED")
            FAILED.append(name)
            return
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0]
        ok = want_op in msg and want_word in msg
        print(f"  {name:20s} refused ({'names op' if ok else 'unnamed'})")
        if not ok:
            print(f"    {msg[:110]}")
            FAILED.append(name)

    # multiply is not a reduction body the executor implements
    expect_refusal("reduce_product", lambda a, b: jnp.prod(a), (x34, w),
                   "reduce_v1", "multiply")
    # mean needs divide, which has no lowering
    expect_refusal("divide", lambda a, b: jnp.mean(a), (x34, w),
                   "divide_v1", "")
    # a 3-channel convolution is not a multiple of the 16-wide panel
    expect_refusal("conv_channels", lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")),
        (np.ones((1, 3, 5, 5), np.float32), np.ones((16, 3, 3, 3), np.float32)),
        "convolution_v1", "input channels")
    # stride 2 is composed by the ONNX runner, not by this plugin
    expect_refusal("conv_stride", lambda a, b: jax.lax.conv_general_dilated(
        a, b, (2, 2), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")),
        (x16, k16), "convolution_v1", "window_strides")
    # control flow is not part of the subset
    def loop(_a, _b):
        return jax.lax.while_loop(lambda i: i < 4, lambda i: i + 1, 0)
    expect_refusal("while_loop", loop, (x, w), "while", "")

    # Cost per call.
    print("\nper-call cost (median of 20, warm):")
    timed_cases = [
        ("mlp_forward", lambda a, b: jnp.sum(jax.nn.relu(a @ b)), (x, w)),
        ("conv_3x3", cases[-2][1], (x16, k16)),
        ("grad_x", grad, (x, w)),
    ]
    for name, fn, args in timed_cases:
        f = lambda a, b, fn=fn: jax.jit(fn, backend="vc4")(a, b)
        ms = timed(f, *args) * 1e3
        info = launches_from_log()
        print(f"  {name:20s} {ms:7.2f} ms  "
              f"{info[1] if info else '?'} launches  "
              f"(cpu {timed(lambda a, b, fn=fn: cpu(fn, a, b), *args) * 1e3:.3f} ms)")

    print()
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        return 1
    print("ALL PJRT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
