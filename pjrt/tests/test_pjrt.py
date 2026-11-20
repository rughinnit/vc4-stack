"""End-to-end checks for the vc4 PJRT plugin.

Runs every case on the QPU device and on jax's CPU backend in the same
process, compares element by element, and reports the launch count the plugin
logged for the last execution.  Exits non-zero on any mismatch, on a missing
device, or on a case that should have been refused but was not.

    PYTHONPATH=pjrt/python JAX_PLATFORMS=vc4 VC4PJRT_LOG=/tmp/pjrt.log \
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


def compare(name, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        FAILED.append(name)
        print(f"  {name:18s} SHAPE {got.shape} != {want.shape}")
        return
    diff = np.abs(got - want)
    over = int(np.count_nonzero(diff > TOL))
    exact = int(np.count_nonzero(got != want))
    status = "ok" if over == 0 else "MISMATCH"
    if over:
        FAILED.append(name)
    print(f"  {name:18s} {status:9s} max|d|={diff.max():.3e} "
          f"over={over}/{got.size} bit-neq={exact}")


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
    ]
    print("\nop subset (vc4 vs cpu):")
    for name, fn, args in cases:
        try:
            got = on_vc4(fn, *args)
            want = cpu(fn, *args)
        except Exception as e:  # noqa: BLE001 - the message is the result
            print(f"  {name:18s} ERROR     {str(e).splitlines()[0][:110]}")
            FAILED.append(name)
            continue
        compare(name, got, want)

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

    # Refusals must name the op.
    print("\nunsupported ops must raise, not fall back:")
    def iota(_a, _b):
        return jnp.arange(6, dtype=jnp.float32)
    try:
        jax.jit(iota, backend="vc4")(x, w)
        print("  iota               NOT REFUSED")
        FAILED.append("iota")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0]
        ok = "iota_v1" in msg or "iota" in msg
        print(f"  iota               refused ({'names op' if ok else 'unnamed'})")
        if not ok:
            FAILED.append("iota-error-text")
    def sliced(_a, _b):
        return jnp.sum(_a.T[:2, :2])
    try:
        jax.jit(sliced, backend="vc4")(x, w)
        print("  slice              NOT REFUSED")
        FAILED.append("slice")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0]
        ok = "slice_v1" in msg or "slice" in msg
        print(f"  slice              refused ({'names op' if ok else 'unnamed'})")
        if not ok:
            FAILED.append("slice-error-text")

    # Cost per call.
    print("\nper-call cost (median of 20, warm):")
    for name, fn, args in (("mlp_forward", lambda a, b: jnp.sum(
            jax.nn.relu(a @ b)), (x, w)),):
        f = lambda a, b: jax.jit(fn, backend="vc4")(a, b)
        ms = timed(f, *args) * 1e3
        info = launches_from_log()
        print(f"  {name:18s} {ms:7.2f} ms  "
              f"{info[1] if info else '?'} launches  "
              f"(cpu {timed(lambda a, b: cpu(fn, a, b), *args) * 1e3:.3f} ms)")
    f = lambda a, b: jax.jit(grad, backend="vc4")(a, b)
    ms = timed(f, x, w) * 1e3
    info = launches_from_log()
    print(f"  {'grad_x':18s} {ms:7.2f} ms  "
          f"{info[1] if info else '?'} launches  "
          f"(cpu {timed(lambda a, b: cpu(grad, a, b), x, w) * 1e3:.3f} ms)")

    print()
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        return 1
    print("ALL PJRT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
