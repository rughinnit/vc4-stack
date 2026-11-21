#!/usr/bin/env python3
"""CPU-side checks for the TensorFlow PluggableDevice plugin.

Stages, each opt-in:

  (default)   the library loads, the platform registers, the device appears in
              tf.config.list_physical_devices / list_logical_devices, and every
              registered op kernel is in TensorFlow's kernel registry for this
              device type with T = float32.  No device memory is touched, so
              nothing reaches the QPUs or the engine arena.
  --copy      the allocator and the H2D/D2H paths: a constant placed on the
              device and read back.  Needs the engine arena (8 MiB of CMA per
              process) or VC4_TF_ARENA=host for a wiring-only run.
  --dispatch  with VC4_TF_ARENA=host: every kernel is reached through eager
              and then refuses (no engine), and the shape, dtype, transpose
              and data_format refusals are checked.  This proves placement and
              kernel registration without a QPU launch.
  --kernels   the real thing: each kernel on the QPU against TF on the CPU, at
              n = 1024 and at a length that is not a multiple of 256.  Needs
              VC4_TF_ALLOW_QPU=1 and an idle board.
  --graph     matmul -> bias -> relu placed entirely on the device, against
              the same graph on the CPU.  Also needs VC4_TF_ALLOW_QPU=1; with
              a host arena it checks only that the graph lands on the device
              (the first op refuses there).

Exits non-zero on a failed check.
"""
import argparse
import ctypes
import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import kernels as tf_kernels
from tensorflow.python.framework.load_library import load_pluggable_device_library

DEVICE_TYPE = os.environ.get("VC4_TF_DEVICE_TYPE", "VCIV")
DEVICE = "/device:%s:0" % DEVICE_TYPE
LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "build", "libtf_vc4_plugin.so")
HOST_ARENA = os.environ.get("VC4_TF_ARENA", "").lower() == "host"
ALLOW_QPU = os.environ.get("VC4_TF_ALLOW_QPU") == "1"

FAILED = []


def check(name, ok, detail=""):
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                        (": " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


def plugin_diag():
    """The plugin's own view: (engine state, last error, arena status)."""
    try:
        lib = ctypes.CDLL(LIB)
    except OSError as exc:
        return 0, str(exc), ""
    lib.vc4_tf_engine_state.restype = ctypes.c_int
    lib.vc4_tf_last_error.restype = ctypes.c_char_p
    lib.vc4_tf_arena_status.restype = ctypes.c_char_p
    state = lib.vc4_tf_engine_state()
    err = (lib.vc4_tf_last_error() or b"").decode()
    arena = (lib.vc4_tf_arena_status() or b"").decode()
    return state, err, arena


_LIB = None


def engine_stats():
    """The engine's counters as the plugin sees them, or None before it starts."""
    global _LIB
    if _LIB is None:
        try:
            _LIB = ctypes.CDLL(LIB)
        except OSError:
            return None
    _LIB.vc4_tf_stats.restype = ctypes.c_char_p
    s = (_LIB.vc4_tf_stats() or b"").decode()
    if not s.startswith("launches="):
        return None
    out = {}
    for field in s.split():
        k, v = field.split("=", 1)
        out[k] = float(v)
    return out


def engine_launches():
    s = engine_stats()
    return None if s is None else int(s["launches"])


def names(devices):
    return [d.name for d in devices]


# Every op TF_InitKernel registers, with its input count.  Mirrors the
# VC4_TF_EW_KERNELS table in src/vc4_plugin.c plus the two shape-driven
# kernels.
OPS = [
    ("Add", 2), ("AddV2", 2), ("Mul", 2), ("Sub", 2), ("Div", 2),
    ("RealDiv", 2), ("Maximum", 2), ("Minimum", 2),
    ("Relu", 1), ("Relu6", 1), ("Sigmoid", 1), ("Tanh", 1), ("Neg", 1),
    ("Exp", 1), ("Log", 1), ("Sqrt", 1),
    ("MatMul", 2), ("BiasAdd", 2),
]


def stage_register():
    print("device type %r, library %s" % (DEVICE_TYPE, LIB))
    before = names(tf.config.list_physical_devices())
    print("physical devices before load:", before)

    load_pluggable_device_library(LIB)

    physical = names(tf.config.list_physical_devices())
    logical = names(tf.config.list_logical_devices())
    print("physical devices after load: ", physical)
    print("logical devices after load:  ", logical)

    want_physical = "/physical_device:%s:0" % DEVICE_TYPE
    check("platform registers %s" % want_physical,
          want_physical in physical)
    check("device %s is visible" % DEVICE, DEVICE in logical)

    phys = [d for d in tf.config.list_physical_devices()
            if d.name == want_physical]
    if phys:
        print("device details:",
              tf.config.experimental.get_device_details(phys[0]))

    missing = []
    for op, _arity in OPS:
        kl = tf_kernels.get_registered_kernels_for_op(op)
        mine = [k for k in kl.kernel if k.device_type == DEVICE_TYPE]
        types = sorted({t for k in mine for c in k.constraint
                        if c.name == "T" for t in c.allowed_values.list.type})
        if not mine:
            missing.append(op)
        check("kernel %s registered on %s" % (op, DEVICE_TYPE), bool(mine),
              "dtypes=%s" % types)
    print("registered kernels: %d of %d%s" % (
        len(OPS) - len(missing), len(OPS),
        "" if not missing else " (missing %s)" % ", ".join(missing)))

    state, err, arena = plugin_diag()
    print("plugin: %s; engine %s%s" % (
        arena, "up" if state == 1 else "not started" if state == 0
        else "down", (" (%s)" % err) if err else ""))


def stage_copy():
    n = 1024
    rng = np.random.default_rng(7)
    src = rng.standard_normal(n).astype(np.float32)
    try:
        with tf.device(DEVICE):
            t = tf.constant(src, dtype=tf.float32)
            back = t.numpy()
    except Exception as exc:   # noqa: BLE001 - the message is the result
        msg = str(exc)
        state, err, arena = plugin_diag()
        if state != 1:
            print("SKIP  device memory unavailable; %s" % arena)
            print("      registration checks above are unaffected")
            return
        check("device constant round-trip", False, msg)
        return

    check("constant landed on %s" % DEVICE, t.device.endswith(DEVICE),
          t.device)
    check("H2D/D2H round-trip is exact", np.array_equal(back, src),
          "max|d|=%g" % float(np.max(np.abs(back - src))))


# --------------------------------------------------------------- dispatch --

def _dispatch_cases():
    """(op, callable run under tf.device(DEVICE)) for every registered op."""
    a = np.linspace(0.5, 2.0, 64, dtype=np.float32)
    b = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    x2 = np.linspace(-1.0, 1.0, 8 * 32, dtype=np.float32).reshape(8, 32)
    w2 = np.linspace(-0.5, 0.5, 32 * 16, dtype=np.float32).reshape(32, 16)
    xb = np.linspace(-1.0, 1.0, 8 * 16, dtype=np.float32).reshape(8, 16)
    bias = np.linspace(-0.1, 0.1, 16, dtype=np.float32)
    one = lambda v: tf.constant(v, dtype=tf.float32)   # noqa: E731
    return [
        ("Add", lambda: tf.raw_ops.Add(x=one(a), y=one(b))),
        ("AddV2", lambda: tf.raw_ops.AddV2(x=one(a), y=one(b))),
        ("Mul", lambda: tf.multiply(one(a), one(b))),
        ("Sub", lambda: tf.subtract(one(a), one(b))),
        ("Div", lambda: tf.raw_ops.Div(x=one(a), y=one(b))),
        ("RealDiv", lambda: tf.divide(one(a), one(b))),
        ("Maximum", lambda: tf.maximum(one(a), one(b))),
        ("Minimum", lambda: tf.minimum(one(a), one(b))),
        ("Relu", lambda: tf.nn.relu(one(a))),
        ("Relu6", lambda: tf.nn.relu6(one(a))),
        ("Sigmoid", lambda: tf.nn.sigmoid(one(a))),
        ("Tanh", lambda: tf.math.tanh(one(a))),
        ("Neg", lambda: tf.math.negative(one(a))),
        ("Exp", lambda: tf.exp(one(a))),
        ("Log", lambda: tf.math.log(one(a))),
        ("Sqrt", lambda: tf.sqrt(one(a))),
        ("MatMul", lambda: tf.matmul(one(x2), one(w2))),
        ("BiasAdd", lambda: tf.nn.bias_add(one(xb), one(bias))),
    ]


def stage_dispatch():
    """Reach every kernel through eager without letting it launch."""
    if not HOST_ARENA:
        print("SKIP  --dispatch needs VC4_TF_ARENA=host (it checks the "
              "dispatch path, not arithmetic)")
        return

    for op, call in _dispatch_cases():
        try:
            with tf.device(DEVICE):
                call().numpy()
        except Exception as exc:   # noqa: BLE001 - the refusal is the result
            msg = str(exc).strip().splitlines()[0]
            ok = ("tf-vc4 %s" % op) in msg and "engine arena" in msg
            check("%s dispatched to the %s kernel" % (op, DEVICE_TYPE), ok,
                  msg[:170])
            continue
        check("%s dispatched to the %s kernel" % (op, DEVICE_TYPE), False,
              "the op ran without an engine; unexpected with a host arena")

    # Broadcasting is a named refusal from the kernel, not a host fallback.
    try:
        with tf.device(DEVICE):
            a44 = tf.constant(np.ones((4, 4), np.float32))
            b4 = tf.constant(np.ones(4, np.float32))
            tf.add(a44, b4).numpy()
        check("broadcast Add is refused by name", False, "it ran")
    except Exception as exc:   # noqa: BLE001
        msg = str(exc).strip().splitlines()[0]
        check("broadcast Add is refused by name",
              "tf-vc4 AddV2: inputs are not the same shape" in msg and
              "broadcasting is not implemented" in msg, msg[:170])

    # An empty tensor is a legal op with no work: it must succeed, and it must
    # not reach the engine (there would be nothing to launch).
    try:
        with tf.device(DEVICE):
            empty = tf.multiply(tf.constant(np.zeros(0, np.float32)),
                                tf.constant(np.zeros(0, np.float32)))
            ok_empty = empty.shape == (0,)
        check("an empty Mul needs no engine", ok_empty, str(empty.shape))
    except Exception as exc:   # noqa: BLE001
        check("an empty Mul needs no engine", False,
              str(exc).strip().splitlines()[0][:170])

    # MatMul with a zero reduction has no engine path; it is a named refusal.
    try:
        with tf.device(DEVICE):
            a = tf.constant(np.zeros((4, 0), np.float32))
            b = tf.constant(np.zeros((0, 5), np.float32))
            tf.matmul(a, b).numpy()
        check("MatMul refuses K == 0", False, "it ran")
    except Exception as exc:   # noqa: BLE001
        msg = str(exc).strip().splitlines()[0]
        check("MatMul refuses K == 0", "K == 0 is not supported" in msg,
              msg[:170])

    # BiasAdd checks the bias length against the last input dimension.
    try:
        with tf.device(DEVICE):
            x = tf.constant(np.ones((8, 16), np.float32))
            bias = tf.constant(np.ones(8, np.float32))
            tf.nn.bias_add(x, bias).numpy()
        check("BiasAdd refuses a bias of the wrong length", False, "it ran")
    except Exception as exc:   # noqa: BLE001
        msg = str(exc).strip().splitlines()[0]
        check("BiasAdd refuses a bias of the wrong length",
              "the bias has 8 elements but the input's last dimension is 16"
              in msg, msg[:170])

    # The registry's dtype constraint: float64 cannot select the VC kernel, so
    # eager places the op on the CPU instead.
    with tf.device(DEVICE):
        t = tf.constant(np.ones(8, np.float64))
        prod = tf.multiply(t, t)
    check("float64 Mul falls back to the CPU (no VC kernel)",
          not prod.device.endswith(DEVICE), prod.device)

    # MatMul refuses a transposed input at construction.
    try:
        with tf.device(DEVICE):
            x = tf.constant(np.ones((4, 8), np.float32))
            w = tf.constant(np.ones((8, 4), np.float32))
            tf.matmul(x, w, transpose_a=True).numpy()
        check("MatMul refuses transpose_a", False, "it ran")
    except Exception as exc:   # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        check("MatMul refuses transpose_a",
              "transpose_a is not supported" in msg, msg[:170])

    # BiasAdd refuses a layout whose bias is not on the last axis.
    try:
        with tf.device(DEVICE):
            x = tf.constant(np.ones((1, 4, 2, 2), np.float32))
            bias = tf.constant(np.ones(4, np.float32))
            tf.raw_ops.BiasAdd(value=x, bias=bias,
                               data_format="NCHW").numpy()
        check("BiasAdd refuses data_format=NCHW", False, "it ran")
    except Exception as exc:   # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        check("BiasAdd refuses data_format=NCHW",
              'data_format="NCHW" is not supported' in msg, msg[:170])


# ---------------------------------------------------------------- kernels --

def _operands(op, n, rng):
    """Inputs that keep every op inside its measured domain."""
    if op in ("Div", "RealDiv"):
        a = rng.standard_normal(n).astype(np.float32)
        b = (rng.uniform(0.5, 2.0, n) *
             rng.choice([-1.0, 1.0], n)).astype(np.float32)
        return a, b
    if op == "Exp":
        return rng.uniform(-20.0, 20.0, n).astype(np.float32), None
    if op == "Log":
        return rng.uniform(1e-3, 100.0, n).astype(np.float32), None
    if op == "Sqrt":
        return rng.uniform(0.0, 100.0, n).astype(np.float32), None
    a = rng.standard_normal(n).astype(np.float32)
    if op in ("Add", "AddV2", "Mul", "Sub", "Maximum", "Minimum"):
        return a, rng.standard_normal(n).astype(np.float32)
    return a, None


# (op, callable, absolute/relative tolerance, launches per call).  The
# tolerances are the measured bounds in docs/ACCURACY.md with a small margin:
# the FPU truncates, so add, mul and sub agree to ~1 ulp rather than bit for
# bit; Neg, Clip and the max/min selects are exact.
#
# The launch count is the engine's group size, not an op property: the bodies
# in kops.HEAVY are generated at 4 vectors per group (64 floats per QPU)
# because their 16-vector copies do not fit the 2048-instruction code slot, so
# 12 QPUs cover 768 floats per launch and 1024 elements need two.  Tanh, Exp,
# Log and Sqrt are those bodies; Exp would fit at 16 vectors (1815 words) but
# the engine's unary table keeps it on the same verified 4-vector path.
KERNELS = [
    ("Add", lambda a, b: tf.raw_ops.Add(x=a, y=b), 1e-6, 1e-6, 1),
    ("AddV2", lambda a, b: tf.raw_ops.AddV2(x=a, y=b), 1e-6, 1e-6, 1),
    ("Mul", tf.multiply, 1e-6, 1e-6, 1),
    ("Sub", tf.subtract, 1e-6, 1e-6, 1),
    ("Div", lambda a, b: tf.raw_ops.Div(x=a, y=b), 1e-6, 1e-6, 1),
    ("RealDiv", tf.divide, 1e-6, 1e-6, 1),
    ("Maximum", tf.maximum, 0.0, 0.0, 1),
    ("Minimum", tf.minimum, 0.0, 0.0, 1),
    ("Relu", tf.nn.relu, 0.0, 0.0, 1),
    ("Relu6", tf.nn.relu6, 0.0, 0.0, 1),
    ("Sigmoid", tf.nn.sigmoid, 1e-6, 1e-6, 1),
    ("Tanh", tf.math.tanh, 2e-6, 1e-6, 2),
    ("Neg", tf.math.negative, 0.0, 0.0, 1),
    ("Exp", tf.exp, 1e-6, 2e-5, 2),
    ("Log", tf.math.log, 1e-6, 1e-6, 2),
    ("Sqrt", tf.sqrt, 1e-6, 1e-6, 2),
]


def _compare(name, got, want, atol, rtol, tag, launches=None, expected=None):
    d = np.abs(got - want)
    tol = atol + rtol * np.abs(want)
    over = int(np.sum(d > tol))
    note = ""
    ok = over == 0
    if expected is not None and launches is not None:
        note = ", launches=%d (expected %d)" % (launches, expected)
        ok = ok and launches == expected
    check("%s %s on %s" % (name, tag, DEVICE), ok,
          "max|d|=%.3g over=%d/%d (tol %g abs + %g rel)%s" % (
              float(np.max(d)) if d.size else 0.0, over, d.size, atol, rtol,
              note))


def _timed(call):
    """Run `call`, then return (result, launches used by this op)."""
    before = engine_launches()
    out = call()
    after = engine_launches()
    used = None if before is None or after is None else after - before
    return out, used


def stage_kernels():
    if not ALLOW_QPU:
        print("SKIP  --kernels dispatches to the QPU; set VC4_TF_ALLOW_QPU=1 "
              "and run it as the only GPU user")
        return

    for n in (1024, 1000):
        rng = np.random.default_rng(n)
        for op, call, atol, rtol, launches in KERNELS:
            a, b = _operands(op, n, rng)
            with tf.device("/cpu:0"):
                ca = tf.constant(a, dtype=tf.float32)
                want = call(ca) if b is None else call(ca, tf.constant(b))
                want = want.numpy()
            with tf.device(DEVICE):
                ta = tf.constant(a, dtype=tf.float32)
                if b is None:
                    got, used = _timed(lambda: call(ta).numpy())
                else:
                    tb = tf.constant(b, dtype=tf.float32)
                    got, used = _timed(lambda: call(ta, tb).numpy())
            _compare(op, got, want, atol, rtol, "n=%d" % n, used, launches)

    # MatMul: 32x32x32 (1024 outputs) and a shape where M, N and K are all
    # non-multiples of the engine's 8/64/16 padding units.
    rng = np.random.default_rng(11)
    for (m, k, n) in ((32, 32, 32), (30, 34, 26)):
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        with tf.device("/cpu:0"):
            want = tf.matmul(tf.constant(a), tf.constant(b)).numpy()
        with tf.device(DEVICE):
            got, used = _timed(lambda: tf.matmul(tf.constant(a),
                                                 tf.constant(b)).numpy())
        _compare("MatMul", got, want, 1e-4, 1e-4, "%dx%dx%d" % (m, k, n),
                 used, 2)

    # BiasAdd: one row (the bias row and x are the same shape, so the add runs
    # without an expansion) and seven rows of 1000 (a copy with a tail).
    for shape, launches in (((1, 1024), 1), ((7, 1000), 5)):
        a = rng.standard_normal(shape).astype(np.float32)
        bias = rng.standard_normal(shape[-1]).astype(np.float32)
        with tf.device("/cpu:0"):
            want = tf.nn.bias_add(tf.constant(a), tf.constant(bias)).numpy()
        with tf.device(DEVICE):
            got, used = _timed(lambda: tf.nn.bias_add(
                tf.constant(a), tf.constant(bias)).numpy())
        _compare("BiasAdd", got, want, 1e-6, 1e-6,
                 "%dx%d" % (shape[0], shape[1]), used, launches)


def stage_graph():
    """matmul -> bias -> relu, all on the device."""
    rng = np.random.default_rng(23)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    w = rng.standard_normal((32, 16)).astype(np.float32)
    bias = rng.standard_normal(16).astype(np.float32)

    if not ALLOW_QPU:
        if not HOST_ARENA:
            print("SKIP  --graph needs VC4_TF_ALLOW_QPU=1 (or "
                  "VC4_TF_ARENA=host for the placement-only check)")
            return
        try:
            with tf.device(DEVICE):
                h = tf.matmul(tf.constant(x), tf.constant(w))
                y = tf.nn.bias_add(h, tf.constant(bias))
                tf.nn.relu(y).numpy()
            check("graph placed on %s" % DEVICE, False,
                  "the graph ran without an engine")
        except Exception as exc:   # noqa: BLE001
            msg = str(exc).strip().splitlines()[0]
            check("graph lands on %s" % DEVICE,
                  "tf-vc4 MatMul" in msg and "engine arena" in msg,
                  msg[:170])
        return

    with tf.device("/cpu:0"):
        cx = tf.constant(x)
        want = tf.nn.relu(tf.nn.bias_add(tf.matmul(cx, tf.constant(w)),
                                         tf.constant(bias))).numpy()
    with tf.device(DEVICE):
        gx = tf.constant(x)
        before = engine_launches()
        h = tf.matmul(gx, tf.constant(w))
        y = tf.nn.bias_add(h, tf.constant(bias))
        z = tf.nn.relu(y)
        got = z.numpy()
        after = engine_launches()
    used = None if before is None or after is None else after - before
    check("matmul -> bias -> relu placed on %s" % DEVICE,
          all(t.device.endswith(DEVICE) for t in (h, y, z)),
          "%s / %s / %s" % (h.device, y.device, z.device))
    _compare("graph", got, want, 1e-4, 1e-4, "8x32x16", used, 5)


def stage_add():
    """The original single-op silicon check, kept for the README's command."""
    if not ALLOW_QPU:
        print("SKIP  --add dispatches to the QPU; set VC4_TF_ALLOW_QPU=1 and "
              "run it as the only GPU user")
        return
    for n in (1024, 1000):
        rng = np.random.default_rng(n)
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        with tf.device("/cpu:0"):
            want = tf.add(tf.constant(a), tf.constant(b)).numpy()
        with tf.device(DEVICE):
            got, used = _timed(lambda: tf.add(tf.constant(a),
                                             tf.constant(b)).numpy())
        _compare("tf.add", got, want, 1e-6, 1e-6, "n=%d" % n, used, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="exercise the allocator and H2D/D2H")
    ap.add_argument("--dispatch", action="store_true",
                    help="reach every kernel with a host arena and check the "
                         "refusals")
    ap.add_argument("--kernels", action="store_true",
                    help="run every kernel on the QPU (needs "
                         "VC4_TF_ALLOW_QPU=1)")
    ap.add_argument("--graph", action="store_true",
                    help="run matmul -> bias -> relu (needs "
                         "VC4_TF_ALLOW_QPU=1)")
    ap.add_argument("--add", action="store_true",
                    help="run tf.add on the QPU (needs VC4_TF_ALLOW_QPU=1)")
    ap.add_argument("--all", action="store_true",
                    help="all of the above")
    args = ap.parse_args()

    stage_register()
    if args.copy or args.all:
        stage_copy()
    if args.dispatch or args.all:
        stage_dispatch()
    if args.kernels or args.all:
        stage_kernels()
    if args.graph or args.all:
        stage_graph()
    if args.add or args.all:
        stage_add()

    s = engine_stats()
    if s:
        print("engine counters: launches=%d kernel_ms=%.3f pack_ms=%.3f "
              "flops=%.3g" % (int(s["launches"]), s["kernel_ms"],
                              s["pack_ms"], s["flops"]))

    print()
    if FAILED:
        print("%d check(s) failed: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
