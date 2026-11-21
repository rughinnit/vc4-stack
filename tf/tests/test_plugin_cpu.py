#!/usr/bin/env python3
"""CPU-side checks for the TensorFlow PluggableDevice plugin.

Stages, each opt-in:

  (default)   the library loads, the platform registers, the device appears in
              tf.config.list_physical_devices / list_logical_devices, and the
              Add/AddV2 kernels are in TensorFlow's kernel registry for this
              device type.  No device memory is touched, so nothing reaches
              the QPUs or the engine arena.
  --copy      the allocator and the H2D/D2H paths: a constant placed on the
              device and read back.  Needs the engine arena (8 MiB of CMA per
              process) or VC4_TF_ARENA=host for a wiring-only run.
  --dispatch  tf.add on the device with VC4_TF_ARENA=host: the kernel is
              reached, then refuses (no engine).  This proves eager placement
              and kernel registration without a QPU launch.
  --add       the real thing: tf.add on the QPU.  Refuses to run unless
              VC4_TF_ALLOW_QPU=1, because it dispatches to the VPUs.

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

DEVICE_TYPE = os.environ.get("VC4_TF_DEVICE_TYPE", "VC")
DEVICE = "/device:%s:0" % DEVICE_TYPE
LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "build", "libtf_vc4_plugin.so")
HOST_ARENA = os.environ.get("VC4_TF_ARENA", "").lower() == "host"

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


def names(devices):
    return [d.name for d in devices]


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

    for op in ("Add", "AddV2"):
        kl = tf_kernels.get_registered_kernels_for_op(op)
        mine = [k for k in kl.kernel if k.device_type == DEVICE_TYPE]
        types = sorted({t for k in mine for c in k.constraint
                        if c.name == "T" for t in c.allowed_values.list.type})
        all_dev = sorted({k.device_type for k in kl.kernel})
        check("kernel %s registered on %s" % (op, DEVICE_TYPE), bool(mine),
              "dtypes=%s (registry has %s)" % (types, all_dev) if mine
              else "not in the registry (has %s)" % all_dev)

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


def stage_dispatch():
    """Reach the Add kernel through eager without letting it launch."""
    if not HOST_ARENA:
        print("SKIP  --dispatch needs VC4_TF_ARENA=host (it checks the "
              "dispatch path, not arithmetic)")
        return
    a = np.arange(64, dtype=np.float32)
    b = np.full(64, 0.5, dtype=np.float32)
    try:
        with tf.device(DEVICE):
            z = tf.add(tf.constant(a), tf.constant(b))
            z.numpy()
    except Exception as exc:   # noqa: BLE001 - the refusal is the result
        msg = str(exc)
        reached = ("VC4_TF_ARENA=host" in msg or
                   "kernel" in msg.lower() and "VC" in msg)
        check("Add dispatched to the %s kernel" % DEVICE_TYPE, reached,
              msg.strip().splitlines()[0][:180])
        return
    check("Add dispatched to the %s kernel" % DEVICE_TYPE, False,
          "tf.add ran without an engine; unexpected with a host arena")


def stage_add():
    if os.environ.get("VC4_TF_ALLOW_QPU") != "1":
        print("SKIP  --add dispatches to the QPU; set VC4_TF_ALLOW_QPU=1 and "
              "run it as the only GPU user")
        return
    for n in (1024, 1000):
        rng = np.random.default_rng(n)
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        want = a + b
        with tf.device(DEVICE):
            x = tf.constant(a, dtype=tf.float32)
            y = tf.constant(b, dtype=tf.float32)
            z = tf.add(x, y)
            got = z.numpy()
        d = float(np.max(np.abs(got - want)))
        check("tf.add n=%d on %s" % (n, DEVICE), d == 0.0,
              "max|d|=%g (expected 0: the QPU add is exact IEEE fp32)" % d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="exercise the allocator and H2D/D2H")
    ap.add_argument("--dispatch", action="store_true",
                    help="reach the Add kernel with a host arena")
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
    if args.add or args.all:
        stage_add()

    print()
    if FAILED:
        print("%d check(s) failed: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
