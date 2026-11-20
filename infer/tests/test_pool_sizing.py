#!/usr/bin/env python3
"""test_pool_sizing.py - G10: the graph pool is sized from the model, not from
an 8 MiB floor.

Before the fix ``Runner.__init__`` asked for

    need = max(eng.arena_bytes - VC4_POOL_RESERVE_MB, 8 MiB)

so with a client arena of 24 MiB and the default 24 MiB reserve the pool was
*always* 8 MiB, and the run died in the middle with

    graph pool exhausted: 13107200 B needed, ... of 8388608 B

which reads like "your model is too big" even though the arena had 16 MiB
free.  After the fix the pool is sized from the model's own peak (float
weights + the longest-lived activations + the packed conv panels), clamped to
what the arena can give beside the engine's scratch, and a model that cannot
fit is refused at plan time with the numbers.

This test builds a two-node model whose largest activation is ~13 MiB - larger
than the old floor, much smaller than the test arena - and checks:

  small arena (16 MiB), model needs ~14 MiB
      -> construction raises UnsupportedNode naming both the need and the
         arena (the honest refusal).  Before the fix: no error, an 8 MiB pool.
  roomy arena, same model
      -> the pool is at least the size of the biggest tensor, and is *not*
         8388608 B.  Before the fix: exactly 8388608 B.

No GPU work: the test only allocates the arena and plans the graph.

usage: test_pool_sizing.py [--arena-mb N] [--tight-mb N]
       (defaults 24 and 8; needs vc4d running or root for the direct path)
"""
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import vc4infer
from vc4infer import _runtime
from vc4infer._errors import UnsupportedNode

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def build_model(path):
    """[1,16,160,160] -> Conv 1x1 -> [1,128,160,160] (13.1 MB of activation)."""
    w = np.zeros((128, 16, 1, 1), np.float32)
    for i in range(16):
        w[i, i, 0, 0] = 1.0
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["y"], name="conv0",
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                         strides=[1, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="relu0"),
    ]
    graph = helper.make_graph(
        nodes, "g10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                       [1, 16, 160, 160])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT,
                                       [1, 128, 160, 160])],
        initializer=[numpy_helper.from_array(w, "w")])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return 160 * 160 * 128 * 4          # the biggest activation, bytes


def open_engine(arena_mb):
    old = os.environ.get("VC4INFER_ARENA_MB")
    os.environ["VC4INFER_ARENA_MB"] = str(arena_mb)
    try:
        return _runtime.open_engine(arena_mb=arena_mb)
    finally:
        if old is None:
            os.environ.pop("VC4INFER_ARENA_MB", None)
        else:
            os.environ["VC4INFER_ARENA_MB"] = old


def close(eng):
    try:
        eng.lib.vc4_shutdown(eng.ctx)
    except Exception:
        pass


def main(argv):
    roomy = 24
    tight = 8
    if "--arena-mb" in argv:
        roomy = int(argv[argv.index("--arena-mb") + 1])
    if "--tight-mb" in argv:
        tight = int(argv[argv.index("--tight-mb") + 1])

    tmp = tempfile.mkdtemp(prefix="vc4-g10-", dir=os.environ.get("TMPDIR") or None)
    model = os.path.join(tmp, "g10.onnx")
    big = build_model(model)
    print("model: %s (largest activation %.1f MiB)" % (model, big / 2 ** 20))

    # ---- 1. an arena that cannot hold the model: refuse with the numbers ---
    try:
        eng = open_engine(tight)
    except Exception as e:                                  # pragma: no cover
        print("FAIL: cannot open a %d MiB engine: %s" % (tight, e))
        return 1
    got = None
    try:
        _runtime.Runner(model, eng)
    except UnsupportedNode as e:
        got = str(e)
    finally:
        close(eng)
    ok = got is not None and "needs at least" in got and \
        ("%d.0 MiB" % tight) in got
    check(ok, "a %d MiB arena is refused at plan time with both numbers: %s"
          % (tight, (got or "no error - the pool was floored instead")))

    # ---- 2. an arena with room: the pool is sized from the model ----------
    try:
        eng = open_engine(roomy)
    except Exception as e:
        print("FAIL: cannot open a %d MiB engine: %s" % (roomy, e))
        return 1
    pool = None
    try:
        r = _runtime.Runner(model, eng)
        pool = r.pool
        plan = getattr(r, "pool_plan", None) or {}
        size = pool.size
        print("        pool %.2f MiB in %d segment(s), estimate %.2f MiB, "
              "arena %.1f MiB" % (size / 2 ** 20, getattr(pool, "segments", 1),
                                  plan.get("estimate_bytes", 0) / 2 ** 20,
                                  plan.get("arena_bytes", 0) / 2 ** 20))
        check(size != 8 << 20,
              "the pool is not the old 8 MiB floor (%d B)" % size)
        check(size >= big,
              "the pool holds the biggest tensor: %.2f MiB >= %.2f MiB"
              % (size / 2 ** 20, big / 2 ** 20))
        check(plan.get("estimate_bytes", 0) > (8 << 20),
              "the plan reports the model's peak (%.2f MiB) rather than a "
              "constant" % (plan.get("estimate_bytes", 0) / 2 ** 20))
    except UnsupportedNode as e:
        check(False, "a %d MiB arena should be enough for a %.1f MiB model: %s"
              % (roomy, big / 2 ** 20, e))
    finally:
        close(eng)

    print("G10 PASS" if not FAILS else "G10 FAIL (%d)" % len(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
