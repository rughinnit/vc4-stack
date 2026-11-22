#!/usr/bin/env python3
"""test_pool_alias.py - a view must not free the tensor it is a view of.

Reshape/Flatten/Squeeze and the 2-D view a batch matmul is split into do not
copy: their Tensor carries the producer's Buf.  The pool's free list is keyed
by (offset, aligned bytes), so a view whose last consumer passes has the same
key as the base tensor and used to free the block *under* it.  The next
allocation then wrote over a tensor that was still live; in TinyStories-1M that
was the attention input, and the value matmul returned the causal mask
(docs/TINY-LLM.md 5.2).

The graph here is the smallest shape of that bug, with the three ingredients
that matter:

  * one tensor with two consumers through a same-size view (Reshape),
  * a constant of the same shape, so the reuse lands on a plausible value
    rather than an obvious NaN,
  * a use of the base tensor after the view's last consumer, so a block freed
    with the view is read back as the wrong tensor.

It checks two things: the result against the ONNX reference evaluator, and a
structural invariant - no pool allocation may overlap a tensor that is still in
the runner's live set.  The second catches the defect even when the pool's
choice of offset happens to miss the live bytes.

usage: test_pool_alias.py     (needs vc4d running, like the other GPU tests)
"""
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import vc4infer
from vc4infer import _onnxrun

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def build_model(path):
    """x -> Reshape(4) twice-consumed, then x used again after the view dies.

    t = Reshape(x)          view of x, same bytes
    p = Add(t, c)           consumer 1
    k = Mul(t, c2)          consumer 2 (last use of the view)
    q = Add(x, m)           allocates here: this reused x's block before the fix
    s = Mul(q, x)           reads x after the view was released
    r = Reshape(p)          a second view, of p
    out = Add(s, r)
    """
    rng = np.random.default_rng(0)
    c = rng.standard_normal(4).astype(np.float32)
    c2 = rng.standard_normal(4).astype(np.float32)
    m = rng.standard_normal((2, 2)).astype(np.float32)
    nodes = [
        helper.make_node("Reshape", ["x", "shape4"], ["t"], name="view_x"),
        helper.make_node("Add", ["t", "c"], ["p"], name="add_view1"),
        helper.make_node("Mul", ["t", "c2"], ["k"], name="mul_view2"),
        helper.make_node("Add", ["x", "m"], ["q"], name="add_mask"),
        helper.make_node("Mul", ["q", "x"], ["s"], name="mul_base"),
        helper.make_node("Reshape", ["p", "shape22"], ["r"], name="view_p"),
        helper.make_node("Add", ["s", "r"], ["out"], name="out"),
    ]
    graph = helper.make_graph(
        nodes, "view_alias",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2])],
        initializer=[numpy_helper.from_array(c, "c"),
                     numpy_helper.from_array(c2, "c2"),
                     numpy_helper.from_array(m, "m"),
                     numpy_helper.from_array(np.array([4], np.int64),
                                             "shape4"),
                     numpy_helper.from_array(np.array([2, 2], np.int64),
                                             "shape22")])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    tmp = tempfile.mkdtemp(prefix="vc4-alias-",
                           dir=os.environ.get("TMPDIR") or None)
    path = os.path.join(tmp, "view_alias.onnx")
    build_model(path)

    # ---- structural check: no allocation over a live tensor ---------------
    RUNNER = [None]
    overlaps = []
    orig_alloc = _onnxrun.Pool.alloc
    align = _onnxrun._align

    def alloc(self, nbytes):
        b = orig_alloc(self, nbytes)
        r = RUNNER[0]
        if r is not None:
            lo = int(b.off)
            hi = lo + align(int(nbytes))
            node = getattr(r, "_node_ctx", None)
            for name, v in list(r.values.items()):
                if getattr(v, "has_buf", True) is False:
                    continue
                vb = getattr(v, "buf", None)
                if vb is None:
                    continue
                vlo, vhi = int(vb.off), int(vb.off) + align(int(vb.bytes))
                if lo < vhi and vlo < hi:
                    overlaps.append((node, name, lo, int(nbytes), vlo, vhi))
        return b

    _onnxrun.Pool.alloc = alloc
    try:
        eng = vc4infer.Engine(path)
        runner = eng._runner
        RUNNER[0] = runner
        x = np.array([[0.5, -1.25], [2.0, 0.75]], np.float32)
        got = np.asarray(list(eng.run({"x": x}).values())[0], np.float32)
    finally:
        _onnxrun.Pool.alloc = orig_alloc

    ref = np.asarray(onnx.reference.ReferenceEvaluator(path).run(
        None, {"x": x})[0], np.float32)
    diff = float(np.abs(got - ref).max())
    print("  engine %s  reference %s  max|diff| %.3g"
          % (got.ravel()[:4], ref.ravel()[:4], diff))
    check(diff < 1e-4, "the graph matches the reference (max %.3g)" % diff)
    check(not overlaps,
          "no allocation overlapped a live tensor (%d overlap(s))"
          % len(overlaps))
    for o in overlaps[:6]:
        node, name, lo, n, vlo, vhi = o
        print("        node %s: %d B at %d overlaps live %r [%d,%d)"
              % (node, n, lo, name, vlo, vhi))

    # the run must not leak the pool: a second run on the same engine is the
    # same answer (the release path also runs at run start for outputs)
    RUNNER[0] = None
    got2 = None
    try:
        got2 = np.asarray(list(eng.run({"x": x}).values())[0], np.float32)
    except Exception as e:                                    # pragma: no cover
        print("FAIL: second run raised %s" % e)
    check(got2 is not None and np.abs(got2 - ref).max() < 1e-4,
          "a second run gives the same answer")
    eng.close()

    if FAILS:
        print("FAIL: %d check(s)" % len(FAILS))
        return 1
    print("PASS: view liveness (no allocation over a live tensor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
