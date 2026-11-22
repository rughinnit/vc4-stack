#!/usr/bin/env python3
"""test_optimizer.py -- phase A/B gates without the device.

The optimiser must be reviewable before it touches the QPUs.  Everything here
runs on the ARM (or on a machine with no VideoCore at all):

  1. the IR round-trip: a model -> Graph -> model keeps every node, attribute,
     input and output, and the result passes ``onnx.checker``;
  2. pass 1 (canonicalise): constants fold, dead nodes go, identities go,
     Div/Sub by a constant lower the way the executor dispatches them;
  3. pass 2 (ew): SiLU, HardSwish and scalar-affine pairs become one dispatch,
     each with a provenance record naming what it replaced;
  4. pass 3 (epilogue): a per-channel Add folds into the conv bias (one launch
     fewer), a Relu/Clip(0,6) attaches to the conv (one dispatch fewer, same
     launch count -- the conv kernel has no epilogue);
  5. the launch model: the elementwise formula reproduces every elementwise
     and conv row of the recorded YOLOv8n-160 profile (when the model and
     profile are present);
  6. the executor's fused dispatch, with a fake engine: the graph rewrite is
     what removes the launch, including under VC4_NO_SILU_FUSE=1;
  7. semantics: onnx.reference gives the same answer before and after every
     pass on the synthetic graphs.

No GPU work, no kernel cache, no ctypes call into libvc4infer.  The real gate
(152/152 operators, YOLOv8n parity) still needs silicon; commands are in
docs/OPTIMIZER.md.

usage: TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_optimizer.py
"""
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "infer", "python"))

import vc4infer                                            # noqa: E402
from vc4infer import optimizer as O                        # noqa: E402
from vc4infer._onnxrun import Runner                       # noqa: E402
from vc4infer._binding import (OP_SILU, UOP_AFFINE,         # noqa: E402
                               UOP_HARDSWISH, LAYOUT_ROW_MAJOR, ACT_RELU)

MODEL = os.path.join(REPO, "infer", "packaging", "vc4infer", "data", "models",
                     "yolov8n_160.onnx")
PROFILE = os.path.join(REPO, "infer", "tools", "onnx", "yolo160_bench.json")
FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def tmpdir():
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="opt-", dir=base)


def save(model, name):
    path = os.path.join(tmpdir(), name)
    onnx.save(model, path)
    onnx.checker.check_model(model)
    return path


# ---------------------------------------------------------------- models
def _model(nodes, inputs, outputs, inits=()):
    g = helper.make_graph(nodes, "opt", inputs, outputs, list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    return m


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def mk_silu(shape=(1, 4, 4, 4)):
    x = _vi("X", shape)
    n = [helper.make_node("Sigmoid", ["X"], ["S"]),
         helper.make_node("Mul", ["X", "S"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape)])


def mk_hardswish(shape=(1, 4, 4, 4)):
    x = _vi("X", shape)
    n = [helper.make_node("HardSigmoid", ["X"], ["S"], alpha=0.2, beta=0.5),
         helper.make_node("Mul", ["X", "S"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape)])


def mk_affine(shape=(1, 4, 4, 4)):
    x = _vi("X", shape)
    c1 = numpy_helper.from_array(np.asarray(0.5, np.float32), "c1")
    c2 = numpy_helper.from_array(np.asarray(2.0, np.float32), "c2")
    n = [helper.make_node("Mul", ["X", "c1"], ["M"]),
         helper.make_node("Add", ["M", "c2"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape)], [c1, c2])


def mk_conv_bias(shape=(1, 16, 8, 8)):
    rng = np.random.default_rng(7)
    w = rng.standard_normal((16, 16, 3, 3)).astype(np.float32) * 0.1
    b = rng.standard_normal((16, 1, 1)).astype(np.float32)
    x = _vi("X", shape)
    n = [helper.make_node("Conv", ["X", "W"], ["C"], pads=[1, 1, 1, 1]),
         helper.make_node("Add", ["C", "B"], ["Y"])]
    inits = [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")]
    return _model(n, [x], [_vi("Y", shape)], inits)


def mk_conv_relu(shape=(1, 16, 8, 8)):
    rng = np.random.default_rng(8)
    w = rng.standard_normal((16, 16, 1, 1)).astype(np.float32) * 0.1
    x = _vi("X", shape)
    n = [helper.make_node("Conv", ["X", "W"], ["C"]),
         helper.make_node("Relu", ["C"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape)],
                  [numpy_helper.from_array(w, "W")])


def mk_conv_relu_nchw(shape=(1, 16, 8, 8)):
    """A 1x3 kernel: the executor takes the legacy NCHW path, no epilogue."""
    rng = np.random.default_rng(10)
    w = rng.standard_normal((16, 16, 1, 3)).astype(np.float32) * 0.1
    x = _vi("X", shape)
    n = [helper.make_node("Conv", ["X", "W"], ["C"], pads=[0, 1, 0, 1]),
         helper.make_node("Relu", ["C"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape)],
                  [numpy_helper.from_array(w, "W")])


def mk_conv_clip6(shape=(1, 16, 8, 8)):
    """Clip(0,6) with min/max as inputs (the opset 11+ form)."""
    rng = np.random.default_rng(11)
    w = rng.standard_normal((16, 16, 3, 3)).astype(np.float32) * 0.1
    x = _vi("X", shape)
    n = [helper.make_node("Conv", ["X", "W"], ["C"], pads=[1, 1, 1, 1]),
         helper.make_node("Clip", ["C", "lo", "hi"], ["Y"])]
    inits = [numpy_helper.from_array(w, "W"),
             numpy_helper.from_array(np.asarray(0.0, np.float32), "lo"),
             numpy_helper.from_array(np.asarray(6.0, np.float32), "hi")]
    return _model(n, [x], [_vi("Y", shape)], inits)


def mk_matmul_clip(shape=((2, 4), (4, 8))):
    x = _vi("X", (1, 8, 4))
    w = np.random.default_rng(12).standard_normal((4, 8)).astype(np.float32)
    n = [helper.make_node("MatMul", ["X", "W"], ["C"]),
         helper.make_node("Clip", ["C"], ["Y"], min=0.0, max=6.0)]
    return _model(n, [x], [_vi("Y", (1, 8, 8))],
                  [numpy_helper.from_array(w, "W")])


def mk_conv_silu(shape=(1, 16, 8, 8)):
    rng = np.random.default_rng(14)
    w = rng.standard_normal((16, 16, 3, 3)).astype(np.float32) * 0.1
    x = _vi("X", shape)
    n = [helper.make_node("Conv", ["X", "W"], ["C"], pads=[1, 1, 1, 1]),
         helper.make_node("Sigmoid", ["C"], ["S"]),
         helper.make_node("Mul", ["C", "S"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape)],
                  [numpy_helper.from_array(w, "W")])


def mk_matmul_relu(shape=((4, 8), (8, 8))):
    """2-D MatMul -> Relu: the executor folds it into one launch."""
    x = _vi("X", shape[0])
    w = np.random.default_rng(16).standard_normal(shape[1]).astype(np.float32)
    n = [helper.make_node("MatMul", ["X", "W"], ["C"]),
         helper.make_node("Relu", ["C"], ["Y"])]
    return _model(n, [x], [_vi("Y", shape[0])],
                  [numpy_helper.from_array(w, "W")])


def mk_canonical(shape=(1, 16, 8, 8)):
    """Dead node, identity, no-op reshape/cast/transpose, Div/Sub by 2."""
    x = _vi("X", shape)
    c2 = numpy_helper.from_array(np.asarray(2.0, np.float32), "two")
    c1 = numpy_helper.from_array(np.asarray(1.0, np.float32), "three")
    n = [
        helper.make_node("Relu", ["X"], ["dead"]),              # dead
        helper.make_node("Identity", ["X"], ["id"]),            # identity
        helper.make_node("Reshape", ["id", "shp"], ["rs"]),     # no-op view
        helper.make_node("Cast", ["rs"], ["cs"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["cs"], ["tp"], perm=[0, 1, 2, 3]),
        helper.make_node("Div", ["tp", "two"], ["d"]),
        helper.make_node("Sub", ["d", "three"], ["Y"]),
    ]
    inits = [c2, c1,
             numpy_helper.from_array(np.asarray(shape, np.int64), "shp")]
    return _model(n, [x], [_vi("Y", shape)], inits)


# ------------------------------------------------------------ fake engine
class FakeStats:
    def __init__(self):
        self.launches = 0
        self.pack_ns = 0
        self.kernel_ns = 0
        self.flops = 0


class FakeEngine:
    """The slice of the engine API the runner uses, recording every dispatch.

    No ctypes, no arena mapping: a bytearray at the same offsets, so views and
    pitched NHWC buffers behave.  ``calls`` is the dispatch sequence the test
    asserts on; ``stats.launches`` counts them the way the real engine does.
    """

    def __init__(self, arena=8 << 20):
        self._arena = bytearray(arena)
        self._next = 64
        self.calls = []
        self._stats = FakeStats()

    @property
    def arena_bytes(self):
        return len(self._arena)

    @property
    def stats(self):
        return self._stats

    def alloc(self, nbytes):
        n = (int(nbytes) + 63) // 64 * 64
        if self._next + n > len(self._arena):
            raise vc4infer.Error("fake arena exhausted")
        b = vc4infer.Buf(self._next, int(nbytes))
        self._next += n
        return b

    def upload(self, arr, buf=None):
        if isinstance(arr, np.ndarray):
            raw = np.ascontiguousarray(arr, dtype=np.float32).ravel()
            data = raw.tobytes()
        else:
            data = bytes(arr)
        if buf is None:
            buf = self.alloc(len(data))
        end = int(buf.off) + len(data)
        self._arena[int(buf.off):end] = data
        return buf

    def download(self, buf, nbytes):
        raw = bytes(self._arena[int(buf.off):int(buf.off) + int(nbytes)])
        return np.frombuffer(raw, dtype=np.float32).copy()

    def _call(self, kind, op):
        self.calls.append((kind, int(op)))
        self._stats.launches += 1

    def elementwise(self, op, a, out, n, b=None):
        self._call("elementwise", op)

    def unary(self, op, a, out, n, p0=0.0, p1=0.0, p2=0.0):
        self._call("unary", op)

    def fill2d(self, *a, **k):
        self._call("fill2d", 0)

    def matmul(self, a, b, c, M, N, K, layout=LAYOUT_ROW_MAJOR, act=0):
        self._call("matmul", act)


# ------------------------------------------------------------------ tests
def test_ir_roundtrip():
    print("IR round-trip")
    if not os.path.isfile(MODEL):
        print("  SKIP: %s is not present" % MODEL)
        return
    m = onnx.load(MODEL)
    g = O.Graph.from_model(m)
    back = g.to_model()
    onnx.checker.check_model(back)
    check(len(back.graph.node) == len(m.graph.node),
          "node count preserved (%d)" % len(m.graph.node))
    check(g.histogram() == O.Graph.from_model(back).histogram(),
          "op histogram preserved")
    check([i.name for i in back.graph.input] ==
          [i.name for i in m.graph.input], "inputs preserved")
    check([o.name for o in back.graph.output] ==
          [o.name for o in m.graph.output], "outputs preserved")
    check(sorted(i.name for i in back.graph.initializer) ==
          sorted(i.name for i in m.graph.initializer),
          "initialisers preserved")


def test_canonicalise():
    print("pass 1: canonicalise")
    m = mk_canonical()
    ops_before = [n.op_type for n in m.graph.node]
    res = O.optimise(m, passes="canonicalise")
    onnx.checker.check_model(res.model)
    ops = [n.op_type for n in res.graph.nodes]
    check("Identity" not in ops, "Identity removed")
    check("Reshape" not in ops, "no-op Reshape removed")
    check("Cast" not in ops, "redundant Cast removed")
    check("Transpose" not in ops, "identity Transpose removed")
    check("Relu" not in ops, "dead Relu removed")
    check("Div" not in ops, "Div by 2 lowered to Mul")
    check("Sub" not in ops, "Sub by 1 lowered to Add")
    check(len(ops) == 2, "two nodes left (Mul, Add): %s" % ops)
    # the reciprocal and the negated constant are the executor's own lowering
    vals = {k: np.asarray(v).ravel().tolist()
            for k, v in res.graph.initializers.items()}
    check(any(v == [0.5] for v in vals.values()), "1/2 written as 0.5")
    check(any(v == [-1.0] for v in vals.values()), "-1 written for Sub")
    _same_answer(m, res.model, {"X": np.random.default_rng(1).standard_normal(
        (1, 16, 8, 8)).astype(np.float32)}, "canonicalise")


def test_ew():
    print("pass 2: elementwise fusion")
    # SiLU
    m = mk_silu()
    res = O.optimise(m, passes="ew")
    onnx.checker.check_model(res.model)
    mul = [n for n in res.graph.nodes if n.op_type == "Mul"][0]
    sig = [n for n in res.graph.nodes if n.op_type == "Sigmoid"][0]
    check(mul.rule("silu") is not None, "Mul records rule=silu")
    check(len(mul.fused["replaces"]) == 1 and
          "Sigmoid" in mul.fused["replaces"][0],
          "Mul records the Sigmoid it replaced (%s)"
          % mul.fused["replaces"])
    check(sig.rule("fused") is not None, "Sigmoid records rule=fused")
    check(sig.rule("fused")["into"] == mul.name, "Sigmoid names the fused Mul")
    p0 = O.plan_of(O.Graph.from_model(m), implicit=False)
    p1 = O.plan_of(res.graph, implicit=False)
    check(p1.total < p0.total, "one fewer launch without the implicit SiLU scan "
          "(%d -> %d)" % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(2).standard_normal(
        (1, 4, 4, 4)).astype(np.float32)}, "silu")
    # HardSwish
    m = mk_hardswish()
    res = O.optimise(m, passes="ew")
    mul = [n for n in res.graph.nodes if n.op_type == "Mul"][0]
    check(mul.rule("hardswish") is not None, "Mul records rule=hardswish")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.total == p0.total - 1, "HardSwish pair 2 -> 1 launch (%d -> %d)"
          % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(3).standard_normal(
        (1, 4, 4, 4)).astype(np.float32)}, "hardswish")
    # scalar affine
    m = mk_affine()
    res = O.optimise(m, passes="ew")
    add = [n for n in res.graph.nodes if n.op_type == "Add"][0]
    check(add.rule("affine") is not None, "Add records rule=affine")
    check(add.rule("affine")["alpha"] == 0.5, "alpha recorded")
    check(add.rule("affine")["beta"] == 2.0, "beta recorded")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.total == p0.total - 1, "affine pair 2 -> 1 launch (%d -> %d)"
          % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(4).standard_normal(
        (1, 4, 4, 4)).astype(np.float32)}, "affine")


def test_epilogue():
    print("pass 3: epilogue fusion")
    m = mk_conv_bias()
    res = O.optimise(m, passes="epilogue")
    onnx.checker.check_model(res.model)
    conv = [n for n in res.graph.nodes if n.op_type == "Conv"][0]
    check(len(conv.proto.input) == 3, "conv now has a bias input")
    check(not [n for n in res.graph.nodes if n.op_type == "Add"],
          "the Add node is gone")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.total == p0.total - 1, "conv+bias is one launch fewer (%d -> %d)"
          % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(5).standard_normal(
        (1, 16, 8, 8)).astype(np.float32)}, "conv-bias", atol=2e-5)
    # activation: the conv's GEMM applies it to the value it stores, so the
    # elementwise pass over the output is gone as well as the dispatch
    m = mk_conv_relu()
    res = O.optimise(m, passes="epilogue")
    conv = [n for n in res.graph.nodes if n.op_type == "Conv"][0]
    relu = [n for n in res.graph.nodes if n.op_type == "Relu"][0]
    check(conv.rule("activation") is not None, "conv records the activation")
    check(conv.rule("activation")["act"] == "relu", "activation is relu")
    check(relu.rule("fused") is not None, "Relu records rule=fused")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.dispatches == p0.dispatches - 1,
          "one dispatch fewer (%d -> %d)" % (p0.dispatches, p1.dispatches))
    check(p1.total == p0.total - 1,
          "the epilogue removes the activation launch (%d -> %d)"
          % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(6).standard_normal(
        (1, 16, 8, 8)).astype(np.float32)}, "conv-relu")

    # a path with no epilogue keeps both launches and records nothing
    m = mk_conv_relu_nchw()
    res = O.optimise(m, passes="epilogue")
    conv = [n for n in res.graph.nodes if n.op_type == "Conv"][0]
    check(conv.rule("activation") is None,
          "a conv the kernel cannot fold is left alone")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.total == p0.total, "no epilogue: launches unchanged")
    # a MatMul folds relu and clip(0,6): one free B register is enough for
    # the bound, not for silu's twelve constants
    m = mk_matmul_clip()
    res = O.optimise(m, passes="epilogue")
    mm = [n for n in res.graph.nodes if n.op_type == "MatMul"][0]
    check(mm.rule("activation") is not None and
          mm.rule("activation")["act"] == "relu6",
          "MatMul + Clip(0,6) folds into the GEMM")
    check(O.plan_of(res.graph).total ==
          O.plan_of(O.Graph.from_model(m)).total - 1,
          "MatMul + clip(0,6) is one launch instead of two")
    gm = O.Graph.from_model(m)
    check(not O.epilogue_ok(gm, gm.nodes[0], "silu"),
          "MatMul + silu is not folded (no room for twelve constants)")
    # the conv folds clip(0,6) as well, and the launch goes with it
    m = mk_conv_clip6()
    res = O.optimise(m, passes="epilogue")
    conv = [n for n in res.graph.nodes if n.op_type == "Conv"][0]
    check(conv.rule("activation")["act"] == "relu6",
          "Clip(0,6) is recorded as relu6")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.total == p0.total - 1,
          "conv + clip(0,6) is one launch instead of two (%d -> %d)"
          % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(13).standard_normal(
        (1, 16, 8, 8)).astype(np.float32)}, "conv-clip6")

    # conv -> silu: the ew pass collapses the pair, the epilogue pass folds it
    m = mk_conv_silu()
    res = O.optimise(m, passes="ew,epilogue")
    conv = [n for n in res.graph.nodes if n.op_type == "Conv"][0]
    check(conv.rule("activation") is not None and
          conv.rule("activation")["act"] == "silu",
          "Conv -> Mul(x, Sigmoid(x)) folds into the conv as silu")
    mul = [n for n in res.graph.nodes if n.op_type == "Mul"][0]
    check(mul.rule("fused") is not None, "the Mul records rule=fused")
    p0 = O.plan_of(O.Graph.from_model(m))
    p1 = O.plan_of(res.graph)
    check(p1.total == p0.total - 1,
          "conv + silu is one launch instead of two (%d -> %d)"
          % (p0.total, p1.total))
    _same_answer(m, res.model, {"X": np.random.default_rng(15).standard_normal(
        (1, 16, 8, 8)).astype(np.float32)}, "conv-silu", atol=2e-6)

    # the epilogue pass alone cannot see a SiLU the ew pass did not collapse
    res = O.optimise(m, passes="epilogue")
    conv = [n for n in res.graph.nodes if n.op_type == "Conv"][0]
    check(conv.rule("activation") is None,
          "epilogue without ew: the conv is left alone")


def test_plan_model():
    print("launch model")
    check(O.elementwise_launches(0) == 0, "0 floats -> 0 launches")
    check(O.elementwise_launches(256) == 1, "256 floats -> 1 launch")
    check(O.elementwise_launches(256 * 12) == 1, "12 groups fit one launch")
    check(O.elementwise_launches(256 * 13) == 2, "13 groups need two")
    check(O.stored_floats((1, 16, 40, 40), ("nhwc", 48)) == 40 * 48 * 16,
          "NHWC pitch is part of the stored extent")
    if not (os.path.isfile(MODEL) and os.path.isfile(PROFILE)):
        print("  SKIP: model or recorded profile not present")
        return
    import json
    m = onnx.load(MODEL)
    plan = O.plan_of(O.Graph.from_model(m))
    with open(PROFILE) as f:
        prof = json.load(f)["profile"]
    checked, exact_bad, copy_bad = O.compare_profile(plan, prof)
    check(checked >= 200, "compared %d recorded nodes" % checked)
    check(not exact_bad, "elementwise/conv rows match the recorded profile "
          "(%d mismatches)" % len(exact_bad))
    print("  note: %d copy-class rows differ; the profile predates the QPU "
          "layout copies" % len(copy_bad))
    # the passes' plan-level effect on the corpus: 57 conv -> silu pairs,
    # 467 launches
    res = O.optimise(m, passes="all")
    p1 = O.plan_of(res.graph)
    check(p1.total == 285,
          "all passes: 752 -> 285 launches (got %d)" % p1.total)
    check(p1.exact == 212, "the exact classes are 212 of them (got %d)"
          % p1.exact)
    check(p1.dispatches == plan.dispatches - 68,
          "11 metadata + 57 activation dispatches fewer (%d -> %d)"
          % (plan.dispatches, p1.dispatches))
    check(p1.nodes == plan.nodes - 11, "11 nodes fewer in the graph")
    # and the switches bisect it: without the epilogue, or without the ew
    # pass that records the SiLU pair, nothing is folded
    for spec in ("all,-epilogue", "all,-ew"):
        p = O.plan_of(O.optimise(m, passes=spec).graph)
        check(p.total == plan.total,
              "%s keeps the launch count (%d)" % (spec, p.total))


def test_dispatch_fake_engine():
    print("fused dispatch (fake engine)")
    saved = os.environ.get("VC4_NO_SILU_FUSE")
    try:
        # baseline with the executor's implicit rewrite switched off
        os.environ["VC4_NO_SILU_FUSE"] = "1"
        m = mk_silu()
        path = save(m, "silu.onnx")
        eng = FakeEngine()
        r = Runner(path, eng, passes="none")
        x = np.random.default_rng(9).standard_normal((1, 4, 4, 4))
        r.run({"X": x.astype(np.float32)})
        base = list(eng.calls)
        check(len(base) == 2, "no pass, no implicit scan: 2 dispatches (%s)"
              % (base,))
        # with the pass, the same graph is one launch even with the scan off
        eng = FakeEngine()
        r = Runner(path, eng, passes="ew")
        r.run({"X": x.astype(np.float32)})
        check(eng.calls == [("elementwise", OP_SILU)],
              "ew pass fuses to one OP_SILU launch (%s)" % (eng.calls,))
    finally:
        if saved is None:
            os.environ.pop("VC4_NO_SILU_FUSE", None)
        else:
            os.environ["VC4_NO_SILU_FUSE"] = saved
    # hardswish: one HARDSWISH unary instead of HARDSIGMOID + MUL
    m = mk_hardswish()
    path = save(m, "hardswish.onnx")
    eng = FakeEngine()
    Runner(path, eng, passes="ew").run(
        {"X": np.random.default_rng(10).standard_normal((1, 4, 4, 4))
         .astype(np.float32)})
    check(eng.calls == [("unary", UOP_HARDSWISH)],
          "HardSwish pair is one unary (%s)" % (eng.calls,))
    # affine: one AFFINE unary instead of MUL + ADD
    m = mk_affine()
    path = save(m, "affine.onnx")
    eng = FakeEngine()
    Runner(path, eng, passes="ew").run(
        {"X": np.random.default_rng(11).standard_normal((1, 4, 4, 4))
         .astype(np.float32)})
    check(eng.calls == [("unary", UOP_AFFINE)],
          "affine pair is one unary (%s)" % (eng.calls,))


def test_runner_markers():
    print("executor marker tables")
    m = mk_silu()
    res = O.optimise(m, passes="ew")
    r = object.__new__(Runner)
    r.graph = res.graph.to_model().graph
    r._find_silu()
    r._collect_fused()
    check(len(r._silu_src) == 1, "one SiLU Mul recorded")
    check(len(r._skip) == 1, "the Sigmoid is skipped")
    check(r._skip_reason, "the skip has a reason: %s"
          % list(r._skip_reason.values())[0])
    # a fused activation moves the activation's output name onto the conv's
    # buffer, or the consumer (and the graph output) would read nothing
    m = mk_conv_relu()
    res = O.optimise(m, passes="epilogue")
    r = object.__new__(Runner)
    r.graph = res.graph.to_model().graph
    r._find_silu()
    r._collect_fused()
    check(len(r._alias) == 1, "one activation alias recorded")
    idx, pair = next(iter(r._alias.items()))
    check(pair == ("Y", "C"), "alias is Relu output -> conv output (%s)"
          % (pair,))
    marker = object()
    r.values = {"C": marker}
    r._apply_alias(idx)
    check(r.values.get("Y") is marker and "C" not in r.values,
          "the output name owns the produced buffer")
    # coverage shows the fusion and the dispatch count
    from vc4infer import coverage
    rep = coverage.analyse_model(res.model)
    check(rep["dispatches"] == 1, "coverage reports 1 dispatch for %d nodes"
          % rep["total"])
    check(any(n["fused"] for n in rep["nodes"]),
          "coverage carries the provenance record")


def test_dispatch_matmul_fold():
    print("matmul epilogue dispatch (fake engine)")
    m = mk_matmul_relu()
    path = save(m, "mmrelu.onnx")
    eng = FakeEngine()
    r = Runner(path, eng, passes="epilogue")
    x = np.random.default_rng(17).standard_normal((4, 8)).astype(np.float32)
    r.run({"X": x})
    check(eng.calls == [("matmul", ACT_RELU)],
          "one matmul call carrying the activation (%s)" % (eng.calls,))


def test_resolve():
    print("pass switches")
    check([p.key for p in O.resolve_passes("")] == [], "default is no passes")
    check([p.key for p in O.resolve_passes("all")] ==
          ["canonicalise", "ew", "epilogue"], "all = pipeline order")
    check([p.key for p in O.resolve_passes("all,-ew")] ==
          ["canonicalise", "epilogue"], "-ew disables one")
    check([p.key for p in O.resolve_passes("epilogue,canonicalise")] ==
          ["canonicalise", "epilogue"], "explicit list keeps pipeline order")
    try:
        O.resolve_passes("eww")
        check(False, "unknown pass name raises")
    except ValueError:
        check(True, "unknown pass name raises")


def _same_answer(m0, m1, feeds, what, atol=1e-6):
    """onnx.reference must agree before and after the rewrite."""
    a = ReferenceEvaluator(m0).run(None, feeds)
    b = ReferenceEvaluator(m1).run(None, feeds)
    ok = all(x.shape == y.shape and np.allclose(x, y, atol=atol, rtol=1e-6)
             for x, y in zip(a, b))
    d = max((float(np.max(np.abs(x - y))) for x, y in zip(a, b)),
            default=0.0)
    check(ok, "%s: same answer (max abs %.2e)" % (what, d))


def test_opmatrix_sweep(directory):
    """Every saved op_matrix case, optimised, against its recorded oracle.

    The device part of the operator sweep cannot run here, but the graph part
    can: build the optimised graph, check it, evaluate it with the same
    reference evaluator, and compare with the oracle the sweep recorded.  That
    is 226 operator graphs x every pass on one CPU, which is what catches a
    rewrite that is wrong for an attribute form YOLO does not use.
    """
    print("op_matrix cases: optimise, check, compare to the recorded oracle")
    if not directory or not os.path.isdir(directory):
        print("  SKIP: no --sweep directory (run op_matrix.py --skip-gpu)")
        return
    import glob
    import pickle
    import signal

    class _Slow(Exception):
        pass

    def _alarm(signum, frame):
        raise _Slow()

    old = signal.signal(signal.SIGALRM, _alarm)
    files = sorted(glob.glob(os.path.join(directory, "*.pkl")))
    checked = failed = slow = gap = 0
    try:
        for path in files:
            with open(path, "rb") as f:
                case = pickle.load(f)
            model = onnx.load(case["path"])
            try:
                signal.setitimer(signal.ITIMER_REAL, 5.0)
                res = O.optimise(model, passes="all")
                onnx.checker.check_model(res.model)
            except _Slow:
                slow += 1
                continue
            except Exception as e:                   # noqa: BLE001
                signal.setitimer(signal.ITIMER_REAL, 0)
                failed += 1
                print("  FAIL  %s: optimise/checker: %s"
                      % (os.path.basename(path), e))
                continue
            try:
                got = ReferenceEvaluator(res.model).run(None, case["feeds"])
                signal.setitimer(signal.ITIMER_REAL, 0)
            except _Slow:
                slow += 1
                continue
            except Exception:                        # noqa: BLE001
                # the op_matrix sweep uses its own numpy oracle for exactly
                # these operators (no reference implementation); not a rewrite
                signal.setitimer(signal.ITIMER_REAL, 0)
                gap += 1
                continue
            ref = case["ref_vals"]
            ok = len(got) == len(ref)
            for a, b in zip(got, ref):
                a, b = np.asarray(a), np.asarray(b)
                if a.shape != b.shape:
                    ok = False
                    break
                if a.dtype.kind in "iub" or b.dtype.kind in "iub":
                    ok = ok and np.array_equal(a, b)
                else:
                    tol = 1e-4 + 1e-4 * np.maximum(np.abs(b.astype(np.float64)),
                                                   1.0)
                    ok = ok and bool(np.all(
                        np.abs(a.astype(np.float64) - b.astype(np.float64))
                        <= tol))
            checked += 1
            if not ok:
                failed += 1
                print("  FAIL  %s: output differs from the recorded oracle"
                      % os.path.basename(path))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    check(failed == 0, "%d graphs checked, %d failed, %d slow, %d without a "
          "reference implementation" % (checked, failed, slow, gap))


def main():
    print("vc4infer optimiser tests (no device)")
    sweep = None
    if "--sweep" in sys.argv:
        sweep = sys.argv[sys.argv.index("--sweep") + 1]
    test_ir_roundtrip()
    test_canonicalise()
    test_ew()
    test_epilogue()
    test_plan_model()
    test_dispatch_fake_engine()
    test_dispatch_matmul_fold()
    test_runner_markers()
    test_resolve()
    test_opmatrix_sweep(sweep)
    print("")
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  - " + f)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
