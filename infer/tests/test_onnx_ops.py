#!/usr/bin/env python3
"""test_onnx_ops.py -- the reduction / comparison / shape-and-arith family.

Every case here is one ONNX graph with one node, synthesised in-process, run
through the shipped runner (``mode="hybrid"``, ``host_policy="device"``) and
compared with ``onnx.reference``.  It covers the forms ``op_matrix.py``
reports on (the generated matrix in docs/OPS-REDUCE.md) *plus* the forms that
matter for these operators and that one generated case per operator cannot
reach: every axes form of a reduction (none, one, several, negative,
keepdims 0/1, noop_with_empty_axes), the four CumSum flag combinations, all
four Pad modes plus the ``axes`` input, the NaN/inf/+-0 comparison edges, and
non-last axes.

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 python3 tests/test_onnx_ops.py
    python3 tests/test_onnx_ops.py --only ReduceSum,CumSum -v

Exit status is 0 only when every case ran on the VC4 kernels and agreed with
the oracle.  A case that raises UnsupportedNode is reported as ``refused`` and
counts as a failure here: this file is the *positive* evidence for the
operators docs/OPS-REDUCE.md claims, not a coverage sweep.
"""
import argparse
import os
import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _binding as vc4infer                  # noqa: E402
from vc4infer._errors import UnsupportedNode               # noqa: E402
from vc4infer._onnxrun import Runner                       # noqa: E402

TOL_ABS = 1e-4
TOL_REL = 1e-4
OPSET = 22

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL


def f32(*a):
    return np.asarray(a, np.float32)


def i64(*a):
    return np.asarray(a, np.int64)


def rnd(rng, shape, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def pos(rng, shape, lo=0.5, hi=1.5):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


class Case:
    def __init__(self, op, name, feeds, inits=None, attrs=None, inputs=None,
                 out_types=None, n_out=1, note="", identity_ref=False):
        self.op = op
        self.name = name
        self.feeds = feeds or {}
        self.inits = inits or {}
        self.attrs = attrs or {}
        self.inputs = inputs
        self.out_types = out_types
        self.n_out = n_out
        self.note = note
        # onnx.reference's reduce classes only treat axes=None as "empty", so
        # an explicit empty axes tensor (with noop_with_empty_axes=1) is
        # reduced over every axis there.  The spec and this engine return the
        # input unchanged, so the expectation is the input itself.
        self.identity_ref = identity_ref
        # a case whose oracle disagrees with the ONNX spec: the test supplies
        # the expectation instead of asking onnx.reference (see `ref_override`)
        self.ref_override = None


def build(case):
    inputs = case.inputs if case.inputs is not None else list(case.feeds) + \
        [k for k in case.inits if k not in case.feeds]
    outs = ["Y%d" % i for i in range(case.n_out)]
    node = helper.make_node(case.op, inputs, outs, name="n", **case.attrs)
    # the runner accepts fp32 graph inputs only: every int64/bool operand is a
    # graph constant (an initialiser), exactly like a real model's axes/pads
    feeds = {k: v for k, v in case.feeds.items()
             if np.asarray(v).dtype == np.float32}
    inits = dict(case.inits)
    inits.update({k: v for k, v in case.feeds.items()
                  if np.asarray(v).dtype != np.float32})
    ins = [helper.make_tensor_value_info(k, _onnx_type(v),
                                         list(np.asarray(v).shape))
           for k, v in feeds.items()]
    inits = [numpy_helper.from_array(np.asarray(v), k)
             for k, v in inits.items()]
    if case.out_types:
        otypes = case.out_types
    else:
        otypes = [F32] * case.n_out
    graph = helper.make_graph([node], "t", ins,
                              [helper.make_tensor_value_info(o, t, None)
                               for o, t in zip(outs, otypes)], initializer=inits)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="vc4infer-test-onnx-ops")
    for vi in model.graph.output:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            tt.shape.SetInParent()
    return model


def _onnx_type(a):
    dt = np.asarray(a).dtype
    return {np.dtype(np.float32): F32, np.dtype(np.int64): I64,
            np.dtype(np.float64): TensorProto.DOUBLE,
            np.dtype(np.bool_): BOOL}[dt]


def compare(ref, got):
    ref = np.asarray(ref)
    got = np.asarray(got)
    if tuple(ref.shape) != tuple(got.shape):
        return "shape %s != %s" % (tuple(got.shape), tuple(ref.shape))
    if ref.dtype.kind in ("i", "u", "b"):
        if not np.array_equal(ref, got):
            return "%d of %d differ (exact)" % (
                int(np.count_nonzero(ref != got)), ref.size)
        return None
    if got.dtype != np.float32:
        got = got.astype(np.float32)
    ref = ref.astype(np.float32)
    one_nan = np.isnan(ref) ^ np.isnan(got)
    inf_bad = (np.isinf(ref) ^ np.isinf(got)) | (
        np.isinf(ref) & np.isinf(got) & (np.sign(ref) != np.sign(got)))
    finite = np.isfinite(ref) & np.isfinite(got)
    diff = np.zeros_like(ref, dtype=np.float64)
    diff[finite] = np.abs(got[finite].astype(np.float64) -
                          ref[finite].astype(np.float64))
    tol = TOL_ABS + TOL_REL * np.maximum(np.abs(ref.astype(np.float64)), 1.0)
    bad = int(np.count_nonzero((finite & (diff > tol)) | one_nan | inf_bad))
    if bad:
        i = int(np.argmax(np.where(finite, diff, -1.0)))
        return "%d of %d over tolerance (worst ref %r got %r)" % (
            bad, ref.size, float(ref.reshape(-1)[i]), float(got.reshape(-1)[i]))
    return None


# =====================================================================
# cases
# =====================================================================
def reduction_cases(rng):
    """Every axes form of the reduction family."""
    cases = []
    X = pos(rng, (2, 3, 4))
    for op in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
               "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum",
               "ReduceLogSumExp", "ReduceSumSquare"):
        src = pos(rng, (2, 3, 4)) if op in ("ReduceLogSum", "ReduceLogSumExp") \
            else pos(rng, (2, 3, 4), 0.5, 1.2)
        cases += [
            Case(op, "all-axes", {"data": src}, n_out=1,
                 inputs=["data"], attrs={"keepdims": 0},
                 note="no axes input: reduce every axis"),
            Case(op, "axes-1-keepdims", {"data": src, "axes": i64(1)},
                 inputs=["data", "axes"]),
            Case(op, "axes-neg1", {"data": src, "axes": i64(-1)},
                 attrs={"keepdims": 0}, inputs=["data", "axes"],
                 note="negative axis, innermost (no layout copy)"),
            Case(op, "axes-0-2", {"data": src, "axes": i64(0, 2)},
                 inputs=["data", "axes"], note="two non-adjacent axes"),
            Case(op, "axes-empty-noop",
                 {"data": src, "axes": np.zeros(0, np.int64)},
                 attrs={"keepdims": 1, "noop_with_empty_axes": 1},
                 inputs=["data", "axes"], identity_ref=True,
                 note="noop_with_empty_axes=1, empty axes tensor"),
            Case(op, "size1-axis", {"data": pos(rng, (3, 1, 5)), "axes": i64(1)},
                 attrs={"keepdims": 0}, inputs=["data", "axes"],
                 note="size-1 axis"),
        ]
    return cases


def compare_cases(rng):
    cases = []
    for op in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
        a = rng.uniform(-2, 2, (2, 3, 4)).astype(np.float32)
        b = a.copy()
        b[0, 0, :] = a[0, 0, ::-1]                 # some equal, some not
        edge = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0,
                         1e-45, -1e-45], np.float32)
        cases += [
            Case(op, "same-shape", {"A": a, "B": b}, inputs=["A", "B"],
                 out_types=[BOOL]),
            Case(op, "edge-values", {"A": edge, "B": edge[::-1].copy()},
                 inputs=["A", "B"], out_types=[BOOL],
                 note="0/-0, +-inf, NaN, denormals"),
            Case(op, "broadcast", {"A": rng.uniform(-1, 1, (2, 1, 4)).astype(
                np.float32), "B": rng.uniform(-1, 1, (3, 4)).astype(np.float32)},
                 inputs=["A", "B"], out_types=[BOOL],
                 note="numpy broadcasting both ways"),
        ]
    cases += [
        Case("IsNaN", "plain", {"X": np.array(
            [1.0, np.nan, np.inf, -np.inf, -0.0, 1e-45], np.float32)},
            inputs=["X"], out_types=[BOOL]),
        Case("IsInf", "both", {"X": np.array(
            [1.0, np.nan, np.inf, -np.inf, -0.0], np.float32)},
            attrs={"detect_negative": 1, "detect_positive": 1},
            inputs=["X"], out_types=[BOOL]),
        Case("IsInf", "positive-only", {"X": np.array(
            [1.0, np.nan, np.inf, -np.inf, -0.0], np.float32)},
            attrs={"detect_negative": 0, "detect_positive": 1},
            inputs=["X"], out_types=[BOOL]),
        Case("IsInf", "negative-only", {"X": np.array(
            [1.0, np.nan, np.inf, -np.inf, -0.0], np.float32)},
            attrs={"detect_negative": 1, "detect_positive": 0},
            inputs=["X"], out_types=[BOOL]),
    ]
    return cases


def where_cases(rng):
    return [
        Case("Where", "cond-from-init",
             {"X": rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32),
              "Y": rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32)},
             inits={"condition": rng.random((2, 3, 4)) < 0.5},
             inputs=["condition", "X", "Y"]),
        Case("Where", "broadcast",
             {"X": rng.uniform(-1, 1, (2, 1, 4)).astype(np.float32),
              "Y": rng.uniform(-1, 1, (3, 1)).astype(np.float32)},
             inits={"condition": rng.random((2, 3, 4)) < 0.5},
             inputs=["condition", "X", "Y"]),
        Case("Where", "all-true-cond",
             {"X": rng.uniform(-1, 1, (5,)).astype(np.float32),
              "Y": rng.uniform(-1, 1, (5,)).astype(np.float32)},
             inits={"condition": np.ones(5, bool)},
             inputs=["condition", "X", "Y"]),
    ]


def arg_cases(rng):
    out = []
    for op in ("ArgMax", "ArgMin"):
        a = rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
        dup = a.copy()
        dup[..., 1] = dup[..., 3]                  # ties, for select_last
        for axis, keepdims in ((1, 1), (-1, 0), (0, 0), (2, 1)):
            out.append(Case(op, "axis%d-keep%d" % (axis, keepdims), {"data": a},
                            attrs={"axis": axis, "keepdims": keepdims},
                            inputs=["data"], out_types=[I64]))
        out.append(Case(op, "select-last", {"data": dup},
                        attrs={"axis": -1, "keepdims": 0,
                               "select_last_index": 1},
                        inputs=["data"], out_types=[I64],
                        note="last index on ties"))
        out.append(Case(op, "size1", {"data": pos(rng, (4, 1))},
                        attrs={"axis": 1, "keepdims": 1},
                        inputs=["data"], out_types=[I64]))
    return out


def cumsum_cases(rng):
    a = rng.uniform(-1, 1, (2, 3, 5)).astype(np.float32)
    out = []
    for axis in (-1, 1):
        for excl in (0, 1):
            for rev in (0, 1):
                out.append(Case(
                    "CumSum", "axis%d-excl%d-rev%d" % (axis, excl, rev),
                    {"x": a, "axis": i64(axis)},
                    attrs={"exclusive": excl, "reverse": rev},
                    inputs=["x", "axis"]))
    out.append(Case("CumSum", "size1-axis",
                    {"x": pos(rng, (3, 1, 4)), "axis": i64(1)},
                    inputs=["x", "axis"]))
    out.append(Case("CumSum", "axis0", {"x": a, "axis": i64(0)},
                    inputs=["x", "axis"]))
    return out


def pow_cases(rng):
    a = pos(rng, (2, 3, 4))
    b = pos(rng, (2, 3, 4))
    return [
        Case("Pow", "tensor-exponent", {"A": a, "B": b}, inputs=["A", "B"]),
        Case("Pow", "scalar-exp-2", {"A": rng.uniform(-1, 1, (2, 3)).astype(
            np.float32)}, inits={"B": f32(2.0)}, inputs=["A", "B"],
            note="exact: one multiply, negative base allowed"),
        Case("Pow", "scalar-exp-3-neg-base",
             {"A": rng.uniform(-1, 1, (2, 3)).astype(np.float32)},
             inits={"B": f32(3.0)}, inputs=["A", "B"]),
        Case("Pow", "scalar-exp-half", {"A": pos(rng, (4,))},
             inits={"B": f32(0.5)}, inputs=["A", "B"], note="sqrt"),
        Case("Pow", "scalar-exp-neg1", {"A": pos(rng, (4,))},
             inits={"B": f32(-1.0)}, inputs=["A", "B"], note="reciprocal"),
        Case("Pow", "broadcast", {"A": pos(rng, (2, 1, 4)), "B": pos(rng, (3, 1))},
             inputs=["A", "B"]),
    ]


def mod_cases(rng):
    # onnx.reference 1.17 evaluates Mod on floats with np.fmod for *both*
    # fmod values (op_mod.py), while the ONNX spec says fmod=0 is Python's
    # modulus.  The expectation below is the spec; the doc records the gap.
    a = rng.uniform(-4, 4, (2, 3, 4)).astype(np.float32)
    b = rng.uniform(0.5, 2.0, (2, 3, 4)).astype(np.float32)
    bneg = (b * np.sign(rng.uniform(-1, 1, (2, 3, 4))).astype(np.float32))
    out = [
        Case("Mod", "int-semantics", {"A": a, "B": b}, inputs=["A", "B"],
             note="fmod=0: sign follows the divisor (spec)"),
        Case("Mod", "int-semantics-neg-divisor", {"A": a, "B": bneg},
             inputs=["A", "B"]),
        Case("Mod", "fmod", {"A": a, "B": b}, attrs={"fmod": 1},
             inputs=["A", "B"], note="fmod=1: sign follows the dividend"),
        Case("Mod", "integer-values", {"A": np.array([-7, -7, 7, 7],
                                                     np.float32),
                                       "B": f32(3.0)},
             attrs={"fmod": 0}, inputs=["A", "B"]),
    ]
    for c in out:
        if int(c.attrs.get("fmod", 0)) == 0:
            c.ref_override = lambda A, B: np.mod(A, B).astype(np.float32)
        else:
            c.ref_override = lambda A, B: np.fmod(A, B).astype(np.float32)
    return out


def shape_cases(rng):
    a = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
    cub = rng.uniform(-1, 1, (1, 2, 3, 3)).astype(np.float32)
    return [
        Case("Expand", "broadcast", {"input": rng.uniform(-1, 1, (2, 1, 4)).astype(
            np.float32), "shape": i64(2, 3, 4)}, inputs=["input", "shape"]),
        Case("Expand", "rank-up", {"input": rng.uniform(-1, 1, (4,)).astype(
            np.float32), "shape": i64(2, 3, 4)}, inputs=["input", "shape"]),
        Case("Expand", "last-axis", {"input": rng.uniform(-1, 1, (3, 1)).astype(
            np.float32), "shape": i64(3, 5)}, inputs=["input", "shape"],
            note="the last axis itself broadcasts"),
        Case("Tile", "2x2", {"input": a, "repeats": i64(2, 2)},
             inputs=["input", "repeats"]),
        Case("Tile", "1x3x2", {"input": rng.uniform(-1, 1, (2, 3)).astype(
            np.float32), "repeats": i64(1, 3)}, inputs=["input", "repeats"]),
        Case("Tile", "size1", {"input": rng.uniform(-1, 1, (1, 5)).astype(
            np.float32), "repeats": i64(3, 1)}, inputs=["input", "repeats"]),
        Case("Pad", "constant", {"data": cub, "pads": i64(0, 0, 1, 1, 0, 0, 1, 1),
                                 "constant_value": f32(0.0)},
             inputs=["data", "pads", "constant_value"]),
        Case("Pad", "constant-nonzero", {
            "data": cub, "pads": i64(0, 0, 0, 1, 0, 0, 2, 0),
            "constant_value": f32(-3.5)}, inputs=["data", "pads",
                                                 "constant_value"]),
        Case("Pad", "reflect", {"data": cub, "pads": i64(0, 0, 1, 1, 0, 0, 1, 1)},
             attrs={"mode": "reflect"}, inputs=["data", "pads"]),
        Case("Pad", "edge", {"data": cub, "pads": i64(0, 0, 2, 1, 0, 0, 1, 2)},
             attrs={"mode": "edge"}, inputs=["data", "pads"]),
        Case("Pad", "wrap", {"data": cub, "pads": i64(0, 0, 2, 1, 0, 0, 1, 2)},
             attrs={"mode": "wrap"}, inputs=["data", "pads"]),
        Case("Pad", "axes-input", {"data": cub, "pads": i64(1, 1, 2, 2),
                                   "axes": i64(2, 3)},
             inputs=["data", "pads", "", "axes"]),
        Case("Compress", "axis0", {"input": a,
                                   "condition": np.array([True, False], bool)},
             attrs={"axis": 0}, inputs=["input", "condition"]),
        Case("Compress", "axis1", {"input": a,
                                   "condition": np.array([False, True, True],
                                                         bool)},
             attrs={"axis": 1}, inputs=["input", "condition"]),
        Case("OneHot", "axis-last", {"indices": i64(0, 2, 3), "depth": i64(4),
                                     "values": f32(0.0, 1.0)},
             attrs={"axis": -1}, inputs=["indices", "depth", "values"]),
        Case("OneHot", "axis0", {"indices": i64(1, 0), "depth": i64(3),
                                 "values": f32(-1.0, 2.0)},
             attrs={"axis": 0}, inputs=["indices", "depth", "values"]),
        Case("Trilu", "upper-k0", {"input": rng.uniform(-1, 1, (3, 4)).astype(
            np.float32), "k": i64(0)}, attrs={"upper": 1},
            inputs=["input", "k"]),
        Case("Trilu", "lower-k1", {"input": rng.uniform(-1, 1, (4, 4)).astype(
            np.float32), "k": i64(1)}, attrs={"upper": 0},
            inputs=["input", "k"]),
        Case("Trilu", "batched", {"input": rng.uniform(-1, 1, (2, 3, 3)).astype(
            np.float32), "k": i64(-1)}, attrs={"upper": 1},
            inputs=["input", "k"]),
        Case("Range", "float", {"start": f32(0.0), "limit": f32(1.0),
                                "delta": f32(0.2)},
             inputs=["start", "limit", "delta"]),
        Case("Range", "negative-delta", {"start": f32(3.0), "limit": f32(0.0),
                                         "delta": f32(-0.75)},
             inputs=["start", "limit", "delta"]),
        Case("EyeLike", "4x4", {"input": rng.uniform(-1, 1, (4, 4)).astype(
            np.float32)}, inputs=["input"]),
        Case("EyeLike", "k2", {"input": np.zeros((3, 5), np.float32)},
             attrs={"k": 2}, inputs=["input"]),
        Case("CastLike", "f32-to-f32", {"input": rng.uniform(-1, 1, (2, 3)
                                                             ).astype(np.float32),
                                        "target": rng.uniform(-1, 1, (1,)
                                                              ).astype(np.float32)},
             inputs=["input", "target"]),
        Case("SpaceToDepth", "b2", {"input": rng.uniform(-1, 1, (1, 4, 8, 8)
                                                         ).astype(np.float32)},
             attrs={"blocksize": 2}, inputs=["input"]),
        Case("SpaceToDepth", "b2-3d-batch", {
            "input": rng.uniform(-1, 1, (2, 2, 4, 6)).astype(np.float32)},
            attrs={"blocksize": 2}, inputs=["input"]),
        Case("Shrink", "default", {"input": rng.uniform(-2, 2, (2, 3, 4)
                                                        ).astype(np.float32)},
             attrs={"bias": 0.5, "lambd": 1.0}, inputs=["input"]),
        Case("ThresholdedRelu", "alpha1", {"input": rng.uniform(-2, 2, (2, 3)
                                                                ).astype(np.float32)},
             attrs={"alpha": 1.0}, inputs=["input"]),
        Case("Celu", "alpha1", {"input": rng.uniform(-2, 2, (2, 3, 4)
                                                      ).astype(np.float32)},
             attrs={"alpha": 1.0}, inputs=["input"]),
        Case("PRelu", "same-shape", {"X": rng.uniform(-2, 2, (2, 3, 4)).astype(
            np.float32), "slope": rng.uniform(0, 0.5, (2, 3, 4)).astype(
                np.float32)}, inputs=["X", "slope"]),
        Case("PRelu", "channel-broadcast", {
            "X": rng.uniform(-2, 2, (1, 4, 2, 2)).astype(np.float32),
            "slope": rng.uniform(0, 0.5, (4, 1, 1)).astype(np.float32)},
            inputs=["X", "slope"]),
        Case("Hardmax", "last-axis", {"input": rng.uniform(-1, 1, (4, 5)
                                                          ).astype(np.float32)},
             inputs=["input"]),
        Case("Hardmax", "axis1-c4", {"input": rng.uniform(-1, 1, (1, 4, 3, 3)
                                                         ).astype(np.float32)},
             attrs={"axis": 1}, inputs=["input"]),
        Case("LogSoftmax", "last-axis", {"input": rng.uniform(-1, 1, (3, 5)
                                                             ).astype(np.float32)},
             inputs=["input"]),
        Case("LogSoftmax", "axis1-c4", {"input": rng.uniform(-1, 1, (1, 4, 3, 3)
                                                            ).astype(np.float32)},
             attrs={"axis": 1}, inputs=["input"]),
        Case("LogSoftmax", "wide-classes", {"input": rng.uniform(-1, 1, (2, 40)
                                                                ).astype(np.float32)},
             inputs=["input"],
             note="no 16-class limit: x - logsumexp(x)"),
    ]


ALL = {}


def _register():
    rng = np.random.default_rng(20261120)
    for fn in (reduction_cases, compare_cases, where_cases, arg_cases,
               cumsum_cases, pow_cases, mod_cases, shape_cases):
        for c in fn(rng):
            ALL.setdefault(c.op, []).append(c)


_register()


def run_case(eng, case, workdir, verbose):
    model = build(case)
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        return "invalid-model", "%s: %s" % (type(e).__name__, e)
    path = os.path.join(workdir, "t_%s_%s.onnx" %
                        (case.op, case.name.replace("/", "_")))
    onnx.save(model, path)
    feeds = {k: v for k, v in case.feeds.items()
             if np.asarray(v).dtype == np.float32}
    try:
        if case.ref_override is not None:
            ref = [case.ref_override(*[feeds[k] for k in feeds])]
        elif case.identity_ref:
            ref = [feeds["data"]]
        else:
            ref = ReferenceEvaluator(model).run(None, dict(feeds))
    except Exception as e:
        return "oracle-error", "%s: %s" % (type(e).__name__, e)
    try:
        r = Runner(path, eng, mode="hybrid", host_policy="device")
        got = r.run({k: np.ascontiguousarray(v, np.float32)
                     for k, v in feeds.items()})
    except UnsupportedNode as e:
        return "refused", str(e)
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return "error", "%s: %s" % (type(e).__name__, e)
    for i, (w, g) in enumerate(zip(ref, got)):
        msg = compare(w, g)
        if msg:
            return "mismatch", "output %d: %s" % (i, msg)
    return "vc4", ""


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    only = [s for s in args.only.replace(",", " ").split() if s]
    ops = [o for o in sorted(ALL) if not only or o in only]
    if only:
        missing = [o for o in only if o not in ALL]
        if missing:
            print("no cases for: %s" % ", ".join(missing))
            return 2
    workdir = args.work_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "vc4ops-models")
    os.makedirs(workdir, exist_ok=True)
    cache = args.cache or os.environ.get("VC4INFER_CACHE") or None
    eng = vc4infer.Engine(cache_dir=cache)
    bad = []
    total = 0
    try:
        for op in ops:
            for case in ALL[op]:
                total += 1
                status, msg = run_case(eng, case, workdir, args.verbose)
                ok = status == "vc4"
                print("%-4s %-16s %-24s %s%s"
                      % ("ok" if ok else "FAIL", op, case.name, status,
                         "" if ok else "  " + msg[:160]))
                if not ok:
                    bad.append((op, case.name, status, msg))
    finally:
        eng.close()
    print("\n%d cases, %d failed" % (total, len(bad)))
    if bad:
        from collections import Counter
        print("by status:", dict(Counter(b[2] for b in bad)))
        for op, name, status, msg in bad:
            print("  %-16s %-24s %-12s %s" % (op, name, status, msg[:120]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
