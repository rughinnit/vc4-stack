#!/usr/bin/env python3
"""test_onnx_norm.py -- the pooling / normalisation / softmax / loss family.

Every case is one ONNX graph with one node, synthesised in-process, run through
the shipped runner (``mode="hybrid"``, ``host_policy="device"``) and compared
with ``onnx.reference`` -- except where that oracle is wrong or missing, in
which case the case supplies the ONNX-spec expectation (``ref_override``) and
says why.  ``op_matrix.py`` covers one generated form per operator; this file
covers the forms that matter and that one generated case cannot reach:

  * the two softmax opset conventions (opset >= 13 per-position over ``axis``,
    opset <= 11 after coercing the input to 2-D), axis 0/1/last, wide class
    counts, ties and -inf for Hardmax;
  * every pooling attribute: count_include_pad 0/1, ceil_mode, dilations,
    asymmetric pads, auto_pad, p=1/2, a single channel, +-inf data;
  * the epsilon placement of each normaliser (BN/IN/GN/LN next to the
    variance, MVN *after* the square root at 1e-9) and its edges (eps=0);
  * the single-channel and all-zero-variance cases;
  * loss reductions none/mean/sum, ignore_index, per-class weights, a rank-4
    score tensor and the optional log_prob output;
  * the refusals that are by design, each with the invariant it names.

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 python3 tests/test_onnx_norm.py
    python3 tests/test_onnx_norm.py --only AveragePool,Softmax -v

Exit status is 0 only when every case either ran on the VC4 kernels and agreed
with its expectation, or was refused with the exact invariant the case names.
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
                 out_types=None, n_out=1, opset=OPSET, note="",
                 ref_override=None, refuse=None):
        self.op = op
        self.name = name
        self.feeds = feeds or {}
        self.inits = inits or {}
        self.attrs = attrs or {}
        self.inputs = inputs
        self.out_types = out_types
        self.n_out = n_out
        self.opset = opset
        self.note = note
        # a callable(*float_feeds) -> array/list: the ONNX-spec expectation
        # for a form onnx.reference gets wrong or cannot evaluate
        self.ref_override = ref_override
        # a substring the refusal must contain (a by-design refusal)
        self.refuse = refuse


def _onnx_type(a):
    dt = np.asarray(a).dtype
    return {np.dtype(np.float32): F32, np.dtype(np.int64): I64,
            np.dtype(np.float64): TensorProto.DOUBLE,
            np.dtype(np.bool_): BOOL}[dt]


def build(case, out_shapes=None, out_types=None):
    inputs = case.inputs if case.inputs is not None else \
        list(case.feeds) + [k for k in case.inits if k not in case.feeds]
    outs = ["Y%d" % i for i in range(case.n_out)]
    node = helper.make_node(case.op, inputs, outs, name="n", **case.attrs)
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
    otypes = out_types or case.out_types or [F32] * case.n_out
    out_vis = []
    for i, (o, t) in enumerate(zip(outs, otypes)):
        shape = None if out_shapes is None else \
            list(np.asarray(out_shapes[i]).shape)
        out_vis.append(helper.make_tensor_value_info(o, t, shape))
    graph = helper.make_graph([node], "t", ins, out_vis, initializer=inits)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("",
                                                                 case.opset)],
                              producer_name="vc4infer-test-onnx-norm")
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception:
        # a dynamic output shape is valid ONNX but shape inference may be
        # unable to place it; re-save without the value info shapes
        for vi in model.graph.output:
            vi.type.tensor_type.shape.SetInParent()
    return model


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
# the ONNX forms onnx.reference 1.17 gets wrong or cannot evaluate
# =====================================================================
def _coerce(x, axis):
    """The opset <= 11 softmax family: 2-D [prod(before), prod(from)]."""
    a = int(axis)
    return x.reshape(int(np.prod(x.shape[:a])), int(np.prod(x.shape[a:])))


def _softmax2(x2):
    m = np.max(x2, axis=1, keepdims=True)
    e = np.exp(x2 - m)
    return e / np.sum(e, axis=1, keepdims=True)


def _logsoftmax2(x2):
    m = np.max(x2, axis=1, keepdims=True)
    return (x2 - m) - np.log(np.sum(np.exp(x2 - m), axis=1, keepdims=True))


def _hardmax2(x2):
    y = np.zeros_like(x2)
    idx = np.argmax(x2, axis=1)
    y[np.arange(x2.shape[0]), idx] = 1.0
    return y


def _lrn_spec(x, size, alpha, beta, bias):
    """ONNX LRN: the channel window [c-floor((s-1)/2), c+ceil((s-1)/2)+1)."""
    c1 = (int(size) - 1) // 2
    c2 = int(np.ceil((int(size) - 1) / 2.0)) + 1
    C = x.shape[1]
    sq = np.zeros_like(x)
    for c in range(C):
        b, e = max(0, c - c1), min(C, c + c2)
        sq[:, c] = np.sum(x[:, b:e, ...] ** 2, axis=1)
    return (x / ((bias + (alpha / size) * sq) ** beta)).astype(np.float32)


def _glp(x, p):
    return (np.sum(np.abs(x) ** p, axis=(2, 3), keepdims=True)
            ** (1.0 / p)).astype(np.float32)


# =====================================================================
# cases
# =====================================================================
def softmax_cases(rng):
    x = rnd(rng, (2, 40, 3, 3))
    wide = rnd(rng, (4, 5))
    tie = np.array([[1.0, 3.0, 3.0, 0.0, -1.0],
                    [-2.0, -2.0, -5.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5, 0.5, 0.5]], np.float32)
    cases = [
        Case("Softmax", "last-axis", {"X": wide}),
        Case("Softmax", "axis1-c4", {"X": rnd(rng, (1, 4, 4, 4))},
             attrs={"axis": 1}),
        Case("Softmax", "axis1-c40", {"X": x}, attrs={"axis": 1},
             note="wide class count on the channel axis"),
        Case("Softmax", "axis0", {"X": rnd(rng, (3, 4))}, attrs={"axis": 0}),
        Case("Softmax", "single-class", {"X": rnd(rng, (2, 1, 3))},
             note="one class: every value is 1"),
        Case("Softmax", "opset11-coerce-axis1", {"X": rnd(rng, (1, 3, 4, 4))},
             attrs={"axis": 1}, opset=11,
             ref_override=lambda x: _softmax2(_coerce(x, 1)).reshape(x.shape),
             note="opset<=11 coerces to 2-D; onnx.reference does not"),
        Case("Softmax", "nan-refused", {"X": np.array([[np.nan, 1.0]],
                                                      np.float32)},
             refuse="NaN"),
        Case("Softmax", "pinf-refused", {"X": np.array([[np.inf, 1.0]],
                                                       np.float32)},
             refuse="non-finite"),
        Case("LogSoftmax", "last-axis", {"X": wide}),
        Case("LogSoftmax", "axis1-c40", {"X": x}, attrs={"axis": 1}),
        Case("LogSoftmax", "wide-magnitude",
             {"X": rnd(rng, (3, 24), -200.0, 200.0)},
             ref_override=lambda x: (x - np.log(np.exp(
                 x - x.max(1, keepdims=True)).sum(1, keepdims=True))
                 - x.max(1, keepdims=True)).astype(np.float32),
             note="outside the direct exp range, so the row-max shift is "
                  "used; onnx.reference forms log(softmax(x)) and underflows "
                  "to -inf there"),
        Case("LogSoftmax", "opset11-coerce-axis1",
             {"X": rnd(rng, (1, 3, 4, 4))}, attrs={"axis": 1}, opset=11,
             ref_override=lambda x: _logsoftmax2(_coerce(x, 1)).reshape(
                 x.shape)),
        Case("LogSoftmax", "pinf-refused", {"X": np.array([[np.inf, 1.0]],
                                                         np.float32)},
             refuse="non-finite"),
        Case("Hardmax", "last-axis", {"X": wide}, out_types=[F32]),
        Case("Hardmax", "axis1-c40", {"X": x}, attrs={"axis": 1}),
        Case("Hardmax", "ties", {"X": tie}),
        Case("Hardmax", "neginf", {"X": np.array([[-np.inf, -1.0, 0.0],
                                                  [0.0, -np.inf, -np.inf]],
                                                 np.float32)}),
        Case("Hardmax", "opset11-coerce-axis1", {"X": rnd(rng, (1, 3, 4, 4))},
             attrs={"axis": 1}, opset=11,
             ref_override=lambda x: _hardmax2(_coerce(x, 1)).reshape(
                 x.shape),
             note="opset<=11 coerces to 2-D; onnx.reference does not"),
        Case("Hardmax", "nan-refused",
             {"X": np.array([[np.nan, 1.0]], np.float32)}, refuse="NaN"),
    ]
    return cases


def pool_cases(rng):
    x = rnd(rng, (1, 3, 8, 8))
    inf = x.copy()
    inf[0, 0, 0, 0] = np.inf
    inf[0, 1, 2, 3] = -np.inf
    cases = [
        Case("AveragePool", "k2s2", {"X": x},
             attrs={"kernel_shape": [2, 2], "strides": [2, 2]}),
        Case("AveragePool", "k3s1p1", {"X": x},
             attrs={"kernel_shape": [3, 3], "strides": [1, 1],
                    "pads": [1, 1, 1, 1]}),
        Case("AveragePool", "k3s1p1-cip", {"X": x},
             attrs={"kernel_shape": [3, 3], "strides": [1, 1],
                    "pads": [1, 1, 1, 1], "count_include_pad": 1}),
        Case("AveragePool", "k3s2p1-ceil", {"X": rnd(rng, (1, 2, 7, 7))},
             attrs={"kernel_shape": [3, 3], "strides": [2, 2],
                    "pads": [1, 1, 1, 1], "ceil_mode": 1}),
        Case("AveragePool", "dilation2", {"X": rnd(rng, (1, 2, 9, 9))},
             attrs={"kernel_shape": [2, 2], "pads": [2, 2, 2, 2],
                    "dilations": [2, 2]}),
        Case("AveragePool", "dilation2-cip", {"X": rnd(rng, (1, 2, 9, 9))},
             attrs={"kernel_shape": [2, 2], "pads": [2, 2, 2, 2],
                    "dilations": [2, 2], "count_include_pad": 1},
             note="the divisor is the dilated window, not kernel_shape"),
        Case("AveragePool", "single-channel", {"X": rnd(rng, (1, 1, 5, 5))},
             attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}),
        Case("AveragePool", "asymmetric-pads",
             {"X": rnd(rng, (2, 2, 7, 5))},
             attrs={"kernel_shape": [3, 3], "pads": [1, 0, 0, 2]}),
        Case("AveragePool", "same-upper", {"X": rnd(rng, (1, 2, 6, 6))},
             attrs={"kernel_shape": [3, 3], "strides": [2, 2],
                    "auto_pad": "SAME_UPPER"}),
        Case("AveragePool", "inf-cip", {"X": inf},
             attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
                    "count_include_pad": 1},
             note="+-inf data with a scalar divisor: the sum and the affine "
                  "divide are exact IEEE"),
        Case("AveragePool", "inf-refused", {"X": inf},
             attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
             refuse="non-finite input"),
        Case("AveragePool", "all-pad-refused", {"X": rnd(rng, (1, 1, 4, 4))},
             attrs={"kernel_shape": [2, 2], "pads": [3, 3, 3, 3]},
             refuse="entirely in the padding"),
        Case("GlobalMaxPool", "c3", {"X": rnd(rng, (1, 3, 4, 4))}),
        Case("GlobalMaxPool", "single-channel-batch2",
             {"X": rnd(rng, (2, 1, 5, 5))}),
        Case("GlobalMaxPool", "inf", {"X": inf},
             note="the max scan is IEEE-exact for +-inf"),
        Case("BatchNormalization", "inf-refused",
             {"X": np.array([[[[np.inf, 1.0]]]], np.float32),
              "scale": f32(1.0), "B": f32(0.0), "mean": f32(0.0),
              "var": f32(1.0)},
             inputs=["X", "scale", "B", "mean", "var"],
             refuse="non-finite input"),
        Case("GlobalMaxPool", "nan-refused",
             {"X": np.array([[[[np.nan, 1.0], [2.0, 3.0]]]], np.float32)},
             refuse="NaN"),
        Case("LpPool", "p1-k3p1", {"X": x},
             attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "p": 1}),
        Case("LpPool", "p2-k2s2", {"X": x},
             attrs={"kernel_shape": [2, 2], "strides": [2, 2], "p": 2}),
        Case("LpPool", "p2-ceil", {"X": rnd(rng, (1, 2, 7, 7))},
             attrs={"kernel_shape": [3, 3], "strides": [2, 2],
                    "pads": [1, 1, 1, 1], "ceil_mode": 1, "p": 2}),
        Case("LpPool", "p2-dilation", {"X": rnd(rng, (1, 2, 9, 9))},
             attrs={"kernel_shape": [2, 2], "pads": [2, 2, 2, 2],
                    "dilations": [2, 2], "p": 2}),
        Case("LpPool", "p3-refused", {"X": x},
             attrs={"kernel_shape": [2, 2], "p": 3}, refuse="p=3"),
        Case("GlobalLpPool", "p1", {"X": rnd(rng, (1, 3, 4, 4))},
             attrs={"p": 1}, ref_override=lambda x: _glp(x, 1),
             note="onnx.reference has no GlobalLpPool implementation"),
        Case("GlobalLpPool", "p2", {"X": rnd(rng, (2, 2, 5, 5))},
             attrs={"p": 2}, ref_override=lambda x: _glp(x, 2)),
        Case("MaxUnpool", "2x2s2",
             {"X": rnd(rng, (1, 2, 3, 3)),
              "I": np.arange(18, dtype=np.int64).reshape(1, 2, 3, 3)},
             inputs=["X", "I"],
             attrs={"kernel_shape": [2, 2], "strides": [2, 2]}),
        Case("MaxUnpool", "permuted-indices",
             {"X": rnd(rng, (1, 1, 2, 2)),
              "I": np.array([[[[10, 3], [8, 12]]]], np.int64)},
             inputs=["X", "I"],
             attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
             note="non-monotone destinations: single-word stores"),
        Case("MaxUnpool", "output-shape",
             {"X": rnd(rng, (1, 1, 2, 2)),
              "I": np.arange(4, dtype=np.int64).reshape(1, 1, 2, 2),
              "S": np.array([1, 1, 4, 4], np.int64)},
             inputs=["X", "I", "S"],
             attrs={"kernel_shape": [2, 2], "strides": [2, 2]}),
        Case("MaxUnpool", "duplicate-refused",
             {"X": rnd(rng, (1, 1, 2, 2)),
              "I": np.array([[[[0, 0], [1, 2]]]], np.int64)},
             inputs=["X", "I"],
             attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
             refuse="duplicate indices"),
    ]
    return cases


def norm_cases(rng):
    c = 3
    ch = lambda n, lo=0.5, hi=1.5: pos(rng, (n,), lo, hi)      # noqa: E731
    cases = [
        Case("BatchNormalization", "c3-batch2",
             {"X": rnd(rng, (2, c, 3, 3)), "scale": ch(c), "B": rnd(rng, (c,)),
              "mean": rnd(rng, (c,)), "var": ch(c)},
             inputs=["X", "scale", "B", "mean", "var"]),
        Case("BatchNormalization", "eps0",
             {"X": rnd(rng, (1, c, 2, 2)), "scale": ch(c), "B": rnd(rng, (c,)),
              "mean": rnd(rng, (c,)), "var": ch(c)},
             inputs=["X", "scale", "B", "mean", "var"],
             attrs={"epsilon": 0.0},
             note="epsilon is inside the sqrt: sqrt(var + 0)"),
        Case("BatchNormalization", "eps1",
             {"X": rnd(rng, (1, c, 2, 2)), "scale": ch(c), "B": rnd(rng, (c,)),
              "mean": rnd(rng, (c,)), "var": ch(c)},
             inputs=["X", "scale", "B", "mean", "var"],
             attrs={"epsilon": 1.0},
             note="a large epsilon: placement is visible"),
        Case("BatchNormalization", "single-channel",
             {"X": rnd(rng, (1, 1, 3, 3)), "scale": f32(2.0), "B": f32(-1.0),
              "mean": f32(0.5), "var": f32(0.25)},
             inputs=["X", "scale", "B", "mean", "var"]),
        Case("BatchNormalization", "training-refused",
             {"X": rnd(rng, (1, c, 2, 2)), "scale": ch(c), "B": rnd(rng, (c,)),
              "mean": rnd(rng, (c,)), "var": ch(c)},
             inputs=["X", "scale", "B", "mean", "var"],
             attrs={"training_mode": 1}, refuse="training_mode=1"),
        Case("InstanceNormalization", "c3-batch2",
             {"X": rnd(rng, (2, c, 4, 4)), "scale": ch(c), "B": rnd(rng, (c,))},
             inputs=["X", "scale", "B"]),
        Case("InstanceNormalization", "eps0",
             {"X": rnd(rng, (1, c, 3, 3)), "scale": ch(c), "B": rnd(rng, (c,))},
             inputs=["X", "scale", "B"], attrs={"epsilon": 0.0}),
        Case("InstanceNormalization", "single-channel-flat",
             {"X": np.full((1, 1, 2, 2), 0.5, np.float32),
              "scale": f32(1.0), "B": f32(0.0)},
             inputs=["X", "scale", "B"],
             note="zero variance: 0/sqrt(eps) is 0, exactly as the reference"),
        Case("GroupNormalization", "groups-eq-c",
             {"X": rnd(rng, (1, 4, 3, 3)), "scale": ch(4), "B": rnd(rng, (4,))},
             inputs=["X", "scale", "B"], attrs={"num_groups": 4}),
        Case("GroupNormalization", "groups1",
             {"X": rnd(rng, (2, 4, 3, 3)), "scale": ch(4), "B": rnd(rng, (4,))},
             inputs=["X", "scale", "B"], attrs={"num_groups": 1}),
        Case("GroupNormalization", "groups2-eps",
             {"X": rnd(rng, (2, 4, 3, 3)), "scale": ch(4), "B": rnd(rng, (4,))},
             inputs=["X", "scale", "B"],
             attrs={"num_groups": 2, "epsilon": 1e-3}),
        Case("LayerNormalization", "last-axis",
             {"X": rnd(rng, (2, 4, 8)), "Scale": ch(8), "B": rnd(rng, (8,))},
             inputs=["X", "Scale", "B"]),
        Case("LayerNormalization", "axis1",
             {"X": rnd(rng, (2, 4, 8)), "Scale": ch(8), "B": rnd(rng, (8,))},
             inputs=["X", "Scale", "B"], attrs={"axis": 1},
             note="rows=2, cols=32: the flattened trailing axes"),
        Case("LayerNormalization", "axis0",
             {"X": rnd(rng, (2, 3, 4)), "Scale": pos(rng, (3, 4)),
              "B": rnd(rng, (3, 4))},
             inputs=["X", "Scale", "B"], attrs={"axis": 0}),
        Case("LayerNormalization", "no-bias",
             {"X": rnd(rng, (3, 5)), "Scale": ch(5)},
             inputs=["X", "Scale"]),
        Case("LayerNormalization", "mean-invstd",
             {"X": rnd(rng, (2, 3, 4)), "Scale": ch(4), "B": rnd(rng, (4,))},
             inputs=["X", "Scale", "B"], n_out=3,
             note="the optional Mean and InvStdDev outputs"),
        Case("LayerNormalization", "broadcast-scale",
             {"X": rnd(rng, (2, 4, 8)),
              "Scale": pos(rng, (1, 8)), "B": rnd(rng, (1, 8))},
             inputs=["X", "Scale", "B"],
             note="Scale/B broadcast over the normalised axes"),
        Case("LayerNormalization", "eps1e-3",
             {"X": rnd(rng, (2, 6)), "Scale": ch(6), "B": rnd(rng, (6,))},
             inputs=["X", "Scale", "B"], attrs={"epsilon": 1e-3}),
        Case("MeanVarianceNormalization", "default-axes",
             {"X": rnd(rng, (1, 3, 4, 4))}, attrs={"axes": [0, 2, 3]}),
        Case("MeanVarianceNormalization", "channel-axis",
             {"X": rnd(rng, (2, 3, 4, 4))}, attrs={"axes": [1]},
             note="channel-axis statistics: the channel axis is made "
                  "innermost on the QPU"),
        Case("MeanVarianceNormalization", "batch-axis",
             {"X": rnd(rng, (4, 3, 4, 4))}, attrs={"axes": [0]},
             note="a batch of 4 keeps the variance group well conditioned; "
                  "ONNX's variance (E[x^2]-mean^2) cancels for two near-equal "
                  "samples and is then only as accurate as that allows"),
        Case("LRN", "size3",
             {"X": rnd(rng, (1, 8, 4, 4))}, attrs={"size": 3},
             ref_override=lambda x: _lrn_spec(x, 3, 1e-4, 0.75, 1.0),
             note="onnx.reference 1.17 indexes LRN's channel window with the "
                  "batch index; the spec formula is supplied"),
        Case("LRN", "size4-even",
             {"X": rnd(rng, (1, 8, 3, 3))}, attrs={"size": 4},
             ref_override=lambda x: _lrn_spec(x, 4, 1e-4, 0.75, 1.0)),
        Case("LRN", "single-channel",
             {"X": rnd(rng, (1, 1, 3, 3))}, attrs={"size": 3},
             ref_override=lambda x: _lrn_spec(x, 3, 1e-4, 0.75, 1.0)),
        Case("LRN", "alpha-beta-bias",
             {"X": rnd(rng, (2, 6, 2, 2))},
             attrs={"size": 5, "alpha": 0.01, "beta": 0.5, "bias": 2.0},
             ref_override=lambda x: _lrn_spec(x, 5, 0.01, 0.5, 2.0)),
        Case("LRN", "size1",
             {"X": rnd(rng, (1, 4, 2, 2))},
             attrs={"size": 1, "alpha": 2.0, "beta": 1.0},
             ref_override=lambda x: _lrn_spec(x, 1, 2.0, 1.0, 1.0)),
    ]
    return cases


def loss_cases(rng):
    score = rnd(rng, (3, 4), -2, 2)
    logs = (score - np.log(np.exp(score).sum(1, keepdims=True))
            ).astype(np.float32)
    tgt = np.array([0, 2, 1], np.int64)
    w = f32(0.5, 2.0, 1.0, 0.25)
    cases = [
        Case("NegativeLogLikelihoodLoss", "mean",
             {"input": logs, "target": tgt}, inputs=["input", "target"]),
        Case("NegativeLogLikelihoodLoss", "none",
             {"input": logs, "target": tgt}, inputs=["input", "target"],
             attrs={"reduction": "none"}),
        Case("NegativeLogLikelihoodLoss", "sum-weighted",
             {"input": logs, "target": tgt, "weight": w},
             inputs=["input", "target", "weight"],
             attrs={"reduction": "sum"}),
        Case("NegativeLogLikelihoodLoss", "mean-ignore",
             {"input": logs, "target": np.array([0, 255, 2], np.int64)},
             inputs=["input", "target"],
             attrs={"ignore_index": 255},
             note="the ignored position leaves both numerator and count"),
        Case("NegativeLogLikelihoodLoss", "mean-ignore-neg1",
             {"input": logs, "target": np.array([0, -1, 2], np.int64)},
             inputs=["input", "target"],
             attrs={"ignore_index": -1},
             ref_override=lambda x: np.array(
                 np.mean([-x[0, 0], -x[2, 2]]), np.float32),
             note="ignore_index=-1: onnx.reference skips its weight path for "
                  "exactly -1 and averages over 3; the spec (and PyTorch) "
                  "divide by the 2 kept samples"),
        Case("NegativeLogLikelihoodLoss", "rank4-none",
             {"input": rnd(rng, (2, 3, 2, 2)),
              "target": np.array([[[0, 1], [2, 0]], [[1, 1], [0, 2]]],
                                 np.int64)},
             inputs=["input", "target"], attrs={"reduction": "none"}),
        Case("SoftmaxCrossEntropyLoss", "mean",
             {"scores": score, "labels": tgt},
             inputs=["scores", "labels"]),
        Case("SoftmaxCrossEntropyLoss", "none",
             {"scores": score, "labels": tgt},
             inputs=["scores", "labels"], attrs={"reduction": "none"}),
        Case("SoftmaxCrossEntropyLoss", "sum-weighted",
             {"scores": score, "labels": tgt, "weights": w},
             inputs=["scores", "labels", "weights"],
             attrs={"reduction": "sum"}),
        Case("SoftmaxCrossEntropyLoss", "log-prob",
             {"scores": score, "labels": tgt},
             inputs=["scores", "labels"], n_out=2),
        Case("SoftmaxCrossEntropyLoss", "mean-ignore",
             {"scores": score, "labels": np.array([0, 255, 1], np.int64)},
             inputs=["scores", "labels"],
             attrs={"ignore_index": 255, "reduction": "mean"}),
        Case("SoftmaxCrossEntropyLoss", "rank3-mean",
             {"scores": rnd(rng, (2, 5, 3), -2, 2),
              "labels": np.array([[0, 1, 4], [3, 2, 2]], np.int64)},
             inputs=["scores", "labels"]),
        Case("Dropout", "inference",
             {"data": rnd(rng, (2, 3, 4))},
             attrs={"seed": 1}),
        Case("Dropout", "inference-ratio",
             {"data": rnd(rng, (2, 3)), "ratio": np.array(0.75, np.float32)},
             inputs=["data", "ratio"], note="ratio is irrelevant in inference"),
        Case("Dropout", "mask",
             {"data": rnd(rng, (2, 3))}, n_out=2,
             out_types=[F32, BOOL],
             note="the inference mask is all ones"),
        Case("Dropout", "training-refused",
             {"data": rnd(rng, (2, 3)),
              "training_mode": np.array(1, np.bool_)},
             inputs=["data", "", "training_mode"],
             refuse="training-mode Dropout"),
    ]
    return cases


ALL = {}


def _register():
    rng = np.random.default_rng(20261120)
    for fn in (softmax_cases, pool_cases, norm_cases, loss_cases):
        for c in fn(rng):
            ALL.setdefault(c.op, []).append(c)


_register()


def run_case(eng, case, workdir, verbose):
    feeds = {k: v for k, v in case.feeds.items()
             if np.asarray(v).dtype == np.float32}
    path = os.path.join(workdir, "n_%s_%s.onnx" %
                        (case.op, case.name.replace("/", "_")))
    # 1. the expectation (a refusal case needs none)
    ref = None
    if case.refuse is None:
        try:
            if case.ref_override is not None:
                ref = [case.ref_override(*[feeds[k] for k in feeds])]
            else:
                ref = ReferenceEvaluator(build(case)).run(None, dict(feeds))
        except Exception as e:
            return "oracle-error", "%s: %s" % (type(e).__name__, e)
    # 2. the graph, with the output shapes it actually has
    model = build(case, None if ref is None else
                  [np.asarray(x) for x in ref])
    onnx.save(model, path)
    # 3. the runner
    try:
        r = Runner(path, eng, mode="hybrid", host_policy="device")
        got = r.run({k: np.ascontiguousarray(v, np.float32)
                     for k, v in feeds.items()})
    except UnsupportedNode as e:
        if case.refuse is not None and case.refuse in str(e):
            return "refused-ok", ""
        if case.refuse is not None:
            return "refused-wrong", "wanted %r, got %s" % (case.refuse, e)
        return "refused", str(e)
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return "error", "%s: %s" % (type(e).__name__, e)
    if case.refuse is not None:
        return "not-refused", "expected a refusal naming %r" % case.refuse
    return _check(ref, got)


def _check(ref, got):
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
        os.environ.get("TMPDIR", "/tmp"), "vc4norm-models")
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
                ok = status in ("vc4", "refused-ok")
                print("%-4s %-26s %-24s %s%s"
                      % ("ok" if ok else "FAIL", op, case.name, status,
                         "" if ok else "  " + msg[:170]))
                if not ok:
                    bad.append((op, case.name, status, msg))
    finally:
        eng.close()
    print("\n%d cases, %d failed" % (total, len(bad)))
    if bad:
        from collections import Counter
        print("by status:", dict(Counter(b[2] for b in bad)))
        for op, name, status, msg in bad:
            print("  %-26s %-24s %-12s %s" % (op, name, status, msg[:140]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
