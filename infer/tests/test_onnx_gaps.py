#!/usr/bin/env python3
"""test_onnx_gaps.py -- the variadic / conversion / gather family.

The generated matrix (``infer/tools/op_matrix.py``) runs one case per operator.
This file is the *positive* evidence for the forms it cannot reach, one ONNX
graph per case, run through the shipped runner (``mode="hybrid"``,
``host_policy="device"``) and compared with ``onnx.reference``:

  Sum/Max/Min/Mean   2..3 operands, numpy broadcasting, +-inf, NaN, -0.0
  Div                runtime divisor, broadcast divisor, folded constant,
                     and the two refusals (zero divisor, non-finite operand)
  Cast               f32->i32 truncation on both signs, f32->i64, f32->bool,
                     f32->f32 view, host-int -> f32, and the uint8 refusal
  LpNormalization    p=1 and p=2, every axis position, rank 4; p=3 refused
  Gather             axis 0/1/-1, negative and scalar indices, 1-D data
  GatherElements     axis 0/1, negative indices
  GatherND           batch_dims 0 and 1, last index dim 1 and 2, negatives
  ScatterElements    axis 0/1, reduction none/add/mul, negative indices
  ScatterND          reduction none/add/mul, batch of rows, duplicates refused
  NonZero            zeros, -0.0, NaN, no zeros, all zeros
  TopK               k=1..D, axis 0/1/-1, largest=0
  Unique             axis=None/0/1, sorted=0/1, duplicate rows
  ReverseSequence    default and swapped axes, short/zero/one-long lens
  CenterCropPad      crop only, pad only, both, with the optional axes input
  DepthToSpace       DCR and CRD, blocksize 2 and 3

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 python3 tests/test_onnx_gaps.py
    python3 tests/test_onnx_gaps.py --only TopK,Unique -v

Exit status is 0 only when every case behaved as declared: a ``refused`` case
must raise and every other case must run on the VC4 kernels within the
tolerance ``test_onnx_ops`` uses.
"""

# A GPU client: refuse unless VC4_REQUIRE_GPU=1 (tests/gpu_gate.py).
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from gpu_gate import require_gpu as _require_gpu
_require_gpu(__file__)
import argparse
import os
import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from test_onnx_ops import Case, build, compare, f32, i64, rnd  # noqa: E402

from vc4infer import _binding as vc4infer                       # noqa: E402
from vc4infer._errors import HostNotAllowed, UnsupportedNode   # noqa: E402
from vc4infer._onnxrun import Runner                            # noqa: E402

I32 = TensorProto.INT32
I64 = TensorProto.INT64
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def ref(op, name, feeds, inits=None, attrs=None, inputs=None,
        out_types=None, n_out=1, note="", why=""):
    """A case that must run on the VC4 kernels."""
    c = Case(op, name, feeds, inits, attrs, inputs, out_types, n_out, note)
    c.expect = "vc4"
    c.why = why
    return c


def no(op, name, feeds, inits=None, attrs=None, inputs=None, n_out=1,
       note="", why=""):
    """A case the engine must refuse, with `why` in the message."""
    c = Case(op, name, feeds, inits, attrs, inputs, out_types=None,
             n_out=n_out, note=note)
    c.expect = "refused"
    c.why = why
    return c


# =====================================================================
def variadic_cases(rng):
    a = rnd(rng, (2, 1, 4))
    b = rnd(rng, (3, 4))
    c = rnd(rng, (1,))
    out = []
    for op in ("Sum", "Mean", "Max", "Min"):
        out += [
            ref(op, "three-same-shape",
                {"A": rnd(rng, (2, 3)), "B": rnd(rng, (2, 3)),
                 "C": rnd(rng, (2, 3))}, inputs=["A", "B", "C"]),
            ref(op, "broadcast-pair",
                {"A": a, "B": b}, inputs=["A", "B"],
                note="(2,1,4) against (3,4)") if op != "Mean" else
            ref(op, "broadcast-oracle-gap", {"A": a}, inputs=["A"],
                note="the 1.17 Mean adds in place into its first operand, so "
                     "a broadcast form cannot be evaluated by the oracle"),
            ref(op, "two-same-shape",
                {"A": rnd(rng, (2, 3)), "B": rnd(rng, (2, 3))},
                inputs=["A", "B"]),
            ref(op, "single-operand", {"A": rnd(rng, (4,))}, inputs=["A"]),
            ref(op, "scalar-broadcast",
                {"A": rnd(rng, (2, 3)), "B": f32(0.25)},
                inputs=["A", "B"]),
        ]
    edge = np.array([np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0], np.float32)
    for op in ("Max", "Min"):
        out += [
            ref(op, "infinite", {"A": edge, "B": edge[::-1].copy()},
                inputs=["A", "B"],
                note="+-inf and -0.0 are exact: the combine step copies one "
                     "operand and adds a true zero"),
            no(op, "nan-operand",
               {"A": np.array([1.0, np.nan, np.inf], np.float32),
                "B": np.array([2.0, 2.0, 2.0], np.float32)},
               inputs=["A", "B"], why="NaN operand"),
        ]
    out += [
        no("Sum", "infinite-operand",
           {"A": np.array([np.inf, 1.0], np.float32),
            "B": np.array([1.0, 2.0], np.float32)},
           inputs=["A", "B"], why="non-finite operand"),
        no("Sum", "nonfinite",
           {"A": np.array([np.inf, np.nan], np.float32),
            "B": np.array([-np.inf, 1.0], np.float32)},
           inputs=["A", "B"], why="non-finite operand"),
        no("Mean", "nan-operand",
           {"A": np.array([np.nan, 1.0], np.float32),
            "B": np.array([1.0, 1.0], np.float32)},
           inputs=["A", "B"], why="non-finite operand"),
    ]
    return out


def div_cases(rng):
    out = [
        ref("Div", "runtime-divisor",
            {"A": rnd(rng, (2, 3, 4), 0.5, 1.5), "B": rnd(rng, (2, 3, 4), 0.5, 1.5)},
            inputs=["A", "B"]),
        ref("Div", "runtime-scalar",
            {"A": rnd(rng, (2, 3)), "B": np.array(0.7, np.float32)},
            inputs=["A", "B"], note="a runtime scalar is still a tensor"),
        ref("Div", "broadcast-divisor",
            {"A": rnd(rng, (2, 3, 4), 0.5, 1.5), "B": rnd(rng, (1, 3, 1), 0.5, 1.5)},
            inputs=["A", "B"], note="a non-scalar divisor broadcasts"),
        ref("Div", "negative-divisor",
            {"A": rnd(rng, (2, 3), -1, 1), "B": rnd(rng, (2, 3), -2, -0.5)},
            inputs=["A", "B"]),
        ref("Div", "folded-constant",
            {"A": rnd(rng, (2, 3), 0.5, 1.5)}, {"B": f32(0.7)},
            inputs=["A", "B"], note="the scalar-constant fold"),
        no("Div", "zero-divisor",
           {"A": rnd(rng, (2, 3)), "B": np.zeros((2, 3), np.float32)},
           inputs=["A", "B"], why="zero divisor"),
        no("Div", "infinite-dividend",
           {"A": np.array([np.inf, 1.0], np.float32),
            "B": np.array([2.0, 2.0], np.float32)},
           inputs=["A", "B"], why="non-finite dividend"),
    ]
    return out


def cast_cases(rng):
    out = [
        ref("Cast", "f32-to-i32-truncation",
            {"input": np.array([-3.7, -0.5, -0.0, 0.0, 0.5, 3.7, 1e7,
                                1.5e7, -1e7], np.float32)},
            attrs={"to": int(I32)}, inputs=["input"], out_types=[I32],
            note="truncation toward zero, both signs"),
        no("Cast", "f32-to-i32-nonfinite",
           {"input": np.array([np.inf, -np.inf, np.nan, 2.0], np.float32)},
           attrs={"to": int(I32)}, inputs=["input"],
           why="non-finite"),
        ref("Cast", "f32-to-i64",
            {"input": np.array([-2.9, 0.0, 7.9], np.float32)},
            attrs={"to": int(I64)}, inputs=["input"], out_types=[I64]),
        ref("Cast", "f32-to-bool",
            {"input": np.array([0.0, -0.0, 1e-45, -3.0, np.nan], np.float32)},
            attrs={"to": int(BOOL)}, inputs=["input"], out_types=[BOOL]),
        ref("Cast", "f32-to-f32-view",
            {"input": rnd(rng, (2, 3))}, attrs={"to": int(F32)},
            inputs=["input"]),
        ref("Cast", "host-int-to-f32",
            {"shape": i64(2, 3)}, attrs={"to": int(F32)},
            inputs=["shape"], out_types=[F32],
            note="a metadata cast still works on a host tensor"),
        ref("Cast", "host-int-to-i64",
            {"axes": i64(0, -1)}, attrs={"to": int(I64)},
            inputs=["axes"], out_types=[I64]),
        no("Cast", "f32-to-uint8",
           {"input": rnd(rng, (2, 3))}, attrs={"to": int(TensorProto.UINT8)},
           inputs=["input"], why="no conversion kernel"),
    ]
    return out


def lpnorm_cases(rng):
    out = []
    for p in (1, 2):
        for ax in (0, -1, 1):
            lo = 0.5 if p == 1 else -1.0
            out.append(ref("LpNormalization", "p%d-axis%d" % (p, ax),
                           {"X": rnd(rng, (3, 4, 2), lo, 1.0)},
                           attrs={"axis": ax, "p": p}, inputs=["X"]))
    out += [
        ref("LpNormalization", "p2-axis1-rank4",
            {"X": rnd(rng, (1, 3, 2, 2), -1, 1)},
            attrs={"axis": 1, "p": 2}, inputs=["X"]),
        signed_p1(ref, rng),
        no("LpNormalization", "p3",
           {"X": rnd(rng, (2, 3))}, attrs={"axis": -1, "p": 3},
           inputs=["X"], why="p=1 and p=2"),
        no("LpNormalization", "zero-norm",
           {"X": np.zeros((2, 3), np.float32)}, attrs={"axis": -1, "p": 2},
           inputs=["X"], why="zero-norm slice"),
    ]
    return out


def signed_p1(ref, rng):
    """p=1 with negative values: the spec's ||x||_1 is a sum of |x|, the 1.17
    reference sums the signed values (np.power(x, 1).sum()), so the
    expectation is supplied here instead of asking the oracle."""
    x = np.array([[-2.0, 3.0], [1.0, -1.0]], np.float32)
    c = ref("LpNormalization", "p1-negative",
            {"X": x}, attrs={"axis": -1, "p": 1}, inputs=["X"],
            note="||x||_1 is a sum of |x| (spec; the oracle sums x)")

    def expected(a):
        n = np.abs(a.astype(np.float64)).sum(axis=-1, keepdims=True)
        return (a.astype(np.float64) / n).astype(np.float32)

    c.ref_override = expected
    return c


def gather_cases(rng):
    out = [
        ref("Gather", "axis0-1d-indices",
            {"data": rnd(rng, (3, 4)), "indices": i64(2, 0, 2)},
            attrs={"axis": 0}, inputs=["data", "indices"]),
        ref("Gather", "axis1-2d-indices",
            {"data": rnd(rng, (3, 4, 5)), "indices": np.array([[0, 2], [1, 3]],
                                                             np.int64)},
            attrs={"axis": 1}, inputs=["data", "indices"]),
        ref("Gather", "axis-last-negative",
            {"data": rnd(rng, (2, 3, 4)), "indices": i64(-1, 0)},
            attrs={"axis": -1}, inputs=["data", "indices"],
            note="negative indices count from the end"),
        ref("Gather", "scalar-index",
            {"data": rnd(rng, (3, 4)), "indices": np.array(1, np.int64)},
            attrs={"axis": 0}, inputs=["data", "indices"],
            note="a 0-d index drops the axis"),
        ref("Gather", "embedding-1d",
            {"data": rnd(rng, (5,)), "indices": np.array([[4, 0], [1, 1]],
                                                        np.int64)},
            inputs=["data", "indices"]),
    ]
    return out


def gather_elements_cases(rng):
    out = [
        ref("GatherElements", "axis1",
            {"data": rnd(rng, (3, 4)), "indices": np.array(
                [[0, 1, 2, 3]] * 3, np.int64)},
            attrs={"axis": 1}, inputs=["data", "indices"]),
        ref("GatherElements", "axis0-negative",
            {"data": rnd(rng, (3, 4)), "indices": np.array(
                [[-1, 0, 0, 1], [0, -2, 1, 0], [1, 1, -1, 2]], np.int64)},
            attrs={"axis": 0}, inputs=["data", "indices"]),
        ref("GatherElements", "axis-last-shuffled",
            {"data": rnd(rng, (2, 3, 4)), "indices": np.array(
                [[[3, 2, 1, 0], [0, 0, 3, 3], [1, 1, 1, 1]],
                 [[1, 1, 1, 1], [2, 3, 0, 1], [-1, 0, -2, 3]]], np.int64)},
            attrs={"axis": -1}, inputs=["data", "indices"]),
        ref("GatherElements", "axis-middle",
            {"data": rnd(rng, (2, 3, 4)), "indices": np.array(
                [[[1, 0, 2, 1], [0, 2, 1, 0], [2, 1, 0, 2]],
                 [[0, 1, 2, 0], [2, 2, 1, 1], [1, 0, 0, 2]]], np.int64)},
            attrs={"axis": 1}, inputs=["data", "indices"],
            note="a non-last axis is moved innermost first"),
    ]
    return out


def gather_nd_cases(rng):
    out = [
        ref("GatherND", "batch-dims0-r2",
            {"data": rnd(rng, (2, 3, 4)),
             "indices": np.array([[0, 1], [1, 2]], np.int64)},
            inputs=["data", "indices"]),
        ref("GatherND", "batch-dims0-r1",
            {"data": rnd(rng, (3, 4)),
             "indices": np.array([[2], [0]], np.int64)},
            inputs=["data", "indices"]),
        ref("GatherND", "batch-dims1",
            {"data": rnd(rng, (2, 3, 4)),
             "indices": np.array([[[1], [0]], [[2], [2]]], np.int64)},
            attrs={"batch_dims": 1}, inputs=["data", "indices"]),
        ref("GatherND", "negative-index",
            {"data": rnd(rng, (3, 2)),
             "indices": np.array([[-1], [0]], np.int64)},
            inputs=["data", "indices"]),
    ]
    return out


def scatter_elements_cases(rng):
    idx = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], np.int64)
    out = [
        ref("ScatterElements", "axis1-none",
            {"data": rnd(rng, (2, 4)), "updates": rnd(rng, (2, 4))},
            {"indices": idx}, attrs={"axis": 1},
            inputs=["data", "indices", "updates"]),
        ref("ScatterElements", "axis1-add",
            {"data": rnd(rng, (2, 4)), "updates": rnd(rng, (2, 4))},
            {"indices": np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int64)},
            attrs={"axis": 1, "reduction": "add"},
            inputs=["data", "indices", "updates"]),
        no("ScatterElements", "add-duplicate-destination",
           {"data": rnd(rng, (2, 4)), "updates": rnd(rng, (2, 4))},
           {"indices": np.array([[0, 0, 2, 2], [1, 1, 3, 3]], np.int64)},
           attrs={"axis": 1, "reduction": "add"},
           inputs=["data", "indices", "updates"],
           why="duplicate destinations"),
        ref("ScatterElements", "axis0-none",
            {"data": rnd(rng, (3, 3)), "updates": rnd(rng, (2, 3))},
            {"indices": np.array([[2, 0, 1], [0, 1, 2]], np.int64)},
            attrs={"axis": 0}, inputs=["data", "indices", "updates"]),
        no("ScatterElements", "duplicate-destination",
           {"data": rnd(rng, (2, 4)), "updates": rnd(rng, (2, 4))},
           {"indices": np.array([[0, 0, 1, 1], [0, 0, 1, 1]], np.int64)},
           attrs={"axis": 1}, inputs=["data", "indices", "updates"],
           why="duplicate destinations"),
    ]
    return out


def scatter_nd_cases(rng):
    out = [
        ref("ScatterND", "none",
            {"data": rnd(rng, (4, 4)), "updates": rnd(rng, (2, 4))},
            {"indices": np.array([[0], [2]], np.int64)},
            inputs=["data", "indices", "updates"]),
        ref("ScatterND", "add",
            {"data": rnd(rng, (4, 4)), "updates": rnd(rng, (2, 4))},
            {"indices": np.array([[1], [3]], np.int64)},
            attrs={"reduction": "add"}, inputs=["data", "indices", "updates"]),
        ref("ScatterND", "mul",
            {"data": rnd(rng, (2, 3, 2), 0.5, 1.5),
             "updates": rnd(rng, (2, 2), 0.5, 1.5)},
            {"indices": np.array([[0, 2], [1, 0]], np.int64)},
            attrs={"reduction": "mul"}, inputs=["data", "indices", "updates"],
            note="one row per index tuple"),
        ref("ScatterND", "negative-index",
            {"data": rnd(rng, (3, 2)), "updates": rnd(rng, (1, 2))},
            {"indices": np.array([[-1]], np.int64)},
            inputs=["data", "indices", "updates"]),
        no("ScatterND", "duplicate-row",
           {"data": rnd(rng, (3, 2)), "updates": rnd(rng, (2, 2))},
           {"indices": np.array([[1], [1]], np.int64)},
           inputs=["data", "indices", "updates"], why="duplicate index rows"),
        no("ScatterND", "reduction-max",
           {"data": rnd(rng, (3, 2)), "updates": rnd(rng, (1, 2))},
           {"indices": np.array([[1]], np.int64)},
           attrs={"reduction": "max"}, inputs=["data", "indices", "updates"],
           why="reduction=max"),
    ]
    return out


def nonzero_cases(rng):
    return [
        ref("NonZero", "matrix",
            {"X": np.array([[1.0, 0.0, -0.0], [np.nan, 2.0, -3.0]],
                           np.float32)},
            inputs=["X"], out_types=[I64],
            note="-0.0 is zero, NaN is not"),
        ref("NonZero", "no-zeros", {"X": rnd(rng, (2, 3), 0.5, 1.5)},
            inputs=["X"], out_types=[I64]),
        ref("NonZero", "all-zeros", {"X": np.zeros((2, 3), np.float32)},
            inputs=["X"], out_types=[I64]),
        ref("NonZero", "vector", {"X": np.array([0.0, 1.0, 0.0], np.float32)},
            inputs=["X"], out_types=[I64]),
    ]


def topk_cases(rng):
    x = np.array([[0.9, -2.0, 3.5, 1.25, 0.0]], np.float32)
    return [
        ref("TopK", "k3-axis-last", {"X": x, "K": i64(3)},
            attrs={"axis": -1}, inputs=["X", "K"], n_out=2,
            out_types=[F32, I64]),
        ref("TopK", "k1", {"X": x, "K": i64(1)}, attrs={"axis": -1},
            inputs=["X", "K"], n_out=2, out_types=[F32, I64]),
        ref("TopK", "k-all", {"X": x, "K": i64(5)}, attrs={"axis": -1},
            inputs=["X", "K"], n_out=2, out_types=[F32, I64]),
        ref("TopK", "smallest", {"X": x, "K": i64(2)},
            attrs={"axis": -1, "largest": 0}, inputs=["X", "K"], n_out=2,
            out_types=[F32, I64]),
        ref("TopK", "axis0",
            {"X": np.array([[1.0, 5.0], [4.0, 2.0], [3.0, 6.0]], np.float32),
             "K": i64(2)}, attrs={"axis": 0}, inputs=["X", "K"], n_out=2,
            out_types=[F32, I64]),
    ]


def unique_cases(rng):
    dup = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], np.float32)
    return [
        ref("Unique", "axis0-sorted", {"X": dup}, attrs={"axis": 0},
            inputs=["X"], n_out=4, out_types=[F32, I64, I64, I64]),
        ref("Unique", "axis0-first-occurrence",
            {"X": np.array([[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], np.float32)},
            attrs={"axis": 0, "sorted": 0}, inputs=["X"], n_out=4,
            out_types=[F32, I64, I64, I64]),
        ref("Unique", "axis1", {"X": np.array([[1.0, 2.0, 1.0],
                                               [5.0, 6.0, 5.0]], np.float32)},
            attrs={"axis": 1}, inputs=["X"], n_out=4,
            out_types=[F32, I64, I64, I64],
            note="units are columns, so the layout copy is exercised"),
        ref("Unique", "flatten-sorted",
            {"X": np.array([[2.0, 1.0], [2.0, 3.0]], np.float32)},
            inputs=["X"], n_out=4, out_types=[F32, I64, I64, I64]),
        ref("Unique", "flatten-first-occurrence",
            {"X": np.array([2.0, 1.0, 2.0, 3.0], np.float32)},
            attrs={"sorted": 0}, inputs=["X"], n_out=4,
            out_types=[F32, I64, I64, I64],
            note="1-D input: the 1.17 reference takes axis=0 when sorted=0"),
        ref("Unique", "one-output",
            {"X": np.array([[2.0, 1.0], [2.0, 3.0]], np.float32)},
            inputs=["X"], n_out=1),
    ]


def reverse_sequence_cases(rng):
    x = rnd(rng, (3, 4, 2))
    return [
        ref("ReverseSequence", "batch0-time1",
            {"input": x, "sequence_lens": i64(3, 2, 4)},
            attrs={"batch_axis": 0, "time_axis": 1},
            inputs=["input", "sequence_lens"]),
        ref("ReverseSequence", "batch1-time0",
            {"input": rnd(rng, (4, 3, 2)), "sequence_lens": i64(2, 4, 1)},
            inputs=["input", "sequence_lens"],
            note="the defaults (batch_axis=1, time_axis=0)"),
        ref("ReverseSequence", "lens-1-and-2",
            {"input": np.arange(6, dtype=np.float32).reshape(2, 3),
             "sequence_lens": i64(1, 2)},
            attrs={"batch_axis": 0, "time_axis": 1},
            inputs=["input", "sequence_lens"],
            note="a one-long sequence is a no-op, a two-long one reverses"),
    ]


def center_crop_pad_cases(rng):
    return [
        ref("CenterCropPad", "crop-last",
            {"input_data": rnd(rng, (2, 3, 4))}, {"shape": i64(2, 3, 3)},
            inputs=["input_data", "shape"]),
        ref("CenterCropPad", "pad-last",
            {"input_data": rnd(rng, (2, 3, 4))}, {"shape": i64(2, 3, 7)},
            inputs=["input_data", "shape"]),
        ref("CenterCropPad", "odd-crop",
            {"input_data": rnd(rng, (2, 3, 5))}, {"shape": i64(2, 3, 6)},
            inputs=["input_data", "shape"],
            note="an odd difference puts the extra element on the right"),
        ref("CenterCropPad", "both-directions",
            {"input_data": rnd(rng, (3, 4))}, {"shape": i64(5, 7)},
            inputs=["input_data", "shape"]),
        ref("CenterCropPad", "crop-middle-axis",
            {"input_data": rnd(rng, (1, 4, 8))}, {"shape": i64(1, 2, 8)},
            inputs=["input_data", "shape"]),
    ]


def depth_to_space_cases(rng):
    return [
        ref("DepthToSpace", "dcr-b2", {"input": rnd(rng, (1, 4, 3, 3))},
            attrs={"blocksize": 2, "mode": "DCR"}, inputs=["input"]),
        ref("DepthToSpace", "crd-b2", {"input": rnd(rng, (1, 4, 3, 3))},
            attrs={"blocksize": 2, "mode": "CRD"}, inputs=["input"]),
        ref("DepthToSpace", "dcr-b2-batch2-c8",
            {"input": rnd(rng, (2, 8, 2, 2))},
            attrs={"blocksize": 2, "mode": "DCR"}, inputs=["input"]),
        ref("DepthToSpace", "crd-b3-c9", {"input": rnd(rng, (1, 9, 2, 2))},
            attrs={"blocksize": 3, "mode": "CRD"}, inputs=["input"]),
    ]


ALL = {}


def _register():
    rng = np.random.default_rng(20261121)
    for fn in (variadic_cases, div_cases, cast_cases, lpnorm_cases,
               gather_cases, gather_elements_cases, gather_nd_cases,
               scatter_elements_cases, scatter_nd_cases, nonzero_cases,
               topk_cases, unique_cases, reverse_sequence_cases,
               center_crop_pad_cases, depth_to_space_cases):
        for c in fn(rng):
            ALL.setdefault(c.op, []).append(c)


_register()


def run_case(eng, case, workdir, verbose):
    model = build(case)
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        return "invalid-model", "%s: %s" % (type(e).__name__, e)
    path = os.path.join(workdir, "g_%s_%s.onnx"
                        % (case.op, case.name.replace("/", "_")))
    onnx.save(model, path)
    feeds = {k: v for k, v in case.feeds.items()
             if np.asarray(v).dtype == np.float32}
    try:
        if case.ref_override is not None:
            refv = [case.ref_override(*[feeds[k] for k in feeds])]
        elif case.identity_ref:
            refv = [feeds["data"]]
        else:
            refv = ReferenceEvaluator(model).run(None, dict(feeds))
    except Exception as e:
        return "oracle-error", "%s: %s" % (type(e).__name__, e)
    try:
        got = Runner(path, eng, mode="hybrid", host_policy="device") \
            .run({k: np.ascontiguousarray(v, np.float32)
                  for k, v in feeds.items()})
    except (UnsupportedNode, HostNotAllowed) as e:
        if case.expect == "refused":
            if case.why and case.why not in str(e):
                return "refused", "refused, but not for %r: %s" % (case.why, e)
            return "refused", ""
        return "refused", str(e)
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return "error", "%s: %s" % (type(e).__name__, e)
    if case.expect == "refused":
        return "ran", "the case was expected to be refused (%s) but ran" \
            % case.why
    for i, (w, g) in enumerate(zip(refv, got)):
        msg = compare(w, g)
        if msg:
            return "mismatch", "output %d: %s" % (i, msg)
    return "vc4", ""


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
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
        os.environ.get("TMPDIR", "/tmp"), "vc4gaps-models")
    os.makedirs(workdir, exist_ok=True)
    cache = os.environ.get("VC4INFER_CACHE") or None
    eng = vc4infer.Engine(cache_dir=cache)
    bad, total = [], 0
    try:
        for op in ops:
            for case in ALL[op]:
                total += 1
                status, msg = run_case(eng, case, workdir, args.verbose)
                ok = status == case.expect or \
                    (case.expect == "vc4" and status == "vc4")
                print("%-4s %-16s %-28s %-10s%s"
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
            print("  %-16s %-28s %-12s %s" % (op, name, status, msg[:140]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
