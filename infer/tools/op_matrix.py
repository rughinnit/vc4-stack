#!/usr/bin/env python3
"""op_matrix.py -- a *generated* ONNX operator-coverage matrix for vc4infer.

Why this exists
---------------
A hand-maintained "supported nodes" table cannot prove that an engine is
complete, and cannot prove that it is correct.  This tool derives the operator
set from the ONNX standard itself (``onnx.defs``), synthesises one minimal
valid graph per operator plus the attribute variants that matter, and runs
every graph twice on the same inputs:

  1. ``vc4infer._onnxrun.Runner`` with ``mode="hybrid", host_policy="device"``
     -- the shipped engine, the policy that forbids runtime host compute;
  2. ``onnx.reference.ReferenceEvaluator`` -- an independent numpy oracle.

Each operator is classified from what actually happened:

  vc4            every required case ran, output within 1e-4 + 1e-4*max(|ref|,1)
  vc4-limited    the general case ran, but at least one required ONNX form was
                 refused (the constraint is named in the report)
  refused        even the general case was refused (UnsupportedNode /
                 HostNotAllowed / NeedsDeviceDispatch), with the engine's text
  mismatch       the engine ran and disagreed with the oracle -- a BUG
  hang           the device did not return, or was lost; never scored as vc4
  invalid-model  our synthesis was wrong (reported, never scored)
  unmeasured     graph built and validated, but no vc4 result in this revision
  untested       no synthesised graph at all; counted against completeness

``--skip-gpu`` produces the graph-validity pass (``onnx.checker`` plus
``onnx.reference``), which needs no vc4 package and no device.  In that mode
every scoreable row is ``unmeasured`` and no percentage is reported: an
unmeasured row is never counted as supported.

The denominator is deliberately adversarial.  Two numbers are reported and
never conflated:

  * of ALL operators in the opset (including shape/metadata, string, sequence
    and control-flow operators that a static fp32 device cannot express);
  * of operators that are not structurally host-only.

Usage
-----
    TMPDIR=$HOME/scratch/tmp python3 infer/tools/op_matrix.py
    python3 infer/tools/op_matrix.py --skip-gpu          # no device needed
    python3 infer/tools/op_matrix.py --only Conv,Resize --skip-gpu
    python3 infer/tools/op_matrix.py --opset 17 --out-dir /tmp/out

The tool never calls the GPU for an operator the policy refuses: ``check_policy``
runs before any node executes, and a graph is a single tiny operator, run
strictly sequentially.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import platform
import re
import signal
import sys
import tempfile
import time
import traceback
import zlib
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

# The graph-validity pass needs onnx + numpy only.  The engine package may be
# mid-purge or mid-reinstall, so importing it is optional and deferred to the
# GPU phase.  A missing package is a first-class, recorded condition, never an
# import crash.
try:
    import vc4infer
    from vc4infer import _runtime
    from vc4infer._errors import (HostNotAllowed, HostNodeError,
                                  NeedsDeviceDispatch, UnsupportedNode)
    VC4_IMPORT_ERROR = None
except Exception as _vc4err:                      # noqa: BLE001
    vc4infer = None
    _runtime = None
    VC4_IMPORT_ERROR = "%s: %s" % (type(_vc4err).__name__, _vc4err)

    class _Vc4Missing(Exception):
        """Placeholder so the refusal handlers stay importable."""

    UnsupportedNode = HostNotAllowed = NeedsDeviceDispatch = \
        HostNodeError = _Vc4Missing

# Set when a run loses the device (timeout, dead daemon).  The remaining cases
# are recorded as ``hang`` instead of failing the whole sweep.
GPU_STATE = {"dead": None}
CASE_TIMEOUT_S = 60.0

TOL_ABS = 1e-4
TOL_REL = 1e-4

# Explanations for mismatches that this tool found and that are reproducible by
# hand.  The measurement is generated; only the explanation is curated.
KNOWN_BUGS = {
    ("Resize", "sizes"):
        "same half_pixel rounding bug: the sizes form also uses ONNX's default "
        "coordinate_transformation_mode=half_pixel.",
    ("Resize", "nearest_half_pixel"):
        "the engine computes floor((s+0.5)*scale-0.5) for every nearest mode; "
        "ONNX's default nearest_mode=round_prefer_floor rounds that coordinate, "
        "so pixels at .75 differ (docs/ONNX.md does not state a nearest_mode "
        "constraint). The floor variant of the same case passes.",
}

NP2ONNX = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int16"): TensorProto.INT16,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("uint16"): TensorProto.UINT16,
    np.dtype("uint32"): TensorProto.UINT32,
    np.dtype("uint64"): TensorProto.UINT64,
    np.dtype("bool"): TensorProto.BOOL,
}

ONNX2NP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.INT64: np.int64,
    TensorProto.INT32: np.int32,
    TensorProto.INT16: np.int16,
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}

def _is_optional(param):
    return str(param.option).endswith("Optional")


def _is_variadic(param):
    return str(param.option).endswith("Variadic")


TENSOR_TYPE_NAMES = {
    "float": TensorProto.FLOAT, "double": TensorProto.DOUBLE,
    "float16": TensorProto.FLOAT16, "bfloat16": TensorProto.BFLOAT16,
    "int64": TensorProto.INT64, "int32": TensorProto.INT32,
    "int16": TensorProto.INT16, "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8, "uint16": TensorProto.UINT16,
    "uint32": TensorProto.UINT32, "uint64": TensorProto.UINT64,
    "bool": TensorProto.BOOL, "string": TensorProto.STRING,
    "complex64": TensorProto.COMPLEX64,
    "complex128": TensorProto.COMPLEX128,
}

# ---------------------------------------------------------------- structural
# Operators that a static, fp32, tensor-in/tensor-out device cannot express
# *by construction*.  This is the only exclusion from the second denominator;
# everything else -- including operators the engine has never heard of -- is
# counted.
HOST_ONLY = {
    # shape / metadata / constant materialisation: no activation arithmetic
    "Shape": ("metadata", "shape metadata only (folded at plan time)"),
    "Size": ("metadata", "element-count metadata only (folded at plan time)"),
    "Constant": ("metadata", "a literal, not a computation"),
    "ConstantOfShape": ("metadata", "constant materialisation only"),
    # control flow: needs a device-side scheduler, not an operator kernel
    "If": ("control-flow", "subgraph branch chosen at run time"),
    "Loop": ("control-flow", "subgraph executed a data-dependent number of times"),
    "Scan": ("control-flow", "subgraph executed along a scanned axis"),
    # string operators: no string type crosses the fp32 arena boundary
    "RegexFullMatch": ("string", "string tensors"),
    "StringConcat": ("string", "string tensors"),
    "StringNormalizer": ("string", "string tensors"),
    "StringSplit": ("string", "string tensors"),
    "TfIdfVectorizer": ("string", "string tensors"),
    # sequence / optional: non-tensor values
    "ConcatFromSequence": ("sequence/optional", "sequence value"),
    "SequenceAt": ("sequence/optional", "sequence value"),
    "SequenceConstruct": ("sequence/optional", "sequence value"),
    "SequenceEmpty": ("sequence/optional", "sequence value"),
    "SequenceErase": ("sequence/optional", "sequence value"),
    "SequenceInsert": ("sequence/optional", "sequence value"),
    "SequenceLength": ("sequence/optional", "sequence value"),
    "SequenceMap": ("sequence/optional", "sequence value and a graph body"),
    "SplitToSequence": ("sequence/optional", "sequence value"),
    "Optional": ("sequence/optional", "optional value"),
    "OptionalGetElement": ("sequence/optional", "optional value"),
    "OptionalHasElement": ("sequence/optional", "optional value"),
    # encoded image I/O: the bytes are produced by the host, not by a kernel
    "ImageDecoder": ("io-decode", "decodes an encoded image; host I/O by nature"),
}

# Operators whose *only* tensor types are integer/bool: they are not
# expressible with an fp32 activation at all (the engine boundary is fp32).
NON_FP32 = {
    "And": "bool tensors", "Or": "bool tensors", "Xor": "bool tensors",
    "Not": "bool tensors",
    "BitShift": "integer tensors", "BitwiseAnd": "integer tensors",
    "BitwiseNot": "integer tensors", "BitwiseOr": "integer tensors",
    "BitwiseXor": "integer tensors",
    "ConvInteger": "int8/uint8 activation (quantized)",
    "MatMulInteger": "int8/uint8 activation (quantized)",
    "QLinearConv": "int8/uint8 activation (quantized)",
    "QLinearMatMul": "int8/uint8 activation (quantized)",
    "DequantizeLinear": "int8/uint8 activation (quantized)",
}

# QuantizeLinear and DynamicQuantizeLinear consume fp32 activations, so they
# stay in the denominator even though they produce int8.


# --------------------------------------------------------------------- spec
@dataclass
class Spec:
    op: str
    variant: str
    ins: list
    feeds: dict
    inits: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    required: bool = True
    edge: bool = False
    opset: int | None = None
    n_out: int | None = None
    out_types: list | None = None
    note: str = ""
    oracle_gap: bool = False


def arr(rng, shape, lo=None, hi=None, dtype=np.float32):
    if lo is None:
        return rng.standard_normal(shape).astype(dtype)
    return rng.uniform(lo, hi, size=shape).astype(dtype)


def i64(*values):
    return np.asarray(values, dtype=np.int64)


def f32(*values):
    return np.asarray(values, dtype=np.float32)


def sp(op, variant, ins, feeds, inits=None, attrs=None, *, required=True,
       edge=False, opset=None, n_out=None, out_types=None, note="",
       oracle_gap=False):
    return Spec(op=op, variant=variant, ins=list(ins), feeds=feeds,
                inits=inits or {}, attrs=attrs or {}, required=required,
                edge=edge, opset=opset, n_out=n_out, out_types=out_types,
                note=note, oracle_gap=oracle_gap)


# --------------------------------------------------------------- model build
def _elem_of(type_str, bindings):
    if type_str in bindings:
        return bindings[type_str]
    m = re.match(r"tensor\(([a-z0-9]+)\)$", type_str)
    if m:
        return TENSOR_TYPE_NAMES.get(m.group(1), TensorProto.UNDEFINED)
    return TensorProto.UNDEFINED


def build_model(spec, schema, default_opset):
    opset = spec.opset or default_opset
    n_out = spec.n_out
    if n_out is None:
        n_out = sum(1 for o in schema.outputs
                    if not _is_optional(o))
        n_out = max(n_out, 1)
    out_names = ["Y%d" % i for i in range(n_out)]

    bindings = {}
    for i, name in enumerate(spec.ins):
        if i >= len(schema.inputs):
            break
        value = spec.feeds.get(name)
        if value is None:
            value = spec.inits.get(name)
        if value is None:
            continue
        bindings[schema.inputs[i].type_str] = NP2ONNX.get(
            np.dtype(np.asarray(value).dtype), TensorProto.UNDEFINED)

    if spec.out_types:
        out_types = list(spec.out_types)
    else:
        out_types = []
        for i in range(n_out):
            ts = schema.outputs[i].type_str if i < len(schema.outputs) else "T"
            out_types.append(_elem_of(ts, bindings))

    node = helper.make_node(spec.op, spec.ins, out_names,
                            name="%s_%s" % (spec.op, spec.variant),
                            **spec.attrs)
    inputs = []
    for name in spec.ins:
        a = spec.feeds.get(name)
        if a is None or name in spec.inits:
            continue
        a = np.ascontiguousarray(a)
        inputs.append(helper.make_tensor_value_info(
            name, NP2ONNX[np.dtype(a.dtype)], list(a.shape)))
    inits = [numpy_helper.from_array(np.asarray(v), name=k)
             for k, v in spec.inits.items()]
    outputs = [helper.make_tensor_value_info(n, t, None)
               for n, t in zip(out_names, out_types)]
    graph = helper.make_graph([node], "opmatrix_%s_%s" % (spec.op, spec.variant),
                              inputs, outputs, initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)],
        producer_name="vc4infer-op-matrix")
    if opset < 11:
        model.ir_version = 8
    try:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=False)
    except Exception:
        pass
    # The ONNX checker requires a shape field on every graph output; an
    # unknown shape is still a shape (rank 0), and a real one is written back
    # from the oracle result below.
    for vi in model.graph.output:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            tt.shape.SetInParent()
    return model, out_names, opset


# ------------------------------------------------------------------ recipes
# Each recipe returns a list of Spec.  Shapes are deliberately tiny: one small
# operator per graph, so the GPU is never loaded for more than a few hundred
# microseconds per case.

def rec_elementwise(op, rng, o):
    """Same-shape float binary operators; PRelu uses slope as the 2nd input."""
    lo = {"Pow": 0.5, "Div": 0.5, "Mod": 0.5}.get(op, -1.0)
    hi = 1.5 if lo == 0.5 else 1.0
    ins = (["X", "slope"] if op == "PRelu" else ["A", "B"])
    out = [sp(op, "same_shape", ins,
              {n: arr(rng, (2, 3, 4), lo, hi) for n in ins})]
    if op == "Div":
        # the documented form: the divisor is a scalar constant that can be
        # folded into a multiply
        out.append(sp(op, "const_scalar_divisor", ["A", "B"],
                      {"A": arr(rng, (2, 3, 4), 0.5, 1.5)},
                      {"B": f32(0.7)}, edge=True,
                      note="documented: QPU Mul by a folded constant"))
    return out


def rec_clip(op, rng, o):
    out = [
        sp(op, "tensor_min_max", ["input", "min", "max"],
           {"input": arr(rng, (2, 3, 4), -2, 2)},
           {"min": f32(-0.5), "max": f32(0.5)}),
        sp(op, "tensor_min_only", ["input", "min"],
           {"input": arr(rng, (2, 3, 4), -2, 2)}, {"min": f32(-0.25)}),
        sp(op, "legacy_attrs_opset6", ["input"],
           {"input": arr(rng, (2, 3, 4), -2, 2)},
           attrs={"min": -0.5, "max": 0.5}, opset=6,
           note="attribute form, removed in opset 11"),
    ]
    return out


def rec_conv(op, rng, o):
    out = []

    def mk(variant, cin, cout, k, s, p, d, g, n=1, hw=8, w=None,
           bias=True, required=True, edge=False, note=""):
        w = w or k
        feeds = {"X": arr(rng, (n, cin, hw, hw), -1, 1)}
        inits = {"W": arr(rng, (cout, cin // g, k, w), -0.4, 0.4)}
        ins = ["X", "W"]
        if bias:
            inits["B"] = arr(rng, (cout,), -0.3, 0.3)
            ins.append("B")
        return sp(op, variant, ins, feeds, inits,
                  {"strides": [s, s], "pads": [p, p, p, p],
                   "dilations": [d, d], "group": g},
                  required=required, edge=edge, note=note)

    out += [
        mk("general_3x3_s1_p1", 16, 16, 3, 1, 1, 1, 1),
        mk("stride2_3x3", 16, 32, 3, 2, 1, 1, 1),
        mk("dil2_3x3", 16, 16, 3, 1, 2, 2, 1),
        mk("group2_3x3", 32, 32, 3, 1, 1, 1, 2, note="16 channels per group"),
        mk("depthwise_3x3", 16, 16, 3, 1, 1, 1, 16),
        mk("1x1", 16, 32, 1, 1, 0, 1, 1),
        mk("5x5_pad2", 16, 16, 5, 1, 2, 1, 1),
        mk("batch2", 16, 16, 3, 1, 1, 1, 1, n=2),
        mk("no_bias", 16, 16, 3, 1, 1, 1, 1, bias=False),
        mk("nonsquare_kernel_3x1", 16, 16, 3, 1, 0, 1, 1, w=1,
           edge=True, note="NCHW fallback only, documented"),
    ]
    return out


def rec_convtranspose(op, rng, o):
    return [sp(op, "3x3_s2_p1", ["X", "W"],
               {"X": arr(rng, (1, 4, 8, 8), -1, 1)},
               {"W": arr(rng, (4, 2, 3, 3), -0.4, 0.4)},
               {"strides": [2, 2], "pads": [1, 1, 1, 1]})]


def rec_conv_group(op, rng, o):
    if op == "DeformConv":
        return [sp(op, "3x3_offset_group1", ["X", "W", "offset", "B", "mask"],
                   {"X": arr(rng, (1, 4, 6, 6), -1, 1)},
                   {"W": arr(rng, (4, 4, 3, 3), -0.3, 0.3),
                    "offset": arr(rng, (1, 2 * 9, 6, 6), -0.2, 0.2),
                    "B": arr(rng, (4,), -0.3, 0.3),
                    "mask": arr(rng, (1, 9, 6, 6), 0.3, 0.7)},
                   {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
                    "strides": [1, 1], "dilations": [1, 1], "group": 1,
                    "offset_group": 1})]
    raise SynthesisError("no recipe for %s" % op)


def rec_matmul(op, rng, o):
    if op == "Gemm":
        return [
            sp(op, "M3_K4_N2_transB", ["A", "B", "C"],
               {"A": arr(rng, (3, 4), -1, 1)},
               {"B": arr(rng, (2, 4), -1, 1), "C": arr(rng, (2,), -1, 1)},
               {"transB": 1}),
            sp(op, "M3_K4_N2_transB0", ["A", "B", "C"],
               {"A": arr(rng, (3, 4), -1, 1)},
               {"B": arr(rng, (4, 2), -1, 1), "C": arr(rng, (3, 2), -1, 1)},
               {"transB": 0}, edge=True, note="C broadcast over M"),
        ]
    return [sp(op, "2d_4x8_8x3", ["A", "B"],
               {"A": arr(rng, (4, 8), -1, 1), "B": arr(rng, (8, 3), -1, 1)})]


def rec_einsum(op, rng, o):
    return [sp(op, "ij_jk_ik", ["A", "B"],
               {"A": arr(rng, (2, 3), -1, 1), "B": arr(rng, (3, 4), -1, 1)},
               attrs={"equation": "ij,jk->ik"})]


def rec_softmax(op, rng, o):
    out = [sp(op, "last_axis_2d", ["X"],
              {"X": arr(rng, (4, 5), -1, 1)})]
    if op in ("Softmax", "LogSoftmax", "Hardmax"):
        out.append(sp(op, "axis1_c16_4d", ["X"],
                      {"X": arr(rng, (1, 16, 4, 4), -1, 1)},
                      attrs={"axis": 1}, required=False,
                      note="16 classes on NCHW (engine's special case)"))
        out.append(sp(op, "axis1_c4_4d", ["X"],
                      {"X": arr(rng, (1, 4, 4, 4), -1, 1)},
                      attrs={"axis": 1},
                      note="four classes along the channel axis"))
    return out


def rec_pool(op, rng, o):
    if op in ("GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool"):
        a = {"p": 2} if op == "GlobalLpPool" else {}
        return [sp(op, "1x16x4x4", ["X"], {"X": arr(rng, (1, 16, 4, 4), -1, 1)},
                   attrs=a)]
    if op == "MaxUnpool":
        # distinct indices into the flattened (N=1,C=2,6,6) output
        ind = np.arange(18, dtype=np.int64).reshape(1, 2, 3, 3)
        return [sp(op, "2x2_s2", ["X", "I"],
                   {"X": arr(rng, (1, 2, 3, 3), -1, 1)},
                   {"I": ind},
                   {"kernel_shape": [2, 2], "strides": [2, 2]})]
    if op == "MaxRoiPool":
        rois = f32(0, 0, 0, 3, 3, 0, 2, 2, 5, 5)
        rois = rois.reshape(2, 5)
        return [sp(op, "2_rois_2x2", ["X", "rois"],
                   {"X": arr(rng, (1, 3, 8, 8), -1, 1)}, {"rois": rois},
                   {"pooled_shape": [2, 2], "spatial_scale": 1.0})]
    out = [
        sp(op, "k2_s2_no_pad", ["X"], {"X": arr(rng, (1, 4, 8, 8), -1, 1)},
           attrs={"kernel_shape": [2, 2], "strides": [2, 2]}),
        sp(op, "k3_s1_pad1", ["X"], {"X": arr(rng, (1, 4, 8, 8), -1, 1)},
           attrs={"kernel_shape": [3, 3], "strides": [1, 1],
                  "pads": [1, 1, 1, 1]}),
        sp(op, "k3_s2_pad1", ["X"], {"X": arr(rng, (1, 4, 8, 8), -1, 1)},
           attrs={"kernel_shape": [3, 3], "strides": [2, 2],
                  "pads": [1, 1, 1, 1]}),
    ]
    if op == "MaxPool":
        out.append(sp(op, "ceil_mode", ["X"],
                      {"X": arr(rng, (1, 4, 7, 7), -1, 1)},
                      attrs={"kernel_shape": [2, 2], "strides": [2, 2],
                             "ceil_mode": 1}, edge=True,
                      note="ceil_mode is documented unsupported"))
    if op == "AveragePool":
        out.append(sp(op, "count_include_pad", ["X"],
                      {"X": arr(rng, (1, 4, 8, 8), -1, 1)},
                      attrs={"kernel_shape": [3, 3], "strides": [1, 1],
                             "pads": [1, 1, 1, 1],
                             "count_include_pad": 1}, edge=True))
    return out


def rec_variadic(op, rng, o):
    """Sum / Mean / Max / Min: variadic element-wise.

    The generic recipe fills one feed per schema input, and a variadic
    operator's schema has a *single* variadic parameter, so it synthesises a
    one-operand node -- for Sum/Max/Min/Mean that is an identity, which a
    composition can pass with no element-wise pass at all.  These are the
    forms that exercise the operator: two operands, and three with numpy
    broadcasting (Mean stays same-shape because the 1.17 reference adds in
    place into its first operand and cannot evaluate a broadcast form).
    """
    shp = (2, 3, 4)
    same = [sp(op, "two_operands", ["A", "B"],
               {"A": arr(rng, shp, -1.0, 1.5), "B": arr(rng, shp, -1.0, 1.5)})]
    three = {"A": arr(rng, (2, 1, 4), -1.0, 1.5),
             "B": arr(rng, (3, 4), -1.0, 1.5)}
    if op == "Mean":
        three = {"A": arr(rng, shp, -1.0, 1.5),
                 "B": arr(rng, shp, -1.0, 1.5)}
        three["C"] = arr(rng, shp, -1.0, 1.5)
    else:
        three["C"] = arr(rng, (1,), 0.5, 1.5)
    same.append(sp(op, "three_operands", ["A", "B", "C"], three,
                   note="numpy broadcasting" if op != "Mean" else ""))
    return same


def rec_norm(op, rng, o):
    if op == "LRN":
        return [sp(op, "size3", ["X"], {"X": arr(rng, (1, 16, 5, 5), -1, 1)},
                   attrs={"size": 3})]
    if op == "BatchNormalization":
        c = 16
        return [sp(op, "inference_c16", ["X", "scale", "B", "input_mean",
                                         "input_var"],
                   {"X": arr(rng, (1, c, 4, 4), -1, 1)},
                   {"scale": arr(rng, (c,), 0.5, 1.5),
                    "B": arr(rng, (c,), -0.5, 0.5),
                    "input_mean": arr(rng, (c,), -0.5, 0.5),
                    "input_var": arr(rng, (c,), 0.5, 1.5)},
                   {"epsilon": 1e-5, "training_mode": 0})]
    if op == "InstanceNormalization":
        c = 4
        return [sp(op, "c4", ["input", "scale", "B"],
                   {"input": arr(rng, (1, c, 4, 4), -1, 1)},
                   {"scale": arr(rng, (c,), 0.5, 1.5),
                    "B": arr(rng, (c,), -0.5, 0.5)})]
    if op == "GroupNormalization":
        c = 4
        return [sp(op, "groups2_c4", ["X", "scale", "B"],
                   {"X": arr(rng, (1, c, 4, 4), -1, 1)},
                   {"scale": arr(rng, (c,), 0.5, 1.5),
                    "B": arr(rng, (c,), -0.5, 0.5)},
                   {"num_groups": 2})]
    if op == "LayerNormalization":
        return [sp(op, "last_axis", ["X", "Scale", "B"],
                   {"X": arr(rng, (2, 4, 8), -1, 1)},
                   {"Scale": arr(rng, (8,), 0.5, 1.5),
                    "B": arr(rng, (8,), -0.5, 0.5)}, {"axis": -1})]
    if op == "MeanVarianceNormalization":
        return [sp(op, "axes_0_2_3", ["X"],
                   {"X": arr(rng, (1, 3, 4, 4), -1, 1)},
                   attrs={"axes": [0, 2, 3]})]
    if op == "LpNormalization":
        return [sp(op, "axis_last_p2", ["X"],
                   {"X": arr(rng, (2, 3, 4), -1, 1)},
                   attrs={"axis": -1, "p": 2})]
    raise SynthesisError("no recipe for %s" % op)


def rec_reduce(op, rng, o):
    shape = (2, 3, 4)
    return [
        sp(op, "all_axes", ["data"],
           {"data": arr(rng, shape, 0.5, 1.5)}),
        sp(op, "axis1_keepdims", ["data", "axes"],
           {"data": arr(rng, shape, 0.5, 1.5)}, {"axes": i64(1)}),
        sp(op, "axis_last_no_keepdims", ["data", "axes"],
           {"data": arr(rng, shape, 0.5, 1.5)}, {"axes": i64(-1)},
           attrs={"keepdims": 0}),
    ]


def rec_arg(op, rng, o):
    return [
        sp(op, "axis1_keepdims", ["data"], {"data": arr(rng, (2, 3, 4))},
           attrs={"axis": 1, "keepdims": 1}),
        sp(op, "axis_last_no_keepdims", ["data"],
           {"data": arr(rng, (2, 3, 4))},
           attrs={"axis": -1, "keepdims": 0}),
    ]


def rec_gather(op, rng, o):
    if op == "Gather":
        return [
            sp(op, "axis1_indices_2x2", ["data", "indices"],
               {"data": arr(rng, (3, 4, 5), -1, 1)},
               {"indices": np.array([[0, 2], [1, 3]], np.int64)},
               {"axis": 1}),
        ]
    if op == "GatherElements":
        return [sp(op, "axis1", ["data", "indices"],
                   {"data": arr(rng, (3, 4), -1, 1)},
                   {"indices": np.array([[0, 1, 2, 3]] * 3, np.int64)},
                   {"axis": 1})]
    if op == "GatherND":
        return [sp(op, "batch_dims0", ["data", "indices"],
                   {"data": arr(rng, (2, 3, 4), -1, 1)},
                   {"indices": np.array([[0, 1], [1, 2]], np.int64)})]
    raise SynthesisError("no recipe for %s" % op)


def rec_scatter(op, rng, o):
    if op == "ScatterND":
        return [sp(op, "rank2", ["data", "indices", "updates"],
                   {"data": arr(rng, (4, 4), -1, 1)},
                   {"indices": np.array([[0], [2]], np.int64),
                    "updates": arr(rng, (2, 4), -1, 1)})]
    idx = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], np.int64)
    return [sp(op, "axis1", ["data", "indices", "updates"],
               {"data": arr(rng, (3, 4), -1, 1)}, {"indices": idx,
                                                    "updates": arr(rng, (3, 4), -1, 1)},
               {"axis": 1})]


def rec_concat(op, rng, o):
    return [
        sp(op, "nchw_axis1", ["A", "B"],
           {"A": arr(rng, (1, 2, 4, 4), -1, 1),
            "B": arr(rng, (1, 3, 4, 4), -1, 1)}, attrs={"axis": 1}),
        sp(op, "rank2_axis1", ["A", "B"],
           {"A": arr(rng, (2, 3), -1, 1), "B": arr(rng, (2, 4), -1, 1)},
           attrs={"axis": 1}, edge=True,
           note="documented: channel axis of 4-D only"),
        sp(op, "axis0", ["A", "B"],
           {"A": arr(rng, (2, 3), -1, 1), "B": arr(rng, (2, 3), -1, 1)},
           attrs={"axis": 0}, edge=True),
    ]


def rec_split(op, rng, o):
    return [
        sp(op, "equal_2way", ["input"],
           {"input": arr(rng, (1, 8, 4, 4), -1, 1)},
           attrs={"axis": 1}, n_out=2),
        sp(op, "explicit_sizes", ["input", "split"],
           {"input": arr(rng, (1, 8, 4, 4), -1, 1)}, {"split": i64(2, 6)},
           attrs={"axis": 1}, n_out=2),
    ]


def rec_slice(op, rng, o):
    return [
        sp(op, "one_axis", ["data", "starts", "ends", "axes"],
           {"data": arr(rng, (1, 4, 8, 8), -1, 1)},
           {"starts": i64(1), "ends": i64(3), "axes": i64(1)}),
        sp(op, "one_axis_negative", ["data", "starts", "ends", "axes"],
           {"data": arr(rng, (1, 4, 8, 8), -1, 1)},
           {"starts": i64(-3), "ends": i64(-1), "axes": i64(2)}),
        sp(op, "two_axes", ["data", "starts", "ends", "axes"],
           {"data": arr(rng, (1, 4, 8, 8), -1, 1)},
           {"starts": i64(1, 2), "ends": i64(3, 6), "axes": i64(1, 2)}),
        sp(op, "step2", ["data", "starts", "ends", "axes", "steps"],
           {"data": arr(rng, (1, 4, 8, 8), -1, 1)},
           {"starts": i64(0), "ends": i64(8), "axes": i64(2),
            "steps": i64(2)}, edge=True),
    ]


def rec_resize(op, rng, o):
    x = (1, 4, 8, 8)
    out = []
    for mode in ("nearest", "linear"):
        for coord in ("asymmetric", "half_pixel"):
            out.append(sp(op, "%s_%s" % (mode, coord), ["X", "", "scales"],
                          {"X": arr(rng, x, -1, 1)},
                          {"scales": f32(1, 1, 2, 2)},
                          {"mode": mode,
                           "coordinate_transformation_mode": coord}))
    out.append(sp(op, "nearest_half_pixel_floor_mode", ["X", "", "scales"],
                  {"X": arr(rng, x, -1, 1)}, {"scales": f32(1, 1, 2, 2)},
                  {"mode": "nearest",
                   "coordinate_transformation_mode": "half_pixel",
                   "nearest_mode": "floor"}, required=False,
                  note="same coordinates, explicit floor rounding"))
    out.append(sp(op, "sizes", ["X", "", "", "sizes"],
                  {"X": arr(rng, x, -1, 1)},
                  {"sizes": i64(1, 4, 16, 16)}, edge=True))
    out.append(sp(op, "cubic", ["X", "", "scales"],
                  {"X": arr(rng, x, -1, 1)}, {"scales": f32(1, 1, 2, 2)},
                  {"mode": "cubic"}, edge=True,
                  note="documented unsupported"))
    out.append(sp(op, "align_corners", ["X", "", "scales"],
                  {"X": arr(rng, x, -1, 1)}, {"scales": f32(1, 1, 2, 2)},
                  {"mode": "linear",
                   "coordinate_transformation_mode": "align_corners"},
                  edge=True))
    return out


def rec_upsample(op, rng, o):
    return [sp(op, "nearest_2x", ["X", "scales"],
               {"X": arr(rng, (1, 4, 8, 8), -1, 1)},
               {"scales": f32(1, 1, 2, 2)}, {"mode": "nearest"})]


def rec_transpose(op, rng, o):
    return [
        sp(op, "perm_0213", ["data"], {"data": arr(rng, (1, 4, 3, 5), -1, 1)},
           attrs={"perm": [0, 2, 1, 3]}),
        sp(op, "perm_0132", ["data"], {"data": arr(rng, (1, 4, 3, 5), -1, 1)},
           attrs={"perm": [0, 1, 3, 2]}),
        sp(op, "rank2", ["data"], {"data": arr(rng, (3, 4), -1, 1)},
           attrs={"perm": [1, 0]}, edge=True),
    ]


def rec_shapeop(op, rng, o):
    if op == "Reshape":
        return [
            sp(op, "4x6", ["data", "shape"],
               {"data": arr(rng, (2, 3, 4), -1, 1)}, {"shape": i64(4, 6)}),
            sp(op, "minus1", ["data", "shape"],
               {"data": arr(rng, (2, 3, 4), -1, 1)}, {"shape": i64(-1, 4)}),
            sp(op, "zero_copy", ["data", "shape"],
               {"data": arr(rng, (2, 3, 4), -1, 1)},
               {"shape": i64(0, 4, -1)}, edge=True),
        ]
    if op == "Flatten":
        return [
            sp(op, "axis1", ["X"], {"X": arr(rng, (2, 3, 4), -1, 1)},
               attrs={"axis": 1}),
            sp(op, "axis_minus1", ["X"], {"X": arr(rng, (2, 3, 4), -1, 1)},
               attrs={"axis": -1}, edge=True),
        ]
    if op == "Squeeze":
        return [
            sp(op, "axes_0_2", ["data", "axes"],
               {"data": arr(rng, (1, 3, 1, 4), -1, 1)}, {"axes": i64(0, 2)}),
            sp(op, "axes_omitted", ["data"],
               {"data": arr(rng, (1, 3, 1, 4), -1, 1)}),
        ]
    if op == "Unsqueeze":
        return [
            sp(op, "axes_0_3", ["data", "axes"],
               {"data": arr(rng, (2, 3), -1, 1)}, {"axes": i64(0, 3)}),
        ]
    raise SynthesisError("no recipe for %s" % op)


def rec_pad(op, rng, o):
    return [
        sp(op, "constant_pads", ["data", "pads", "constant_value"],
           {"data": arr(rng, (1, 2, 3, 3), -1, 1)},
           {"pads": i64(0, 0, 1, 1, 0, 0, 1, 1),
            "constant_value": f32(0.0)}),
        sp(op, "reflect", ["data", "pads"],
           {"data": arr(rng, (1, 2, 3, 3), -1, 1)},
           {"pads": i64(0, 0, 1, 1, 0, 0, 1, 1)}, {"mode": "reflect"},
           edge=True),
        sp(op, "axes", ["data", "pads", "", "axes"],
           {"data": arr(rng, (1, 2, 3, 3), -1, 1)},
           {"pads": i64(1, 1, 1, 1), "axes": i64(2, 3)}, edge=True),
    ]


def rec_tile(op, rng, o):
    return [sp(op, "2x2", ["input", "repeats"],
               {"input": arr(rng, (2, 3), -1, 1)}, {"repeats": i64(2, 2)})]


def rec_expand(op, rng, o):
    return [sp(op, "broadcast", ["input", "shape"],
               {"input": arr(rng, (2, 1, 4), -1, 1)},
               {"shape": i64(2, 3, 4)})]


def rec_topk(op, rng, o):
    return [sp(op, "k3_axis_last", ["X", "K"],
               {"X": arr(rng, (2, 4, 5), -1, 1)}, {"K": i64(3)},
               {"axis": -1}, n_out=2)]


def rec_onehot(op, rng, o):
    return [sp(op, "depth4", ["indices", "depth", "values"],
               {"indices": f32(0.0, 2.0, 3.0), "depth": f32(4.0),
                "values": f32(0.0, 1.0)}, attrs={"axis": -1})]


def rec_nms(op, rng, o):
    return [sp(op, "4_boxes", ["boxes", "scores",
                               "max_output_boxes_per_class", "iou_threshold",
                               "score_threshold"],
               {"boxes": arr(rng, (1, 4, 4), 0, 1),
                "scores": arr(rng, (1, 1, 4), 0, 1)},
               {"max_output_boxes_per_class": i64(2),
                "iou_threshold": f32(0.5), "score_threshold": f32(0.0)})]


def rec_recurrent(op, rng, o):
    t, b, i, h = 2, 2, 3, 2
    n = 1
    if op == "LSTM":
        gates = 4
    elif op == "GRU":
        gates = 3
    else:
        gates = 1
    feeds = {"X": arr(rng, (t, b, i), -1, 1)}
    inits = {"W": arr(rng, (n, gates * h, i), -0.4, 0.4),
             "R": arr(rng, (n, gates * h, h), -0.4, 0.4),
             "B": arr(rng, (n, 2 * gates * h), -0.3, 0.3)}
    return [sp(op, "t2_b2_h2", ["X", "W", "R", "B"], feeds, inits,
               {"hidden_size": h}, n_out=2)]


def rec_quantize(op, rng, o):
    if op == "QuantizeLinear":
        return [sp(op, "scale01_zp0", ["x", "y_scale", "y_zero_point"],
                   {"x": arr(rng, (2, 3, 4), -2, 2)},
                   {"y_scale": f32(0.1), "y_zero_point": np.asarray(0, np.uint8)},
                   out_types=[TensorProto.UINT8])]
    if op == "DynamicQuantizeLinear":
        return [sp(op, "float_input", ["x"],
                   {"x": arr(rng, (2, 3, 4), -2, 2)}, n_out=3)]
    raise SynthesisError("no recipe for %s" % op)


def rec_affinegrid(op, rng, o):
    return [sp(op, "theta_1x2x3", ["theta", "size"],
               {"theta": arr(rng, (1, 2, 3), -0.5, 0.5)},
               {"size": i64(1, 3, 4, 5)})]


def rec_gridsample(op, rng, o):
    return [sp(op, "bilinear", ["X", "grid"],
               {"X": arr(rng, (1, 3, 6, 6), -1, 1),
                "grid": arr(rng, (1, 5, 5, 2), -0.9, 0.9)})]


def rec_roialign(op, rng, o):
    return [sp(op, "2_rois", ["X", "rois", "batch_indices"],
               {"X": arr(rng, (1, 3, 8, 8), -1, 1),
                "rois": arr(rng, (2, 4), 0, 6)},
               {"batch_indices": i64(0, 0)},
               {"output_height": 2, "output_width": 2,
                "sampling_ratio": 2})]


def rec_col2im(op, rng, o):
    return [sp(op, "4x4_blocks2", ["input", "image_shape", "block_shape"],
               {"input": arr(rng, (1, 8, 9), -1, 1)},
               {"image_shape": i64(1, 2, 4, 4),
                "block_shape": i64(2, 2)},
               attrs={"dilations": [1, 1], "pads": [0, 0, 0, 0],
                      "strides": [1, 1]}, oracle_gap=True,
               note="onnx.reference 1.17 Col2Im passes a list where a 2-D "
                    "pads array is expected")]


def rec_centercrop(op, rng, o):
    return [sp(op, "crop_last_axis", ["input_data", "shape"],
               {"input_data": arr(rng, (2, 3, 4), -1, 1)},
               {"shape": i64(2, 3, 6)})]


def rec_compress(op, rng, o):
    return [sp(op, "axis0", ["input", "condition"],
               {"input": arr(rng, (2, 3), -1, 1)},
               {"condition": np.asarray([True, False], np.bool_)},
               {"axis": 0})]


def rec_unique(op, rng, o):
    return [sp(op, "axis0", ["X"],
               {"X": np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]],
                                np.float32)}, attrs={"axis": 0}, n_out=4)]


def rec_det(op, rng, o):
    return [sp(op, "batched_3x3", ["X"], {"X": arr(rng, (2, 3, 3), -1, 1)})]


def rec_dft(op, rng, o):
    return [sp(op, "real_onesided", ["input", "dft_length", "axis"],
               {"input": arr(rng, (2, 4, 5, 1), -1, 1)},
               {"dft_length": i64(4), "axis": i64(1)},
               attrs={"onesided": 1}, oracle_gap=True,
               note="onnx.reference 1.17 DFT passes a 0-d axis to numpy.fft")]


def rec_stft(op, rng, o):
    return [sp(op, "frame8_step4", ["signal", "frame_step", "window"],
               {"signal": arr(rng, (2, 16, 2), -1, 1)},
               {"frame_step": i64(4), "window": arr(rng, (8,), 0.5, 1.0)},
               oracle_gap=True,
               note="onnx.reference 1.17 STFT passes a 0-d frame_step to "
                    "range()")]


def rec_mel(op, rng, o):
    return [sp(op, "4mel_dft8", ["num_mel_bins", "dft_length", "sample_rate",
                                 "lower_edge_hertz", "upper_edge_hertz"],
               {}, {"num_mel_bins": i64(4), "dft_length": i64(8),
                    "sample_rate": i64(8000),
                    "lower_edge_hertz": f32(20.0),
                    "upper_edge_hertz": f32(4000.0)},
               out_types=[TensorProto.FLOAT], oracle_gap=True,
               note="onnx.reference 1.17 MelWeightMatrix passes 0-d ints to "
                    "np.zeros")]


def rec_window(op, rng, o):
    return [sp(op, "size8", ["size"], {}, {"size": i64(8)},
               out_types=[TensorProto.FLOAT])]


def rec_range(op, rng, o):
    return [sp(op, "float_0_1_0p2", ["start", "limit", "delta"],
               {}, {"start": f32(0.0), "limit": f32(1.0),
                    "delta": f32(0.2)})]


def rec_constantofshape(op, rng, o):
    return [sp(op, "shape_2x3", ["input"], {},
               {"input": i64(2, 3)},
               attrs={"value": numpy_helper.from_array(f32(1.0))})]


def rec_constant(op, rng, o):
    return [sp(op, "value_2x3", [], {},
               attrs={"value": numpy_helper.from_array(arr(rng, (2, 3)))})
            ]


def rec_cumsum(op, rng, o):
    return [sp(op, "axis1", ["x", "axis"],
               {"x": arr(rng, (2, 3, 4), -1, 1)}, {"axis": i64(1)})]


def rec_cast(op, rng, o):
    return [
        sp(op, "f32_to_i32", ["input"], {"input": arr(rng, (2, 3, 4), -2, 2)},
           attrs={"to": TensorProto.INT32},
           out_types=[TensorProto.INT32]),
        sp(op, "f32_to_f32", ["input"], {"input": arr(rng, (2, 3, 4), -2, 2)},
           attrs={"to": TensorProto.FLOAT}, edge=True,
           note="no-op cast; classified host-meta by the engine"),
    ]


def rec_castlike(op, rng, o):
    return [sp(op, "f32_to_f32", ["input", "target_type"],
               {"input": arr(rng, (2, 3, 4), -2, 2),
                "target_type": arr(rng, (1,), -1, 1)})]


def rec_dropout(op, rng, o):
    return [sp(op, "inference_identity", ["data"],
               {"data": arr(rng, (2, 3, 4), -1, 1)})]


def rec_where(op, rng, o):
    return [sp(op, "cond_init", ["condition", "X", "Y"],
               {"X": arr(rng, (2, 3, 4), -1, 1),
                "Y": arr(rng, (2, 3, 4), -1, 1)},
               {"condition": rng.random((2, 3, 4)) < 0.5})]


def rec_eyelike(op, rng, o):
    return [sp(op, "4x4", ["input"], {"input": arr(rng, (4, 4), -1, 1)},
               out_types=[TensorProto.FLOAT])]


def rec_trilu(op, rng, o):
    return [sp(op, "upper_k0", ["input", "k"],
               {"input": arr(rng, (3, 4), -1, 1)}, {"k": i64(0)},
               {"upper": 1})]


def rec_reverseseq(op, rng, o):
    return [sp(op, "batch0_time1", ["input", "sequence_lens"],
               {"input": arr(rng, (3, 4, 2), -1, 1)},
               {"sequence_lens": i64(3, 2, 3)},
               {"batch_axis": 0, "time_axis": 1})]


def rec_space2depth(op, rng, o):
    return [sp(op, "blocksize2", ["input"],
               {"input": arr(rng, (1, 4, 8, 8), -1, 1)},
               attrs={"blocksize": 2})]


def rec_depth2space(op, rng, o):
    return [sp(op, "blocksize2_dcr", ["input"],
               {"input": arr(rng, (1, 4, 4, 4), -1, 1)},
               attrs={"blocksize": 2, "mode": "DCR"})]


def rec_loss(op, rng, o):
    if op == "SoftmaxCrossEntropyLoss":
        return [sp(op, "mean", ["scores", "labels"],
                   {"scores": arr(rng, (3, 4), -2, 2)},
                   {"labels": i64(0, 2, 1)}, {"reduction": "mean"})]
    return [sp(op, "mean", ["input", "target"],
               {"input": arr(rng, (3, 4), -2, 2)},
               {"target": i64(0, 2, 1)}, {"reduction": "mean"})]


def rec_bernoulli(op, rng, o):
    return [sp(op, "p_uniform", ["input"],
               {"input": arr(rng, (2, 3), 0.0, 1.0)},
               attrs={"dtype": TensorProto.FLOAT, "seed": 0.0})]


def rec_random(op, rng, o):
    if op.endswith("Like"):
        return [sp(op, "like", ["input"], {"input": arr(rng, (2, 3), -1, 1)},
                   attrs={"dtype": TensorProto.FLOAT, "seed": 0.0})]
    return [sp(op, "shape_2x3", [], {},
               attrs={"shape": [2, 3], "dtype": TensorProto.FLOAT,
                      "seed": 0.0})]


def rec_multinomial(op, rng, o):
    return [sp(op, "2x4_sample3", ["input"],
               {"input": arr(rng, (2, 4), 0.0, 1.0)},
               attrs={"sample_size": 3, "seed": 0.0},
               out_types=[TensorProto.INT64])]


def rec_ismatch(op, rng, o):
    attrs = {}
    if op == "IsInf":
        attrs = {"detect_negative": 1, "detect_positive": 1}
    return [sp(op, "default", ["X"], {"X": arr(rng, (2, 3, 4), -2, 2)},
               attrs=attrs)]


def rec_globallppool(op, rng, o):
    return [sp(op, "p2", ["X"], {"X": arr(rng, (1, 16, 4, 4), -1, 1)},
               attrs={"p": 2})]


RECIPES = {
    "Conv": rec_conv, "ConvTranspose": rec_convtranspose,
    "DeformConv": rec_conv_group,
    "MatMul": rec_matmul, "Gemm": rec_matmul, "Einsum": rec_einsum,
    "Softmax": rec_softmax, "LogSoftmax": rec_softmax, "Hardmax": rec_softmax,
    "Clip": rec_clip,
    "AveragePool": rec_pool, "MaxPool": rec_pool, "LpPool": rec_pool,
    "GlobalAveragePool": rec_pool, "GlobalMaxPool": rec_pool,
    "GlobalLpPool": rec_globallppool, "MaxUnpool": rec_pool,
    "MaxRoiPool": rec_pool,
    "LRN": rec_norm, "BatchNormalization": rec_norm,
    "InstanceNormalization": rec_norm, "GroupNormalization": rec_norm,
    "LayerNormalization": rec_norm, "MeanVarianceNormalization": rec_norm,
    "LpNormalization": rec_norm,
    "Sum": rec_variadic, "Mean": rec_variadic, "Max": rec_variadic,
    "Min": rec_variadic,
    "ReduceL1": rec_reduce, "ReduceL2": rec_reduce,
    "ReduceLogSum": rec_reduce, "ReduceLogSumExp": rec_reduce,
    "ReduceMax": rec_reduce, "ReduceMean": rec_reduce,
    "ReduceMin": rec_reduce, "ReduceProd": rec_reduce,
    "ReduceSum": rec_reduce, "ReduceSumSquare": rec_reduce,
    "ArgMax": rec_arg, "ArgMin": rec_arg,
    "Gather": rec_gather, "GatherElements": rec_gather,
    "GatherND": rec_gather,
    "Scatter": rec_scatter, "ScatterElements": rec_scatter,
    "ScatterND": rec_scatter,
    "Concat": rec_concat, "Split": rec_split, "Slice": rec_slice,
    "Resize": rec_resize, "Upsample": rec_upsample,
    "Transpose": rec_transpose,
    "Reshape": rec_shapeop, "Flatten": rec_shapeop,
    "Squeeze": rec_shapeop, "Unsqueeze": rec_shapeop,
    "Pad": rec_pad, "Tile": rec_tile, "Expand": rec_expand,
    "TopK": rec_topk, "OneHot": rec_onehot,
    "NonMaxSuppression": rec_nms,
    "RNN": rec_recurrent, "GRU": rec_recurrent, "LSTM": rec_recurrent,
    "QuantizeLinear": rec_quantize,
    "DynamicQuantizeLinear": rec_quantize,
    "AffineGrid": rec_affinegrid, "GridSample": rec_gridsample,
    "RoiAlign": rec_roialign, "Col2Im": rec_col2im,
    "CenterCropPad": rec_centercrop, "Compress": rec_compress,
    "Unique": rec_unique, "Det": rec_det, "DFT": rec_dft, "STFT": rec_stft,
    "MelWeightMatrix": rec_mel,
    "BlackmanWindow": rec_window, "HammingWindow": rec_window,
    "HannWindow": rec_window,
    "Range": rec_range, "ConstantOfShape": rec_constantofshape,
    "Constant": rec_constant, "CumSum": rec_cumsum,
    "Cast": rec_cast, "CastLike": rec_castlike, "Dropout": rec_dropout,
    "Where": rec_where, "EyeLike": rec_eyelike, "Trilu": rec_trilu,
    "ReverseSequence": rec_reverseseq, "SpaceToDepth": rec_space2depth,
    "DepthToSpace": rec_depth2space,
    "SoftmaxCrossEntropyLoss": rec_loss,
    "NegativeLogLikelihoodLoss": rec_loss,
    "Bernoulli": rec_bernoulli, "RandomNormal": rec_random,
    "RandomNormalLike": rec_random, "RandomUniform": rec_random,
    "RandomUniformLike": rec_random, "Multinomial": rec_multinomial,
    "IsInf": rec_ismatch, "IsNaN": rec_ismatch,
}


class SynthesisError(Exception):
    pass


# Input domains for the elementwise long tail.  These are the operator's
# mathematical domain, not an engine constraint: log/sqrt/reciprocal get
# positive data, acos/asin/atanh get (-1, 1), acosh gets > 1.
DOMAIN = {
    "Acos": (-0.9, 0.9), "Acosh": (1.1, 2.0), "Asin": (-0.9, 0.9),
    "Atanh": (-0.9, 0.9), "Log": (0.5, 1.5), "Reciprocal": (0.5, 1.5),
    "Sqrt": (0.5, 1.5), "Pow": (0.5, 1.5), "Div": (0.5, 1.5),
    "Mod": (0.5, 1.5), "LogSoftmax": (-1.0, 1.0),
    "Softmax": (-1.0, 1.0), "Softplus": (-2.0, 2.0),
    "Hardmax": (-1.0, 1.0),
}


def generic_recipe(op, rng, schema, opset):
    """Schema-driven fallback for the elementwise long tail."""
    inputs = []
    feeds = {}
    inits = {}
    bindings = {}
    for p in schema.inputs:
        if _is_optional(p):
            continue
        ts = p.type_str
        if ts not in bindings:
            allowed = []
            for tc in schema.type_constraints:
                if tc.type_param_str == ts:
                    allowed = list(tc.allowed_type_strs)
            if not allowed:
                raise SynthesisError("%s: no type constraint for %s" % (op, ts))
            elem = None
            for cand in ("tensor(float)", "tensor(int64)", "tensor(int32)",
                         "tensor(bool)"):
                if cand in allowed:
                    elem = TENSOR_TYPE_NAMES[cand.split("(")[1][:-1]]
                    break
            if elem is None:
                raise SynthesisError("%s: no synthesised dtype for %s" %
                                     (op, ts))
            bindings[ts] = elem
        elem = bindings[ts]
        name = p.name
        if elem in (TensorProto.FLOAT, TensorProto.DOUBLE,
                    TensorProto.FLOAT16):
            lo, hi = DOMAIN.get(op, (-1.0, 1.0))
            feeds[name] = arr(rng, (2, 3, 4), lo, hi)
        else:
            inits[name] = np.zeros((1,), ONNX2NP[elem])
        inputs.append(name)
    if not inputs:
        raise SynthesisError("no inputs")
    return [sp(op, "generic", inputs, feeds, inits)]


# ------------------------------------------------------------------ constants
def load_schema(op, opset):
    return defs.get_schema(op, opset, "")


# ------------------------------------------------------------------ execution
class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("case exceeded the wall-clock guard")


def compare(ref, got, tol_abs=TOL_ABS, tol_rel=TOL_REL):
    """Return (ok, stats) for one output tensor."""
    ref = np.asarray(ref)
    got = np.asarray(got)
    if tuple(ref.shape) != tuple(got.shape):
        return False, {"reason": "shape %s != %s" % (tuple(got.shape),
                                                     tuple(ref.shape))}
    if ref.dtype.kind in ("i", "u", "b"):
        same = np.array_equal(ref, got)
        return bool(same), {"exact": True, "max_abs": 0.0 if same else None,
                            "mismatched": int(np.count_nonzero(ref != got))}
    if got.dtype != np.float32:
        got = got.astype(np.float32)
    ref = ref.astype(np.float32)
    if ref.shape == ():
        ref = ref.reshape(1)
        got = got.reshape(1)
    one_nan = np.isnan(ref) ^ np.isnan(got)
    inf_bad = (np.isinf(ref) ^ np.isinf(got)) | (
        np.isinf(ref) & np.isinf(got) & (np.sign(ref) != np.sign(got)))
    finite = np.isfinite(ref) & np.isfinite(got)
    diff = np.zeros_like(ref, dtype=np.float64)
    diff[finite] = np.abs(got[finite].astype(np.float64) -
                          ref[finite].astype(np.float64))
    tol = tol_abs + tol_rel * np.maximum(np.abs(ref.astype(np.float64)), 1.0)
    over = finite & (diff > tol)
    over |= one_nan | inf_bad
    bad = int(np.count_nonzero(over))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    rel = diff / denom
    stats = {
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "max_rel": float(np.max(rel[finite])) if np.any(finite) else 0.0,
        "elements": int(ref.size),
        "over_tolerance": bad,
        "nan_mismatch": int(np.count_nonzero(one_nan)),
    }
    if bad:
        worst = int(np.argmax(np.where(over, diff, -1.0)))
        stats["worst_index"] = list(np.unravel_index(worst, ref.shape))
        stats["worst_ref"] = float(ref.reshape(-1)[worst])
        stats["worst_got"] = float(got.reshape(-1)[worst])
    return not bool(bad), stats


# Known gaps in onnx.reference 1.17 that fire on *valid* graphs.  Each was
# reproduced independently of this tool (see docs/COMPLETENESS.md).  Scalar
# tensor inputs arrive as 0-d numpy arrays and the implementations below use
# them as Python ints.
_REFERENCE_GAPS = (
    "only integer scalar arrays can be converted to a scalar index",
    "No registered implementation for operator",
    "No registered schema for operator",
    "evaluator_cls must be specified",
)
_REFERENCE_GAP_TYPES = ("RuntimeImplementationError", "RuntimeContextError")


def _reference_limitation(exc):
    msg = str(exc)
    if "No registered implementation" in msg or \
            "No registered schema" in msg or \
            "evaluator_cls must be specified" in msg:
        return True
    return type(exc).__name__ in _REFERENCE_GAP_TYPES


def refusal_info(exc):
    info = {"type": type(exc).__name__, "text": str(exc)}
    for attr in ("detail", "reason", "feature", "node", "op", "policy"):
        v = getattr(exc, attr, None)
        if v:
            info[attr] = str(v)
    if "reason" in info:
        info["invariant"] = info["reason"]
    elif "detail" in info:
        info["invariant"] = info["detail"]
    elif "feature" in info:
        info["invariant"] = info["feature"]
    else:
        info["invariant"] = info["text"]
    return info


def run_case(spec, schema, default_opset, workdir, eng, do_engine=True,
             cache=None):
    """Synthesise, validate against the oracle, and (optionally) run on vc4.

    ``cache`` replays a case saved by an earlier synthesis-only pass: the
    model, feeds and oracle result come from disk, so the GPU phase can run in
    a short, controlled window and can be resumed after a reboot.
    """
    if cache is not None:
        return _engine_phase(spec, cache["rec"], cache["feeds"],
                             cache["ref_vals"], cache["oracle"], eng,
                             cache["path"], do_engine)
    rec = {"op": spec.op, "variant": spec.variant, "required": spec.required,
           "edge": spec.edge, "opsop": spec.opset or default_opset,
           "note": spec.note, "status": "untested"}
    t_build = time.perf_counter()
    try:
        model, out_names, opset = build_model(spec, schema, default_opset)
    except Exception as e:
        rec.update(status="untested", stage="synthesis",
                   error="%s: %s" % (type(e).__name__, e))
        return rec
    path = os.path.join(workdir, "%s__%s.onnx" % (
        spec.op, re.sub(r"[^A-Za-z0-9_.-]", "_", spec.variant)))
    onnx.save(model, path)
    rec["model"] = os.path.basename(path)
    rec["opset"] = opset
    rec["shapes"] = {k: list(np.asarray(v).shape)
                     for k, v in list(spec.feeds.items()) +
                     list(spec.inits.items())}
    rec["dtypes"] = {k: str(np.asarray(v).dtype)
                     for k, v in list(spec.feeds.items()) +
                     list(spec.inits.items())}
    rec["attrs"] = _jsonable(spec.attrs)
    rec["inputs"] = list(spec.ins)
    rec["build_ms"] = round((time.perf_counter() - t_build) * 1e3, 3)

    ref_vals = None
    oracle = "unavailable"
    t0 = time.perf_counter()
    try:
        ev = ReferenceEvaluator(model)
        ref_vals = ev.run(None, {k: np.asarray(v) for k, v in spec.feeds.items()})
        oracle = "ok"
    except Exception as e:
        msg = str(e)
        if _reference_limitation(e) or spec.oracle_gap:
            oracle = "unavailable"
            rec["oracle_error"] = "%s: %s" % (type(e).__name__, msg)
            rec["oracle_note"] = ("onnx.reference %s cannot evaluate this "
                                  "checker-valid graph (%s)" %
                                  (onnx.__version__, type(e).__name__))
        else:
            rec.update(status="invalid-model", stage="oracle",
                       oracle="error",
                       error="%s: %s" % (type(e).__name__, msg))
            rec["traceback"] = traceback.format_exc(limit=6)
            return rec
    rec["oracle"] = oracle
    rec["oracle_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    if ref_vals is not None:
        rec["ref"] = [_summary(v) for v in ref_vals]

    # ---- validity: the ONNX checker, not the oracle, decides whether the
    # graph is well formed.  A refusal must be of a valid graph.  Output shapes
    # that shape inference could not derive are taken from the oracle result
    # (the operator's own definition), then the model is re-saved.
    if ref_vals is not None:
        for vi, a in zip(model.graph.output, ref_vals):
            shape = vi.type.tensor_type.shape
            del shape.dim[:]
            for d in np.asarray(a).shape:
                shape.dim.add().dim_value = int(d)
        onnx.save(model, path)
    try:
        onnx.checker.check_model(model)
        rec["checker"] = "ok"
    except Exception as e:
        rec.update(status="invalid-model", stage="checker",
                   checker="failed",
                   error="%s: %s" % (type(e).__name__, e))
        rec["traceback"] = traceback.format_exc(limit=6)
        return rec

    if not do_engine:
        rec["checker"] = "ok"
        rec["status"] = "unmeasured"
        rec["measured"] = False
        with open(path[:-5] + ".pkl", "wb") as f:
            pickle.dump({"rec": rec, "feeds": spec.feeds,
                         "ref_vals": ref_vals, "oracle": oracle,
                         "path": path}, f)
        return rec

    return _engine_phase(spec, rec, spec.feeds, ref_vals, oracle, eng, path,
                         do_engine)


def _daemon_dead(e):
    """The daemon itself is gone.  That is *not* an operator result: the case
    and every case after it must be recorded as unmeasured."""
    msg = str(e).lower()
    return any(k in msg for k in (
        "daemon is dead", "daemon not running", "daemon is not running",
        "vc4d", "connect to", "connection refused", "broken pipe",
        "socket", "no such device", "device lost"))


def _wedge(e):
    """The daemon is alive but the launch did not complete; name it hang."""
    msg = str(e).lower()
    return "timed out" in msg or "timeout" in msg


def _mark_unmeasured(rec, e, stage):
    GPU_STATE["dead"] = "%s: %s" % (stage, e)
    rec.update(status="unmeasured", stage=stage, measured=False,
               daemon_dead="%s: %s" % (type(e).__name__, e))
    return rec


def _engine_phase(spec, rec, feeds, ref_vals, oracle, eng, path, do_engine):
    """Run one already-validated graph on the VC4 runner and compare."""
    if GPU_STATE["dead"]:
        rec.update(status="unmeasured", stage="run", measured=False,
                   daemon_dead="skipped: vc4d died earlier (%s)" %
                               GPU_STATE["dead"])
        return rec
    # ---- engine
    runner = None
    t0 = time.perf_counter()
    try:
        runner = _runtime.Runner(path, eng, mode="hybrid",
                                 host_policy="device")
        rec["tier"] = [n.get("tier") for n in
                       runner.coverage_report()["nodes"]]
        rec["tier_reason"] = [n.get("tier_reason") for n in
                              runner.coverage_report()["nodes"]]
        runner.check_policy()
    except (UnsupportedNode, HostNotAllowed, NeedsDeviceDispatch) as e:
        if _daemon_dead(e):
            return _mark_unmeasured(rec, e, "plan")
        if _wedge(e):
            rec.update(status="hang", stage="plan", measured=False,
                       hang={"reason": "launch did not complete",
                             "text": str(e)})
            return rec
        rec.update(status="refused", stage="plan", refusal=refusal_info(e))
        rec["engine_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return rec
    except Exception as e:
        if _daemon_dead(e):
            return _mark_unmeasured(rec, e, "plan")
        if _wedge(e):
            rec.update(status="hang", stage="plan", measured=False,
                       hang={"reason": "launch did not complete",
                             "text": "%s: %s" % (type(e).__name__, e)})
        else:
            rec.update(status="error", stage="plan",
                       error="%s: %s" % (type(e).__name__, e))
            rec["traceback"] = traceback.format_exc(limit=6)
        return rec

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, CASE_TIMEOUT_S)
    try:
        out = runner.run({k: np.ascontiguousarray(v, np.float32)
                          for k, v in feeds.items()}, profile=True)
        rec["engine_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    except (UnsupportedNode, HostNotAllowed, NeedsDeviceDispatch,
            HostNodeError) as e:
        if _daemon_dead(e):
            return _mark_unmeasured(rec, e, "run")
        if _wedge(e):
            rec.update(status="hang", stage="run", measured=False,
                       hang={"reason": "launch did not complete",
                             "text": str(e)})
            return rec
        rec.update(status="refused", stage="run", refusal=refusal_info(e))
        rec["engine_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return rec
    except Timeout as e:
        # A single-operator wedge: name it, keep the sweep alive.  The next
        # case re-opens the graph; if the daemon is gone that open is a
        # ``hang`` too and GPU_STATE stops the rest immediately.
        rec.update(status="hang", stage="run", measured=False,
                   hang={"reason": "no return within %.0f s" % CASE_TIMEOUT_S,
                         "text": str(e)})
        return rec
    except Exception as e:
        if _daemon_dead(e):
            return _mark_unmeasured(rec, e, "run")
        if _wedge(e):
            rec.update(status="hang", stage="run", measured=False,
                       hang={"reason": "launch did not complete",
                             "text": "%s: %s" % (type(e).__name__, e)})
        else:
            rec.update(status="error", stage="run",
                       error="%s: %s" % (type(e).__name__, e))
            rec["traceback"] = traceback.format_exc(limit=8)
        return rec
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)

    prof = list(getattr(runner, "profile", []) or [])
    rec["gpu_ms"] = round(sum(p.get("gpu_ms", 0.0) for p in prof), 4)
    rec["wall_ms"] = round(sum(p.get("wall_ms", 0.0) for p in prof), 4)
    rec["launches"] = int(sum(p.get("launches", 0) for p in prof))
    rec["backends"] = [p.get("backend") for p in prof]
    rec["measured"] = True
    # A metadata node the engine resolved at plan time produced a value with
    # no QPU work.  That is not a vc4 run and must not be scored as one.
    folded = bool(prof) and all(p.get("backend") == "folded" for p in prof)
    rec["got"] = [_summary(v) for v in out]

    if oracle != "ok":
        rec["status"] = "folded" if folded else "vc4-unverified"
        return rec

    all_ok = True
    stats = []
    for i, (r, g) in enumerate(zip(ref_vals, out)):
        ok, s = compare(r, g)
        s["output"] = i
        stats.append(s)
        all_ok = all_ok and ok
    rec["compare"] = stats
    if not all_ok:
        rec["status"] = "mismatch"
    elif folded:
        rec["status"] = "folded"
    else:
        rec["status"] = "vc4"
    return rec


def _summary(a):
    a = np.asarray(a)
    d = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.dtype.kind == "f" and a.size:
        if np.any(np.isfinite(a)):
            lo = float(np.nanmin(a))
            hi = float(np.nanmax(a))
            d["min"] = lo if np.isfinite(lo) else None
            d["max"] = hi if np.isfinite(hi) else None
    return d


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, onnx.TensorProto):
        return "<TensorProto %s>" % v.name
    return v


# ------------------------------------------------------------------ classify
def enumerate_ops(opset):
    hist = [s for s in defs.get_all_schemas_with_history()
            if s.domain in ("", "ai.onnx")]
    names = sorted({s.name for s in hist if defs.has(s.name, opset, "")})
    out = []
    for name in names:
        s = defs.get_schema(name, opset, "")
        out.append({"name": name, "since": s.since_version,
                    "last_version": s.since_version,
                    "domain": s.domain or "ai.onnx",
                    "deprecated": bool(s.deprecated)})
    return out


def structural_bucket(name, schema):
    if getattr(schema, "deprecated", False):
        return ("deprecated",
                "removed from the opset (last schema v%d); an opset-%d model "
                "may not contain it" % (schema.since_version,
                                        defs.onnx_opset_version()))
    if name in HOST_ONLY:
        return HOST_ONLY[name]
    if name in NON_FP32:
        return ("non-fp32",
                "%s; the engine arena is fp32 activations only" %
                NON_FP32[name])
    for a in schema.attributes.values():
        if a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
            return ("control-flow", "graph-valued attribute (subgraph)")
    for p in list(schema.inputs) + list(schema.outputs):
        if p.type_str.startswith("seq(") or p.type_str.startswith("optional("):
            return ("sequence/optional", "non-tensor value (%s)" % p.type_str)
    allowed = set()
    for tc in schema.type_constraints:
        allowed.update(tc.allowed_type_strs)
    if allowed:
        tensor_types = [a for a in allowed if a.startswith("tensor(")]
        if tensor_types and all(
                a in ("tensor(string)", "tensor(bool)") or
                re.match(r"tensor\((u?int(8|16|32|64))\)$", a)
                for a in tensor_types):
            kind = ("string" if any("string" in a for a in tensor_types)
                    else "bool" if any("bool" in a for a in tensor_types)
                    else "integer")
            return ("%s-only" % kind,
                    "all allowed tensor types are %s" % kind)
    return None


def op_status(cases):
    if any(c["status"] == "mismatch" for c in cases):
        return "mismatch"
    if any(c["status"] == "invalid-model" for c in cases):
        return "invalid-model"
    live = [c for c in cases if c["status"] != "untested"]
    if not live:
        return "untested"
    if any(c["status"] == "hang" for c in live):
        return "hang"
    measured = [c for c in live if c["status"] in
                ("vc4", "vc4-unverified", "refused", "error", "folded")]
    unmeasured = [c for c in live if c["status"] == "unmeasured"]
    if unmeasured and not measured:
        return "unmeasured"
    if not measured:
        return "unmeasured"
    if all(c["status"] == "folded" for c in measured):
        return "folded"
    ran = [c for c in measured if c["status"] in ("vc4", "vc4-unverified")]
    if not ran:
        return "refused"
    req = [c for c in measured if c["required"] and not c["edge"]]
    if any(c["status"] in ("refused", "error") for c in req):
        return "vc4-limited"
    if all(c["status"] in ("vc4",) for c in req):
        return "vc4"
    return "vc4-unverified"


# --------------------------------------------------------------------- corpus
DEFAULT_CORPUS = [
    "/usr/share/vc4infer/models/*.onnx",
    "infer/packaging/vc4infer/data/models/*.onnx",
    "infer/tools/onnx/models/*.onnx",
    "/home/raspi/work/torch-vc4/.venv/lib/python3.13/site-packages/"
    "onnx/backend/test/data/light/*.onnx",
]


def scan_corpus(patterns, repo_root):
    usage = {}
    models = []
    seen = set()
    for pat in patterns:
        full = pat if os.path.isabs(pat) else os.path.join(repo_root, pat)
        for path in sorted(glob.glob(full)):
            # the same model is shipped in several trees (installed copy,
            # in-tree copy); one model is one architecture, counted once.
            key = os.path.basename(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                m = onnx.load(path)
            except Exception:
                continue
            counts = {}
            for n in m.graph.node:
                counts[n.op_type] = counts.get(n.op_type, 0) + 1
            name = os.path.basename(path)
            models.append({"file": name, "path": path,
                           "nodes": len(m.graph.node),
                           "opset": (m.opset_import[0].version
                                     if m.opset_import else None)})
            for op, n in counts.items():
                e = usage.setdefault(op, {"models": 0, "nodes": 0,
                                          "in": []})
                e["models"] += 1
                e["nodes"] += n
                e["in"].append(name)
    return usage, models


# --------------------------------------------------------------------- report
def pct(a, b):
    return None if not b else round(100.0 * a / b, 2)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=False,
                  default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def write_markdown_validity(path, payload):
    """The no-GPU pass: denominators + graph validity, zero scored support."""
    v = payload["validity"]
    s = payload["summary"]
    lines = []
    add = lines.append
    add("# ONNX operator completeness: graph-validity pass (generated)")
    add("")
    add("This file is **generated** by `infer/tools/op_matrix.py`; do not edit")
    add("it by hand.  This revision contains the operator enumeration and the")
    add("graph-validity pass only.  **No row is scored as supported:** the")
    add("vc4 sweep is deferred while the installed stack is being rebuilt.")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Generated | %s |" % payload["generated"])
    if payload.get("versions", {}).get("vc4infer"):
        add("| Engine | vc4infer %s (imported, **not exercised**) |" %
            payload["versions"]["vc4infer"])
    else:
        add("| Engine | not imported (`%s`) |" %
            _esc(str(payload.get("vc4_import_error"))))
    add("| Validity | `onnx.checker.check_model` + "
        "`onnx.reference.ReferenceEvaluator` %s |"
        % payload["versions"]["onnx"])
    add("| ONNX / opset | onnx %s, default domain opset **%s** |" %
        (payload["versions"]["onnx"], payload["opset"]))
    add("| Tolerance (reserved for the sweep) | "
        "`abs(got-ref) <= %g + %g*max(|ref|,1)` |" %
        (payload["tolerance"]["abs"], payload["tolerance"]["rel"]))
    add("| Hardware | %s, kernel %s, stock clocks |" %
        (payload["hardware"]["model"], payload["hardware"]["kernel"]))
    add("| Scope | ONNX default domain (`ai.onnx`); other domains are not "
        "enumerated |")
    if payload.get("measurement_note"):
        add("| Provenance | %s |" % _esc(payload["measurement_note"]))
    add("")
    add("> An `unmeasured` row is **not** a supported row.  It stays outside")
    add("> the numerator until a sweep on a stable stack records `vc4`,")
    add("> `refused`, `hang` or `mismatch` for it.")
    add("")

    add("## Denominators")
    add("")
    add("| denominator | ops | graphs built | checker-valid | "
        "oracle-verified | with an oracle gap | invalid | measured on vc4 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in (s["all"], s["expressible"]):
        add("| %s | %d | %d | %d | %d | %d | %d | **0 (unmeasured)** |" % (
            row["label"], row["total"], v["ops_with_graph"],
            v["ops_checker_ok"], v["ops_oracle_ok"],
            v["ops_with_graph"] - v["ops_oracle_ok"],
            row["invalid-model"]))
    add("")
    add("| cases | built | checker-valid | oracle-verified | oracle gap "
        "| invalid | unmeasured |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    add("| %d | %d | %d | %d | %d | %d | %d |" % (
        v["cases"], v["cases"], v["cases_checker_ok"], v["cases_oracle_ok"],
        v["cases_oracle_gap"], v["cases_invalid"], v["cases_unmeasured"]))
    add("")
    add("`checker-valid` is the graph-validity guarantee: the model passes")
    add("`onnx.checker` (attributes, types, ranks, output shapes).  "
        "`oracle-verified`")
    add("additionally means `onnx.reference` executed it and produced a")
    add("reference tensor.  The %d-case gap is recorded below." %
        v["cases_oracle_gap"])
    add("")
    add("Structural exclusions from the second denominator: **%d** operators "
        "(%s)." % (len(payload["excluded"]),
                   ", ".join("%s %d" % (k, n) for k, n in
                             sorted(s["exclusions"].items()))))
    add("")

    add("## Operators by domain")
    add("")
    for dom, group in payload["domains"].items():
        add("### `%s`" % dom)
        add("")
        add("| ops | unmeasured | invalid | host-only |")
        add("|---:|---:|---:|---:|")
        add("| %d | %d | %d | %d |" % (
            group["total"], group["untested"],
            sum(1 for fam in group["families"].values() for e in fam
                if e["status"] == "invalid-model"),
            group["host_only"]))
        add("")
        for fam, ops in group["families"].items():
            add("#### %s" % fam)
            add("")
            add("| op | since | graph | checker | oracle | vc4 | note |")
            add("|---|---:|---|---|---|---|---|")
            for e in ops:
                graph = ("missing" if e.get("synth_error") else
                         "built" if e["cases"] else "-")
                checker = "ok" if (e["cases"] and
                                   all(c.get("checker") == "ok"
                                       for c in e["cases"])) else "-"
                orc = ("verified" if (e["cases"] and
                                      all(c.get("oracle") == "ok"
                                          for c in e["cases"])) else
                       "gap" if any(c.get("oracle") == "unavailable"
                                    for c in e["cases"]) else "-")
                note = (e.get("synth_error") or
                        (e["cases"][0].get("oracle_note", "")
                         if e["cases"] and
                         e["cases"][0].get("oracle") == "unavailable"
                         else ""))
                add("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                    e["name"], e["since"], graph, checker, orc,
                    _badge(e["status"]), _esc(note)))
            add("")

    add("## Excluded operators (with the reason)")
    add("")
    add("These are not part of the second denominator.  Nothing is excluded")
    add("because the engine fails it; the reason is structural.")
    add("")
    add("| op | kind | reason |")
    add("|---|---|---|")
    for e in payload["excluded"]:
        add("| `%s` | %s | %s |" % (e["op"], e["kind"], _esc(e["reason"])))
    add("")

    gaps = [(e["name"], c) for e in payload["ops"] for c in e["cases"]
            if c.get("oracle") == "unavailable"]
    if gaps:
        add("## Oracle gaps")
        add("")
        add("These graphs are checker-valid but `onnx.reference` cannot")
        add("evaluate them in this onnx release, so a future sweep can score")
        add("a refusal but cannot verify a computed result.  The exact error")
        add("is in `completeness.json`.")
        add("")
        add("| op | variant | reference evaluator error |")
        add("|---|---|---|")
        for opn, c in gaps:
            add("| `%s` | %s | `%s` |" % (
                opn, c["variant"],
                _esc(str(c.get("oracle_error", ""))[:180])))
        add("")

    add("## Unmeasured operators that real models use")
    add("")
    add("Ranked over a local corpus of %d distinct ONNX models (%s).  This "
        "is the order in which the deferred sweep matters, not a result."
        % (s["corpus"]["models"], ", ".join(s["corpus"]["files"])))
    add("")
    add("| # | op | models | nodes |")
    add("|---:|---|---:|---:|")
    for i, r in enumerate(s.get("top_unmeasured", []), 1):
        add("| %d | `%s` | %d | %d |" % (i, r["op"], r["models"], r["nodes"]))
    add("")

    add("## Method")
    add("")
    add("* Operator list: `onnx.defs.get_all_schemas_with_history()`, "
        "filtered to the default domain and to schemas in force at opset %s."
        % payload["opset"])
    add("* One node per graph, seeded inputs.  Structural parameters (axes,")
    add("  pads, shape, starts/ends) are initialisers; only fp32 activations")
    add("  are graph inputs.")
    add("* Every graph is checked with `onnx.checker.check_model` first.  If")
    add("  the checker rejects it, the case is `invalid-model` and is never")
    add("  scored.")
    add("* The graph then runs through `onnx.reference.ReferenceEvaluator` to")
    add("  produce the oracle tensor.  A reference-implementation gap is")
    add("  recorded, not treated as an engine result.")
    add("* The vc4 sweep is the same set of graphs through")
    add("  `Runner(mode='hybrid', host_policy='device')`, compared with the")
    add("  tolerance above.  It was **not run** for this revision.")
    add("")
    add("## Command for the deferred sweep")
    add("")
    add("Run only when the installed package is stable and no a concurrent job is")
    add("using the GPU (one process, one arena, one operator at a time):")
    add("")
    add("```sh")
    add("cd ~/work/vc4-stack")
    add("TMPDIR=$HOME/scratch/tmp python3 infer/tools/op_matrix.py \\")
    add("    --note \"vc4infer $(python3 -c 'import vc4infer;print("
        "vc4infer.__version__)'), clean kernel cache\"")
    add("```")
    add("")
    add("A subset is `--only Conv,Resize,Softmax`; the no-GPU pass is")
    add("`--skip-gpu`.  Every accepted graph runs sequentially; a case that")
    add("does not return in `--case-timeout` seconds is recorded as `hang`,")
    add("and a lost daemon stops the sweep instead of failing the report.")
    add("")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    os.replace(tmp, path)


def write_markdown(path, payload):
    if not payload.get("measured", True):
        return write_markdown_validity(path, payload)
    s = payload["summary"]
    lines = []
    add = lines.append
    add("# ONNX operator completeness (generated)")
    add("")
    add("This file is **generated** by `infer/tools/op_matrix.py`; do not edit")
    add("it by hand.  It is the result of running the shipped runner against")
    add("synthesised single-operator graphs, not of reading a table.")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Generated | %s |" % payload["generated"])
    add("| Engine | vc4infer %s |" % payload["versions"]["vc4infer"])
    add("| Runner | `mode=%s`, `host_policy=%s` |" %
        (payload["runner"]["mode"], payload["runner"]["host_policy"]))
    add("| ONNX / opset | onnx %s, default domain opset **%s** |" %
        (payload["versions"]["onnx"], payload["opset"]))
    add("| Oracle | onnx.reference.ReferenceEvaluator %s |" %
        payload["versions"]["onnx"])
    add("| Tolerance | `abs(got-ref) <= %g + %g*max(|ref|,1)` |" %
        (payload["tolerance"]["abs"], payload["tolerance"]["rel"]))
    add("| Hardware | %s, kernel %s, stock clocks |" %
        (payload["hardware"]["model"], payload["hardware"]["kernel"]))
    add("| Scope | ONNX default domain (`ai.onnx`); other domains are not "
        "enumerated |")
    if payload.get("measurement_note"):
        add("| Provenance | %s |" % _esc(payload["measurement_note"]))
    add("")

    add("## Headline")
    add("")
    add("Two different denominators.  The first is the whole standard at")
    add("opset %s; the second removes only the operators a static fp32 device"
        % payload["opset"])
    add("cannot express *structurally* (shape/metadata, string, sequence or")
    add("optional values, control flow, and integer/bool-only tensor ops, "
        "including int8/uint8 quantized operators, because the runner accepts "
        "fp32 graph inputs only).  The")
    add("second number is never presented as the first.")
    add("")
    add("| denominator | ops | **complete (vc4)** | vc4-limited | refused "
        "| mismatch | hang | folded | unmeasured | invalid | excluded "
        "| **pct_complete** | pct_expressible |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in (s["all"], s["expressible"]):
        add("| %s | %d | **%d** | %d | %d | %d | %d | %d | %d | %d | %d "
            "| **%s** | %s |" % (
                row["label"], row["total"], row["complete"],
                row["vc4-limited"], row["refused"], row["mismatch"],
                row["hang"], row["folded"], row["unmeasured"],
                row["invalid-model"], row["excluded"],
                _pct(row["pct_complete"]),
                _pct(row["pct_expressible"])))
    add("")
    add("`folded` = resolved by the engine at plan time with the host")
    add("reference evaluator (constant table generators).  It is reported,")
    add("but never counted as vc4.  `complete (vc4)` = every required form")
    add("tested here, including the attribute variants listed below, ran")
    add("within tolerance.  `pct_expressible`")
    add("counts a working general case (`vc4` + `vc4-limited`); it is not the")
    add("same number as `pct_complete` and is never presented as it.")
    add("")
    if payload.get("daemon_dead"):
        add("> **The sweep stopped early: vc4d died.**  `%s`  Every case")
        add("> after that point is `unmeasured`, never `refused` or `hang`;")
        add("> a dead daemon is not an engine result." %
            _esc(str(payload["daemon_dead"])))
        add("")
    add("Structural exclusions from the second denominator: **%d** operators "
        "(%s).  They are listed with their exact reason in "
        "`completeness.json` under `excluded`." % (
            len(payload["excluded"]),
            ", ".join("%s %d" % (k, v) for k, v in
                      sorted(s["exclusions"].items()))))
    add("")
    _nf = sum(1 for e in payload["excluded"] if e["kind"] == "non-fp32")
    _den = len(payload["ops"]) - len(payload["excluded"]) + _nf
    add("Sensitivity: if the %d integer/bool-only operators (including the "
        "quantized ones) were counted as expressible, the denominator would "
        "be **%d** and the complete figure **%d/%d = %.2f%%**.  They are "
        "excluded because `Runner.__init__` accepts fp32 graph inputs only, "
        "so no valid instance of those operators can reach the arena." % (
            _nf, _den, s["expressible"]["complete"], _den,
            pct(s["expressible"]["complete"], _den)))
    add("")

    if s["mismatches"]:
        add("## MISMATCHES (bugs, highest priority)")
        add("")
        add("| op | variant | max abs | max rel | worst ref -> got | "
            "explanation |")
        add("|---|---|---:|---:|---|---|")
        for m in s["mismatches"]:
            add("| `%s` | %s | %s | %s | %s -> %s | %s |" % (
                m["op"], m["variant"], _fmt(m.get("max_abs")),
                _fmt(m.get("max_rel")), _fmt(m.get("worst_ref")),
                _fmt(m.get("worst_got")),
                _esc(KNOWN_BUGS.get((m["op"], m["variant"]), ""))))
        add("")

    if s.get("engine_errors"):
        add("## Internal engine errors (bug reports)")
        add("")
        add("These are not policy refusals: the node was dispatched, the")
        add("engine's own call failed, and the runner wrapped the library")
        add("error in `UnsupportedNode`.  No output was produced, so they are")
        add("scored as `refused`, but they are engine bugs, not constraints.")
        add("")
        add("| op | variant | engine error |")
        add("|---|---|---|")
        for e in s["engine_errors"]:
            add("| `%s` | %s | `%s` |" % (e["op"], e["variant"],
                                          _esc(e["message"])))
        add("")

    add("## Excluded operators (with the reason)")
    add("")
    add("These are not part of the second denominator.  Nothing is excluded")
    add("because the engine fails it; the reason is structural.")
    add("")
    add("| op | kind | reason |")
    add("|---|---|---|")
    for e in payload["excluded"]:
        add("| `%s` | %s | %s |" % (e["op"], e["kind"], _esc(e["reason"])))
    add("")

    add("## Operators by domain")
    add("")
    for dom, group in payload["domains"].items():
        add("### `%s`" % dom)
        add("")
        add("| ops | vc4 | vc4-limited | refused | mismatch | untested "
            "| host-only |")
        add("|---:|---:|---:|---:|---:|---:|---:|")
        add("| %d | %d | %d | %d | %d | %d | %d |" % (
            group["total"], group["vc4"], group["limited"],
            group["refused"], group["mismatch"], group["untested"],
            group["host_only"]))
        add("")
        for fam, ops in group["families"].items():
            add("#### %s" % fam)
            add("")
            add("| op | since | status | required forms | edge forms "
                "| max abs | max rel | gpu ms | refusal / constraint |")
            add("|---|---:|---|---:|---:|---:|---:|---:|---|")
            for e in ops:
                req = [c for c in e["cases"] if c["required"] and
                       not c["edge"]]
                edge = [c for c in e["cases"] if c["edge"]]
                req_ok = sum(1 for c in req
                             if c["status"] in ("vc4", "vc4-unverified"))
                edge_ok = sum(1 for c in edge
                              if c["status"] in ("vc4", "vc4-unverified"))
                gpu = [c.get("gpu_ms") for c in e["cases"]
                       if c.get("gpu_ms")]
                cmps = [s for c in e["cases"]
                        for s in (c.get("compare") or [])]
                mabs = [s.get("max_abs") for s in cmps
                        if s.get("max_abs") is not None]
                mrel = [s.get("max_rel") for s in cmps
                        if s.get("max_rel") is not None]
                why = e.get("primary_refusal") or ""
                if e["status"] == "mismatch":
                    why = "see MISMATCHES above"
                add("| `%s` | %s | %s | %d/%d | %d/%d | %s | %s | %s | %s |" % (
                    e["name"], e["since"], _badge(e["status"]),
                    req_ok, len(req), edge_ok, len(edge),
                    _fmt(max(mabs) if mabs else None),
                    _fmt(max(mrel) if mrel else None),
                    _fmt(max(gpu) if gpu else None), _esc(why)))
            add("")

    add("## Refusals, verbatim")
    add("")
    add("Every row is the engine's own exception text for a graph that the")
    add("ReferenceEvaluator accepted.  `plan` means the policy refused before")
    add("any node executed; `run` means it refused during execution.")
    add("")
    add("| op | variant | phase | exception | invariant |")
    add("|---|---|---|---|---|")
    for e in payload["ops"]:
        for c in e["cases"]:
            if c["status"] != "refused":
                continue
            r = c["refusal"]
            add("| `%s` | %s | %s | `%s` | `%s` |" % (
                e["name"], c["variant"], c.get("stage", ""),
                _esc(r.get("type", "")), _esc(r.get("invariant", ""))))
    add("")

    gaps = [(e["name"], c) for e in payload["ops"] for c in e["cases"]
            if c.get("oracle") == "unavailable"]
    if gaps:
        add("## Oracle gaps")
        add("")
        add("These graphs are **valid** (the ONNX checker accepts them) but")
        add("`onnx.reference` cannot evaluate them, so only the engine's")
        add("refusal could be scored.  The exact reference error is in")
        add("`completeness.json` (`oracle_error`).")
        add("")
        add("| op | variant | reference evaluator error |")
        add("|---|---|---|")
        for opn, c in gaps:
            add("| `%s` | %s | `%s` |" % (
                opn, c["variant"],
                _esc(str(c.get("oracle_error", ""))[:180])))
        add("")

    add("## Top refusals by real-model usage")
    add("")
    add("Ranked over a local corpus of %d distinct real ONNX models (%s): the "
        "number of models that contain the operator, then node count." %
        (s["corpus"]["models"], ", ".join(s["corpus"]["files"])))
    if len(s["top_refusals"]) < 10:
        add("")
        add("The corpus contains **%d refused operators** in total; all of "
            "them are listed (there is no tenth)." %
            len(s["top_refusals"]))
    add("")
    add("| # | op | status | models | nodes |")
    add("|---:|---|---|---:|---:|")
    for i, r in enumerate(s["top_refusals"], 1):
        add("| %d | `%s` | %s | %d | %d |" % (
            i, r["op"], _badge(r["status"]), r["models"], r["nodes"]))
    add("")

    add("## Method")
    add("")
    add("* Operator list: `onnx.defs.get_all_schemas_with_history()`, filtered")
    add("  to the default domain and to schemas in force at opset %s." %
        payload["opset"])
    add("* Each graph is one node with seeded random inputs.  Conv weights and")
    add("  all structural parameters (axes, pads, shape, starts/ends) are")
    add("  initialisers; only fp32 activations are graph inputs.")
    add("* The oracle runs first.  If it rejects the graph, the case is")
    add("  `invalid-model` (a bug in this tool, never scored) and the run")
    add("  stops there.")
    add("* Then `Runner(...).check_policy()` runs.  A refusal at this point")
    add("  allocates no GPU work.  Only accepted graphs execute.")
    add("* Non-float outputs must match exactly; float outputs use the")
    add("  tolerance above, with NaN==NaN and matching infinities allowed.")
    add("* `gpu_ms` is the engine's own pack+kernel time for the node, not wall")
    add("  time, and is reported per case in `completeness.json`.")
    add("")
    add("## What is not measured")
    add("")
    add("* Other ONNX domains (`ai.onnx.ml`, `ai.onnx.preview.training`,")
    add("  vendor domains).")
    add("* Multi-node graphs: fusion, layout propagation and allocation are")
    add("  out of scope; every case is a single operator.")
    add("* Performance: `gpu_ms` is reported for information only.")
    add("* Operators excluded as structurally host-only are listed with their")
    add("  reason in `completeness.json` (`excluded`).")
    add("")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    os.replace(tmp, path)


def _pct(v):
    return "not measured" if v is None else "%.2f%%" % v


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return "%.2e" % v
        return "%.3g" % v
    return str(v)


def _esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def _badge(status):
    return {"vc4": "**vc4**", "vc4-limited": "vc4-limited",
            "refused": "refused", "mismatch": "**MISMATCH**",
            "hang": "**HANG**", "folded": "folded (host)",
            "invalid-model": "invalid-model",
            "untested": "unmeasured (no graph)",
            "unmeasured": "unmeasured",
            "vc4-unverified": "vc4-unverified"}.get(status, status)


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the ONNX operator-coverage matrix for vc4infer")
    ap.add_argument("--opset", type=int, default=defs.onnx_opset_version())
    ap.add_argument("--only", default="",
                    help="comma-separated operator names")
    ap.add_argument("--skip", default="")
    ap.add_argument("--stage", choices=("synth", "all"), default="all",
                    help="synth: build+validate graphs only (no vc4)")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="alias for --stage synth; the vc4 import is optional "
                         "and no device is opened")
    ap.add_argument("--case-timeout", type=float, default=60.0,
                    help="wall-clock guard per graph (seconds); a case that "
                         "does not return is recorded as 'hang'")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--reuse", default=None,
                    help="replay the cases saved by --stage synth in this dir")
    ap.add_argument("--corpus", default="",
                    help="comma-separated glob patterns")
    ap.add_argument("--note", default="",
                    help="provenance note printed in the report header")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    global CASE_TIMEOUT_S
    CASE_TIMEOUT_S = max(1.0, float(args.case_timeout))
    gpu = (args.stage == "all") and not args.skip_gpu
    if args.reuse and not gpu:
        print("ERROR: --reuse replays the GPU phase; it cannot be combined "
              "with --skip-gpu", file=sys.stderr)
        return 3
    if gpu and vc4infer is None:
        print("ERROR: the vc4infer package is not importable (%s).\n"
              "Run the graph-validity pass with --skip-gpu; run the GPU "
              "sweep only when the installed stack is stable."
              % VC4_IMPORT_ERROR, file=sys.stderr)
        return 3

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    out_dir = args.out_dir or os.path.join(repo, "docs")
    os.makedirs(out_dir, exist_ok=True)

    tmp_root = os.environ.get("TMPDIR") or os.path.expanduser("~/scratch/tmp")
    os.makedirs(tmp_root, exist_ok=True)
    if args.work_dir:
        workdir = args.work_dir
        os.makedirs(workdir, exist_ok=True)
    else:
        workdir = tempfile.mkdtemp(prefix="opmatrix-", dir=tmp_root)

    ops = enumerate_ops(args.opset)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if only:
        ops = [o for o in ops if o["name"] in only]
    ops = [o for o in ops if o["name"] not in skip]
    opmap = {o["name"]: o for o in ops}

    # --- structural classification (the honest denominator)
    records = {}
    for o in ops:
        schema = load_schema(o["name"], args.opset)
        bucket = structural_bucket(o["name"], schema)
        o["exclude"] = None
        if bucket:
            o["exclude"] = {"kind": bucket[0], "reason": bucket[1]}
        records[o["name"]] = {"name": o["name"], "since": o["since"],
                              "last_version": o.get("last_version"),
                              "domain": o["domain"],
                              "deprecated": o["deprecated"],
                              "excluded": o["exclude"], "cases": []}

    # --- synthesis + oracle (no GPU)
    synth_cache = {}
    skipped_synth = {}
    if args.reuse:
        for o in ops:
            name = o["name"]
            if o["exclude"]:
                continue
            if glob.glob(os.path.join(args.reuse, "%s__*.pkl" % name)):
                synth_cache[name] = [None]
            else:
                skipped_synth[name] = "no saved cases in %s" % args.reuse
        if args.stage != "all":
            print("ERROR: --reuse requires --stage all", file=sys.stderr)
            return 3
    else:
        for o in ops:
            name = o["name"]
            if o["exclude"]:
                continue
            schema = load_schema(name, args.opset)
            rng = np.random.default_rng(zlib.crc32(
                ("%s/%s" % (name, args.opset)).encode()) & 0x7FFFFFFF)
            try:
                fn = RECIPES.get(name, generic_recipe)
                if fn is generic_recipe:
                    specs = generic_recipe(name, rng, schema, args.opset)
                else:
                    specs = fn(name, rng, args.opset)
            except SynthesisError as e:
                skipped_synth[name] = str(e)
                continue
            except Exception as e:
                skipped_synth[name] = "%s: %s" % (type(e).__name__, e)
                continue
            synth_cache[name] = specs
            if not args.quiet:
                print("[synth] %-28s %d case(s)" % (name, len(specs)),
                      file=sys.stderr, flush=True)

    # --- engine.  The daemon serves one arena; another client makes the open
    # fail with NOMEM.  Retry for a few minutes rather than colliding or
    # giving up, and never run two operators at once.
    eng = None
    if gpu and synth_cache:
        last = None
        for attempt in range(24):
            try:
                eng = _runtime.open_engine(retry_s=2.0)
                break
            except Exception as e:
                last = e
                print("[gpu] arena busy (%s); retry %d/24" % (e, attempt + 1),
                      file=sys.stderr, flush=True)
                time.sleep(3)
        if eng is None:
            print("ERROR: cannot open the VC4 engine: %s" % last,
                  file=sys.stderr)
            return 3

    results = []
    try:
        for o in ops:
            name = o["name"]
            rec = records[name]
            if o["exclude"]:
                results.append(rec)
                continue
            if args.reuse:
                pkls = sorted(glob.glob(os.path.join(
                    args.reuse, "%s__*.pkl" % name)))
                if not pkls:
                    results.append(rec)
                    continue
                for pkl in pkls:
                    with open(pkl, "rb") as f:
                        cached = pickle.load(f)
                    spec = _spec_from_cache(cached, name, args.opset)
                    if not args.quiet:
                        print("[run]   %-28s %-32s" % (name, spec.variant),
                              end="", file=sys.stderr, flush=True)
                    t0 = time.perf_counter()
                    r = run_case(spec, None, args.opset, workdir, eng,
                                 do_engine=True, cache=cached)
                    rec["cases"].append(r)
                    if not args.quiet:
                        print(" -> %-14s %7.1f ms" % (
                            r["status"], (time.perf_counter() - t0) * 1e3),
                            file=sys.stderr, flush=True)
            else:
                if name in skipped_synth:
                    rec["synth_error"] = skipped_synth[name]
                    rec["cases"] = [{"op": name, "variant": "-",
                                     "status": "untested",
                                     "stage": "synthesis",
                                     "error": skipped_synth[name],
                                     "required": True, "edge": False}]
                    results.append(rec)
                    continue
                if name not in synth_cache:
                    results.append(rec)
                    continue
                schema = load_schema(name, args.opset)
                for spec in synth_cache[name]:
                    if not args.quiet:
                        print("[run]   %-28s %-32s" % (name, spec.variant),
                              end="", file=sys.stderr, flush=True)
                    t0 = time.perf_counter()
                    r = run_case(spec, schema, args.opset, workdir, eng,
                                 do_engine=gpu)
                    rec["cases"].append(r)
                    if not args.quiet:
                        print(" -> %-14s %7.1f ms" % (
                            r["status"], (time.perf_counter() - t0) * 1e3),
                            file=sys.stderr, flush=True)
            rec["status"] = op_status(rec["cases"])
            rec["gpu_ms"] = sum(c.get("gpu_ms") or 0.0 for c in rec["cases"])
            rec["primary_refusal"] = _primary_refusal(rec)
            results.append(rec)
    finally:
        if eng is not None:
            try:
                eng.close()
            except Exception:
                pass

    # --- ranking against a real-model corpus
    patterns = ([p.strip() for p in args.corpus.split(",") if p.strip()]
                or DEFAULT_CORPUS)
    usage, corpus_models = scan_corpus(patterns, repo)

    for rec in results:
        if rec.get("excluded"):
            rec["status"] = "excluded"
    for rec in results:
        if not rec.get("status"):
            rec["status"] = op_status(rec["cases"]) if rec["cases"] else \
                "untested"
    by_name = {r["name"]: r for r in results}
    for rec in results:
        u = usage.get(rec["name"])
        rec["usage"] = u or {"models": 0, "nodes": 0, "in": []}

    refused_ops = [r for r in results if r["status"] == "refused"]
    used = [r for r in refused_ops if r["usage"]["models"] > 0]
    top = sorted(used, key=lambda r: (-r["usage"]["models"],
                                      -r["usage"]["nodes"], r["name"]))[:10]
    unmeasured_ops = [r for r in results if r["status"] == "unmeasured"
                      and not r["excluded"]]
    top_unmeasured = sorted(
        [r for r in unmeasured_ops if r["usage"]["models"] > 0],
        key=lambda r: (-r["usage"]["models"], -r["usage"]["nodes"],
                       r["name"]))[:10]

    # --- summary
    host_only = [r for r in results if r["excluded"]]
    scored = [r for r in results if not r["excluded"]]
    exclusions = {}
    for r in host_only:
        k = r["excluded"]["kind"]
        exclusions[k] = exclusions.get(k, 0) + 1

    def tally(group):
        t = {"total": len(group)}
        for st in ("vc4", "vc4-limited", "refused", "mismatch", "hang",
                   "folded", "invalid-model", "untested", "unmeasured",
                   "vc4-unverified"):
            t[st] = sum(1 for r in group if r["status"] == st)
        t["complete"] = t["vc4"]
        t["expressible"] = t["vc4"] + t["vc4-limited"]
        # anything unverified or invalid is not claimed as working
        t["pct_complete"] = pct(t["complete"], t["total"])
        t["pct_expressible"] = pct(t["expressible"], t["total"])
        return t

    all_t = tally(results)
    all_t["label"] = "all ops in opset %s" % args.opset
    all_t["excluded"] = len(host_only)
    all_t["exclusions"] = exclusions
    expr_t = tally(scored)
    expr_t["label"] = "ops not structurally host-only"
    expr_t["excluded"] = 0
    expr_t["exclusions"] = exclusions
    if not gpu:
        # A percentage of zero would read as a measurement.  It is not one.
        for t in (all_t, expr_t):
            t["measured"] = False
            t["pct_complete"] = None
            t["pct_expressible"] = None

    engine_errors = []
    for r in results:
        for c in r["cases"]:
            ref = c.get("refusal") or {}
            if c["status"] == "refused" and \
                    str(ref.get("detail", "")).startswith("engine error"):
                engine_errors.append({
                    "op": r["name"], "variant": c["variant"],
                    "message": ref.get("detail"), "phase": c.get("stage")})
    mismatches = []
    for r in results:
        for c in r["cases"]:
            if c["status"] != "mismatch":
                continue
            cm = (c.get("compare") or [{}])
            bad = [x for x in cm if not x.get("over_tolerance", 0) == 0]
            s = bad[0] if bad else cm[0]
            mismatches.append({
                "op": r["name"], "variant": c["variant"],
                "max_abs": s.get("max_abs"), "max_rel": s.get("max_rel"),
                "worst_ref": s.get("worst_ref"),
                "worst_got": s.get("worst_got"),
                "worst_index": s.get("worst_index"),
                "refusal": None})

    # --- per-domain, per-family grouping
    families = {
        "NN / convolution / pooling": (
            "Conv", "ConvTranspose", "DeformConv", "ConvInteger",
            "AveragePool", "MaxPool", "LpPool", "GlobalAveragePool",
            "GlobalMaxPool", "GlobalLpPool", "MaxUnpool", "MaxRoiPool",
            "LRN", "BatchNormalization", "InstanceNormalization",
            "GroupNormalization", "LayerNormalization",
            "MeanVarianceNormalization", "LpNormalization", "Dropout",
            "RoiAlign", "GridSample", "Col2Im", "SpaceToDepth",
            "DepthToSpace"),
        "Matrix / reduction / loss": (
            "MatMul", "Gemm", "Einsum", "Det", "ReduceL1", "ReduceL2",
            "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
            "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
            "ArgMax", "ArgMin", "SoftmaxCrossEntropyLoss",
            "NegativeLogLikelihoodLoss", "CumSum", "TopK", "Unique"),
        "Activation / elementwise": (
            "Relu", "Sigmoid", "Tanh", "Silu", "HardSigmoid", "HardSwish",
            "LeakyRelu", "Elu", "Celu", "Selu", "PRelu", "ThresholdedRelu",
            "Mish", "Gelu", "Softplus", "Softsign", "Softmax", "LogSoftmax",
            "Hardmax", "Abs", "Neg", "Sign", "Clip", "Ceil", "Floor", "Round",
            "Exp", "Log", "Sqrt", "Reciprocal", "Erf", "Sin", "Cos", "Tan",
            "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh",
            "Shrink", "IsInf", "IsNaN", "Add", "Sub", "Mul", "Div", "Pow",
            "Mod", "Max", "Min", "Mean", "Sum"),
        "Tensor / shape manipulation": (
            "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose",
            "Concat", "Split", "Slice", "Pad", "Tile", "Expand", "Gather",
            "GatherElements", "GatherND", "Scatter", "ScatterElements",
            "ScatterND", "Resize", "Upsample", "Compress", "OneHot",
            "EyeLike", "Trilu", "ReverseSequence", "CenterCropPad",
            "ConstantOfShape", "Range", "Shape", "Size", "Constant",
            "Identity", "Cast", "CastLike"),
        "Quantization": ("QuantizeLinear", "DequantizeLinear",
                         "DynamicQuantizeLinear", "QLinearConv",
                         "QLinearMatMul", "MatMulInteger"),
        "Logical / comparison": ("And", "Or", "Xor", "Not", "Equal",
                                 "Greater", "GreaterOrEqual", "Less",
                                 "LessOrEqual", "Where", "BitShift",
                                 "BitwiseAnd", "BitwiseNot", "BitwiseOr",
                                 "BitwiseXor"),
        "Recurrent": ("RNN", "GRU", "LSTM"),
        "Signal / random / other": ("DFT", "STFT", "MelWeightMatrix",
                                    "BlackmanWindow", "HammingWindow",
                                    "HannWindow", "Bernoulli", "RandomNormal",
                                    "RandomNormalLike", "RandomUniform",
                                    "RandomUniformLike", "Multinomial",
                                    "NonMaxSuppression", "NonZero",
                                    "AffineGrid", "ImageDecoder"),
        "Control flow / sequence / string": (
            "If", "Loop", "Scan", "SequenceMap", "ConcatFromSequence",
            "SequenceAt", "SequenceConstruct", "SequenceEmpty",
            "SequenceErase", "SequenceInsert", "SequenceLength",
            "SplitToSequence", "Optional", "OptionalGetElement",
            "OptionalHasElement", "RegexFullMatch", "StringConcat",
            "StringNormalizer", "StringSplit", "TfIdfVectorizer"),
    }
    fam_of = {}
    for fam, names in families.items():
        for n in names:
            fam_of[n] = fam

    domains = {}
    for r in results:
        d = r["domain"]
        g = domains.setdefault(d, {"total": 0, "vc4": 0, "limited": 0,
                                   "refused": 0, "mismatch": 0,
                                   "untested": 0, "host_only": 0,
                                   "families": {}})
        g["total"] += 1
        if r["excluded"]:
            g["host_only"] += 1
        else:
            key = {"vc4": "vc4", "vc4-limited": "limited",
                   "refused": "refused", "mismatch": "mismatch"}.get(
                       r["status"], "untested")
            g[key] += 1
        fam = fam_of.get(r["name"], "Other")
        g["families"].setdefault(fam, []).append(r)
    for g in domains.values():
        for fam in g["families"]:
            g["families"][fam].sort(key=lambda r: r["name"])

    # ---- graph-validity accounting (the trustworthy denominator)
    scored = [r for r in results if not r["excluded"]]
    cases = [c for r in scored for c in r["cases"]]
    validity = {
        "ops_with_graph": sum(1 for r in scored if r["cases"]),
        "ops_checker_ok": sum(
            1 for r in scored if r["cases"] and
            all(c.get("checker") == "ok" for c in r["cases"])),
        "ops_oracle_ok": sum(
            1 for r in scored if r["cases"] and
            all(c.get("oracle") == "ok" for c in r["cases"])),
        "cases": len(cases),
        "cases_checker_ok": sum(1 for c in cases
                                if c.get("checker") == "ok"),
        "cases_oracle_ok": sum(1 for c in cases
                               if c.get("oracle") == "ok"),
        "cases_oracle_gap": sum(1 for c in cases
                                if c.get("oracle") == "unavailable"),
        "cases_invalid": sum(1 for c in cases
                             if c["status"] == "invalid-model"),
        "cases_unmeasured": sum(1 for c in cases
                                if c["status"] == "unmeasured"),
        "cases_synthesis_failed": sum(
            1 for r in scored if r.get("synth_error")),
    }

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured": bool(gpu),
        "mode": "full" if gpu else "graph-validity",
        "vc4_import_error": VC4_IMPORT_ERROR,
        "daemon_dead": GPU_STATE["dead"],
        "case_timeout_s": CASE_TIMEOUT_S,
        "tool": "infer/tools/op_matrix.py",
        "opset": args.opset,
        "domain": "ai.onnx",
        "tolerance": {"abs": TOL_ABS, "rel": TOL_REL},
        "runner": {"mode": "hybrid", "host_policy": "device",
                   "mode_note": "host_policy=device forbids runtime host "
                                "compute; only plan-time constant/shape "
                                "folding is allowed"},
        "engine_artefacts": _engine_artefacts(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "vc4infer": getattr(vc4infer, "__version__", None),
            "vc4infer_file": getattr(vc4infer, "__file__", None),
        },
        "engine": _engine_artefacts(),
        "hardware": {
            "model": _model_name(),
            "kernel": platform.release(),
            "clocks": "stock (no overclock applied by this tool)",
        },
        "work_dir": workdir,
        "measurement_note": args.note,
        "validity": validity,
        "summary": {
            "all": all_t,
            "expressible": expr_t,
            "mismatches": mismatches,
            "engine_errors": engine_errors,
            "exclusions": exclusions,
            "corpus": {"models": len(corpus_models),
                       "files": [m["file"] for m in corpus_models],
                       "patterns": patterns},
            "top_unmeasured": [{"op": r["name"],
                                "models": r["usage"]["models"],
                                "nodes": r["usage"]["nodes"]}
                               for r in top_unmeasured],
            "top_refusals": [{"op": r["name"], "status": r["status"],
                              "models": r["usage"]["models"],
                              "nodes": r["usage"]["nodes"]} for r in top],
        },
        "corpus": {"models": corpus_models, "usage": usage},
        "excluded": [{"op": r["name"], "kind": r["excluded"]["kind"],
                      "reason": r["excluded"]["reason"]}
                     for r in host_only],
        "domains": domains,
        "ops": results,
    }
    json_path = os.path.join(out_dir, "completeness.json")
    md_path = os.path.join(out_dir, "COMPLETENESS.md")
    write_json(json_path, payload)
    if args.stage == "all":
        write_markdown(md_path, payload)

    print("")
    print("ops: %d  structurally excluded: %d  expressible denominator: %d" % (
        all_t["total"], len(host_only), expr_t["total"]))
    if gpu:
        print("complete   %d/%d = %.2f%%  (of all ops: %.2f%%)" % (
            expr_t["complete"], expr_t["total"], expr_t["pct_complete"],
            all_t["pct_complete"]))
        print("general    %d/%d = %.2f%%  (of all ops: %.2f%%)" % (
            expr_t["expressible"], expr_t["total"],
            expr_t["pct_expressible"], all_t["pct_expressible"]))
        print("refused %d  mismatch %d  hang %d  invalid-model %d  "
              "untested %d" % (
                  expr_t["refused"], expr_t["mismatch"], expr_t["hang"],
                  expr_t["invalid-model"], expr_t["untested"]))
    else:
        print("measured on vc4: 0 (deferred; every row is unmeasured)")
        print("graphs built %d/%d ops  cases %d  checker-valid %d  "
              "oracle-verified %d  oracle gaps %d  invalid %d" % (
                  validity["ops_with_graph"], expr_t["total"],
                  validity["cases"], validity["cases_checker_ok"],
                  validity["cases_oracle_ok"], validity["cases_oracle_gap"],
                  validity["cases_invalid"]))
    print("wrote %s" % json_path)
    if args.stage == "all":
        print("wrote %s" % md_path)
    print("models kept in %s" % workdir)
    if GPU_STATE["dead"]:
        print("SWEEP STOPPED: vc4d died (%s); the remaining cases are "
              "recorded as unmeasured" % GPU_STATE["dead"], file=sys.stderr)

    if GPU_STATE["dead"]:
        return 4
    if expr_t["invalid-model"]:
        print("ERROR: %d operators have an invalid synthesised model; fix the "
              "tool before scoring" % expr_t["invalid-model"], file=sys.stderr)
        return 2
    return 0


def _spec_from_cache(cache, op, opset):
    """Rebuild the fields the replay needs from a saved record."""
    rec = cache["rec"]
    return Spec(op=op, variant=rec.get("variant", "cached"),
                ins=rec.get("inputs") or [], feeds=cache["feeds"],
                inits={}, required=rec.get("required", True),
                edge=rec.get("edge", False),
                opset=rec.get("opset", opset), note=rec.get("note", ""))


def _primary_refusal(rec):
    for c in rec["cases"]:
        if c["status"] == "refused" and c.get("required") and \
                not c.get("edge"):
            r = c["refusal"]
            return r.get("invariant") or r.get("text")
    for c in rec["cases"]:
        if c["status"] == "refused":
            r = c["refusal"]
            return r.get("invariant") or r.get("text")
    return ""


def _engine_artefacts():
    import hashlib
    out = {}
    for key, path in (("lib", _lib_path()),
                      ("kops", os.path.join(
                          os.path.dirname(vc4infer.__file__), "kgen", "kops.py")),
                      ("kgen", os.path.join(
                          os.path.dirname(vc4infer.__file__), "kgen", "kgen.py"))):
        if not path:
            continue
        try:
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            out[key] = {"path": path, "sha256": h,
                        "mtime": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(os.path.getmtime(path)))}
        except OSError:
            pass
    return out


def _lib_path():
    try:
        from vc4infer import _binding
        return _binding._LIB
    except Exception:                             # noqa: BLE001
        return None


def _engine_artefacts():
    import hashlib
    if vc4infer is None:
        return {"import_error": VC4_IMPORT_ERROR}
    out = {}
    pkg = os.path.dirname(vc4infer.__file__)
    for key, path in (("lib", _lib_path()),
                      ("kops", os.path.join(pkg, "kgen", "kops.py")),
                      ("kgen", os.path.join(pkg, "kgen", "kgen.py"))):
        try:
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            out[key] = {"path": path, "sha256": h}
        except (OSError, TypeError):
            pass
    return out


def _lib_path():
    from vc4infer import _binding
    return _binding._LIB


def _model_name():
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(p, "rb") as f:
                return f.read().replace(b"\x00", b"").decode(errors="replace")
        except OSError:
            pass
    return platform.machine()


if __name__ == "__main__":
    sys.exit(main())
