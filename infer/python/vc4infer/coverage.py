"""Static node coverage and host-tier classification for the vc4infer executor.

This mirrors the dispatch table and the constraint checks in ``onnxrun.py``.
It is a *static* report: it reads the graph, the initialiser shapes and the
shape-inferred value_info, and flags a node as unsupported when the executor
would raise on it.  The executor stays authoritative -- ``vc4infer run`` is what
actually decides -- but the report answers "will this model run?" in a second
and names the node that will fail.

Status values:

    supported    the executor has an implementation for this op and these
                 attributes/shapes
    unsupported  the executor would raise ``UnsupportedNode``; ``reason`` says
                 why
    unknown      the op is implemented but a constraint depends on a shape that
                 static inference could not produce (a data-dependent shape)

Every node is also classified into exactly one execution tier:

    vc4                 the VC4 executor runs it (its kernels may do byte
                        movement on the ARM, as the engine always has)
    host-meta           metadata / shape arithmetic only (Shape, Size,
                        Constant, ConstantOfShape, and integer or boolean
                        tensor plumbing): no activation arithmetic
    host-control        If / Loop / Scan: the executor is host-scheduled, so the
                        node runs in the reference evaluator
    host-unwritten      a numeric operator with no VC4 kernel yet.  This is
                        *debt*: structural hybrid refuses it, ``host_policy=
                        "any"`` is the explicit escape hatch
    unsupported-graph   invalid or unsupported types/domains that cannot be run
                        at all, on either backend

``backend_plan`` turns the tiers into the per-node ``vc4 | host-meta |
host-control | host-unwritten | unsupported-graph`` report used by
``Engine.node_backends`` and ``vc4infer nodes``.
"""
import os
import onnx
import numpy as np
from onnx import TensorProto, numpy_helper

# op -> short implementation note (mirrors onnxrun.Runner._dispatch)
OPS = {
    # the geometry / detection family (_dispatch_geom.py,
    # docs/OPS-GEOMETRY.md).  Each entry names the kernels the handler composes.
    "ConvTranspose": "QPU gemm (W^T @ X) + QPU tap scatter (col2im), bias QPU",
    "Col2Im": "QPU k^2 tap scatter (copy2d lane stride) + QPU vector add",
    "AffineGrid": "QPU affine x/y planes + one-word strided store interleave",
    "GridSample": "QPU denormalise/reflect/floor, index read-back, 4 copy2d taps, QPU bilinear weights",
    "RoiAlign": "QPU ROI sample coordinates, 4 copy2d taps per sample, QPU GEMM avg / max chain",
    "MaxRoiPool": "QPU integer ROI bins, window row copies + QPU max chain",
    "NonMaxSuppression": "QPU IoU matrix + pairwise score rank (GEMM), QPU triangular suppression predicate",
    "Det": "QPU permutation gathers + QPU mul/add chain (Leibniz expansion)",
    "Einsum": "QPU copy2d permutations + QPU gemm contraction (lowered, not a kernel)",
    "DeformConv": "QPU offset coordinates + 4-tap bilinear im2col + QPU gemm",
    "Conv": "QPU conv: NHWC 1x1/3x3/5x5 s1/s2, dilation 1..8, groups",
    "Relu": "QPU elementwise",
    "Abs": "QPU unary (vc4_unary UOP_ABS, exact)",
    "Neg": "QPU unary (UOP_NEG, exact)",
    "Sign": "QPU unary (UOP_SIGN, exact)",
    "Clip": "QPU unary (UOP_CLIP, scalar limits in uniforms)",
    "LeakyRelu": "QPU unary (UOP_LEAKYRELU, alpha in a uniform)",
    "HardSigmoid": "QPU unary (UOP_HARDSIGMOID, alpha/beta)",
    "HardSwish": "QPU unary (UOP_HARDSWISH)",
    "Floor": "QPU unary (UOP_FLOOR, exact; |x| >= 2^23 passes through)",
    "Ceil": "QPU unary (UOP_CEIL, exact; |x| >= 2^23 passes through)",
    "Softsign": "QPU unary (UOP_SOFTSIGN, x/(1+|x|), max err 1.6e-7)",
    "Sigmoid": "QPU elementwise",
    "Tanh": "QPU sigmoid composition: 2*sigmoid(2x)-1",
    "Silu": "QPU elementwise (fused)",
    "Add": "QPU elementwise",
    "Mul": "QPU elementwise",
    "Sub": "QPU Mul(-1)+Add",
    "Div": "QPU Mul by folded constant",
    "MaxPool": "QPU maxpool_nhwc + QPU -inf border",
    "Concat": "QPU copy2d, pitched channel concat",
    "Split": "QPU copy2d, channel-strided",
    "Slice": "QPU copy2d last-axis runs",
    "Resize": "QPU nearest (2x exact) / Mul+Add (bilinear)",
    "Transpose": "QPU copy2d ([0,2,1,3] only)",
    "Softmax": "QPU softmax_fwd (fused <=16 classes) / QPU reduce+exp+div (any class count)",
    "MatMul": "QPU gemm",
    "Gemm": "QPU gemm + Add",
    "GlobalAveragePool": "QPU gemm over 1/HW column",
    "Reshape": "view (no copy)",
    "Flatten": "view (no copy)",
    "Squeeze": "view (no copy)",
    "Unsqueeze": "view (no copy)",
    "Cast": "type change",
    "Identity": "view (no copy)",
    "Constant": "initialiser",
    "Shape": "folded at plan time (no runtime work)",
    "Gather": "folded at plan time when it indexes shapes",
    "Exp": "QPU unary (UOP_EXP, |x|<=88, max rel 5.7e-6)",
    "Log": "QPU unary (UOP_LOG, 2^-126<=x<inf, max rel 4.8e-7; refused outside)",
    "Sqrt": "QPU unary (UOP_SQRT, 0<=x<inf, max rel 1.1e-7; refused outside)",
    "Reciprocal": "QPU unary (UOP_RECIPROCAL, sign-split, max rel 5.7e-8)",
    "Erf": "QPU unary (UOP_ERF, A&S 7.1.26, max abs 4.1e-7)",
    "Sin": "QPU unary (UOP_SIN, Cody-Waite, |x|<=1e5, max abs 4.8e-6; refused outside)",
    "Cos": "QPU unary (UOP_COS, Cody-Waite, |x|<=1e5, max abs 4.8e-6; refused outside)",
    "Round": "QPU unary (UOP_ROUND, half-to-even, exact)",
    "Softplus": "QPU unary (UOP_SOFTPLUS, |x|<inf, max abs 1.3e-6)",
    "Elu": "QPU unary (UOP_ELU, alpha uniform, max abs 5.9e-7)",
    "Selu": "QPU unary (UOP_SELU, alpha/gamma uniforms, max abs 8.7e-7)",
    "Gelu": "QPU unary (UOP_GELU tanh / composed erf, max abs 5.0e-7)",
    "Mish": "QPU Softplus+Tanh+MUL composition (max abs 4.1e-7)",
    # inverse / hyperbolic (docs/OPS-UNARY.md): one kernel each, nv4 bodies
    "Asin": "QPU unary (UOP_ASIN, 2-branch poly, -1<=x<=1, max abs 2.9e-7; refused outside)",
    "Acos": "QPU unary (UOP_ACOS, pi/2 - asin, -1<=x<=1, max abs 2.9e-7; refused outside)",
    "Atan": "QPU unary (UOP_ATAN, deg-13 odd poly + 1/x fold, finite x, max abs 4.6e-7)",
    "Atanh": "QPU unary (UOP_ATANH, log core of (1+x)/(1-x), -1<x<1, max abs 1.7e-6; refused outside)",
    "Asinh": "QPU unary (UOP_ASINH, log core of x+sqrt(x^2+1), finite x, max abs 1.6e-6 (rel 1.7e-7))",
    "Acosh": "QPU unary (UOP_ACOSH, log core of x+sqrt((x-1)(x+1)), x>=1, max abs 1.6e-6 (rel 1.7e-7); refused below 1)",
    "Sinh": "QPU unary (UOP_SINH, (e^|x| - e^-|x|)/2, |x|<=88, max rel 9.3e-6; refused outside)",
    "Cosh": "QPU unary (UOP_COSH, (e^|x| + e^-|x|)/2, |x|<=88, max rel 9.6e-6; refused outside)",
    "Tan": "QPU unary (UOP_TAN, Cody-Waite + tan poly, |x|<=1e5 and |cos|>=0.05, max abs 4.9e-6 (rel 1.1e-6); refused outside)",
    # the reduction / comparison / shape-and-arith family (_dispatch_extra.py,
    # docs/OPS-REDUCE.md).  Each entry names the kernels the handler composes.
    "ReduceSum": "QPU reduce (sum) + copy2d axis-to-last",
    "ReduceMean": "QPU reduce (sum) + QPU affine 1/count",
    "ReduceMax": "QPU reduce (max) + copy2d axis-to-last",
    "ReduceMin": "QPU reduce (min) + copy2d axis-to-last",
    "ReduceProd": "QPU MUL chain over the reduced slices + copy2d",
    "ReduceL1": "QPU Abs + reduce (sum)",
    "ReduceL2": "QPU MUL + reduce (sum) + QPU Sqrt",
    "ReduceSumSquare": "QPU MUL + reduce (sum)",
    "ReduceLogSum": "QPU reduce (sum) + QPU Log",
    "ReduceLogSumExp": "QPU Exp + reduce (sum) + QPU Log (|x| <= 88)",
    "Equal": "QPU ew bit-compare (exact), bool output",
    "Greater": "QPU ew bit-compare (exact), bool output",
    "GreaterOrEqual": "QPU ew bit-compare (exact), bool output",
    "Less": "QPU ew bit-compare (exact), bool output",
    "LessOrEqual": "QPU ew bit-compare (exact), bool output",
    "Where": "QPU select3 (where)",
    "PRelu": "QPU select3 (prelu)",
    "IsNaN": "QPU ew bit-test (|bits| > 0x7F800000)",
    "IsInf": "QPU ew bit-test + QPU ReluBwd sign filter",
    "ArgMax": "QPU reduce (argmax), int64 index output",
    "ArgMin": "QPU reduce (argmin), int64 index output",
    "CumSum": "QPU cumsum + copy2d transpose/padding",
    "Expand": "QPU copy2d broadcast (lane copies)",
    "Tile": "QPU copy2d repeat (lane copies)",
    "Pad": "QPU fill2d + copy2d (constant/reflect/edge/wrap)",
    "Compress": "QPU copy2d row select",
    "OneHot": "host-built constant pattern + QPU affine",
    "Trilu": "host-built triangle mask + QPU Mul",
    "Range": "host-built constant sequence",
    "EyeLike": "host-built constant identity",
    "CastLike": "view (fp32) / dtype conversion",
    "SpaceToDepth": "QPU copy2d laned gather",
    "Shrink": "QPU ew bit-compare + Mul/Add",
    "ThresholdedRelu": "QPU affine + ReluBwd",
    "Celu": "QPU Relu/Exp/Affine composition",
    "Hardmax": "QPU reduce (argmax) + host-built mask + copy2d",
    "LogSoftmax": "QPU Exp + reduce + Log (stable shift when |x| > 80)",
    # the pooling / normalisation / softmax-variant / loss family
    # (_dispatch_norm.py, docs/OPS-NORM.md).  Each entry names the kernels the
    # handler composes.
    "AveragePool": "QPU copy2d window taps + QPU Add/Mul (integer divisor map)",
    "LpPool": "QPU copy2d window taps + QPU Abs/Mul/Sqrt",
    "GlobalMaxPool": "QPU max chain over the plane (copy2d slice columns)",
    "GlobalLpPool": "QPU Abs/Mul + QPU GEMM sum + QPU Sqrt (p=1, 2)",
    "MaxUnpool": "int64 index map + QPU copy2d single-word scatter",
    "BatchNormalization": "QPU batchnorm_infer (eps inside the sqrt), inference",
    "InstanceNormalization": "QPU layernorm_fwd per channel plane + QPU channel affine",
    "GroupNormalization": "QPU layernorm_fwd per group + QPU channel affine",
    "LayerNormalization": "QPU layernorm_fwd over the normalised axes",
    "MeanVarianceNormalization": "QPU GEMM mean/E[x^2] + QPU Sqrt + Affine",
    "LRN": "QPU GEMM channel band sum + QPU Log/Exp",
    "Dropout": "identity (ONNX inference; training mode refused)",
    "SoftmaxCrossEntropyLoss": "QPU LogSoftmax + one-hot GEMM sum + reduction",
    "NegativeLogLikelihoodLoss": "QPU one-hot GEMM sum + reduction",
    "Pow": "QPU Log/Exp/Reciprocal chain (small integer exponents are exact "
           "multiply chains)",
    "Mod": "QPU Reciprocal/Floor/Sign + Mul/Sub (fmod and floor semantics)",
    "DepthToSpace": "QPU element layout map (DCR/CRD), laned copies",
    # the variadic / conversion / gather family (_dispatch_extra.py,
    # docs/OPS-GAPS.md).  Each entry names the kernels the handler composes.
    "Sum": "QPU elementwise Add chain (broadcast operands)",
    "Mean": "QPU elementwise Add chain + QPU Affine 1/n",
    "Max": "QPU exact bit select over the operands (NaN wins, +-inf copied)",
    "Min": "QPU exact bit select over the operands (NaN wins, +-inf copied)",
    "Div": "QPU Reciprocal + correctly-rounded MUL (constant divisor folds)",
    "Cast": "view (fp32) / QPU Floor+Ceil truncation (int) + dtype boundary",
    "LpNormalization": "QPU Abs/MUL + GEMM sum + QPU Sqrt/Reciprocal + MUL",
    "Gather": "QPU copy2d row gather (activation data and host shape tensors)",
    "GatherElements": "QPU copy2d laned word gather, axis moved innermost",
    "GatherND": "QPU copy2d row gather over an integer index map",
    "ScatterElements": "QPU copy2d word store + QPU Add/MUL for reductions",
    "ScatterND": "QPU copy2d row store + QPU Add/MUL for reductions",
    "NonZero": "QPU exact bit-compare mask + int64 coordinates on the ARM",
    "TopK": "QPU exact-select scan per k + int32 index read-back",
    "Unique": "QPU pairwise equality/order matrices + QPU row gather",
    "ReverseSequence": "QPU copy2d laned gather over the time axis",
    "CenterCropPad": "QPU fill2d + copy2d laned gather (crop/pad map)",
    # the recurrent family (_dispatch_rnn.py, docs/OPS-RNN.md): the static
    # sequence is unrolled on the ARM, each timestep is GEMM + elementwise
    "RNN": "QPU GEMM (X@W^T, H@R^T) + QPU activation per timestep, unrolled",
    "GRU": "QPU GEMM (X@W^T, H@R^T) + QPU gate arithmetic per timestep",
    "LSTM": "QPU GEMM (X@W^T, H@R^T) + QPU gate arithmetic per timestep",
    # the signal / quantise / random family (_dispatch_sig.py, docs/OPS-RNN.md)
    "QuantizeLinear": "QPU affine + Round + Clip; int8/uint8 storage only",
    "DynamicQuantizeLinear": "QPU max/min tree + affine/round/clip",
    "DFT": "QPU GEMM against a constant cos/sin twiddle matrix",
    "STFT": "QPU copy2d framing + window MUL + the DFT GEMM",
    "RandomUniform": "QPU VC4_UOP_RNG (counter-based hash)",
    "RandomUniformLike": "QPU VC4_UOP_RNG (counter-based hash)",
    "RandomNormal": "QPU VC4_UOP_RNG x2 + Box-Muller (Log/Sqrt/Cos/MUL)",
    "RandomNormalLike": "QPU VC4_UOP_RNG x2 + Box-Muller (Log/Sqrt/Cos/MUL)",
    "Bernoulli": "QPU VC4_UOP_RNG + exact fp32 CMP_LT",
    "Multinomial": "QPU Exp/GEMM cumsum + one VC4_UOP_RNG draw per sample",
}

# Registered after the 2026-11-20 clean-window verification.  Set
# VC4_UNARY_VERIFIED=0 to stage the 13 ops out again (matching the C gate).
if os.environ.get("VC4_UNARY_VERIFIED", "1") == "0":
    for _op in ("Exp", "Log", "Sqrt", "Reciprocal", "Erf", "Sin", "Cos",
                "Round", "Softplus", "Elu", "Selu", "Gelu", "Mish"):
        OPS.pop(_op, None)

# tiers ---------------------------------------------------------------------
META_OPS = frozenset(("Shape", "Size", "Constant", "ConstantOfShape"))
CONTROL_OPS = frozenset(("If", "Loop", "Scan"))

_FLOAT_TYPES = frozenset(
    t for t in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16,
                getattr(TensorProto, "BFLOAT16", -1),
                getattr(TensorProto, "FLOAT8E4M3FN", -2),
                getattr(TensorProto, "FLOAT8E5M2", -3))
    if t >= 0)
_UNSUPPORTED_TYPES = frozenset(
    t for t in (TensorProto.STRING, TensorProto.FLOAT16,
                getattr(TensorProto, "BFLOAT16", -1),
                getattr(TensorProto, "FLOAT8E4M3FN", -2),
                getattr(TensorProto, "FLOAT8E5M2", -3),
                TensorProto.COMPLEX64, TensorProto.COMPLEX128)
    if t >= 0)

_INT_ATTR = ("group", "axis", "ceil_mode", "transA", "transB")


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        if a.type == onnx.AttributeProto.INT:
            return int(a.i)
        if a.type == onnx.AttributeProto.FLOAT:
            return float(a.f)
        if a.type == onnx.AttributeProto.STRING:
            return a.s.decode()
        if a.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(a.t)
        if a.type == onnx.AttributeProto.FLOATS:
            return list(a.floats)
        return default
    return default


def _shape_map(model):
    """(name -> list of dims, name -> TensorProto elem_type) via inference."""
    try:
        m = onnx.shape_inference.infer_shapes(model, strict_mode=False,
                                              data_prop=False)
    except Exception:
        m = model
    out = {}
    types = {}
    vals = list(m.graph.input) + list(m.graph.value_info) + \
        list(m.graph.output)
    for vi in vals:
        tt = vi.type.tensor_type
        if tt.elem_type:
            types[vi.name] = int(tt.elem_type)
        if not tt.HasField("shape"):
            continue
        dims = []
        for d in tt.shape.dim:
            dims.append(int(d.dim_value) if d.HasField("dim_value") else -1)
        out[vi.name] = dims
        if tt.elem_type:
            types.setdefault(vi.name, int(tt.elem_type))
    return out, types


def _init_shapes(model):
    """(name -> shape, name -> elem type) for initialisers and Constant."""
    shapes = {}
    types = {}
    for init in model.graph.initializer:
        try:
            shapes[init.name] = list(numpy_helper.to_array(init).shape)
        except Exception:
            pass
        types[init.name] = int(init.data_type)
    for n in model.graph.node:                 # Constant nodes
        if n.op_type != "Constant" or not n.output:
            continue
        for a in n.attribute:
            if a.name == "value" and a.t is not None:
                try:
                    arr = numpy_helper.to_array(a.t)
                    shapes[n.output[0]] = list(arr.shape)
                    types[n.output[0]] = int(a.t.data_type)
                except Exception:
                    pass
    return shapes, types


def _rank(shapes, name):
    s = shapes.get(name)
    return len(s) if s else None


def _is_float(elem):
    return elem in _FLOAT_TYPES


def _check_conv(node, shapes, inits):
    ks = _attr(node, "kernel_shape")
    w = inits.get(node.input[1]) if len(node.input) > 1 else None
    if w is None:
        return "unknown", "weight is not an initialiser"
    kh, kw = (int(ks[0]), int(ks[1])) if ks else (int(w[2]), int(w[3]))
    strides = _attr(node, "strides", [1, 1])
    pads = _attr(node, "pads", [0, 0, 0, 0])
    dil = _attr(node, "dilations", [1, 1])
    group = int(_attr(node, "group", 1))
    Cout = int(w[0])
    xs = shapes.get(node.input[0])
    Cin = int(xs[1]) if xs and len(xs) == 4 and xs[1] > 0 else int(w[1]) * group
    N = int(xs[0]) if xs and len(xs) == 4 and xs[0] > 0 else 1

    if int(dil[0]) != int(dil[1]):
        return "unsupported", "non-square dilation %s" % (list(dil),)
    dilv = int(dil[0])
    if not 1 <= dilv <= 8:
        return "unsupported", "dilation %d (1..8)" % dilv
    if int(strides[0]) != int(strides[1]):
        return "unsupported", "non-square stride %s" % (list(strides),)
    sh = int(strides[0])
    if sh not in (1, 2):
        return "unsupported", "stride %d (only 1, 2)" % sh
    if xs and len(xs) != 4:
        return "unsupported", "input rank %d" % len(xs)

    symm = pads[0] == pads[1] == pads[2] == pads[3]
    depthwise = group == Cin
    grouped = group > 1 and not depthwise
    native_group = (grouped and Cin % group == 0 and Cout % group == 0
                    and (Cin // group) % 16 == 0
                    and (Cout // group) % 16 == 0)
    nhwc = (kh in (1, 3, 5) and kw in (1, 3, 5) and symm
            and (group == 1 or native_group or depthwise))
    if nhwc:
        if depthwise and Cout != Cin:
            return "unsupported", ("depthwise Cout %d != Cin %d (channel "
                                  "multiplier)" % (Cout, Cin))
        if depthwise and kh != kw:
            return "unsupported", "non-square depthwise %dx%d" % (kh, kw)
        if sh * Cin * 4 > 8191:
            return "unsupported", ("stride*Cin*4 = %d B > 8191 B im2col pitch"
                                   % (sh * Cin * 4))
        kind = ("depthwise" if depthwise else
                "grouped(%d)" % group if grouped else "dense")
        return "supported", ("%dx%d s%d d%d g%d %s NHWC, batch %d"
                             % (kh, kw, sh, dilv, group, kind, N))

    # legacy NCHW fallback: batch 1, no dilation, group 1 or depthwise
    if N != 1:
        return "unsupported", ("batch %d needs the NHWC conv path" % N)
    if dilv != 1:
        return "unsupported", "dilation %d needs the NHWC conv path" % dilv
    if group not in (1, Cin):
        return "unsupported", "group %d needs 16-aligned per-group blocks" % group
    if kh != kw or kh not in (1, 3):
        return "unsupported", "kernel %dx%d (NCHW fallback: 1x1, 3x3)" % (kh, kw)
    if depthwise and sh != 1:
        return "unsupported", "stride-2 depthwise"
    p = int(pads[0])
    if sh == 2 and not ((kh == 3 and list(pads) == [1, 1, 1, 1])
                        or (kh == 1 and list(pads) == [0, 0, 0, 0])):
        return "unsupported", "stride 2 with pads %s" % (list(pads),)
    if xs and sh == 1:
        H, W = int(xs[2]), int(xs[3])
        if H > 0 and W > 0 and symm and (H + 2 * p) * (
                (int(W) + 2 * p + 15) & ~15) * 4 > 8191:
            return "unsupported", ("NCHW im2col pitch %dx%d > 8191 B"
                                   % (H + 2 * p, W + 2 * p))
    return "supported", "%dx%d s%d g%d NCHW fallback" % (kh, kw, sh, group)


def _check_maxpool(node, shapes, inits):
    ks = _attr(node, "kernel_shape")
    strides = _attr(node, "strides", [1, 1])
    if not ks:
        return "unknown", "no kernel_shape"
    if ks[0] != ks[1] or strides[0] != strides[1]:
        return "unsupported", "non-square window %s / stride %s" % (ks, strides)
    if _attr(node, "ceil_mode", 0):
        return "unsupported", "ceil_mode"
    if list(_attr(node, "dilations", [1, 1])) != [1, 1]:
        return "unsupported", "dilations"
    return "supported", "%dx%d s%d" % (ks[0], ks[0], strides[0])


def _check_softmax(node, shapes, inits):
    rank = _rank(shapes, node.input[0])
    if not rank:
        return "unknown", "input rank unknown"
    axis = int(_attr(node, "axis", -1))
    if axis < 0:
        axis += rank
    xs = shapes.get(node.input[0])
    if not xs or len(xs) <= axis or xs[axis] <= 0:
        return "unknown", "axis %d extent unknown" % axis
    c = xs[axis]
    # the fused kernel covers <= 16 classes; any other class count and any
    # other axis run through the reduce/exp/div composition
    return "supported", "%d classes along axis %d" % (c, axis)


def _check_resize(node, shapes, inits):
    mode = _attr(node, "mode", "nearest")
    coord = _attr(node, "coordinate_transformation_mode", "half_pixel")
    nmode = _attr(node, "nearest_mode", "round_prefer_floor")
    if mode not in ("nearest", "linear"):
        return "unsupported", "mode=%s (nearest, linear)" % mode
    if coord not in ("half_pixel", "half_pixel_symmetric",
                     "pytorch_half_pixel", "asymmetric", "align_corners",
                     "tf_crop_and_resize"):
        return "unsupported", "coordinate_transformation_mode=%s" % coord
    if mode == "nearest" and nmode not in ("round_prefer_floor",
                                           "round_prefer_ceil", "floor",
                                           "ceil"):
        return "unsupported", "nearest_mode=%s" % nmode
    if int(_attr(node, "antialias", 0)):
        return "unsupported", "antialias=1"
    if _attr(node, "axes", None):
        return "unsupported", "axes (spatial H/W only)"
    if mode == "nearest":
        return "supported", "nearest/%s/%s" % (coord, nmode)
    return "supported", "linear/%s" % coord


def _check_gemm(node, shapes, inits):
    if int(_attr(node, "transA", 0)):
        return "unsupported", "transA=1"
    if float(_attr(node, "alpha", 1.0)) != 1.0:
        return "unsupported", "alpha != 1"
    beta = float(_attr(node, "beta", 1.0))
    if beta not in (0.0, 1.0):
        return "unsupported", "beta=%g" % beta
    return "supported", "gemm"


def _check_matmul(node, shapes, inits):
    for name in node.input[:2]:
        rank = _rank(shapes, name)
        if rank is not None and rank != 2:
            return "unsupported", "rank %d operand (2-D only)" % rank
    return "supported", "gemm"


def _check_concat(node, shapes, inits):
    axis = int(_attr(node, "axis", 0))
    rank = _rank(shapes, node.input[0])
    if rank == 4 and axis != 1:
        return "unsupported", "axis %d on a 4-D tensor (channel axis 1)" % axis
    if rank == 3 and axis not in (1, -1, 2):
        return "unsupported", "axis %d on a 3-D tensor" % axis
    return "supported", "axis %d" % axis


def _check_split(node, shapes, inits):
    axis = int(_attr(node, "axis", 0))
    rank = _rank(shapes, node.input[0])
    if rank is not None and rank != 4:
        return "unknown", "rank %d (Split is channel-axis 4-D)" % rank
    if axis != 1:
        return "unsupported", "axis %d (channel axis 1)" % axis
    return "supported", "axis 1"


def _check_clip(node, shapes, inits):
    for i, nm in ((1, "min"), (2, "max")):
        if len(node.input) > i and node.input[i]:
            if node.input[i] not in inits:
                return "unsupported", ("runtime tensor limit %s (scalar "
                                       "constant limits only)"
                                       % node.input[i])
            v = inits[node.input[i]]
            if getattr(v, "size", 1) != 1:
                return "unsupported", "limit %s has %d elements" % (nm, v.size)
    lo = _attr(node, "min", None)
    hi = _attr(node, "max", None)
    if lo is not None and hi is not None and float(lo) > float(hi):
        return "unsupported", "min > max"
    return "supported", "scalar limits"


def _check_reduce(node, shapes, inits):
    axes = _attr(node, "axes", None)
    if len(node.input) > 1 and node.input[1]:
        axes = inits.get(node.input[1])
        if axes is None:
            return "unknown", "axes is a runtime tensor"
    if axes is not None and len(list(np.asarray(axes).ravel())) == 0:
        return "supported", "no axes (noop_with_empty_axes)"
    return "supported", "%s axes" % ("all" if axes is None else
                                     list(np.asarray(axes).ravel()))


def _check_pad(node, shapes, inits):
    mode = _attr(node, "mode", "constant")
    if mode not in ("constant", "reflect", "edge", "wrap"):
        return "unsupported", "mode=%s (constant, reflect, edge, wrap)" % mode
    if len(node.input) > 3 and node.input[3] and node.input[3] not in inits:
        return "unsupported", "axes is a runtime tensor"
    return "supported", mode


def _check_compress(node, shapes, inits):
    if _attr(node, "axis", None) is None:
        return "unsupported", ("axis=None selects single elements from the "
                               "flattened input (a per-element scatter)")
    return "supported", "axis %d" % int(_attr(node, "axis", 0))


def _check_cumsum(node, shapes, inits):
    if len(node.input) > 1 and node.input[1] and node.input[1] not in inits:
        return "unsupported", "axis is a runtime tensor"
    return "supported", "prefix sum, axis in a constant"


def _check_variadic(node, shapes, inits):
    if not [i for i in node.input if i]:
        return "unsupported", "no operands"
    return "supported", "%d operands, numpy broadcast" % len(
        [i for i in node.input if i])


def _check_div(node, shapes, inits):
    b = node.input[1] if len(node.input) > 1 else None
    if b and b in inits and getattr(inits[b], "size", 0) == 1:
        return "supported", "QPU Mul by the folded reciprocal"
    return "supported", "QPU Reciprocal + MUL (finite, non-zero divisor)"


def _check_cast(node, shapes, inits):
    to = int(_attr(node, "to", TensorProto.FLOAT))
    if to in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT32,
              TensorProto.INT64, TensorProto.BOOL):
        return "supported", "to %s" % TensorProto.DataType.Name(to)
    return "unsupported", ("Cast to %s: the arena is fp32 activations and the "
                           "integer outputs are host index tensors (FLOAT, "
                           "DOUBLE, INT32, INT64, BOOL are wired)"
                           % TensorProto.DataType.Name(to))


def _check_lpnorm(node, shapes, inits):
    p = int(_attr(node, "p", 2))
    if p not in (1, 2):
        return "unsupported", "p=%d (only p=1 and p=2)" % p
    return "supported", "p=%d" % p


def _check_gather(node, shapes, inits):
    idx = node.input[1] if len(node.input) > 1 else None
    if idx and idx not in inits and idx not in shapes:
        return "unknown", "indices is a runtime tensor"
    return "supported", "QPU copy2d rows"


def _check_scatter(node, shapes, inits):
    red = _attr(node, "reduction", "none") or "none"
    if red not in ("none", "add", "mul"):
        return "unsupported", ("reduction=%s (none, add, mul; max/min need a "
                               "per-word comparison pass)" % red)
    return "supported", "reduction %s" % red


def _check_nonzero(node, shapes, inits):
    return "supported", "QPU bit-compare mask + int64 coordinates"


def _check_unique(node, shapes, inits):
    return "supported", "pairwise equality/order matrices (bounded size)"


def _check_topk(node, shapes, inits):
    if len(node.input) > 1 and node.input[1] not in inits:
        return "unsupported", "K is a runtime tensor"
    return "supported", "k selection passes"


def _check_reverseseq(node, shapes, inits):
    if len(node.input) > 1 and node.input[1] not in inits:
        return "unsupported", "sequence_lens is a runtime tensor"
    return "supported", "laned gather over the time axis"


def _check_centercrop(node, shapes, inits):
    if len(node.input) > 1 and node.input[1] not in inits:
        return "unsupported", "shape is a runtime tensor"
    return "supported", "crop/pad index map"


def _check_eargmax(node, shapes, inits):
    if int(_attr(node, "select_last_index", 0)):
        return "supported", "select_last_index=1 (reversed copy)"
    return "supported", "arg reduce"


_CHECKS = {
    "ReduceSum": _check_reduce,
    "ReduceMean": _check_reduce,
    "ReduceMax": _check_reduce,
    "ReduceMin": _check_reduce,
    "ReduceProd": _check_reduce,
    "ReduceL1": _check_reduce,
    "ReduceL2": _check_reduce,
    "ReduceLogSum": _check_reduce,
    "ReduceLogSumExp": _check_reduce,
    "ReduceSumSquare": _check_reduce,
    "Pad": _check_pad,
    "Compress": _check_compress,
    "CumSum": _check_cumsum,
    "ArgMax": _check_eargmax,
    "ArgMin": _check_eargmax,
    "Conv": _check_conv,
    "Clip": _check_clip,
    "MaxPool": _check_maxpool,
    "Softmax": _check_softmax,
    "Resize": _check_resize,
    "Gemm": _check_gemm,
    "MatMul": _check_matmul,
    "Concat": _check_concat,
    "Split": _check_split,
    "Sum": _check_variadic,
    "Mean": _check_variadic,
    "Max": _check_variadic,
    "Min": _check_variadic,
    "Div": _check_div,
    "Cast": _check_cast,
    "LpNormalization": _check_lpnorm,
    "Gather": _check_gather,
    "GatherElements": _check_gather,
    "GatherND": _check_gather,
    "ScatterElements": _check_scatter,
    "ScatterND": _check_scatter,
    "NonZero": _check_nonzero,
    "TopK": _check_topk,
    "Unique": _check_unique,
    "ReverseSequence": _check_reverseseq,
    "CenterCropPad": _check_centercrop,
}


def _macs(node, shapes, inits):
    """Static multiply-accumulate count for the arithmetic ops.

    Only true MACs are counted (Conv, MatMul, Gemm, GlobalAveragePool); data
    movement and elementwise ops are 0.  This is the denominator for
    ``arithmetic_on_vc4``.
    """
    op = node.op_type
    try:
        if op == "Conv":
            w = inits.get(node.input[1]) if len(node.input) > 1 else None
            if w is None:
                return 0
            kh, kw = int(w[2]), int(w[3])
            cout, cin = int(w[0]), int(w[1]) * int(_attr(node, "group", 1)
                                                   or 1)
            xs = shapes.get(node.input[0])
            if xs and len(xs) == 4 and all(d > 0 for d in xs):
                N, _, H, W = (int(d) for d in xs)
                sh = int((_attr(node, "strides", [1, 1]) or [1, 1])[0])
                pads = _attr(node, "pads", [0, 0, 0, 0])
                dil = int((_attr(node, "dilations", [1, 1]) or [1, 1])[0])
                ho = (H + int(pads[0]) + int(pads[2]) - dil * (kh - 1) - 1) \
                    // sh + 1
                wo = (W + int(pads[1]) + int(pads[3]) - dil * (kw - 1) - 1) \
                    // sh + 1
                return max(0, N * cout * ho * wo * cin * kh * kw)
            return 0
        if op in ("MatMul", "Gemm"):
            a = shapes.get(node.input[0])
            b = shapes.get(node.input[1])
            if a and b and len(a) == 2 and len(b) == 2:
                return max(0, int(a[0]) * int(b[1]) * int(a[1]))
            return 0
        if op == "GlobalAveragePool":
            xs = shapes.get(node.input[0])
            if xs and all(d > 0 for d in xs):
                n = 1
                for d in xs:
                    n *= int(d)
                return n
            return 0
    except Exception:
        return 0
    return 0


def _defined(model, node):
    try:
        return bool(onnx.defs.has(node.op_type, node.domain or ""))
    except Exception:
        return False


def _tier(node, status, reason, dtypes, model):
    """Classify one node into exactly one execution tier."""
    op = node.op_type
    data_in = [i for i in node.input if i]
    outs = [o for o in node.output if o]
    elems = [dtypes.get(n) for n in data_in + outs]
    known = [e for e in elems if e is not None]

    if op in META_OPS:
        return "host-meta", "metadata: %s (no activation arithmetic)" % op
    if op == "MelWeightMatrix":
        # every input is a scalar parameter (num_mel_bins, dft_length,
        # sample_rate, the two edges), so the output is a constant table: it
        # is generated once at plan time and folded, never run at run time
        return ("host-meta",
                "constant table generator: a mel filterbank from scalar "
                "parameters, folded at plan time")
    if op in CONTROL_OPS:
        return ("host-control",
                "control flow %s: subgraphs are host-scheduled (a persistent "
                "kernel + device-side dispatch would be needed on VC4)" % op)

    if any(e in _UNSUPPORTED_TYPES for e in known):
        return ("unsupported-graph",
                "tensor type %s cannot cross the fp32 engine boundary"
                % "/".join(sorted(set(TensorProto.DataType.Name(e)
                                      for e in known
                                      if e in _UNSUPPORTED_TYPES))))

    # Shape / index plumbing on integer or boolean tensors is metadata, even
    # when the op (Concat, Mul, Slice, ...) would otherwise be a VC4 kernel.
    if data_in and all(e is not None and not _is_float(e) for e in
                       [dtypes.get(n) for n in data_in]):
        return ("host-meta",
                "metadata/shape arithmetic on non-float tensors (%s)"
                % op)

    if op == "Cast":
        to = int(_attr(node, "to", TensorProto.FLOAT))
        src = dtypes.get(node.input[0]) if node.input else None
        if src is not None and _is_float(src):
            if to in (TensorProto.FLOAT, TensorProto.DOUBLE,
                      TensorProto.INT32, TensorProto.INT64, TensorProto.BOOL):
                return ("vc4",
                        "QPU conversion of an fp32 activation (Cast to %s)"
                        % TensorProto.DataType.Name(to))
            return ("host-unwritten",
                    "Cast fp32 -> %s: no conversion kernel (the arena is fp32 "
                    "activations and the integer outputs are host index "
                    "tensors)%s"
                    % (TensorProto.DataType.Name(to),
                       "" if _defined(model, node) else ""))
        return ("host-meta",
                "metadata/control type conversion (Cast to %s)"
                % TensorProto.DataType.Name(to))

    if op not in OPS and not _defined(model, node):
        return ("unsupported-graph",
                "no ONNX definition for op %r in domain %r"
                % (op, node.domain or "ai.onnx"))

    if status == "supported":
        return "vc4", reason
    if status == "unknown":
        return "vc4", "unverified: %s (runtime decides)" % reason
    return ("host-unwritten",
            "numeric op with no VC4 kernel: %s" % reason)


def analyse_model(model):
    """Analyse an in-memory ModelProto and return the coverage report."""
    g = model.graph
    shapes, dtypes = _shape_map(model)
    inits, init_types = _init_shapes(model)
    dtypes.update(init_types)

    # producer/consumer map for the SiLU rewrite (Mul(x, Sigmoid(x)))
    producer = {}
    for n in g.node:
        for o in n.output:
            if o:
                producer[o] = n
    producer_idx = {}
    for idx, n in enumerate(g.node):
        for o in n.output:
            if o:
                producer_idx[o] = idx
    consumers = {}
    for n in g.node:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)
    gouts = set(o.name for o in g.output)
    silu_pairs = 0
    silu_skip = []                 # Sigmoid node indices folded into a Mul
    silu_src = {}                  # Mul index -> x tensor name
    for idx, n in enumerate(g.node):
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        for sig, x in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
            p = producer.get(sig)
            if p is None or p.op_type != "Sigmoid" or list(p.input) != [x]:
                continue
            if len(consumers.get(sig, [])) == 1 and sig not in gouts:
                silu_pairs += 1
                sig_idx = producer_idx.get(sig)
                if sig_idx is not None:
                    silu_skip.append(sig_idx)
                silu_src[idx] = x
                break

    nodes = []
    for idx, n in enumerate(g.node):
        name = n.name or "%s_%d" % (n.op_type, idx)
        if n.op_type not in OPS:
            status, reason = "unsupported", "no implementation"
        else:
            fn = _CHECKS.get(n.op_type)
            status, reason = (fn(n, shapes, inits) if fn else
                              ("supported", OPS[n.op_type]))
        tier, tier_reason = _tier(n, status, reason, dtypes, model)
        nodes.append({"index": idx, "name": name, "op": n.op_type,
                      "status": status, "reason": reason,
                      "tier": tier, "tier_reason": tier_reason,
                      "macs": _macs(n, shapes, inits),
                      "outputs": [o for o in n.output if o]})

    counts = {}
    for e in nodes:
        c = counts.setdefault(e["op"], {"op": e["op"], "nodes": 0,
                                        "supported": 0, "unsupported": 0,
                                        "unknown": 0, "note": OPS.get(e["op"], "")})
        c["nodes"] += 1
        c[e["status"]] += 1

    inputs = []
    for vi in g.input:
        if vi.name in inits:
            continue
        tt = vi.type.tensor_type
        inputs.append({
            "name": vi.name,
            "dtype": onnx.TensorProto.DataType.Name(tt.elem_type),
            "shape": [d.dim_value if d.HasField("dim_value") else -1
                      for d in tt.shape.dim],
        })
    outputs = []
    for vi in g.output:
        tt = vi.type.tensor_type
        outputs.append({
            "name": vi.name,
            "dtype": onnx.TensorProto.DataType.Name(tt.elem_type),
            "shape": [d.dim_value if d.HasField("dim_value") else -1
                      for d in tt.shape.dim],
        })

    import os
    opset = None
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            opset = o.version
    return {
        "path": None,
        "bytes": None,
        "producer": (model.producer_name + " " +
                     model.producer_version).strip(),
        "opset": opset,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "counts": sorted(counts.values(), key=lambda c: (-c["nodes"], c["op"])),
        "total": len(nodes),
        "supported": sum(1 for e in nodes if e["status"] == "supported"),
        "unsupported": sum(1 for e in nodes if e["status"] == "unsupported"),
        "unknown": sum(1 for e in nodes if e["status"] == "unknown"),
        "silu_pairs": silu_pairs,
        "silu_skip": silu_skip,
        "silu_src": silu_src,
        "dispatches": len(nodes) - silu_pairs,
        "shapes": shapes,
        "macs_total": sum(int(n.get("macs") or 0) for n in nodes),
        "tiers": {t: sum(1 for e in nodes if e["tier"] == t)
                  for t in ("vc4", "host-meta", "host-control",
                            "host-unwritten", "unsupported-graph")},
    }


def analyse(path):
    """Read a model and return the coverage report as a dict."""
    import os
    model = onnx.load(path)
    rep = analyse_model(model)
    rep["path"] = path
    try:
        rep["bytes"] = os.path.getsize(path)
    except OSError:
        rep["bytes"] = None
    return rep


def backend_plan(rep, mode="hybrid", host_policy="device",
                 silu_skip=None, silu_src=None, folded=None):
    """Per-node backend report: ``vc4`` / ``folded`` / ``host-meta`` /
    ``host-control`` / ``host-unwritten`` / ``unsupported-graph``.

    ``folded`` is the set of node indices resolved at plan time under the
    ``device`` policy (constant/shape folding).  Folded nodes cost zero runtime
    work and are not executed nodes.  Skipped ``Sigmoid`` nodes are part of the
    fused VC4 SiLU, not host nodes.
    """
    skip = set(silu_skip if silu_skip is not None else rep.get("silu_skip", ()))
    src = silu_src if silu_src is not None else rep.get("silu_src", {})
    folded = set(folded or ())
    out = []
    for n in rep["nodes"]:
        tier = n.get("tier") or "host-unwritten"
        reason = n.get("tier_reason") or n["reason"]
        if n["index"] in skip:
            tier = "vc4"
            reason = "fused into the consumer Mul as one SiLU launch"
        elif n["index"] in folded:
            tier = "folded"
            reason = ("plan-time constant/shape resolution (device policy): "
                      "no runtime work")
        elif n["index"] in src:
            reason += " (fused SiLU)"
        out.append({"index": n["index"], "name": n["name"], "op": n["op"],
                    "backend": tier, "reason": reason,
                    "status": n["status"], "macs": int(n.get("macs") or 0),
                    "mode": mode, "host_policy": host_policy})
    return out


def host_debt(rep):
    """Only the host-unwritten nodes (numeric ops with no VC4 kernel)."""
    return [n for n in rep["nodes"] if n.get("tier") == "host-unwritten"]


def _mel_table(node, known):
    """MelWeightMatrix as a plan-time constant (ONNX's formula, verbatim).

    num_spectrogram_bins = dft_length // 2 + 1; each of the num_mel_bins
    triangles sits between consecutive points of a mel-spaced grid mapped back
    to DFT bins.  Every input is a scalar; there are no activations.
    """
    def arg(name, i):
        if len(node.input) > i and node.input[i]:
            if node.input[i] in known:
                return float(np.asarray(known[node.input[i]]).ravel()[0])
        return float(_attr(node, name))
    try:
        mel = int(arg("num_mel_bins", 0))
        dft_length = int(arg("dft_length", 1))
        sample_rate = int(arg("sample_rate", 2))
        low = float(arg("lower_edge_hertz", 3))
        high = float(arg("upper_edge_hertz", 4))
    except Exception:                                # noqa: BLE001
        return None
    if mel < 1 or dft_length < 2 or sample_rate <= 0:
        return None
    bins = dft_length // 2 + 1
    f = np.arange(mel + 2)
    lo_mel = 2595.0 * np.log10(1.0 + low / 700.0)
    hi_mel = 2595.0 * np.log10(1.0 + high / 700.0)
    step = (hi_mel - lo_mel) / f.shape[0]
    f = f * step + lo_mel
    f = 700.0 * (np.power(10.0, f / 2595.0) - 1.0)
    f = ((dft_length + 1) * f) // sample_rate
    f = f.astype(int)
    out = np.zeros((bins, mel), np.float64)
    for i in range(mel):
        a, b, c = int(f[i]), int(f[i + 1]), int(f[i + 2])
        lc = b - a
        if lc == 0:
            if 0 <= b < bins:
                out[b, i] = 1.0
        else:
            for j in range(a, b + 1):
                if 0 <= j < bins:
                    out[j, i] = float(j - a) / float(lc)
        ch = c - b
        if ch > 0:
            for j in range(b, c):
                if 0 <= j < bins:
                    out[j, i] = float(c - j) / float(ch)
    return out.astype(np.float32)


def fold_metadata(model, rep):
    """Plan-time constant/shape resolution for the device policy.

    Returns ``(folded_indices, values)``.  ``values`` maps tensor name to the
    numpy array resolved at plan time.  Only ``host-meta`` nodes are folded;
    they cost zero runtime work and never appear as executed nodes.
    """
    from ._hybrid import HostEvaluator
    shapes = rep.get("shapes", {})
    known = {}
    for init in model.graph.initializer:
        try:
            known[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass
    ev = HostEvaluator(model)
    folded = set()
    for idx, node in enumerate(model.graph.node):
        if rep["nodes"][idx]["tier"] != "host-meta":
            continue
        op = node.op_type
        outs = [o for o in node.output if o]
        if any(a.type in (onnx.AttributeProto.GRAPH,
                          onnx.AttributeProto.GRAPHS)
               for a in node.attribute):
            continue
        if op == "Constant":
            v = _attr(node, "value")
            if v is None or not outs:
                continue
            known[outs[0]] = np.asarray(v)
            folded.add(idx)
            continue
        if op == "Shape":
            src = shapes.get(node.input[0])
            if not src or any(int(d) < 0 for d in src) or not outs:
                continue
            known[outs[0]] = np.asarray(src, dtype=np.int64)
            folded.add(idx)
            continue
        if op == "Size":
            src = shapes.get(node.input[0])
            if not src or any(int(d) < 0 for d in src) or not outs:
                continue
            n = 1
            for d in src:
                n *= int(d)
            known[outs[0]] = np.asarray(n, dtype=np.int64)
            folded.add(idx)
            continue
        if op == "MelWeightMatrix":
            # onnx.reference 1.17 crashes on this graph (it passes the 0-d int
            # inputs to np.zeros), so the table is generated here from the
            # same formula -- it is a constant generator, not tensor maths.
            vals = _mel_table(node, known)
            if vals is not None:
                known[outs[0]] = vals
                folded.add(idx)
            continue
        feeds = {}
        ok = True
        for name in node.input:
            if not name:
                continue
            if name in known:
                feeds[name] = np.asarray(known[name])
            else:
                ok = False
                break
        if not ok:
            continue
        try:
            vals = ev.run(node, feeds, idx)
        except Exception:                            # noqa: BLE001
            continue
        if len(vals) != len(outs):
            continue
        for name, arr in zip(outs, vals):
            known[name] = np.asarray(arr)
        folded.add(idx)
    return folded, known

