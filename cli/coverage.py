"""Static node coverage for the vc4infer executor.

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
"""
import onnx
from onnx import numpy_helper

# op -> short implementation note (mirrors onnxrun.Runner._dispatch)
OPS = {
    "Conv": "QPU conv: NHWC 1x1/3x3 s1/s2, depthwise s1; NCHW fallback",
    "Relu": "QPU elementwise",
    "Sigmoid": "QPU elementwise",
    "Silu": "QPU elementwise (fused)",
    "Add": "QPU elementwise",
    "Mul": "QPU elementwise",
    "Sub": "QPU Mul(-1)+Add",
    "Div": "QPU Mul by folded constant",
    "MaxPool": "QPU maxpooln / maxpool_nhwc",
    "Concat": "host byte concat (channel axis)",
    "Split": "host byte gather (channel axis)",
    "Slice": "host gather",
    "Resize": "host gather (nearest)",
    "Transpose": "host permute",
    "Softmax": "QPU softmax_fwd (<=16 classes)",
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
    "Shape": "host int64",
    "Gather": "host int64",
}

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
        return default
    return default


def _shape_map(model):
    """name -> list of dims (0/negative = unknown) from shape inference."""
    try:
        m = onnx.shape_inference.infer_shapes(model, strict_mode=False,
                                              data_prop=False)
    except Exception:
        m = model
    out = {}
    vals = list(m.graph.input) + list(m.graph.value_info) + \
        list(m.graph.output)
    for vi in vals:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = []
        for d in tt.shape.dim:
            dims.append(int(d.dim_value) if d.HasField("dim_value") else -1)
        out[vi.name] = dims
        out.setdefault(vi.name + "#dtype", tt.elem_type)
    return out


def _init_shapes(model):
    shapes = {}
    for init in model.graph.initializer:
        try:
            shapes[init.name] = list(numpy_helper.to_array(init).shape)
        except Exception:
            pass
    for n in model.graph.node:                 # Constant nodes
        if n.op_type != "Constant" or not n.output:
            continue
        for a in n.attribute:
            if a.name == "value" and a.t is not None:
                try:
                    shapes[n.output[0]] = list(
                        numpy_helper.to_array(a.t).shape)
                except Exception:
                    pass
    return shapes


def _rank(shapes, name):
    s = shapes.get(name)
    return len(s) if s else None


def _check_conv(node, shapes, inits):
    ks = _attr(node, "kernel_shape")
    w = inits.get(node.input[1]) if len(node.input) > 1 else None
    if w is None:
        return "unknown", "weight is not an initialiser"
    kh, kw = (ks[0], ks[1]) if ks else (w[2], w[3])
    strides = _attr(node, "strides", [1, 1])
    pads = _attr(node, "pads", [0, 0, 0, 0])
    dil = _attr(node, "dilations", [1, 1])
    group = int(_attr(node, "group", 1))
    Cout = w[0]
    xs = shapes.get(node.input[0])
    Cin = xs[1] if xs and len(xs) == 4 and xs[1] > 0 else w[1] * group

    if list(dil) != [1, 1]:
        return "unsupported", "dilations %s" % (list(dil),)
    if strides[0] != strides[1]:
        return "unsupported", "non-square stride %s" % (list(strides),)
    sh = int(strides[0])
    if sh not in (1, 2):
        return "unsupported", "stride %d (only 1, 2)" % sh
    if kh != kw or kh not in (1, 3):
        return "unsupported", "kernel %dx%d (only 1x1, 3x3)" % (kh, kw)
    if group not in (1, Cin):
        return "unsupported", "group %d (only 1 or depthwise)" % group
    if group == Cin:
        if Cout != Cin:
            return "unsupported", "depthwise Cin %d != Cout %d" % (Cin, Cout)
        if sh != 1:
            return "unsupported", "stride-2 depthwise"
    symm = pads[0] == pads[1] == pads[2] == pads[3]
    if sh == 2 and not symm:
        return "unsupported", "stride 2 with asymmetric pads %s" % (list(pads),)
    if sh * Cin * 4 > 8191:
        return "unsupported", ("stride*Cin*4 = %d B > 8191 B im2col pitch"
                               % (sh * Cin * 4))
    if xs and len(xs) == 4 and xs[0] > 1:
        return "unsupported", "batch %d (only 1)" % xs[0]
    if xs and len(xs) != 4:
        return "unsupported", "input rank %d" % len(xs)
    path = "NHWC" if (sh == 1 or group == 1) else "NCHW"
    return "supported", "%dx%d s%d g%d %s" % (kh, kw, sh, group, path)


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
    if c > 16:
        return "unsupported", "%d classes along axis %d (max 16)" % (c, axis)
    return "supported", "%d classes" % c


def _check_resize(node, shapes, inits):
    mode = _attr(node, "mode", "nearest")
    coord = _attr(node, "coordinate_transformation_mode", "half_pixel")
    if mode != "nearest":
        return "unsupported", "mode=%s (nearest only)" % mode
    if coord not in ("asymmetric", "half_pixel"):
        return "unsupported", "coordinate_transformation_mode=%s" % coord
    return "supported", "nearest/%s" % coord


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


_CHECKS = {
    "Conv": _check_conv,
    "MaxPool": _check_maxpool,
    "Softmax": _check_softmax,
    "Resize": _check_resize,
    "Gemm": _check_gemm,
    "MatMul": _check_matmul,
    "Concat": _check_concat,
    "Split": _check_split,
}


def analyse(path):
    """Read a model and return the coverage report as a dict."""
    model = onnx.load(path)
    g = model.graph
    shapes = _shape_map(model)
    inits = _init_shapes(model)

    # producer/consumer map for the SiLU rewrite (Mul(x, Sigmoid(x)))
    producer = {}
    for n in g.node:
        for o in n.output:
            if o:
                producer[o] = n
    consumers = {}
    for n in g.node:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)
    gouts = set(o.name for o in g.output)
    silu_pairs = 0
    for n in g.node:
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        for sig, x in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
            p = producer.get(sig)
            if p is None or p.op_type != "Sigmoid" or list(p.input) != [x]:
                continue
            if len(consumers.get(sig, [])) == 1 and sig not in gouts:
                silu_pairs += 1
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
        nodes.append({"index": idx, "name": name, "op": n.op_type,
                      "status": status, "reason": reason,
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
        "path": path,
        "bytes": os.path.getsize(path),
        "producer": (model.producer_name + " " + model.producer_version).strip(),
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
        "dispatches": len(nodes) - silu_pairs,
    }
