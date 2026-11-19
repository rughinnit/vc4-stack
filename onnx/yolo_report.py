#!/usr/bin/env python3
"""Static + dynamic coverage report for a real YOLOv8n ONNX export.

The static pass walks every node with shapes from onnx.shape_inference and
classifies it against exactly what onnxrun.py implements (op type *and*
attributes *and* the engine's plane-pitch limit).  The dynamic pass then runs
the graph on the VC4 engine and reports the first node that fails, which is
the honest end-to-end answer.

    python yolo_report.py models/yolov8n.onnx
"""
import os
import sys
from collections import Counter, OrderedDict

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from onnxrun import Runner, UnsupportedNode, _attr, ENGINE  # noqa: E402
import vc4infer  # noqa: E402

RUNNER_OPS = {
    "Conv", "Relu", "Sigmoid", "Add", "Mul", "MaxPool", "Concat", "Resize",
    "MatMul", "Gemm", "Softmax", "GlobalAveragePool", "Reshape", "Flatten",
    "Identity", "Constant",
}


def align16(v):
    return (int(v) + 15) & ~15


def shapes_of(model):
    m = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    vi = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + \
            list(m.graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else -1
                for d in v.type.tensor_type.shape.dim]
        vi[v.name] = dims
    return vi


def classify(node, vi, inits):
    """Return (ok, reason) for one node against onnxrun.py's implementation."""
    op = node.op_type
    if op not in RUNNER_OPS:
        return False, "op type not implemented"
    if not node.input:
        return True, ""
    shape = vi.get(node.input[0])
    if op == "Conv":
        w = inits.get(node.input[1])
        if w is None:
            return False, "non-constant weight"
        wd = list(w.dims)
        group = int(_attr(node, "group", 1))
        ks = _attr(node, "kernel_shape", wd[2:])
        kh, kw = int(ks[0]), int(ks[1])
        strides = _attr(node, "strides", [1, 1])
        sh, sw = int(strides[0]), int(strides[1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        if list(_attr(node, "dilations", [1, 1])) != [1, 1]:
            return False, "dilations"
        if sh != sw:
            return False, "non-square stride"
        if group not in (1, shape[1]):
            return False, "groups=%d" % group
        if group == shape[1] and wd[0] != shape[1]:
            return False, "depthwise multiplier"
        if group == shape[1] and sh != 1:
            return False, "stride-%d depthwise" % sh
        if kh == 1 and kw == 1:
            if pads != [0, 0, 0, 0]:
                return False, "1x1 with padding"
            if sh not in (1, 2):
                return False, "1x1 stride %d" % sh
        elif kh == 3 and kw == 3:
            if sh == 2 and pads != [1, 1, 1, 1]:
                return False, "3x3 stride 2 pads %s" % pads
            if sh not in (1, 2):
                return False, "3x3 stride %d" % sh
            p = pads[0]
            if pads != [p] * 4:
                return False, "asymmetric pads"
            if sh == 1:
                hp, wp = shape[2] + 2 * p, align16(shape[3] + 2 * p)
            else:
                hp = (shape[2] + 1) // 2 + 2
                wp = align16(((shape[3] + 1) // 2) + 2)
            if hp * wp * 4 > 8191:
                return False, ("plane %dx%d = %d B > 8191 B pitch limit"
                               % (hp, wp, hp * wp * 4))
        else:
            return False, "kernel %dx%d" % (kh, kw)
        return True, ""
    if op in ("Relu", "Sigmoid"):
        return True, ""
    if op in ("Add", "Mul"):
        a = vi.get(node.input[0])
        b = vi.get(node.input[1])
        if a is None or b is None or a != b:
            return False, "broadcast %s vs %s" % (a, b)
        return True, ""
    if op == "MaxPool":
        if int(_attr(node, "ceil_mode", 0)):
            return False, "ceil_mode"
        if list(_attr(node, "dilations", [1, 1])) != [1, 1]:
            return False, "dilations"
        return True, ""
    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        if len(shape) != 4 or axis not in (1, -3):
            return False, "axis %d" % axis
        return True, ""
    if op == "Resize":
        if _attr(node, "mode", "nearest") != "nearest":
            return False, "mode=%s" % _attr(node, "mode")
        return True, ""
    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        if axis not in (-1, len(shape) - 1):
            return False, "axis %d" % axis
        if shape[-1] > 16:
            return False, "%d classes > 16" % shape[-1]
        return True, ""
    if op == "MatMul":
        if len(shape) != 2 or len(vi.get(node.input[1], [])) != 2:
            return False, "non-2D"
        return True, ""
    if op == "Gemm":
        a, b = vi.get(node.input[0]), vi.get(node.input[1])
        if not a or not b or len(a) != 2 or len(b) != 2:
            return False, "non-2D"
        if int(_attr(node, "transA", 0)) or float(_attr(node, "alpha", 1)) != 1:
            return False, "transA/alpha"
        if float(_attr(node, "beta", 1)) not in (0.0, 1.0):
            return False, "beta"
        return True, ""
    if op == "GlobalAveragePool":
        return (len(shape) == 4), "not NCHW"
    if op == "Reshape":
        return True, ""
    return True, ""


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, "models", "yolov8n.onnx")
    model = onnx.load(path)
    g = model.graph
    vi = shapes_of(model)
    inits = {i.name: i for i in g.initializer}

    counts = Counter(n.op_type for n in g.node)
    print("graph: %d nodes, %d initializers, opset %s" %
          (len(g.node), len(g.initializer),
           [(o.domain or "ai.onnx", o.version) for o in model.opset_import]))
    print("input:  %s" % ([(i.name, vi.get(i.name)) for i in g.input]))
    print("output: %s" % ([(o.name, vi.get(o.name)) for o in g.output]))
    print("\nop census:")
    for op, n in counts.most_common():
        print("  %-20s %4d  %s" % (op, n,
              "runner op" if op in RUNNER_OPS else "NOT IMPLEMENTED"))

    gaps = OrderedDict()
    ok = 0
    first_bad = None
    for n in g.node:
        good, why = classify(n, vi, inits)
        if good:
            ok += 1
            continue
        gaps.setdefault((n.op_type, why.split("=")[0].split(" ")[0]), []).append(n)
        if first_bad is None:
            first_bad = (n, why)
    print("\nstatic verdict: %d/%d nodes pass the runner's checks, %d do not" %
          (ok, len(g.node), len(g.node) - ok))
    if first_bad:
        n, why = first_bad
        print("first node that cannot run: %r (%s): %s" %
              (n.name, n.op_type, why))
    print("\ngaps:")
    for (op, kind), ns in gaps.items():
        names = ", ".join(repr(n.name) for n in ns[:3])
        print("  %-16s %-18s %3d node(s)  e.g. %s" %
              (op, kind, len(ns), names))

    # ---- dynamic: actually try to run it on the GPU ----------------------
    print("\ndynamic run on the VC4 engine:")
    eng = vc4infer.Engine(cache_dir=os.environ.get(
        "VC4INFER_CACHE", os.path.join(ENGINE, "kcache")))
    try:
        r = Runner(path, eng)
        name, dims = r.float_inputs[0]
        x = np.zeros([d if d > 0 else 1 for d in dims], np.float32)
        r.run({name: x}, profile=False)
        print("  ran to completion")
    except UnsupportedNode as e:
        print("  FAILED: %s" % e)
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
