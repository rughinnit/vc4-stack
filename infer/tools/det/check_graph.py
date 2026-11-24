#!/usr/bin/env python3
"""Check `student.py` against the ONNX that `build.py` writes.

    /mnt/Ext/env-tsl2/bin/python infer/tools/det/check_graph.py \
        --reference /mnt/Ext/bccd-student/onnx/vc4det-n3_96.onnx \
        --exported /mnt/Ext/bccd-student/runs/s1/vc4det-n3_96.onnx --nc 3

Two checks, both failure-loud:

1. **Architecture.**  The reference graph's 8 conv weights and biases are
   loaded, in topological order, into `student.fuse(Student(nc))`; both graphs
   then see the same random input.  If kernel sizes, strides, paddings,
   channel counts, layer order, activations or the head differ, the outputs
   differ.  This is what makes "the ONNX is the specification" a test rather
   than a claim.

2. **Export shape.**  Operator histogram, node count and output shape of an
   exported student are compared with the reference (Conv/Sigmoid/Mul counts,
   no BatchNormalization).

Exit status is non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import student as S  # noqa: E402


def conv_order(model):
    """Conv nodes in topological order, with their weight/bias initializers.

    torch 1.13 aliases tensors that hold equal values through Identity nodes
    (a freshly fused graph has zero conv biases, so several of them are the same
    tensor), hence the alias resolution below.
    """
    init = {i.name: i for i in model.graph.initializer}
    alias = {n.output[0]: n.input[0] for n in model.graph.node
             if n.op_type == "Identity"}

    def resolve(name):
        seen = 0
        while name in alias and seen < 16:
            name = alias[name]
            seen += 1
        return name

    out = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        w = init[resolve(node.input[1])]
        b = init[resolve(node.input[2])] if len(node.input) > 2 else None
        out.append((node.name, w, b))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--exported", default=None)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--tol", type=float, default=2e-5)
    args = ap.parse_args(argv)

    import onnx
    import onnxruntime as ort

    ref = onnx.load(args.reference)
    convs = conv_order(ref)
    m = S.fuse(S.Student(nc=args.nc))
    mine = [m.stem, m.s1a, m.s1b, m.s2a, m.s2b, m.p3a, m.p3b, m.head]
    assert len(convs) == len(mine), "reference has %d convs, module has %d" % (
        len(convs), len(mine))
    for (name, w, b), block in zip(convs, mine):
        mod = block.conv
        ww = np.asarray(onnx.numpy_helper.to_array(w), np.float32)
        bb = (np.zeros(ww.shape[0], np.float32) if b is None else
              np.asarray(onnx.numpy_helper.to_array(b), np.float32))
        assert tuple(ww.shape) == tuple(mod.weight.shape), (
            "%s: onnx weight %r vs module %r" % (name, ww.shape,
                                                 tuple(mod.weight.shape)))
        mod.weight.data.copy_(torch_from(ww))
        mod.bias.data.copy_(torch_from(bb))

    import torch
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        a = m(x).numpy()
    sess = ort.InferenceSession(args.reference,
                                providers=["CPUExecutionProvider"])
    b = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(a - b).max())
    ok = diff <= args.tol
    doc = {"reference": os.path.abspath(args.reference), "convs": len(convs),
           "out_shape": list(a.shape), "ref_out_shape": list(b.shape),
           "max_abs_diff": diff, "tol": args.tol,
           "architecture": "match" if ok else "MISMATCH"}

    if args.exported:
        hist = {}
        for n in onnx.load(args.exported).graph.node:
            hist[n.op_type] = hist.get(n.op_type, 0) + 1
        rhist = {}
        for n in ref.graph.node:
            rhist[n.op_type] = rhist.get(n.op_type, 0) + 1
        doc["exported"] = os.path.abspath(args.exported)
        doc["ops_exported"] = dict(sorted(hist.items()))
        doc["ops_reference"] = dict(sorted(rhist.items()))
        doc["nodes_exported"] = sum(hist.values())
        doc["nodes_reference"] = sum(rhist.values())
        for k in ("Conv", "Sigmoid", "Mul"):
            if hist.get(k, 0) != rhist.get(k, 0):
                ok = False
                doc["architecture"] = "MISMATCH (op count %s)" % k
        if "BatchNormalization" in hist:
            ok = False
            doc["architecture"] = "MISMATCH (BatchNormalization survives)"
    print(json.dumps(doc, indent=1))
    return 0 if ok else 1


def torch_from(a):
    import torch
    return torch.from_numpy(a)


if __name__ == "__main__":
    sys.exit(main())
