#!/usr/bin/env python3
"""Build the fixed-shape ONNX graphs the new conv paths are verified on.

    batched.onnx    input [2,16,20,20]   Conv 16->16 k3 s1 p1, Relu, 1x1
    grouped.onnx    input [1,64,16,16]   Conv 64->64 g=4 k3 s1 p1, Relu
    dilated.onnx    input [1,16,24,24]   Conv 16->16 k3 d=2 p=2
    depthwise.onnx  input [1,32,16,16]   Conv 32->32 g=32 k3 s1 p1
    stride2.onnx    input [1,32,32,32]   Conv 32->32 k3 s2 p1 (native)

Weights are deterministic, so the .onnx + numpy input is reproducible.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models")


def conv_node(name, x, w, b, cout, cin, k, s=1, p=1, d=1, group=1):
    return helper.make_node(
        "Conv", [x, w, b], [name],
        name=name, kernel_shape=[k, k], strides=[s, s], pads=[p, p, p, p],
        dilations=[d, d], group=group)


def build(name, shape, convs):
    """convs: list of (cin, cout, k, s, p, d, group, relu)"""
    rng = np.random.default_rng((sum(map(ord, name)) * 7919) % 2 ** 31)
    inits, nodes = [], []
    prev = "x"
    for i, (cin, cout, k, s, p, d, g, relu) in enumerate(convs):
        wn = "w%d" % i
        bn = "b%d" % i
        w = (rng.standard_normal((cout, cin // g, k, k)) * 0.15).astype(np.float32)
        b = (rng.standard_normal(cout) * 0.1).astype(np.float32)
        inits += [numpy_helper.from_array(w, wn),
                  numpy_helper.from_array(b, bn)]
        on = "c%d" % i
        nodes.append(conv_node(on, prev, wn, bn, cout, cin, k, s, p, d, g))
        prev = on
        if relu:
            rn = "r%d" % i
            nodes.append(helper.make_node("Relu", [prev], [rn], name=rn))
            prev = rn
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(prev, TensorProto.FLOAT, None)],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    m = onnx.shape_inference.infer_shapes(m)
    path = os.path.join(OUT, name + ".onnx")
    onnx.save(m, path)
    x = rng.standard_normal(shape).astype(np.float32)
    np.save(os.path.join(OUT, name + ".in.npy"), x)
    print("%-14s %s -> x %s" % (name, path, shape))


def main():
    os.makedirs(OUT, exist_ok=True)
    build("batched", [2, 16, 20, 20],
          [(16, 16, 3, 1, 1, 1, 1, True), (16, 16, 1, 1, 0, 1, 1, False)])
    build("grouped", [1, 64, 16, 16], [(64, 64, 3, 1, 1, 1, 4, True)])
    build("dilated", [1, 16, 24, 24], [(16, 16, 3, 1, 2, 2, 1, False)])
    build("depthwise", [1, 32, 16, 16], [(32, 32, 3, 1, 1, 1, 32, False)])
    build("stride2", [1, 32, 32, 32], [(32, 32, 3, 2, 1, 1, 1, False)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
