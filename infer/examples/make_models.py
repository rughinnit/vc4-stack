#!/usr/bin/env python3
"""Generate two small non-YOLO models for the Engine demos.

    python3 infer/examples/make_models.py [outdir]

Writes, deterministically:

    mlp.onnx         x[1,16] -> Gemm 16->8 -> Relu -> Gemm 8->4 -> Softmax
    mlp.in.npy / mlp.ref.npy
    two_input.onnx   a[1,3,8,8] + b[1,3,8,8] -> Add -> Relu ->
                     GlobalAveragePool -> Flatten -> Gemm 3->5
    two_input.a.npy / two_input.b.npy / two_input.ref.npy

The references are computed here with numpy (float64, cast to float32); they
are the *model definition*, not a runtime fallback.  Run them with
examples/non_yolo.py.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name=name)


def mlp(path, rng):
    w1 = (rng.standard_normal((8, 16)) * 0.3).astype(np.float32)
    b1 = (rng.standard_normal(8) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((4, 8)) * 0.4).astype(np.float32)
    b2 = (rng.standard_normal(4) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h"], name="fc1",
                         transB=1),
        helper.make_node("Relu", ["h"], ["h2"], name="relu"),
        helper.make_node("Gemm", ["h2", "w2", "b2"], ["logits"], name="fc2",
                         transB=1),
        helper.make_node("Softmax", ["logits"], ["probs"], name="softmax",
                         axis=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 4])],
        [_init("w1", w1), _init("b1", b1), _init("w2", w2), _init("b2", b2)])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)

    x = rng.standard_normal((1, 16)).astype(np.float32)
    h = np.maximum(x @ w1.T + b1, 0)
    logits = h @ w2.T + b2
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    ref = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    np.save(path.replace(".onnx", ".in.npy"), x)
    np.save(path.replace(".onnx", ".ref.npy"), ref)
    return x, ref


def two_input(path, rng):
    w = (rng.standard_normal((5, 3)) * 0.5).astype(np.float32)
    b = (rng.standard_normal(5) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node("Add", ["a", "b"], ["s"], name="sum"),
        helper.make_node("Relu", ["s"], ["r"], name="relu"),
        helper.make_node("GlobalAveragePool", ["r"], ["g"], name="gap"),
        helper.make_node("Flatten", ["g"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "w", "bias"], ["logits"], name="fc",
                         transB=1),
    ]
    graph = helper.make_graph(
        nodes, "two_input",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 8, 8]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 5])],
        [_init("w", w), _init("bias", b)])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)

    a = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    bb = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    g = np.maximum(a + bb, 0).mean(axis=(2, 3), keepdims=True)
    ref = (g.reshape(1, 3) @ w.T + b).astype(np.float32)
    np.save(path.replace(".onnx", ".a.npy"), a)
    np.save(path.replace(".onnx", ".b.npy"), bb)
    np.save(path.replace(".onnx", ".ref.npy"), ref)
    return a, bb, ref


def main(argv):
    out = argv[0] if argv else os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261120)
    mlp(os.path.join(out, "mlp.onnx"), rng)
    two_input(os.path.join(out, "two_input.onnx"), rng)
    print("wrote %s/{mlp,two_input}.onnx (+ .npy inputs/references)" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
