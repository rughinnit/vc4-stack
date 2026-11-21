#!/usr/bin/env python3
"""Build the small ONNX graphs used by the hybrid-mode tests.

    python make_hybrid_models.py [models_dir]

Writes, under ``$HYBRID_MODELS`` or ``/home/raspi/work/onnx/models``:

    hybrid_mixed.onnx   Split/Shape/Gather + Einsum + TopK + NonZero + LSTM.
                        All four compute operators are QPU kernels now; the
                        graph keeps them together with plan-time metadata so a
                        regression in either tier is visible.
    hybrid_meta.onnx    Shape/Gather/Slice/Concat shape arithmetic feeding a
                        Resize with computed sizes (host-meta, structural OK)
    hybrid_erf.onnx     Conv/Relu/Erf/GAP/Gemm/Softmax.  Erf used to be
                        numeric host debt and is now a QPU kernel.
    hybrid_if.onnx      Slice+Cast predicate into If with captured x
                        (host-control, structural OK; two inputs)

Each model gets deterministic `.in.npy` input(s) and `.ref.npy` outputs from a
full `onnx.reference.ReferenceEvaluator` run, so the tests compare against an
independent implementation of the whole graph.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

F = TensorProto.FLOAT


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def model(nodes, inputs, outputs, inits, name, opset=13):
    g = helper.make_graph(nodes, name, inputs, outputs,
                          initializer=list(inits))
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)],
                             producer_name="vc4infer-hybrid-tests")


def rng(shape, scale=1.0, seed=0):
    r = np.random.default_rng(seed)
    return (r.standard_normal(shape) * scale).astype(np.float32)


# --------------------------------------------------------------------- mixed
def build_mixed():
    x = vi("x", F, [1, 3, 16, 16])
    outs = [vi("p", F, [1, 8]), vi("r3", F, [1, 16, 16, 16]),
            vi("p2", F, [1, 4]), vi("nz", TensorProto.INT64, None),
            vi("shape_out", TensorProto.INT64, [4]),
            vi("gather_out", TensorProto.INT64, [1])]
    inits = [
        init("w1", rng((16, 3, 3, 3), 0.3)), init("b1", np.zeros(16)),
        init("w2", rng((16, 8, 1, 1), 0.5)), init("b2", np.zeros(16)),
        init("w3", rng((16, 3, 1, 1), 0.5)), init("b3", np.zeros(16)),
        init("w4", rng((16, 2, 1, 1), 0.5)), init("b4", np.zeros(16)),
        init("wg", rng((16, 8), 0.4)), init("bg", np.zeros(8)),
        init("wg2", rng((80, 4), 0.2)), init("bg2", np.zeros(4)),
        init("k", np.array([3], np.int64)),
        init("gi", np.array([2], np.int64)),
        init("unsq", np.array([2, 3], np.int64)),
        init("w_lstm", rng((1, 8, 3), 0.4)),
        init("r_lstm", rng((1, 8, 2), 0.4)),
        init("b_lstm", rng((1, 16), 0.1)),
        init("x_lstm", rng((5, 1, 3), 0.7)),
        init("lstm_shape", np.array([1, 2, 5, 1], np.int64)),
    ]
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], name="conv1",
                         pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node("Split", ["r1"], ["a", "b"], name="split",
                         axis=1, split=[8, 8]),
        helper.make_node("Shape", ["r1"], ["shape_out"], name="shape"),
        helper.make_node("Gather", ["shape_out", "gi"], ["gather_out"],
                         name="gather", axis=0),
        helper.make_node("Einsum", ["a"], ["e"], name="einsum",
                         equation="nchw->nc"),
        helper.make_node("Unsqueeze", ["e", "unsq"], ["u"], name="unsqueeze"),
        helper.make_node("Conv", ["u", "w2", "b2"], ["c2"], name="conv2"),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
        helper.make_node("Flatten", ["r2"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["g"], name="gemm"),
        helper.make_node("Softmax", ["g"], ["p"], name="softmax", axis=1),
        helper.make_node("TopK", ["b", "k"], ["tv", "ti"], name="topk",
                         axis=1),
        helper.make_node("Conv", ["tv", "w3", "b3"], ["c3"], name="conv3"),
        helper.make_node("Relu", ["c3"], ["r3"], name="relu3"),
        helper.make_node("NonZero", ["r1"], ["nz"], name="nonzero"),
        helper.make_node("LSTM", ["x_lstm", "w_lstm", "r_lstm", "b_lstm"],
                         ["ly0"], name="lstm"),
        helper.make_node("Reshape", ["ly0", "lstm_shape"], ["ly"],
                         name="lstm_reshape"),
        helper.make_node("Conv", ["ly", "w4", "b4"], ["c4"], name="conv4"),
        helper.make_node("Relu", ["c4"], ["r4"], name="relu4"),
        helper.make_node("Flatten", ["r4"], ["f4"], name="flatten2"),
        helper.make_node("Gemm", ["f4", "wg2", "bg2"], ["g2"], name="gemm2"),
        helper.make_node("Softmax", ["g2"], ["p2"], name="softmax2", axis=1),
    ]
    m = model(nodes, [x], outs, inits, "hybrid_mixed")
    return m, {"x": rng((1, 3, 16, 16), 0.8)}


# ---------------------------------------------------------------------- meta
def build_meta():
    x = vi("x", F, [1, 3, 8, 8])
    outs = [vi("p", F, [1, 4]), vi("sizes", TensorProto.INT64, [4]),
            vi("up", F, [1, 16, 16, 16])]
    inits = [
        init("w1", rng((16, 3, 3, 3), 0.3)), init("b1", np.zeros(16)),
        init("w2", rng((16, 16, 3, 3), 0.05)), init("b2", np.zeros(16)),
        init("wg", rng((16, 4), 0.4)), init("bg", np.zeros(4)),
        init("starts", np.array([0], np.int64)),
        init("ends", np.array([2], np.int64)),
        init("axes", np.array([0], np.int64)),
        init("idx2", np.array([2, 2], np.int64)),
        init("two", np.array([2], np.int64)),
    ]
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["c"], name="conv",
                         pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node("Shape", ["r"], ["sh"], name="shape"),
        helper.make_node("Slice", ["sh", "starts", "ends", "axes"], ["nc"],
                         name="slice_nc"),
        helper.make_node("Gather", ["sh", "idx2"], ["hh"], name="gather_hw",
                         axis=0),
        helper.make_node("Mul", ["hh", "two"], ["hh2"], name="mul_hw"),
        helper.make_node("Concat", ["nc", "hh2"], ["sizes"], name="concat",
                         axis=0),
        helper.make_node("Resize", ["r", "", "", "sizes"], ["up"], name="resize",
                         mode="nearest",
                         coordinate_transformation_mode="asymmetric"),
        helper.make_node("Conv", ["up", "w2", "b2"], ["c2"], name="conv2",
                         pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
        helper.make_node("GlobalAveragePool", ["r2"], ["gap"], name="gap"),
        helper.make_node("Flatten", ["gap"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["g"], name="gemm"),
        helper.make_node("Softmax", ["g"], ["p"], name="softmax", axis=1),
    ]
    m = model(nodes, [x], outs, inits, "hybrid_meta")
    return m, {"x": rng((1, 3, 8, 8), 0.8)}


# ----------------------------------------------------------------------- erf
def build_erf():
    x = vi("x", F, [1, 3, 8, 8])
    outs = [vi("p", F, [1, 4])]
    inits = [init("w", rng((16, 3, 3, 3), 0.3)), init("b", np.zeros(16)),
             init("wg", rng((16, 4), 0.4)), init("bg", np.zeros(4))]
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], name="conv",
                         pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node("Erf", ["r"], ["e"], name="erf"),
        helper.make_node("GlobalAveragePool", ["e"], ["gap"], name="gap"),
        helper.make_node("Flatten", ["gap"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["g"], name="gemm"),
        helper.make_node("Softmax", ["g"], ["p"], name="softmax", axis=1),
    ]
    m = model(nodes, [x], outs, inits, "hybrid_erf")
    return m, {"x": rng((1, 3, 8, 8), 0.8)}


# ------------------------------------------------------------------------ if
def build_if():
    x = vi("x", F, [1, 5])
    outs = [vi("y", F, [1, 5])]
    inits = [init("starts", np.array([4], np.int64)),
             init("ends", np.array([5], np.int64)),
             init("axes", np.array([1], np.int64))]
    then_g = helper.make_graph(
        [helper.make_node("Add", ["x", "one"], ["y"], name="then_add")],
        "then", [], [vi("y", F, [1, 5])],
        initializer=[init("one", np.array([1.0], np.float32))])
    else_g = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["y"], name="else_mul")],
        "else", [], [vi("y", F, [1, 5])],
        initializer=[init("two", np.array([2.0], np.float32))])
    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["c"],
                         name="slice_cond"),
        helper.make_node("Cast", ["c"], ["cb"], name="cast", to=TensorProto.BOOL),
        helper.make_node("Squeeze", ["cb"], ["cs"], name="squeeze"),
        helper.make_node("If", ["cs"], ["y"], name="if",
                         then_branch=then_g, else_branch=else_g),
    ]
    m = model(nodes, [x], outs, inits, "hybrid_if")
    a = rng((1, 5), 0.8)
    b = a.copy()
    a[0, 4] = 1.0
    b[0, 4] = -1.0
    return m, {"true": {"x": a}, "false": {"x": b}}


def reference(m, feeds_by_case):
    ev = ReferenceEvaluator(m)
    out = {}
    for case, feeds in feeds_by_case.items():
        out[case] = [np.asarray(z) for z in ev.run(None, feeds)]
    return out


def save(m, path):
    onnx.save(m, path)
    return path


def ensure(models_dir, force=None):
    """Create the models and references that are missing; return metadata.

    Rebuilding unconditionally runs ReferenceEvaluator over the whole LSTM
    graph in every sub-test.  That is tens of seconds of ARM work and heat (it
    can push the board into the thermal guard) for files that are already
    correct, so an existing model, input and reference set is left alone.
    Set HYBRID_REBUILD=1 (or pass force=True) to regenerate them.
    """
    if force is None:
        force = os.environ.get("HYBRID_REBUILD", "") not in ("", "0")
    os.makedirs(models_dir, exist_ok=True)
    made = {}
    specs = {
        "hybrid_mixed": build_mixed,
        "hybrid_meta": build_meta,
        "hybrid_erf": build_erf,
        "hybrid_if": build_if,
    }
    for stem, fn in specs.items():
        m, cases = fn()
        if "x" in cases:
            cases = {"default": cases}
        path = os.path.join(models_dir, stem + ".onnx")
        needed = [path]
        for case in cases:
            suffix = "" if case == "default" else "." + case
            needed.append(os.path.join(models_dir, stem + suffix + ".in.npy"))
            needed.append(os.path.join(models_dir, stem + suffix + ".ref.npz"))
        if not force and all(os.path.isfile(f) for f in needed):
            made[stem] = {"model": path, "cases": list(cases)}
            continue
        save(m, path)
        refs = reference(m, cases)
        for case, feeds in cases.items():
            suffix = "" if case == "default" else "." + case
            ip = os.path.join(models_dir, stem + suffix + ".in.npy")
            rp = os.path.join(models_dir, stem + suffix + ".ref.npz")
            np.save(ip, feeds["x"])
            np.savez(rp, **{"out%d" % i: a
                            for i, a in enumerate(refs[case])})
        made[stem] = {"model": path, "cases": list(cases)}
    return made


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.environ.get("HYBRID_MODELS", "/home/raspi/work/onnx/models")
    import json
    print(json.dumps(ensure(d), indent=1))
