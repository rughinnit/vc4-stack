#!/usr/bin/env python3
"""test_onnx_breadth.py -- the operator forms four YOLO families need.

Four shapes were refused by the planner before the arena mattered:

  * Split on axis 2 of a rank-4 [B][heads][C][HW] tensor (v10, v11, v12,
    YOLO26 attention) and on any other non-channel axis;
  * MatMul with rank-4 operands (the two batched products inside attention);
  * Slice on a pitched NHWC conv output (every v9 C3k2 branch);
  * a 6x6 stride-2 convolution (the YOLOv5 stem).

Each case is a small graph, run through the shipped runner
(``mode="hybrid"``, ``host_policy="device"``) and compared with
``onnx.reference``.  Graphs that need a conv-produced NHWC buffer start with a
Conv, because a one-node graph always sees an uploaded compact NCHW input and
cannot reach that path.

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 python3 tests/test_onnx_breadth.py
    python3 tests/test_onnx_breadth.py --only Split,MatMul -v

Exit status is 0 only when every case ran on the VC4 kernels and agreed with
the oracle.
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


def rnd(rng, shape, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def i64(*a):
    return np.asarray(a, np.int64)


def compare(ref, got):
    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.shape != got.shape:
        return "shape %s vs %s" % (ref.shape, got.shape)
    d = np.abs(ref - got)
    lim = TOL_ABS + TOL_REL * np.maximum(np.abs(ref), 1.0)
    bad = d > lim
    if bad.any():
        i = np.unravel_index(int(np.argmax(d / lim)), d.shape)
        return ("max abs %.3g at %s (ref %.6g got %.6g)" %
                (float(d[i]), i, float(ref[i]), float(got[i])))
    return ""


class Case:
    def __init__(self, name, feeds, inits, nodes, out_names, out_types=None,
                 note="", opset=OPSET):
        self.name = name
        self.feeds = feeds
        self.inits = inits
        self.nodes = nodes
        self.out_names = out_names
        self.out_types = out_types or [TensorProto.FLOAT] * len(out_names)
        self.note = note
        self.opset = opset


def node(op, ins, outs, **attrs):
    return helper.make_node(op, ins, outs, name="%s_%d" % (op, id(outs)),
                            **attrs)


def build(case):
    inputs = []
    seen = set()
    produced = {o for n in case.nodes for o in n.output}
    for n in case.nodes:
        for name in n.input:
            if name in seen or name in case.inits or name in produced:
                continue
            a = case.feeds.get(name)
            if a is None:
                continue
            seen.add(name)
            inputs.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT, list(a.shape)))
    inits = [numpy_helper.from_array(np.asarray(v), name=k)
             for k, v in case.inits.items()]
    outputs = [helper.make_tensor_value_info(n, t, None)
               for n, t in zip(case.out_names, case.out_types)]
    graph = helper.make_graph(case.nodes, "breadth", inputs, outputs,
                              initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", case.opset)],
        producer_name="vc4infer-test-breadth")
    if case.opset < 11:
        model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False,
                                              data_prop=False)
    for vi in model.graph.output:
        if not vi.type.tensor_type.HasField("shape"):
            vi.type.tensor_type.shape.SetInParent()
    onnx.checker.check_model(model)
    return model


def _conv(x, w, b=None, pads=(1, 1, 1, 1), strides=(1, 1)):
    ins = [x, w] + ([b] if b else [])
    return node("Conv", ins, ["XP"],
                pads=list(pads), strides=list(strides),
                dilations=[1, 1], group=1)


def cases(rng):
    out = []

    # --- Split: attention q/k/v, and the other axes ----------------------
    x = rnd(rng, (4, 2, 96, 25))
    out.append(Case(
        "split_attention_axis2_b4", {"X": x}, {"split": i64(32, 32, 32)},
        [node("Split", ["X", "split"], ["Y0", "Y1", "Y2"], axis=2)],
        ["Y0", "Y1", "Y2"], note="v12: [4][2][96][25] -> 3x[4][2][32][25]"))
    x = rnd(rng, (1, 2, 128, 25))
    out.append(Case(
        "split_attention_axis2_uneven", {"X": x}, {"split": i64(32, 32, 64)},
        [node("Split", ["X", "split"], ["Y0", "Y1", "Y2"], axis=2)],
        ["Y0", "Y1", "Y2"], note="v10/v11: q, k, v with C=32,32,64"))
    x = rnd(rng, (2, 6, 4))
    out.append(Case(
        "split_rank3_axis1", {"X": x}, {"split": i64(1, 2, 3)},
        [node("Split", ["X", "split"], ["Y0", "Y1", "Y2"], axis=1)],
        ["Y0", "Y1", "Y2"], note="rank 3, uneven axis 1"))
    x = rnd(rng, (4, 4, 8, 8))
    out.append(Case(
        "split_axis0_negative_attr", {"X": x}, {},
        [node("Split", ["X"], ["Y0", "Y1"], axis=-4, split=[1, 3])],
        ["Y0", "Y1"], note="axis 0, attribute split form (opset 11)",
        opset=11))

    # --- MatMul: batched ------------------------------------------------
    out.append(Case(
        "matmul_rank4_qk", {"A": rnd(rng, (4, 2, 25, 32)),
                     "B": rnd(rng, (4, 2, 32, 25))}, {},
        [node("MatMul", ["A", "B"], ["Y0"])], ["Y0"],
        note="v12 attention scores, B*H = 8 GEMMs"))
    out.append(Case(
        "matmul_rank4_vs", {"A": rnd(rng, (1, 2, 64, 25)),
                     "B": rnd(rng, (1, 2, 25, 25))}, {},
        [node("MatMul", ["A", "B"], ["Y0"])], ["Y0"],
        note="v10/v11 value @ scores"))
    out.append(Case(
        "matmul_rank3", {"A": rnd(rng, (3, 4, 8)), "B": rnd(rng, (3, 8, 5))}, {},
        [node("MatMul", ["A", "B"], ["Y0"])], ["Y0"]))

    # --- Slice on a pitched NHWC conv output ---------------------------
    w = rnd(rng, (32, 16, 3, 3), -0.3, 0.3)
    b = rnd(rng, (32,), -0.3, 0.3)
    x = rnd(rng, (1, 16, 8, 8))
    out.append(Case(
        "slice_conv_channel_half", {"X": x}, {"W": w, "B": b,
                                        "starts": i64(0), "ends": i64(16),
                                        "axes": i64(1)},
        [_conv("X", "W", "B"),
         node("Slice", ["XP", "starts", "ends", "axes"], ["Y0"])],
        ["Y0"], note="v9: NHWC conv output pitched to 16, slice 0:16"))
    out.append(Case(
        "slice_conv_channel_second_half", {"X": x}, {"W": w, "B": b,
                                               "starts": i64(16),
                                               "ends": i64(32),
                                               "axes": i64(1)},
        [_conv("X", "W", "B"),
         node("Slice", ["XP", "starts", "ends", "axes"], ["Y0"])],
        ["Y0"], note="v9: second half, channel offset 16"))
    out.append(Case(
        "slice_conv_channel_24", {"X": x}, {"W": w, "B": b,
                                           "starts": i64(0), "ends": i64(24),
                                           "axes": i64(1)},
        [_conv("X", "W", "B"),
         node("Slice", ["XP", "starts", "ends", "axes"], ["Y0"])],
        ["Y0"], note="32 -> 24: one full block and one partial (8 words)"))
    out.append(Case(
        "slice_conv_spatial_gather", {"X": x}, {"W": w, "B": b,
                                          "starts": i64(1, 2, 4, 6),
                                          "ends": i64(7, 6, 8, 8),
                                          "axes": i64(0, 1, 2, 3)},
        [_conv("X", "W", "B"),
         node("Slice", ["XP", "starts", "ends", "axes"], ["Y0"])],
        ["Y0"], note="several axes on NHWC: storage-coordinate gather"))

    # --- 6x6 stride-2 conv (YOLOv5 stem) -------------------------------
    out.append(Case(
        "conv_6x6_stem", {"X": rnd(rng, (1, 3, 16, 16))},
        {"W": rnd(rng, (16, 3, 6, 6), -0.3, 0.3),
         "B": rnd(rng, (16,), -0.3, 0.3)},
        [node("Conv", ["X", "W", "B"], ["Y0"], pads=[2, 2, 2, 2],
              strides=[2, 2], dilations=[1, 1], group=1)],
        ["Y0"], note="Cin 3 padded to 16, K = 16*6*6"))
    out.append(Case(
        "conv_cout24", {"X": rnd(rng, (1, 16, 10, 10))},
        {"W": rnd(rng, (24, 16, 3, 3), -0.3, 0.3),
         "B": rnd(rng, (24,), -0.3, 0.3)},
        [node("Conv", ["X", "W", "B"], ["Y0"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1)],
        ["Y0"], note="Cout 24: GEMM computes 32, trim is a partial block"))
    out.append(Case(
        "conv_repad_24", {"X": rnd(rng, (1, 24, 10, 10))},
        {"W1": rnd(rng, (24, 24, 3, 3), -0.2, 0.2),
         "B1": rnd(rng, (24,), -0.2, 0.2),
         "W2": rnd(rng, (16, 24, 3, 3), -0.2, 0.2),
         "B2": rnd(rng, (16,), -0.2, 0.2)},
        [node("Conv", ["X", "W1", "B1"], ["H1"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Conv", ["H1", "W2", "B2"], ["Y0"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1)],
        ["Y0"], note="Cout 24 then Cin 24: trim and NHWC re-pad to 32"))
    out.append(Case(
        "conv_concat_24", {"X": rnd(rng, (1, 16, 10, 10))},
        {"W": rnd(rng, (24, 16, 3, 3), -0.2, 0.2),
         "B": rnd(rng, (24,), -0.2, 0.2)},
        [node("Conv", ["X", "W", "B"], ["H1"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Concat", ["H1", "H1"], ["Y0"], axis=1)],
        ["Y0"], note="NHWC concat of two 24-channel conv outputs"))
    out.append(Case(
        "concat_box_axis2", {"X": rnd(rng, (1, 16, 8, 8))},
        {"W": rnd(rng, (4, 16, 3, 3), -0.2, 0.2),
         "B": rnd(rng, (4,), -0.2, 0.2), "shp": i64(1, 4, 64)},
        [node("Conv", ["X", "W", "B"], ["H1"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Reshape", ["H1", "shp"], ["H2"]),
         node("Concat", ["H2", "H2"], ["Y0"], axis=2)],
        ["Y0"], note="YOLO26 box head: 4 channels, concat on the anchor axis"))
    out.append(Case(
        "concat_box_3scales", {"X0": rnd(rng, (1, 16, 20, 20)),
                               "X1": rnd(rng, (1, 16, 10, 10)),
                               "X2": rnd(rng, (1, 16, 5, 5))},
        {"W": rnd(rng, (4, 16, 3, 3), -0.2, 0.2),
         "B": rnd(rng, (4,), -0.2, 0.2),
         "s0": i64(1, 4, 400), "s1": i64(1, 4, 100), "s2": i64(1, 4, 25)},
        [node("Conv", ["X0", "W", "B"], ["C0"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Conv", ["X1", "W", "B"], ["C1"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Conv", ["X2", "W", "B"], ["C2"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Reshape", ["C0", "s0"], ["R0"]),
         node("Reshape", ["C1", "s1"], ["R1"]),
         node("Reshape", ["C2", "s2"], ["R2"]),
         node("Concat", ["R0", "R1", "R2"], ["Y0"], axis=2)],
        ["Y0"], note="YOLO26 head exactly: 4x400 + 4x100 + 4x25 on axis 2"))
    out.append(Case(
        "reshape_conv4_5x5", {"X": rnd(rng, (1, 16, 5, 5))},
        {"W": rnd(rng, (4, 16, 3, 3), -0.2, 0.2),
         "B": rnd(rng, (4,), -0.2, 0.2), "shp": i64(1, 4, 25)},
        [node("Conv", ["X", "W", "B"], ["H1"], pads=[1, 1, 1, 1],
              strides=[1, 1], dilations=[1, 1], group=1),
         node("Reshape", ["H1", "shp"], ["Y0"])],
        ["Y0"], note="4-channel NHWC pitch 16 -> NCHW 1x4x25"))
    out.append(Case(
        "concat_rank3_525", {"A": rnd(rng, (1, 4, 400)),
                             "B": rnd(rng, (1, 4, 100)),
                             "C": rnd(rng, (1, 4, 25))}, {},
        [node("Concat", ["A", "B", "C"], ["Y0"], axis=2)],
        ["Y0"], note="row length 525 is not a 16-byte multiple"))
    out.append(Case(
        "concat_rank3_528", {"A": rnd(rng, (1, 4, 400)),
                             "B": rnd(rng, (1, 4, 100)),
                             "C": rnd(rng, (1, 4, 28))}, {},
        [node("Concat", ["A", "B", "C"], ["Y0"], axis=2)],
        ["Y0"], note="control: row length 528 is 16-byte aligned"))
    out.append(Case(
        "conv_6x6_stem_odd", {"X": rnd(rng, (1, 3, 15, 17))},
        {"W": rnd(rng, (16, 3, 6, 6), -0.3, 0.3),
         "B": rnd(rng, (16,), -0.3, 0.3)},
        [node("Conv", ["X", "W", "B"], ["Y0"], pads=[2, 2, 2, 2],
              strides=[2, 2], dilations=[1, 1], group=1)],
        ["Y0"], note="odd 15x17 input: phase pads differ by one"))
    return out


def run_case(eng, case, workdir, verbose):
    model = build(case)
    path = os.path.join(workdir, case.name + ".onnx")
    onnx.save(model, path)
    feeds = {k: np.ascontiguousarray(v, np.float32)
             for k, v in case.feeds.items()}
    ref = ReferenceEvaluator(model).run(None, feeds)
    r = Runner(path, eng, mode="hybrid", host_policy="device")
    r.check_policy()
    got = r.run(feeds)
    for i, (w, g) in enumerate(zip(ref, got)):
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
    workdir = args.work_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "vc4breadth-models")
    os.makedirs(workdir, exist_ok=True)
    rng = np.random.default_rng(20261122)
    eng = vc4infer.Engine(cache_dir=os.environ.get("VC4INFER_CACHE") or None)
    bad = []
    total = 0
    try:
        for case in cases(rng):
            if only and not any(o.lower() in case.name for o in only):
                continue
            total += 1
            status, msg = run_case(eng, case, workdir, args.verbose)
            ok = status == "vc4"
            print("%-4s %-28s %-12s %s%s"
                  % ("ok" if ok else "FAIL", case.name, status,
                     "" if ok else msg[:200], "" if ok else ""))
            if not ok:
                bad.append((case.name, status, msg))
    finally:
        eng.close()
    print("\n%d cases, %d failed" % (total, len(bad)))
    for name, status, msg in bad:
        print("  %-28s %-12s %s" % (name, status, msg[:160]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
