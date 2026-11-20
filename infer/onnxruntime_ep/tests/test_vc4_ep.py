#!/usr/bin/env python3
"""Acceptance tests for the VC4 ONNX Runtime provider.

    # on the Pi, with onnxruntime installed in a venv:
    PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
        infer/onnxruntime_ep/tests/test_vc4_ep.py --reps 3

Cases
  cnn        a small CNN exported by torch.onnx.export, fully claimed by VC4,
             compared with ORT CPU alone (0 over 1e-4 + 1e-4*max(|ref|,1))
  split      a model with one deliberately unsupported node (Einsum), proving
             the partition is reported and the result still matches ORT CPU
  yolo160    the shipped YOLOv8n-160, compared with the stored CPU reference
             (0/44100 over 5e-4 when fully claimed)
  passthrough providers=["CPUExecutionProvider"] stays pure ORT

Every case prints the node table, so a pass is not just a boolean: it shows
which nodes ran where and what share of the MACs that was.
"""
import argparse
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import vc4infer
from vc4infer.ort import InferenceSession

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def tol_1e4(ref):
    return 1e-4 + 1e-4 * np.maximum(np.abs(ref), 1.0)


def tol_5e4(ref):
    return 5e-4 + 5e-4 * np.maximum(np.abs(ref), 1.0)


def compare(got, ref, tol):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    d = np.abs(got - ref)
    over = int((d > tol).sum())
    return {"elements": int(d.size), "over": over,
            "max_abs": float(d.max()) if d.size else 0.0,
            "max_scaled": float((d / tol).max()) if d.size else 0.0}


def make_split_model(path, seed=7):
    """Conv -> Relu -> Einsum -> Conv -> Relu, with Einsum in the middle.

    vc4infer has no Einsum kernel, so a correct provider must cut the graph in
    three and say so.  ORT CPU evaluates the Einsum; the two Conv/Relu islands
    stay on the QPUs.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((16, 3, 3, 3), dtype=np.float32) * 0.1
    b1 = rng.standard_normal((16,), dtype=np.float32) * 0.1
    w2 = rng.standard_normal((8, 16, 1, 1), dtype=np.float32) * 0.1
    b2 = rng.standard_normal((8,), dtype=np.float32) * 0.1
    inits = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(b2, "b2"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"],
                         name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node("Einsum", ["r1", "r1"], ["e1"], name="einsum",
                         equation="nchw,nchw->nchw"),
        helper.make_node("Conv", ["e1", "w2", "b2"], ["c2"],
                         name="conv2"),
        helper.make_node("Relu", ["c2"], ["output"], name="relu2"),
    ]
    graph = helper.make_graph(
        nodes, "split_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, 8, 16, 16])],
        inits)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _timing(sess, feed, reps):
    sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = sess.run(None, feed)
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    return out, ms


def case_cnn(data_dir, work_dir, reps):
    path = os.path.join(work_dir, "smallcnn.onnx")
    if not os.path.isfile(path):
        print("SKIP cnn: %s missing (run export_cnn.py with torch)" % path)
        return None
    x = np.load(os.path.join(data_dir, "smallcnn.in.npy")) \
        if os.path.isfile(os.path.join(data_dir, "smallcnn.in.npy")) \
        else np.random.default_rng(0).standard_normal(
            (1, 3, 32, 32)).astype(np.float32)
    ref_sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref, ref_ms = _timing(ref_sess, {"input": x}, reps)

    with InferenceSession(path, providers=["vc4"]) as sess:
        got, vc4_ms = _timing(sess, {"input": x}, reps)
        sess.profile = True          # one profiled run for the node table
        sess.run(None, {"input": x})
        s = sess.report(per_node=True)
        assert s["claimed_all"], "CNN was not fully claimed: %s" % s["counts"]
        c = compare(got[0], ref[0], tol_1e4(ref[0]))
        assert c["over"] == 0, c
        print("  cnn: %d/%d over tol, max abs %.3g; ORT+vc4 %.2f ms, "
              "ORT CPU %.2f ms (%.1fx)"
              % (c["over"], c["elements"], c["max_abs"], vc4_ms, ref_ms,
                 ref_ms / vc4_ms))
        sess.close()
    return {"name": "cnn", **c, "vc4_ms": vc4_ms, "ort_ms": ref_ms}


def case_split(data_dir, work_dir, reps):
    path = os.path.join(work_dir, "split_einsum.onnx")
    if not os.path.isfile(path):
        make_split_model(path)
    x = np.random.default_rng(11).standard_normal(
        (1, 3, 16, 16)).astype(np.float32)
    ref_sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref, ref_ms = _timing(ref_sess, {"input": x}, reps)
    with InferenceSession(path, providers=["vc4"]) as sess:
        got, vc4_ms = _timing(sess, {"input": x}, reps)
        sess.profile = True          # one profiled run for the node table
        sess.run(None, {"input": x})
        s = sess.report(per_node=True)
        assert s["counts"]["ort-cpu"] == 1, s["counts"]
        assert s["counts"]["vc4"] >= 3, s["counts"]
        c = compare(got[0], ref[0], tol_1e4(ref[0]))
        assert c["over"] == 0, c
        print("  split: reported %d vc4 + %d ort-cpu node(s); %d/%d over tol; "
              "max abs %.3g; %.2f ms vs %.2f ms"
              % (s["counts"]["vc4"], s["counts"]["ort-cpu"], c["over"],
                 c["elements"], c["max_abs"], vc4_ms, ref_ms))
        sess.close()

    # The hard guarantee: refuse to fall back at all.
    try:
        InferenceSession(path, providers=[("vc4", {"fallback": "error"})])
    except vc4infer.UnsupportedNode as e:
        print("  split: fallback='error' refused: %s" % str(e).splitlines()[0])
    else:
        raise AssertionError("fallback='error' did not refuse the Einsum node")
    return {"name": "split", **c, "vc4_ms": vc4_ms, "ort_ms": ref_ms}


def _yolo_paths(models_dir):
    p = os.path.join(models_dir, "yolov8n_160.onnx")
    i = os.path.join(models_dir, "yolov8n_160.in.npy")
    r = os.path.join(models_dir, "yolov8n_160.ref.npy")
    if all(os.path.isfile(q) for q in (p, i, r)):
        return p, i, r
    return None, None, None


def case_yolo(data_dir, work_dir, reps):
    path, inp, refp = _yolo_paths(data_dir)
    if not path:
        print("SKIP yolo160: models not found in %s" % data_dir)
        return None
    x = np.load(inp).astype(np.float32)
    ref = np.load(refp).astype(np.float32)
    with InferenceSession(path, providers=["vc4"]) as sess:
        got, ms = _timing(sess, {"images": x}, reps)
        s = sess.report(per_node=False)
        assert s["claimed_all"], "YOLO not fully claimed: %s" % s["counts"]
        c = compare(got[0], ref, tol_5e4(ref))
        assert c["over"] == 0, c
        print("  yolo160: %d/%d over 5e-4, max abs %.4g, max scaled %.3g; "
              "%.2f ms/inf = %.2f FPS"
              % (c["over"], c["elements"], c["max_abs"], c["max_scaled"],
                 ms, 1000.0 / ms))
        sess.close()
    return {"name": "yolo160", **c, "ms": ms, "fps": 1000.0 / ms}


def case_passthrough(data_dir, work_dir, reps):
    path = os.path.join(work_dir, "smallcnn.onnx")
    if not os.path.isfile(path):
        print("SKIP passthrough: no smallcnn.onnx")
        return None
    x = np.random.default_rng(3).standard_normal(
        (1, 3, 32, 32)).astype(np.float32)
    with InferenceSession(path, providers=["CPUExecutionProvider"]) as sess:
        s = sess.report()
        assert s["counts"]["ort-cpu"] == s["nodes"], s["counts"]
        assert sess.get_providers() == ["CPUExecutionProvider"]
        sess.run(None, {"input": x})
    print("  passthrough: %d/%d nodes on ort-cpu" % (s["counts"]["ort-cpu"],
                                                     s["nodes"]))
    return {"name": "passthrough"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        ROOT, "..", "packaging", "vc4infer", "data", "models"))
    ap.add_argument("--work", default=os.path.join(
        os.path.expanduser("~"), "scratch", "tmp", "vc4ort"))
    ap.add_argument("--only", default="cnn,split,yolo160,passthrough")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    data = os.path.abspath(args.data)
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    print("vc4infer.ort tests: data=%s work=%s, providers=%s"
          % (data, work, ort.get_available_providers()))
    results = []
    for name in args.only.split(","):
        fn = {"cnn": case_cnn, "split": case_split, "yolo160": case_yolo,
              "passthrough": case_passthrough}.get(name)
        if fn is None:
            raise SystemExit("unknown case %r" % name)
        print("[%s]" % name)
        r = fn(data, work, args.reps)
        if r:
            results.append(r)
    print("\nOK: %d case(s) passed" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
