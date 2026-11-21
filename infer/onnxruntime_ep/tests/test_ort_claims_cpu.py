#!/usr/bin/env python3
"""CPU-side claim table and partition-equivalence test for ``vc4infer.ort``.

No daemon, no engine, no QPU dispatch: this test measures what the provider
decides and whether that decision preserves the graph, and nothing else.

    PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
        infer/onnxruntime_ep/tests/test_ort_claims_cpu.py

For each model it

* prints the per-node claim list (``vc4`` / ``vc4-folded`` / ``ort-cpu``),
  the number of partitions and the MAC share, from ``vc4infer.ort.plan()``;
* executes every partition of that plan on ORT's CPU EP in dependency order
  and compares the result with the whole model on ORT's CPU EP.  That isolates
  the partitioning (sub-model surgery, boundary order, initialiser capture)
  from the QPU kernels: a difference here is a partitioner bug, not a kernel
  approximation;
* checks the refusal path, ``fallback="error"``, names the first node that
  would have run on ORT.

The on-device acceptance run (``test_vc4_ep.py``) is separate because it needs
vc4d and the GPU to itself.  This file is safe to run while the GPU is busy.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import vc4infer
import vc4infer.ort as vc4ort

TOL_1E4 = (1e-4, 1e-4)
TOL_5E4 = (5e-4, 5e-4)


def tol(pair, ref):
    a, r = pair
    return a + r * np.maximum(np.abs(ref), 1.0)


def compare(got, ref, pair):
    d = np.abs(np.asarray(got, np.float64) - np.asarray(ref, np.float64))
    t = tol(pair, ref)
    return {"elements": int(d.size), "over": int((d > t).sum()),
            "max_abs": float(d.max()) if d.size else 0.0,
            "max_scaled": float((d / t).max()) if d.size else 0.0}


# --------------------------------------------------------------- fixtures
def make_split_model(path):
    """Conv -> Relu -> Resize(cubic) -> Conv -> Relu.

    ``Resize`` with ``mode="cubic"`` has no VC4 form (nearest and linear only),
    so the graph must split in three: two VC4 islands around one ORT node.  The
    ``up`` value_info is what an exporter with shape information emits; without
    it the dynamic boundary would poison the downstream island (see the
    ``poison`` case).
    """
    rng = np.random.default_rng(7)
    w1 = rng.standard_normal((8, 3, 3, 3), dtype=np.float32) * 0.1
    b1 = rng.standard_normal((8,), dtype=np.float32) * 0.1
    w2 = rng.standard_normal((4, 8, 1, 1), dtype=np.float32) * 0.1
    b2 = rng.standard_normal((4,), dtype=np.float32) * 0.1
    inits = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(b2, "b2"),
        numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"],
                         name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node("Resize", ["r1", "", "scales"], ["up"],
                         name="resize_cubic", mode="cubic"),
        helper.make_node("Conv", ["up", "w2", "b2"], ["c2"], name="conv2"),
        helper.make_node("Relu", ["c2"], ["output"], name="relu2"),
    ]
    graph = helper.make_graph(
        nodes, "split_cubic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, 4, 32, 32])], inits)
    graph.value_info.append(helper.make_tensor_value_info(
        "up", TensorProto.FLOAT, [1, 8, 32, 32]))
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def make_poison_model(path):
    """Conv -> Relu -> Compress(axis=None) -> Reshape -> Conv -> Relu.

    ``Compress`` with ``axis=None`` has no VC4 form and its output length is
    data-dependent, so the tensor leaving it has no static shape.  The
    partitioner then refuses the downstream island as well: a data-dependent
    boundary costs more than the one node that produced it.
    """
    rng = np.random.default_rng(8)
    w1 = rng.standard_normal((8, 3, 3, 3), dtype=np.float32) * 0.1
    b1 = rng.standard_normal((8,), dtype=np.float32) * 0.1
    w2 = rng.standard_normal((4, 8, 1, 1), dtype=np.float32) * 0.1
    b2 = rng.standard_normal((4,), dtype=np.float32) * 0.1
    inits = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(b2, "b2"),
        numpy_helper.from_array(np.ones(8 * 16 * 16, np.bool_), "keep"),
        numpy_helper.from_array(np.array([1, 8, 16, 16], np.int64), "shape"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"],
                         name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node("Compress", ["r1", "keep"], ["flat"], name="compress",
                         axis=None),
        helper.make_node("Reshape", ["flat", "shape"], ["rs"], name="reshape"),
        helper.make_node("Conv", ["rs", "w2", "b2"], ["c2"], name="conv2"),
        helper.make_node("Relu", ["c2"], ["output"], name="relu2"),
    ]
    graph = helper.make_graph(
        nodes, "dynamic_boundary",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, 4, 16, 16])], inits)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def make_control_model(path):
    """An If node: control flow is never split, so nothing is claimed."""
    then_g = helper.make_graph(
        [helper.make_node("Add", ["x", "one"], ["t"])], "then",
        [], [helper.make_tensor_value_info("t", TensorProto.FLOAT, [1, 4])])
    else_g = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["e"])], "else",
        [], [helper.make_tensor_value_info("e", TensorProto.FLOAT, [1, 4])])
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["output"], then_branch=then_g,
                          else_branch=else_g, name="if")],
        "control",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(np.ones((1, 4), np.float32), "x"),
         numpy_helper.from_array(np.ones((), np.bool_), "cond"),
         numpy_helper.from_array(np.array(1.0, np.float32), "one"),
         numpy_helper.from_array(np.array(2.0, np.float32), "two")])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


# ------------------------------------------------------------- execution
class PlanRunner:
    """Execute a plan's sub-models on ORT's CPU EP, in dependency order."""

    def __init__(self, plan):
        self.plan = plan
        self.sessions = {}
        for p in plan.partitions:
            blob = p.model_bytes
            if blob is None:
                with open(p.path, "rb") as f:
                    blob = f.read()
            self.sessions[p.index] = ort.InferenceSession(
                blob, providers=["CPUExecutionProvider"])

    def run(self, feed):
        values = {k: np.ascontiguousarray(v) for k, v in feed.items()}
        t0 = time.perf_counter()
        for p in self.plan.partitions:
            outs = self.sessions[p.index].run(
                None, {t: values[t] for t in p.inputs})
            for name, arr in zip(p.outputs, outs):
                values[name] = arr
        ms = (time.perf_counter() - t0) * 1e3
        return [values[n] for n in [o.name for o in self.plan.model.graph.output]], ms


def ort_cpu_session(path_or_bytes):
    return ort.InferenceSession(path_or_bytes,
                                providers=["CPUExecutionProvider"])


def best_ms(fn, reps=3):
    fn()                     # warm-up: ORT initialises kernels on first Run
    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return out, min(times)


def claim_row(plan):
    s = plan.summary()
    return {
        "nodes": s["nodes"],
        "counts": s["counts"],
        "pct_macs": s["pct_macs"],
        "macs_total": s["macs_total"],
        "parts_vc4": s["parts_vc4"],
        "parts_ort": s["parts_ort"],
        "claimed_all": s["claimed_all"],
    }


def show(plan, limit):
    rows = plan.rows()
    print("  %-6s %-26s %-14s %-11s %-10s %s" % (
        "index", "node", "op", "backend", "MACs", "reason"))
    for r in rows[:limit]:
        print("  %-6d %-26s %-14s %-11s %-10d %s" % (
            r["index"], r["name"][:26], r["op"][:14], r["backend"],
            r["macs"], (r["reason"] or "")[:58]))
    if len(rows) > limit:
        print("  ... %d more node(s)" % (len(rows) - limit))


def check(name, plan, feed, ref_pair, limit=12, extra=None):
    r = PlanRunner(plan)
    whole = ort_cpu_session(plan.model_bytes)
    got, plan_ms = best_ms(lambda: r.run(feed)[0])
    ref, ort_ms = best_ms(lambda: whole.run(None, feed))
    c = compare(got[0], ref[0], ref_pair)
    claim = claim_row(plan)
    print("  claim: %d node(s) -> %s, %d vc4 partition(s), %d ort partition(s)"
          % (claim["nodes"], claim["counts"], claim["parts_vc4"],
             claim["parts_ort"]))
    print("  MACs:  %.1f%% on vc4, %.1f%% on vc4-folded, %.1f%% on ort-cpu"
          % (claim["pct_macs"]["vc4"] or 0.0,
             claim["pct_macs"]["vc4-folded"] or 0.0,
             claim["pct_macs"]["ort-cpu"] or 0.0))
    print("  plan on CPU: %d/%d over tol, max abs %.3g; %.2f ms vs %.2f ms "
          "whole-model ORT CPU (%.2fx)"
          % (c["over"], c["elements"], c["max_abs"], plan_ms, ort_ms,
             plan_ms / ort_ms if ort_ms else float("nan")))
    show(plan, limit)
    if extra:
        extra(plan)
    assert c["over"] == 0, "%s: %s" % (name, c)
    return {"case": name, **claim, **c, "plan_ms": plan_ms, "ort_ms": ort_ms}


# ------------------------------------------------------------------ cases
def case_cnn(path, limit):
    plan = vc4ort.plan(path)
    claim = claim_row(plan)
    assert claim["counts"]["vc4"] == claim["nodes"], claim
    assert claim["claimed_all"], claim
    x = np.random.default_rng(0).standard_normal(
        (1, 3, 32, 32)).astype(np.float32)
    return check("cnn", plan, {"input": x}, TOL_1E4, limit)


def case_split(path, limit):
    plan = vc4ort.plan(path)
    claim = claim_row(plan)
    assert claim["counts"]["ort-cpu"] == 1, claim
    assert claim["counts"]["vc4"] == 4, claim
    assert claim["parts_vc4"] == 2 and claim["parts_ort"] == 1, claim
    x = np.random.default_rng(11).standard_normal(
        (1, 3, 16, 16)).astype(np.float32)
    out = check("split", plan, {"input": x}, TOL_1E4, limit)

    try:
        vc4ort.plan(path, fallback="error")
    except vc4infer.UnsupportedNode as e:
        print("  fallback='error': refused: %s" % str(e).splitlines()[0])
        out["refusal"] = str(e).splitlines()[0]
    else:
        raise AssertionError("fallback='error' did not refuse Resize(cubic)")
    return out


def case_poison(path, limit):
    plan = vc4ort.plan(path)
    claim = claim_row(plan)
    assert claim["counts"]["ort-cpu"] == 4, claim
    assert claim["counts"]["vc4"] == 2, claim
    x = np.random.default_rng(12).standard_normal(
        (1, 3, 16, 16)).astype(np.float32)
    out = check("poison", plan, {"input": x}, TOL_1E4, limit)
    return out


def case_control(path, limit):
    plan = vc4ort.plan(path)
    claim = claim_row(plan)
    assert claim["counts"]["vc4"] == 0 and not claim["claimed_all"], claim
    feed = {"input": np.random.default_rng(13).standard_normal(
        (1, 4)).astype(np.float32)}
    out = check("control", plan, feed, TOL_1E4, limit)
    try:
        vc4ort.plan(path, fallback="error")
    except vc4infer.UnsupportedNode as e:
        print("  fallback='error': refused: %s" % str(e).splitlines()[0])
        out["refusal"] = str(e).splitlines()[0]
    else:
        raise AssertionError("fallback='error' did not refuse control flow")
    return out


def case_yolo(path, inp, ref_path, limit):
    plan = vc4ort.plan(path)
    claim = claim_row(plan)
    assert claim["nodes"] == 263, claim
    assert claim["counts"] == {"vc4": 233, "vc4-folded": 30,
                               "ort-cpu": 0}, claim
    assert claim["parts_vc4"] == 1 and claim["parts_ort"] == 0, claim
    x = np.load(inp).astype(np.float32)
    out = check("yolo160", plan, {"images": x}, TOL_5E4, limit)
    if ref_path and os.path.isfile(ref_path):
        ref = np.load(ref_path).astype(np.float32)
        c = compare(PlanRunner(plan).run({"images": x})[0][0], ref, TOL_5E4)
        print("  stored CPU reference: %d/%d over tol, max abs %.3g"
              % (c["over"], c["elements"], c["max_abs"]))
        out["stored_ref"] = c
        assert c["over"] == 0, c
    return out


def case_no_daemon(cnn_path):
    """The provider refuses to open a session without vc4d; it does not
    silently run the graph on ORT and call it acceleration."""
    from vc4infer import _runtime
    from vc4infer.ort import InferenceSession
    old = _runtime.SOCKET
    _runtime.SOCKET = "/nonexistent/vc4d.sock"
    try:
        try:
            InferenceSession(cnn_path, providers=["vc4"])
        except vc4infer.DeviceError as e:
            msg = str(e).splitlines()[0]
            print("  no daemon: DeviceError: %s" % msg)
        else:
            raise AssertionError("session opened without vc4d")
    finally:
        _runtime.SOCKET = old
    with InferenceSession(cnn_path, providers=["CPUExecutionProvider"]) as s:
        sum_ = vc4ort.summary(s)
        assert sum_["counts"]["ort-cpu"] == sum_["nodes"]
        print("  no daemon: providers=['CPUExecutionProvider'] still runs "
              "(whole graph on ORT, no vc4 claim)")
    return {"case": "no_daemon", "refused": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/usr/share/vc4infer/models")
    ap.add_argument("--cnn", default=os.path.join(
        os.path.expanduser("~"), "scratch", "tmp", "vc4ort", "smallcnn.onnx"))
    ap.add_argument("--work", default=os.path.join(
        os.path.expanduser("~"), "scratch", "tmp", "vc4ort"))
    ap.add_argument("--only", default="cnn,split,poison,control,yolo160,"
                                    "no_daemon")
    ap.add_argument("--limit", type=int, default=12,
                    help="node rows printed per case (0 = all)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    print("vc4infer %s, onnxruntime %s, providers=%s"
          % (vc4infer.__version__, ort.__version__,
             ort.get_available_providers()))

    results = []
    wanted = args.only.split(",")
    for name in wanted:
        print("[%s]" % name)
        if name == "cnn":
            if not os.path.isfile(args.cnn):
                print("  SKIP: %s missing (export_cnn.py needs torch)"
                      % args.cnn)
                continue
            results.append(case_cnn(args.cnn, args.limit or 10 ** 9))
        elif name == "split":
            p = make_split_model(os.path.join(work, "split_cubic.onnx"))
            results.append(case_split(p, args.limit or 10 ** 9))
        elif name == "poison":
            p = make_poison_model(os.path.join(work, "split_poison.onnx"))
            results.append(case_poison(p, args.limit or 10 ** 9))
        elif name == "control":
            p = make_control_model(os.path.join(work, "control_if.onnx"))
            results.append(case_control(p, args.limit or 10 ** 9))
        elif name == "yolo160":
            p = os.path.join(args.models, "yolov8n_160.onnx")
            if not os.path.isfile(p):
                print("  SKIP: %s missing" % p)
                continue
            results.append(case_yolo(
                p, os.path.join(args.models, "yolov8n_160.in.npy"),
                os.path.join(args.models, "yolov8n_160.ref.npy"),
                args.limit or 10 ** 9))
        elif name == "no_daemon":
            results.append(case_no_daemon(args.cnn))
        else:
            raise SystemExit("unknown case %r" % name)

    print("\nOK: %d CPU-side case(s) passed" % len(results))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, sort_keys=True)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
