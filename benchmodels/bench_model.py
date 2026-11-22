#!/usr/bin/env python3
"""Measure one ONNX model on ONNX Runtime CPU and on the VC4 engine.

Protocol (matching docs/BENCHMARKS.md): three warm-ups and five timed runs on
each backend, identical feeds, ONNX Runtime ``CPUExecutionProvider`` with four
intra-op threads, managed backend (``VC4INFER_DAEMON=1``), one client at a
time.  The engine runs in strict mode: a graph that plans is timed; a graph
that does not is recorded with the refusal it raised and is CPU-only.

A model with symbolic batch/height/width is written to a fixed-shape copy
first (the engine accepts fixed-shape fp32 only); both backends then run the
same file.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "infer", "python"))
import vc4infer                                                   # noqa: E402

WARMUP = 3
RUNS = 5


def _default_dims(vi, size):
    out = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            out.append(d.dim_value)
        elif d.dim_param.lower() in ("height", "h", "in_h"):
            out.append(size)
        elif d.dim_param.lower() in ("width", "w", "in_w"):
            out.append(size)
        else:
            out.append(1)          # batch and anything else
    return out


def fix_shapes(path, size, out_dir):
    m = onnx.load(path)
    changed = False
    for vi in list(m.graph.input) + list(m.graph.output):
        for d, v in zip(vi.type.tensor_type.shape.dim,
                        _default_dims(vi, size)):
            if not (d.HasField("dim_value") and d.dim_value == v):
                d.ClearField("dim_param")
                d.dim_value = v
                changed = True
    if not changed:
        return path
    m = onnx.shape_inference.infer_shapes(m)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0]
                       + "-fixed.onnx")
    onnx.save(m, dst)
    return dst


def _cs_shape_map(model):
    """ConstantOfShape output -> element count, from its int64 shape input."""
    init = {t.name: t for t in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type != "ConstantOfShape" or not n.input:
            continue
        t = init.get(n.input[0])
        if t is None or t.data_type != onnx.TensorProto.INT64:
            continue
        vals = onnx.numpy_helper.to_array(t)
        if vals.size:
            out[n.output[0]] = int(np.prod(vals))
    return out


def count_params(model):
    params = 0
    for t in model.graph.initializer:
        params += int(np.prod(t.dims)) if t.dims else 1
    # Topology-only ONNX test models synthesise weights with ConstantOfShape;
    # those are real shapes and real bytes once materialised.
    params += sum(_cs_shape_map(model).values())
    return params


def feed_specs(model):
    init = {t.name for t in model.graph.initializer}
    specs = []
    for i in model.graph.input:
        if i.name in init or not i.type.HasField("tensor_type"):
            continue
        if i.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            continue
        specs.append((i.name, [d.dim_value for d in
                               i.type.tensor_type.shape.dim]))
    return specs


def make_feeds(specs):
    rng = np.random.default_rng(1234)
    return {name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in specs}


def time_ort(sess, feeds):
    for _ in range(WARMUP):
        sess.run(None, feeds)
    runs, outs = [], None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        outs = sess.run(None, feeds)
        runs.append((time.perf_counter() - t0) * 1e3)
    return {"ok": True, "runs_ms": runs, "wall_ms": statistics.mean(runs),
            "min_ms": min(runs)}, outs


def time_engine(eng, feeds):
    for _ in range(WARMUP):
        eng.run(feeds)
    runs, launches, qpu = [], [], []
    out = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = eng.run(feeds, profile=True)
        runs.append((time.perf_counter() - t0) * 1e3)
        launches.append(int(eng.stats["launches"]))
        qpu.append(float(eng.stats["gpu_ms"]))
    return {"ok": True, "runs_ms": runs, "wall_ms": statistics.mean(runs),
            "min_ms": min(runs), "launches": launches[0],
            "qpu_ms": statistics.mean(qpu)}, out


def compare(got, ref):
    max_abs = 0.0
    max_scaled = 0.0
    elems = 0
    for a, b in zip(got, ref):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"shape_mismatch": [list(a.shape), list(b.shape)]}
        d = np.abs(a - b)
        max_abs = max(max_abs, float(d.max()) if d.size else 0.0)
        max_scaled = max(max_scaled,
                         float((d / (1.0 + np.abs(b))).max()) if d.size
                         else 0.0)
        elems += d.size
    return {"max_abs": max_abs, "max_scaled": max_scaled, "elements": elems}


def mi(b):
    return round(b / 2 ** 20, 1) if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--size", type=int, default=224,
                    help="value for symbolic height/width")
    ap.add_argument("--arena-mb", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-only", action="store_true")
    a = ap.parse_args()

    rec = {"model": os.path.basename(a.model), "task": a.task,
           "source": a.source, "file": a.model}
    fixed = fix_shapes(a.model, a.size,
                       os.path.join(os.path.dirname(os.path.abspath(a.model)),
                                    "fixed"))
    m = onnx.load(fixed)
    rec["fixed_file"] = os.path.basename(fixed) if fixed != a.model else None
    rec["opset"] = [o.version for o in m.opset_import if o.domain == ""]
    rec["nodes"] = len(m.graph.node)
    rec["params"] = count_params(m)
    rec["inputs"] = [[i.name, [d.dim_value for d in
                               i.type.tensor_type.shape.dim]]
                     for i in m.graph.input][:4]
    rec["n_inputs"] = len(m.graph.input)
    rec["outputs"] = [[o.name, [d.dim_value for d in
                                o.type.tensor_type.shape.dim]]
                      for o in m.graph.output]
    rec["ops"] = sorted({n.op_type for n in m.graph.node})
    specs = feed_specs(m)
    feeds = make_feeds(specs)
    rec["n_feeds"] = len(specs)
    rec["feed_elements"] = int(sum(int(np.prod(s)) for _, s in specs))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(fixed, so,
                                providers=["CPUExecutionProvider"])
    try:
        cpu, cpu_out = time_ort(sess, feeds)
    except Exception as e:                                      # noqa: BLE001
        cpu = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    rec["cpu"] = cpu

    eng = None
    if a.cpu_only:
        rec["plan"] = {"ok": False, "refusal": "not attempted (--cpu-only)"}
    else:
        try:
            eng = vc4infer.Engine(fixed, arena_mb=a.arena_mb)
            rec["engine_inputs"] = [[i.name, list(i.shape)] for i in eng.inputs]
            rec["plan"] = eng.plan().to_dict()
            rec["plan"]["ok"] = True
        except Exception as e:                                  # noqa: BLE001
            rec["plan"] = {"ok": False,
                           "refusal": "%s: %s" % (type(e).__name__, e)}
    if a.cpu_only:
        rec["engine"] = {"ok": False, "refusal": "not attempted (--cpu-only)"}
    elif eng is None:
        rec["engine"] = {"ok": False,
                         "refusal": "engine did not open: %s"
                                    % rec["plan"].get("refusal", "")}
    else:
        ef = {i.name: feeds[i.name] for i in eng.inputs
              if i.name in feeds}
        missing = [i.name for i in eng.inputs if i.name not in feeds]
        if not rec["plan"].get("ok"):
            rec["engine"] = {"ok": False,
                             "refusal": rec["plan"]["refusal"],
                             "stage": "plan"}
        elif missing:
            rec["engine"] = {"ok": False,
                             "refusal": "no feed for engine input(s) %s"
                                        % ", ".join(missing)}
        else:
            try:
                eng_out, out = time_engine(eng, ef)
                got = [out[o.name] for o in eng.outputs]
                eng_out["error"] = compare(got, cpu_out) if cpu["ok"] else {}
                eng_out["memory"] = {
                    "arena_mb": mi(eng.memory()["arena_bytes"]),
                    "pool_mb": mi(eng.memory()["pool_bytes"]),
                    "peak_mb": mi(eng.memory()["peak_bytes"]),
                    "estimate_mb": mi(eng.memory()["estimate_bytes"]),
                    "scratch_mb": mi(eng.memory()["scratch_bytes"]),
                    "segments": eng.memory()["pool_segments"],
                }
                eng_out["outputs"] = [
                    [o.name, list(v.shape)] for o, v in zip(eng.outputs, got)]
                rec["engine"] = eng_out
            except Exception as e:                              # noqa: BLE001
                rec["engine"] = {"ok": False,
                                 "refusal": "%s: %s" % (type(e).__name__,
                                                         e)}
    if eng is not None:
        try:
            eng.close()
        except Exception:                                       # noqa: BLE001
            pass
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    e = rec["engine"]
    print("%-32s plan=%s vc4=%s cpu=%s"
          % (rec["model"],
             "yes" if rec["plan"].get("ok", "nodes" in rec["plan"]) else "no",
             ("%.1f ms" % e["wall_ms"]) if e.get("ok") else "refused",
             ("%.1f ms" % rec["cpu"]["wall_ms"]) if rec["cpu"]["ok"]
             else "failed"))


if __name__ == "__main__":
    main()
