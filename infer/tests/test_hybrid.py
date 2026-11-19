#!/usr/bin/env python3
"""Hybrid-mode acceptance tests (GPU, one engine at a time).

    cd vc4-stack
    VC4INFER_DAEMON=1 PYTHONPATH=infer/python \
        python3 infer/tests/test_hybrid.py

Covers the three host tiers and the policy:

(a) hybrid_meta   Shape/Gather/Slice/Mul/Concat -> computed Resize sizes.
                  Structural hybrid runs it; those nodes are host-meta and the
                  output matches a full ReferenceEvaluator run.
(b) hybrid_erf    Erf is a numeric op with no VC4 kernel.  Strict raises
                  UnsupportedNode; structural hybrid raises HostNotAllowed;
                  host_policy="any" runs it and matches the reference.
    hybrid_mixed  Split/Shape/Gather + Einsum/TopK/NonZero/LSTM.  Under `any`
                  everything runs and matches; structural names the first debt
                  node (Einsum).
(c) hybrid_if     Slice+Cast predicate into If with a captured x: host-control
                  under structural, both branches verified.
(d) YOLOv8n       structural hybrid must have zero host debt and identical
                  output/launches to strict (100% of compute nodes on VC4).

The models are built by ``infer/tools/onnx/make_hybrid_models.py``; their
references come from a full ``onnx.reference.ReferenceEvaluator`` run, i.e. an
implementation independent of both the VC4 executor and this test.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(INFER, "python"))
sys.path.insert(0, os.path.join(INFER, "tools", "onnx"))

import vc4infer                                             # noqa: E402
import make_hybrid_models as gen                            # noqa: E402

ATOL = 1e-4
RTOL = 1e-4
TOL = 5e-4                       # YOLO reference tolerance (docs)
MODELS = os.environ.get("HYBRID_MODELS", "/home/raspi/work/onnx/models")
YOLO = os.environ.get("YOLO_MODEL",
                      "/home/raspi/work/onnx/models/yolov8n_160.onnx")

FAILED = []


def check(cond, what):
    if cond:
        print("  ok   %s" % what)
    else:
        print("  FAIL %s" % what)
        FAILED.append(what)


def compare(got, ref, atol=ATOL, rtol=RTOL):
    """fp32 tolerance used throughout the engine: atol + rtol*max(|ref|,1)."""
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    if g.shape != r.shape:
        return {"shape_mismatch": (g.shape, r.shape)}
    if r.dtype.kind in "iu" or np.asarray(ref).dtype.kind in "iu":
        bad = int((g != r).sum())
        return {"max_abs": float(np.abs(g - r).max()) if g.size else 0.0,
                "max_rel": 0.0, "over": bad, "elements": int(g.size),
                "atol": 0.0, "rtol": 0.0}
    d = np.abs(g - r)
    den = np.maximum(np.abs(r), 1.0)
    tol = atol + rtol * den
    return {"max_abs": float(d.max()) if d.size else 0.0,
            "max_rel": float((d / np.maximum(np.abs(r), 1e-30)).max())
            if d.size else 0.0,
            "max_rel_scaled": float((d / den).max()) if d.size else 0.0,
            "over": int((d > tol).sum()), "elements": int(d.size),
            "atol": atol, "rtol": rtol}


def load_case(stem, case="default"):
    suffix = "" if case == "default" else "." + case
    ip = os.path.join(MODELS, stem + suffix + ".in.npy")
    rp = os.path.join(MODELS, stem + suffix + ".ref.npz")
    x = np.load(ip)
    z = np.load(rp)
    return x, [z["out%d" % i] for i in range(len(z.files))]


def run_model(path, x, **kw):
    with vc4infer.Engine(path, **kw) as eng:
        outs = eng.run({"x": x}, profile=True)
        names = [o.name for o in eng.outputs]
        return ([np.asarray(outs[n]) for n in names],
                eng.node_backends(), eng.last_profile, dict(eng.last_boundary),
                len(eng.last_profile))


def report(label, results):
    for i, c in enumerate(results):
        if "shape_mismatch" in c:
            print("    out%d SHAPE %s" % (i, c["shape_mismatch"]))
        else:
            print("    out%-2d max abs %.3e  max rel %.3e  over %d/%d"
                  % (i, c["max_abs"], c["max_rel"], c["over"], c["elements"]))
    return all(c.get("over", 1) == 0 and "shape_mismatch" not in c
               for c in results)


# ------------------------------------------------------------------- (a) meta
def test_meta():
    print("\n(a) Shape/Gather/Slice/Mul/Concat -> Resize sizes (host-meta)")
    models = gen.ensure(MODELS)
    p = models["hybrid_meta"]["model"]
    x, ref = load_case("hybrid_meta")
    with vc4infer.Engine(p, mode="hybrid",
                         host_policy="device") as eng:
        plan = eng.plan()
        check(not eng.host_debt(), "no host debt")
        folded = {b.name for b in plan.backends if b.backend == "folded"}
        check({"shape", "slice_nc", "gather_hw", "mul_hw", "concat"} <= folded,
              "Shape/Slice/Gather/Mul/Concat folded at plan time")
        check(any(b.name == "resize" and b.backend == "vc4"
                  for b in plan.backends), "Resize stays on VC4")
        check(eng.host_compute_nodes() == [], "host_compute_nodes == []")
        outs = eng.run({"x": x}, profile=True)
        got = [np.asarray(outs[o.name]) for o in eng.outputs]
        prof = eng.last_profile
        check(eng.host_compute_nodes() == [], "no runtime host compute")
        check(all(p.get("backend") for p in prof),
              "every profile row has a backend")
    res = [compare(g, r) for g, r in zip(got, ref)]
    check(report("meta", res), "structural output matches ReferenceEvaluator")


# ---------------------------------------------------------------- (b) numeric
def test_erf_and_mixed():
    print("\n(b) numeric ops with no VC4 kernel: strict / structural / any")
    models = gen.ensure(MODELS)
    x, ref = load_case("hybrid_erf")
    ep = models["hybrid_erf"]["model"]

    with vc4infer.Engine(ep) as eng:
        try:
            eng.run({"x": x})
            check(False, "strict raises UnsupportedNode")
        except vc4infer.UnsupportedNode as e:
            check(e.node == "erf" and e.op == "Erf",
                  "strict raises UnsupportedNode naming erf/Erf: %s" % e)

    with vc4infer.Engine(ep, mode="hybrid",
                         host_policy="device") as eng:
        debt = eng.host_debt()
        check([b.name for b in debt] == ["erf"],
              "host_debt() == [erf]")
        try:
            eng.plan()
            check(False, "device plan() raises HostNotAllowed")
        except vc4infer.HostNotAllowed as e:
            check(e.node == "erf" and e.op == "Erf",
                  "device plan() names erf: %s" % e)
        try:
            eng.run({"x": x})
            check(False, "device run() raises HostNotAllowed")
        except vc4infer.HostNotAllowed as e:
            check(e.node == "erf", "device run() names erf")
    with vc4infer.Engine(ep, mode="hybrid",
                         host_policy="structural") as eng:
        try:
            eng.run({"x": x})
            check(False, "structural still refuses Erf")
        except vc4infer.HostNotAllowed as e:
            check(e.node == "erf", "structural names erf")

    got, backs, prof, boundary, n = run_model(ep, x, mode="hybrid",
                                              host_policy="any")
    check(any(b.name == "erf" and b.backend == "host-unwritten"
              for b in backs), "any: erf reported host-unwritten")
    check(boundary["tensors"] > 0,
          "any: boundary copies counted (%d)" % boundary["tensors"])
    res = [compare(g, r) for g, r in zip(got, ref)]
    check(report("erf", res), "any: output matches ReferenceEvaluator")

    # ---- mixed: Split/Shape/Gather + Einsum/TopK/NonZero/LSTM
    print("\n(b2) hybrid_mixed: Split + Einsum/TopK/NonZero/LSTM")
    x, ref = load_case("hybrid_mixed")
    mp = models["hybrid_mixed"]["model"]
    with vc4infer.Engine(mp) as eng:
        try:
            eng.run({"x": x})
            check(False, "strict raises on the first unwritten node")
        except vc4infer.UnsupportedNode as e:
            check(e.node == "einsum" and e.op == "Einsum",
                  "strict names einsum/Einsum: %s" % e)
    with vc4infer.Engine(mp, mode="hybrid",
                         host_policy="device") as eng:
        debt = {b.name for b in eng.host_debt()}
        check(debt == {"einsum", "topk", "nonzero", "lstm"},
              "device debt lists Einsum/TopK/NonZero/LSTM: %s"
              % sorted(debt))
        try:
            eng.run({"x": x})
            check(False, "device refuses hybrid_mixed")
        except vc4infer.HostNotAllowed as e:
            check(e.node == "einsum", "device names einsum first")
    got, backs, prof, boundary, n = run_model(mp, x, mode="hybrid",
                                              host_policy="any")
    un = [b.name for b in backs if b.backend == "host-unwritten"]
    check(set(un) == {"einsum", "topk", "nonzero", "lstm"},
          "any: four host-unwritten nodes reported: %s" % un)
    check(boundary["vc4_to_host"] > 0 and boundary["host_to_vc4"] > 0,
          "any: both boundary directions counted (%d/%d)"
          % (boundary["host_to_vc4"], boundary["vc4_to_host"]))
    res = [compare(g, r) for g, r in zip(got, ref)]
    check(report("mixed", res), "any: all outputs match ReferenceEvaluator")


# ------------------------------------------------------------- (c) host-control
def test_if():
    print("\n(c) If with a data-dependent branch (host-control)")
    models = gen.ensure(MODELS)
    p = models["hybrid_if"]["model"]
    with vc4infer.Engine(p, mode="hybrid",
                         host_policy="device") as eng:
        try:
            eng.plan()
            check(False, "device plan() raises NeedsDeviceDispatch for If")
        except vc4infer.NeedsDeviceDispatch as e:
            check(e.node == "if" and e.op == "If",
                  "device plan() names if/If as NeedsDeviceDispatch")
    with vc4infer.Engine(p, mode="hybrid",
                         host_policy="structural") as eng:
        plan = eng.plan()
        ctl = [b for b in plan.backends if b.backend == "host-control"]
        check([b.name for b in ctl] == ["if"],
              "If classified host-control")
        check(not eng.host_debt(), "no host debt")
        for case in ("true", "false"):
            x, ref = load_case("hybrid_if", case)
            outs = eng.run({"x": x}, profile=True)
            got = [np.asarray(outs[o.name]) for o in eng.outputs]
            res = [compare(g, r) for g, r in zip(got, ref)]
            check(report("if/" + case, res),
                  "If(%s) matches ReferenceEvaluator" % case)
            check(eng.last_boundary["vc4_to_host"] >= 1,
                  "If(%s): captured x crossed the boundary" % case)
    with vc4infer.Engine(p) as eng:
        try:
            eng.plan()
            check(False, "strict plan() refuses If")
        except vc4infer.UnsupportedNode as e:
            check(e.node == "if" and e.op == "If",
                  "strict plan() raises UnsupportedNode naming if/If")
        x, _ = load_case("hybrid_if", "true")
        try:
            eng.run({"x": x})
            check(False, "strict still refuses the bool Cast")
        except vc4infer.UnsupportedNode as e:
            check(e.node == "cast" and e.op == "Cast",
                  "strict run() stops at the earlier Cast (unchanged): %s" % e)


# ---------------------------------------------------------------- (d) YOLOv8n
def test_yolo():
    print("\n(d) YOLOv8n: device hybrid must be 100%% compute-on-VC4")
    if not os.path.isfile(YOLO):
        print("  skip (no %s)" % YOLO)
        return
    x = np.random.default_rng(0).standard_normal((1, 3, 160, 160))
    x = x.astype(np.float32)
    import time
    with vc4infer.Engine(YOLO) as eng:
        t0 = time.perf_counter()
        strict = eng.run({"images": x}, profile=True)
        strict_wall = (time.perf_counter() - t0) * 1e3
        strict_name = eng.outputs[0].name
        strict_launch = sum(p["launches"] for p in eng.last_profile)
        strict_gpu = eng.stats["gpu_ms"]
    with vc4infer.Engine(YOLO, mode="hybrid",
                         host_policy="device") as eng:
        check(not eng.host_debt(), "no host debt")
        backs = eng.node_backends()
        tiers = {}
        for b in backs:
            tiers[b.backend] = tiers.get(b.backend, 0) + 1
        check(tiers.get("host-control", 0) == 0 and
              tiers.get("host-unwritten", 0) == 0 and
              tiers.get("host-meta", 0) == 0,
              "zero runtime host nodes")
        compute = tiers.get("vc4", 0)
        check(compute == sum(v for k, v in tiers.items()
                             if k != "folded"),
              "100%% of compute nodes on VC4 (%d)" % compute)
        check(eng.host_compute_nodes() == [], "host_compute_nodes == []")
        t0 = time.perf_counter()
        yh = eng.run({"images": x}, profile=True)
        hybrid_wall = (time.perf_counter() - t0) * 1e3
        hybrid_launch = sum(p["launches"] for p in eng.last_profile)
        hybrid_gpu = eng.stats["gpu_ms"]
        util = eng.utilization(reps=1, feeds={"images": x})
        prof = eng.last_profile
    d = np.abs(np.asarray(strict[strict_name], np.float64) -
               np.asarray(yh[strict_name], np.float64))
    check(d.size > 0 and float(d.max()) == 0.0,
          "hybrid output identical to strict (max abs %.3e)" % float(d.max()))
    check(hybrid_launch == strict_launch,
          "same launch count (%d == %d)" % (hybrid_launch, strict_launch))
    print("    strict: %.1f ms wall / %.1f ms GPU / %d launches"
          % (strict_wall, strict_gpu, strict_launch))
    print("    hybrid: %.1f ms wall / %.1f ms GPU / %d launches"
          % (hybrid_wall, hybrid_gpu, hybrid_launch))
    check(all(p.get("backend") in ("vc4", "folded") for p in prof),
          "profile rows are vc4/folded only")
    check(util["host_compute_nodes"] == 0, "util host_compute_nodes == 0")
    check(util["arithmetic_on_vc4"] == 100.0,
          "arithmetic_on_vc4 == 100%%")
    print("    util: bytes_copied_by_host=%d  qpu_duty=%.1f%%  "
          "launches=%d  launch_path=%.1f%%"
          % (util["bytes_copied_by_host"], util["qpu_duty"],
             util["launches"], util["launch_path_pct"]))
    if util["bytes_copied_by_host"]:
        top = sorted(util["host_copy_sites"], key=lambda s: -s["bytes"])[:8]
        print("    OUTSTANDING host copies:")
        for s in top:
            print("      %-34s %-10s %-14s %8d B x%d"
                  % (s["name"][:34], s["op"], s["site"], s["bytes"],
                     s["count"]))
    # Reference at 5e-4, when the fixed input/reference files are present.
    ref = "/tmp/yolo160_ref.npy"
    inp = "/tmp/yolo160_in.npy"
    if os.path.isfile(ref) and os.path.isfile(inp):
        xr = np.load(inp).astype(np.float32)
        yr = np.load(ref).astype(np.float32)
        with vc4infer.Engine(YOLO, mode="hybrid",
                             host_policy="structural") as eng:
            out = eng.run({eng.inputs[0].name: xr})[eng.outputs[0].name]
        res = compare(out, yr, atol=TOL, rtol=TOL)
        print("    ref@5e-4: max abs %.4g  max rel %.3e  over %d/%d"
              % (res["max_abs"], res["max_rel_scaled"], res["over"],
                 res["elements"]))
        check(res["over"] == 0, "YOLO reference 0/%d over 5e-4"
              % res["elements"])
    else:
        print("  skip reference (no /tmp/yolo160_ref.npy)")


def main():
    print("hybrid acceptance tests: models=%s" % MODELS)
    test_meta()
    test_erf_and_mixed()
    test_if()
    test_yolo()
    print()
    if FAILED:
        print("HYBRID TESTS FAILED: %d check(s)" % len(FAILED))
        for f in FAILED:
            print("  - %s" % f)
        return 1
    print("HYBRID TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
