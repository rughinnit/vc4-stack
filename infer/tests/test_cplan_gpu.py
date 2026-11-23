#!/usr/bin/env python3
"""test_cplan_gpu.py -- a real model, plan on vs plan off, byte-identical.

The host tests in test_cplan.py replay against a stub engine, and the
structure test there only inspects the plan.  Neither executes a real frame,
which is how a weight panel overwritten between recording and replay reached
silicon: the stub conv ignored the panel and the stub allocator never gave a
freed range back to a panel.  This test runs the real engine, on the real
YOLOv8n graph and on a shipped model, and compares every output tensor of a
non-profile replay frame with the same frame without the plan.

It needs the QPUs, so it refuses to run unless VC4_REQUIRE_GPU=1.

    VC4_REQUIRE_GPU=1 python3 infer/tests/test_cplan_gpu.py
    VC4_REQUIRE_GPU=1 python3 infer/tests/test_cplan_gpu.py --models modelB
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "infer", "python"))
sys.path.insert(0, HERE)

from gpu_gate import require_gpu                            # noqa: E402

SHIPPED = os.path.join(REPO, "infer", "packaging", "vc4infer", "data",
                       "models")
BENCH = os.path.expanduser("~/work/onnx/models")
FAILS = []


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)
    return cond


def run_frames(model_path, arr, frames, plan):
    """Run `frames` frames with the plan on or off; return outputs + stats."""
    import vc4infer
    old = {k: os.environ.get(k) for k in ("VC4_CPLAN", "VC4_NO_CPLAN")}
    try:
        if plan:
            os.environ["VC4_CPLAN"] = "1"
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = "1"
            os.environ.pop("VC4_CPLAN", None)
        with vc4infer.Engine(model_path, arena_mb=64, passes="all") as eng:
            name = eng.inputs[0].name
            out = None
            for _ in range(frames):
                out = eng.run({name: arr})
            launches = eng.stats["launches"]
            r = eng._runner
            return ([np.asarray(v) for _, v in sorted(out.items())],
                    launches, r.cplan, r.cplan_deferrals, r.plan_error)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_model(label, path, arr, expect_plan):
    print("%s" % label)
    ref, ref_launches, _, _, _ = run_frames(path, arr, 3, plan=False)
    got, launches, plan, deferrals, err = run_frames(path, arr, 5, plan=True)
    check(err is None, "no fallback (%s)" % err)
    if expect_plan:
        check(plan is not None, "the plan was used")
    if plan is not None:
        check(plan.c_nodes > 0, "%d C nodes, %d Python nodes, %d deferral(s)"
              % (plan.c_nodes, len(plan.py_nodes), deferrals))
    else:
        print("     no plan for this graph (nothing to replay)")
    check(launches == ref_launches,
          "same launch count (%d vs %d)" % (launches, ref_launches))
    same = len(ref) == len(got) and all(
        a.shape == b.shape and np.array_equal(a, b)
        for a, b in zip(ref, got))
    if not same:
        for i, (a, b) in enumerate(zip(ref, got)):
            if not np.array_equal(a, b):
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                print("     output %d: max abs %.3g, %d of %d differ"
                      % (i, d.max(), int(np.count_nonzero(a != b)), a.size))
    check(same, "every output is byte-identical")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", help="comma-separated subset")
    args = ap.parse_args(argv)
    require_gpu("test_cplan_gpu.py")
    todo = []
    for name in ("modelB", "modelC"):
        p = os.path.join(SHIPPED, name + ".onnx")
        if os.path.isfile(p):
            todo.append((name, p,
                         np.load(os.path.join(SHIPPED, name + ".in.npy"))))
    p = os.path.join(SHIPPED, "yolov8n_160.onnx")
    if os.path.isfile(p):
        todo.append(("yolov8n_160", p, np.random.default_rng(0)
                     .standard_normal((1, 3, 160, 160)).astype(np.float32)))
    for size in (96, 128, 160):
        p = os.path.join(BENCH, "yolov8n_%d.onnx" % size)
        if os.path.isfile(p):
            todo.append(("yolov8n_%d" % size, p,
                         np.random.default_rng(0).standard_normal(
                             (1, 3, size, size)).astype(np.float32)))
    if args.models:
        want = {s.strip() for s in args.models.split(",")}
        todo = [t for t in todo if t[0] in want]
    for label, path, arr in todo:
        test_model(label, path, arr,
                   expect_plan=label.startswith("yolov8n") or
                   label == "modelB")
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("OK: %d model(s) byte-identical with the plan on" % len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
