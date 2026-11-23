#!/usr/bin/env python3
"""artifact_plan.py -- an engine artifact with the frozen plan, checked and timed.

Two questions, measured in that order:

  --check   the artifact's plan replays the same frame as the ONNX path with
            the plan on, byte for byte, and the artifact still runs the Python
            path under VC4_NO_CPLAN=1;
  --time    artifact load, first result, steady wall and CPU per frame, and
            launches against the ONNX path with the plan on.

Needs the board.  One client at a time; run under tools/thermal-guard.sh.

    tools/thermal-guard.sh --max 75 -- \
        python3 tools/cost/artifact_plan.py --check
    tools/thermal-guard.sh --max 75 -- \
        python3 tools/cost/artifact_plan.py --time --runs 8
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "infer", "python"))
sys.path.insert(0, HERE)

import common                                              # noqa: E402
import numpy as np                                         # noqa: E402

SHIPPED = os.path.join(REPO, "infer", "packaging", "vc4infer", "data",
                       "models")
BENCH = os.path.expanduser("~/work/onnx/models")
ART = os.path.expanduser("~/scratch/artifacts")
FAILS = []


def check(cond, what, detail=""):
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", what,
                         ("  [%s]" % detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(what)
    return cond


def models(names):
    from onnx import load
    out = []
    for name in names:
        if name.startswith("yolov8n"):
            size = int(name.split("_")[1])
            path = os.path.join(BENCH, "yolov8n_%d.onnx" % size)
            arr = np.random.default_rng(0).standard_normal(
                (1, 3, size, size)).astype(np.float32)
        else:
            path = os.path.join(SHIPPED, name + ".onnx")
            arr = np.load(os.path.join(SHIPPED, name + ".in.npy"))
        inp = load(path).graph.input[0]
        out.append({
            "name": name, "onnx": path,
            "artifact": os.path.join(ART, name + ".vc4x"),
            "feed": {inp.name: arr},
        })
    return out


def onnx_frame(m, frames=4):
    """Run the ONNX graph with the plan on; return (outputs, launches)."""
    import vc4infer
    with vc4infer.Engine(m["onnx"], arena_mb=64, passes="all") as eng:
        out = None
        for _ in range(frames):
            out = eng.run(m["feed"])
        return ([np.asarray(v) for _, v in sorted(out.items())],
                eng.stats["launches"], eng._runner.cplan is not None)


def artifact_frame(m, frames=2, no_plan=False):
    import vc4infer
    old = os.environ.get("VC4_NO_CPLAN")
    if no_plan:
        os.environ["VC4_NO_CPLAN"] = "1"
    else:
        os.environ.pop("VC4_NO_CPLAN", None)
    try:
        with vc4infer.Engine(m["artifact"], arena_mb=64) as eng:
            out = None
            for _ in range(frames):
                out = eng.run(m["feed"])
            return ([np.asarray(v) for _, v in sorted(out.items())],
                    eng.stats["launches"], eng._runner.cplan is not None,
                    eng._runner.plan_error)
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old


def check_model(m):
    print(m["name"])
    ref, launches_ref, plan_ref = onnx_frame(m)
    got, launches, plan, err = artifact_frame(m)
    if plan_ref:
        check(err is None, "the artifact's plan loaded (%s)" % err)
        check(plan, "the artifact replayed its plan, not the Python path")
    else:
        # A graph with no representable node has nothing to freeze; both
        # paths are the Python executor and the loader says so.
        check(not plan, "no plan on either path (nothing to replay)")
        check(err is not None, "the loader says why: %s" % err)
    same = len(ref) == len(got) and all(
        a.shape == b.shape and np.array_equal(a, b)
        for a, b in zip(ref, got))
    if not same:
        for i, (a, b) in enumerate(zip(ref, got)):
            if not np.array_equal(a, b):
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                print("     output %d: max abs %.3g, %d of %d differ"
                      % (i, d.max(), int(np.count_nonzero(a != b)), a.size))
    check(same, "byte-identical to the ONNX path with the plan on")
    check(launches == launches_ref,
          "same launches per frame (%d vs %d)" % (launches, launches_ref))
    py, launches_py, plan_py, err_py = artifact_frame(m, no_plan=True)
    check(not plan_py, "VC4_NO_CPLAN=1 leaves the artifact on Python")
    check(err_py is not None, "the refusal is recorded: %s" % err_py)
    same_py = len(ref) == len(py) and all(
        a.shape == b.shape and np.array_equal(a, b)
        for a, b in zip(ref, py))
    check(same_py, "the Python path is byte-identical too")
    check(launches_py == launches_ref,
          "same launches on the Python path (%d vs %d)"
          % (launches_py, launches_ref))


def timed(fn):
    """(seconds, result) for a call."""
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def cell(m, runs):
    import vc4infer
    # -- artifact: construction is load, the first run is first result
    old = os.environ.get("VC4_NO_CPLAN")
    os.environ.pop("VC4_NO_CPLAN", None)
    try:
        load_s, eng = timed(lambda: vc4infer.Engine(m["artifact"],
                                                     arena_mb=64))
        dp = common.vc4d_pid()
        c0 = common.proc_cpu_s()
        d0 = common.proc_cpu_s(dp) if dp else 0.0
        t0 = time.perf_counter()
        eng.run(m["feed"])
        first_s = time.perf_counter() - t0
        c1 = common.proc_cpu_s()
        d1 = common.proc_cpu_s(dp) if dp else 0.0
        first_cpu = (c1 - c0) + max(0.0, d1 - d0)
        c0, d0 = c1, d1
        walls = []
        for _ in range(runs):
            t = time.perf_counter()
            eng.run(m["feed"])
            walls.append((time.perf_counter() - t) * 1e3)
        c1 = common.proc_cpu_s()
        d1 = common.proc_cpu_s(dp) if dp else 0.0
        launches = eng.stats["launches"]
        eng.close()
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old
    artifact = {
        "load_s": load_s, "first_s": first_s, "first_cpu_s": first_cpu,
        "first_total_s": load_s + first_s,
        "wall_ms": statistics.median(walls),
        "client_cpu_s": (c1 - c0) / runs,
        "daemon_cpu_s": max(0.0, d1 - d0) / runs,
        "launches": launches,
        "runs": runs,
    }
    artifact["total_cpu_s"] = (artifact["client_cpu_s"] +
                               artifact["daemon_cpu_s"])
    # -- ONNX with the plan on: three warm-up frames (panels, record, replay)
    setup_s, eng = timed(lambda: vc4infer.Engine(m["onnx"], arena_mb=64,
                                                 passes="all"))
    t0 = time.perf_counter()
    c0 = common.proc_cpu_s()
    d0 = common.proc_cpu_s(dp) if dp else 0.0
    eng.run(m["feed"])
    first_s = time.perf_counter() - t0
    c1 = common.proc_cpu_s()
    d1 = common.proc_cpu_s(dp) if dp else 0.0
    first_cpu = (c1 - c0) + max(0.0, d1 - d0)
    for _ in range(2):
        eng.run(m["feed"])
    c0 = common.proc_cpu_s()
    d0 = common.proc_cpu_s(dp) if dp else 0.0
    walls = []
    for _ in range(runs):
        t = time.perf_counter()
        eng.run(m["feed"])
        walls.append((time.perf_counter() - t) * 1e3)
    c1 = common.proc_cpu_s()
    d1 = common.proc_cpu_s(dp) if dp else 0.0
    launches_onnx = eng.stats["launches"]
    plan_on = eng._runner.cplan is not None
    eng.close()
    onnx = {
        "setup_s": setup_s, "first_s": first_s, "first_cpu_s": first_cpu,
        "first_total_s": setup_s + first_s,
        "wall_ms": statistics.median(walls),
        "client_cpu_s": (c1 - c0) / runs,
        "daemon_cpu_s": max(0.0, d1 - d0) / runs,
        "launches": launches_onnx,
        "plan": plan_on,
        "runs": runs,
    }
    onnx["total_cpu_s"] = onnx["client_cpu_s"] + onnx["daemon_cpu_s"]

    # -- the same artifact with the plan disabled: the Python path it ran
    # before frozen plans, which is the baseline the plan has to beat
    old = os.environ.get("VC4_NO_CPLAN")
    os.environ["VC4_NO_CPLAN"] = "1"
    try:
        load_py, eng = timed(lambda: vc4infer.Engine(m["artifact"],
                                                     arena_mb=64))
        eng.run(m["feed"])                 # first result, not timed
        c0 = common.proc_cpu_s()
        d0 = common.proc_cpu_s(dp) if dp else 0.0
        walls = []
        for _ in range(runs):
            t = time.perf_counter()
            eng.run(m["feed"])
            walls.append((time.perf_counter() - t) * 1e3)
        c1 = common.proc_cpu_s()
        d1 = common.proc_cpu_s(dp) if dp else 0.0
        launches_py = eng.stats["launches"]
        plan_py = eng._runner.cplan is not None
        eng.close()
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old
    py = {
        "load_s": load_py, "wall_ms": statistics.median(walls),
        "client_cpu_s": (c1 - c0) / runs,
        "daemon_cpu_s": max(0.0, d1 - d0) / runs,
        "launches": launches_py, "plan": plan_py, "runs": runs,
    }
    py["total_cpu_s"] = py["client_cpu_s"] + py["daemon_cpu_s"]
    return artifact, onnx, py


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = (args.models.split(",") if args.models else
             ["modelA", "modelB", "modelC", "yolov8n_96", "yolov8n_128",
              "yolov8n_160"])
    todo = models([n.strip() for n in names if n.strip()])
    do_check = args.check or not args.time
    do_time = args.time or not args.check
    for m in todo:
        missing = [p for p in (m["onnx"], m["artifact"]) if not os.path.isfile(p)]
        if missing:
            print("SKIP %s: missing %s" % (m["name"], missing[0]))
            continue
        if do_check:
            check_model(m)
    if do_time:
        rows = {}
        print("\n%-12s %7s %8s %8s %8s %8s %8s %6s" %
              ("model", "load s", "first s", "wall ms", "client", "daemon",
               "total", "launch"))
        for m in todo:
            if not os.path.isfile(m["artifact"]):
                continue
            a, o, py = cell(m, args.runs)
            rows[m["name"]] = {"artifact": a, "onnx_plan": o,
                               "artifact_python": py}
            print("%-12s %7.2f %8.2f %8.1f %8.4f %8.4f %8.4f %6d"
                  % (m["name"], a["load_s"], a["first_s"], a["wall_ms"],
                     a["client_cpu_s"], a["daemon_cpu_s"],
                     a["total_cpu_s"], a["launches"]))
            print("%-12s %7.2f %8.2f %8.1f %8.4f %8.4f %8.4f %6d  (ONNX, plan %s)"
                  % ("", o["setup_s"], o["first_s"], o["wall_ms"],
                     o["client_cpu_s"],
                     o["daemon_cpu_s"], o["total_cpu_s"], o["launches"],
                     "on" if o["plan"] else "off"))
            print("%-12s %7.2f %8s %8.1f %8.4f %8.4f %8.4f %6d  (artifact, no plan)"
                  % ("", py["load_s"], "", py["wall_ms"], py["client_cpu_s"],
                     py["daemon_cpu_s"], py["total_cpu_s"], py["launches"]))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1, sort_keys=True)
            print("\nwrote %s" % args.out)
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
