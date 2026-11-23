#!/usr/bin/env python3
"""bench_cplan.py -- host cost of a frame with the device calls stubbed out.

This is not the client/daemon measurement of docs/CPU-BREAKDOWN.md; that one
needs the QPUs and a client.  What this measures is the part a frozen plan
changes: the Python executor's per-node work on a real graph, with every
engine call reduced to a recorded no-op.  The stub removes the engine's own
cost identically from both sides, so the difference between the two columns
is the executor and node-host-code work the plan removes - the two rows the
account lists as addressable.

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/bench_cplan.py \
        --model infer/packaging/vc4infer/data/models/yolov8n_160.onnx
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "infer", "python"))
sys.path.insert(0, HERE)

from test_cplan import FakeEngine                              # noqa: E402
from vc4infer._onnxrun import Runner                           # noqa: E402


class NullEngine(FakeEngine):
    """Every device call is a counter bump; no arithmetic, no bytes moved."""

    def elementwise(self, op, a, out, n, b=None):
        self._call("elementwise", int(op))

    def unary(self, op, a, out, n, p0=0.0, p1=0.0, p2=0.0):
        self._call("unary", int(op))

    def copy2d_x(self, a, out, rows, vpr, srow, drow, svec, dvec, slane,
                 nlane=16, nblk=1, sbstep=64, dbstep=64, units=4, depth=16):
        self._call("copy2d_x", int(rows), int(vpr))

    def gather(self, tab, out, n, fill=0.0):
        self._call("gather", int(n))

    def fill2d(self, out, rows, vpr, drow, dvec, value=0.0):
        # the real border fill writes -inf for MaxPool; a stub that leaves it
        # there would trip the Softmax finite-input guard
        self._call("fill2d", 0)

    def matmul(self, a, b, c, M, N, K, layout=0, act=0):
        self._call("matmul", int(M))


def feed_for(model):
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return {model.graph.input[0].name:
            np.zeros(shape, np.float32)}


def frame_times(model_path, runs, plan):
    old = os.environ.get("VC4_NO_CPLAN")
    if plan:
        os.environ.pop("VC4_NO_CPLAN", None)
    else:
        os.environ["VC4_NO_CPLAN"] = "1"
    try:
        eng = NullEngine(arena=96 << 20)
        r = Runner(model_path, eng, passes="all")
        feed = feed_for(r.model)
        # Frame 0 packs the panels; a plan is not recorded from it.  Frame 1
        # is the recording frame, and that is the build cost.
        r.run(feed)
        build = 0.0
        times = []
        for i in range(runs + 1):
            t0 = time.thread_time()
            n0 = len(eng.calls)
            r.run(feed)
            dt = time.thread_time() - t0
            if i == 0:
                build = dt
            else:
                times.append(dt)
        return {"build": build, "median": statistics.median(times),
                "min": min(times), "max": max(times),
                "per_frame": len(eng.calls) - n0,
                "plan": r.cplan.summary() if r.cplan else None}
    finally:
        if old is None:
            os.environ.pop("VC4_NO_CPLAN", None)
        else:
            os.environ["VC4_NO_CPLAN"] = old


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.path.join(
        REPO, "infer", "packaging", "vc4infer", "data", "models",
        "yolov8n_160.onnx"))
    ap.add_argument("--runs", type=int, default=9)
    args = ap.parse_args(argv)
    print("model: %s  (device calls stubbed; engine time excluded)"
          % args.model)
    py = frame_times(args.model, args.runs, plan=False)
    pl = frame_times(args.model, args.runs, plan=True)
    print("%-10s %10s %10s %10s" % ("", "host ms", "min", "max"))
    print("%-10s %10.2f %10.2f %10.2f" % ("python", py["median"] * 1e3,
                                          py["min"] * 1e3,
                                          py["max"] * 1e3))
    print("%-10s %10.2f %10.2f %10.2f" % ("plan", pl["median"] * 1e3,
                                          pl["min"] * 1e3,
                                          pl["max"] * 1e3))
    if pl["plan"]:
        s = pl["plan"]
        print("plan: %d steps, %d C nodes with %d ops, %d Python nodes"
              % (s["steps"], s["c_nodes"], s["c_ops"], s["py_nodes"]))
    print("calls/frame: python %d, plan %d"
          % (py["per_frame"], pl["per_frame"]))
    delta = py["median"] - pl["median"]
    print("host delta: %.2f ms/frame (%.1f %%)"
          % (delta * 1e3, 100.0 * delta / py["median"]))
    print("build frame: python %.2f ms, plan %.2f ms"
          % (py["build"] * 1e3, pl["build"] * 1e3))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
