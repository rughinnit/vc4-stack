#!/usr/bin/env python3
"""check_plan_equiv.py -- byte equality for the frozen execution plan.

Runs each model in two processes - once with the plan disabled
($VC4_NO_CPLAN=1) and once with it enabled - and compares the graph outputs
with ``np.array_equal``.  The enabled run executes three frames, so the
comparison covers a replay frame, not the frame the plan was recorded from.
The child reports whether a plan was used and whether it fell back, so a
model that silently took the Python path cannot be counted as a pass.

Needs the board: this is the first silicon gate for docs/CPU-BREAKDOWN.md
section 6.  One client, nothing else on the GPU.

    tools/thermal-guard.sh --max 75
    python3 tools/cost/check_plan_equiv.py
    python3 tools/cost/check_plan_equiv.py --models modelA,modelB,modelC
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHIPPED = os.path.join(REPO, "infer", "packaging", "vc4infer", "data",
                       "models")
BENCH = os.path.expanduser("~/work/onnx/models")

CHILD = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(sys.argv[1], "infer", "python"))
import onnx, vc4infer
model, inp, dest = sys.argv[2:5]
m = onnx.load(model)
name = m.graph.input[0].name
dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
x = np.load(inp).astype(np.float32) if inp != "-" else \
    np.zeros(dims, np.float32)
eng = vc4infer.Engine(model, arena_mb=64, passes="all")
try:
    res = None
    for _ in range(3):
        res = eng.run({name: x})
    arr = np.asarray(list(res.values())[0])
    np.save(dest, arr)
    r = eng._runner
    plan = r.cplan.summary() if r.cplan is not None else None
    print(json.dumps({"used": plan is not None, "plan": plan,
                      "plan_error": r.plan_error,
                      "launches": int(eng.stats["launches"]),
                      "shape": list(arr.shape)}))
finally:
    eng.close()
'''


def models(sel):
    out = []
    for name in ("modelA", "modelB", "modelC"):
        p = os.path.join(SHIPPED, name + ".onnx")
        if os.path.isfile(p):
            out.append((name, p, os.path.join(SHIPPED, name + ".in.npy")))
    p = os.path.join(SHIPPED, "yolov8n_160.onnx")
    if os.path.isfile(p):
        out.append(("yolov8n_160", p, "-"))
    for size in (96, 128, 160):
        p = os.path.join(BENCH, "yolov8n_%d.onnx" % size)
        if os.path.isfile(p):
            out.append(("yolov8n_%d" % size, p, "-"))
    if sel:
        want = {s.strip() for s in sel.split(",") if s.strip()}
        out = [m for m in out if m[0] in want]
    return out


def run_child(tmp, tag, model, inp, env_extra):
    dest = os.path.join(tmp, "%s.npy" % tag)
    env = dict(os.environ)
    env.update(env_extra)
    env["TMPDIR"] = tmp
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, REPO, model, inp, dest],
        capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return dest, json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", help="comma-separated subset")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args(argv)
    todo = models(args.models)
    if not todo:
        print("no models found under %s" % SHIPPED)
        return 2
    tmp = args.tmp or tempfile.mkdtemp(prefix="cplan.")
    failures = 0
    for name, model, inp in todo:
        try:
            py, ipy = run_child(tmp, name + ".py", model, inp,
                                {"VC4_NO_CPLAN": "1"})
            pl, ipl = run_child(tmp, name + ".pl", model, inp, {})
        except Exception as exc:                          # noqa: BLE001
            print("FAIL %-12s %s" % (name, exc))
            failures += 1
            continue
        a = np.load(py)
        b = np.load(pl)
        same = a.shape == b.shape and np.array_equal(a, b)
        note = ""
        if not ipl["used"]:
            note = " (no plan: %s)" % (ipl["plan_error"] or "not built")
        elif ipl["plan_error"]:
            note = " (fell back: %s)" % ipl["plan_error"]
        print("%-4s %-12s %s  plan %s%s"
              % ("PASS" if same else "FAIL", name,
                 "byte-identical" if same else "DIFFERS", ipl["plan"], note))
        if not same:
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print("     max abs %.3g, %d elements differ"
                  % (d.max(), int(np.count_nonzero(a != b))))
            failures += 1
        del ipy
    print()
    if failures:
        print("FAILED: %d model(s)" % failures)
        return 1
    print("OK: %d model(s) byte-identical with and without the plan"
          % len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
