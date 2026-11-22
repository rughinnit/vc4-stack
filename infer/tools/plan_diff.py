#!/usr/bin/env python3
"""plan_diff.py -- print the optimiser's plan before and after every pass.

No device, no QPU launch, no ctypes: the tool builds the graph IR, runs the
selected passes, and reports node count, dispatch count, the op histogram and
the modelled launch count at each step.  That is what bisects a regression to
one rewrite before a board run:

    python3 infer/tools/plan_diff.py MODEL --passes all
    python3 infer/tools/plan_diff.py MODEL --passes all,-epilogue
    python3 infer/tools/plan_diff.py MODEL --passes ew --verbose

The launch model is ``vc4infer.optimizer.plan``: one elementwise launch per 12
groups of 256 floats, a conv at two launches, a depthwise conv at one, and NHWC
storage pitched to ``align16(wo)``.  It reproduces every elementwise and conv
row of a recorded runner profile exactly (134 elementwise nodes, 525 launches;
64 convs, 128 launches); ``--profile`` re-checks that against
``yolo_bench.py --json``, which is the command that produces a fresh profile on
the board:

    python3 infer/tools/onnx/yolo_bench.py MODEL --input in.npy --ref ref.npy \\
        --json /tmp/yolo.json
    python3 infer/tools/plan_diff.py MODEL --passes all --profile /tmp/yolo.json

A pass prints ``launches -N`` only when the model says the rewrite removes
launches.  A rewrite that removes a dispatch but keeps its kernel launch says
so (``dispatches -N, launches +0``), which is the honest split for a fused
activation on this engine: the conv kernel has no epilogue.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from vc4infer import optimizer as O          # noqa: E402


def _ops_line(ops):
    return ", ".join("%s %d" % (op, n) for op, n in list(ops.items())[:12])


def _print_plan(plan, indent="  "):
    print("%snodes %d  dispatches %d  launches %d (exact %d + copy ~%d "
          "over %d nodes)"
          % (indent, plan.nodes, plan.dispatches, plan.total, plan.exact,
             plan.copy, plan.copy_nodes))
    print("%sops: %s" % (indent, _ops_line(plan.histogram)))
    for op, n in plan.by_op.items():
        print("%s  %-14s %5d" % (indent, op, n))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Print the vc4infer optimiser plan, pass by pass")
    ap.add_argument("model")
    ap.add_argument("--passes", default="all",
                    help="pass spec (default all; 'none' leaves the graph "
                         "alone; -name disables one)")
    ap.add_argument("--profile", default=None,
                    help="a yolo_bench.py --json profile to check the launch "
                         "model against")
    ap.add_argument("--out", default=None,
                    help="write the optimised model here")
    ap.add_argument("--verbose", action="store_true",
                    help="name every node each rule removed or rewrote")
    ap.add_argument("--json", default=None,
                    help="write the per-pass plan and diffs as JSON")
    args = ap.parse_args(argv)

    model = onnx.load(args.model)
    graph0 = O.Graph.from_model(model)
    plan0 = O.plan_of(graph0)
    nosilu0 = O.plan_of(graph0, implicit=False).total
    print("model %s" % args.model)
    print("baseline")
    _print_plan(plan0, "  ")
    if nosilu0 != plan0.total:
        print("  without the executor's implicit SiLU scan: %d launches"
              % nosilu0)

    seen = []

    def hook(res, graph):
        seen.append((res, O.plan_of(graph)))

    final = O.optimise(model, passes=args.passes, on_pass=hook)
    report = {"model": args.model, "baseline": plan0.summary(), "passes": []}
    prev = plan0
    for res, plan in seen:
        d = prev.diff(plan)
        print("")
        print("after %s: %s" % (res.key, res.title))
        _print_plan(plan, "  ")
        print("  delta: nodes %+d  dispatches %+d  launches %+d "
              "(exact %+d, copy %+d over %+d nodes)"
              % (d["nodes"], d["dispatches"], d["launches"], d["exact"],
                 d["copy"], d["copy_nodes"]))
        if d["ops"]:
            print("  launches by op: %s"
                  % ", ".join("%s %+d" % (k, v) for k, v in d["ops"].items()))
        for st in res.rules:
            if not st.matched:
                continue
            line = "  %-15s matched %d" % (st.rule, st.matched)
            if st.notes:
                line += "  (%s)" % "; ".join(st.notes[:3])
            print(line)
            if args.verbose:
                for nm in st.removed:
                    print("    removed %s" % nm)
                for nm in st.added:
                    print("    added   %s" % nm)
        report["passes"].append({"pass": res.key, "before": prev.summary(),
                                 "after": plan.summary(), "delta": d,
                                 "rules": [st.as_dict() for st in res.rules]})
        prev = plan

    print("")
    print("total: nodes %+d  dispatches %+d  launches %+d"
          % (prev.nodes - plan0.nodes, prev.dispatches - plan0.dispatches,
             prev.total - plan0.total))
    if nosilu0 != plan0.total:
        print("without the implicit SiLU scan: %d -> %d launches (%+d from the "
              "recorded rewrites)" % (nosilu0, O.plan_of(final.graph,
                                                        implicit=False).total,
                                     O.plan_of(final.graph, implicit=False).total
                                     - nosilu0))

    if args.profile:
        with open(args.profile) as f:
            prof = json.load(f)
        rows = prof.get("profile", prof)
        checked, exact_bad, copy_bad = O.compare_profile(plan0, rows)
        print("")
        print("launch model vs %s: %d nodes checked, %d exact-class "
              "mismatches, %d copy-class mismatches"
              % (args.profile, checked, len(exact_bad), len(copy_bad)))
        for row in exact_bad:
            print("  MISMATCH %s %s modelled %d recorded %d" % row)
        if exact_bad:
            print("  the model no longer matches the executor; fix plan.py "
                  "before trusting a delta")
            return 2

    if args.out:
        onnx.save(final.model, args.out)
        print("wrote %s (%d nodes)" % (args.out, len(final.graph.nodes)))

    if args.json:
        report["total"] = {"nodes": prev.nodes - plan0.nodes,
                           "dispatches": prev.dispatches - plan0.dispatches,
                           "launches": prev.total - plan0.total}
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
