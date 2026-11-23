#!/usr/bin/env python3
"""mem_plan.py -- print the plan-time liveness and arena layout of a model.

No device, no QPU launch, no ctypes: the planner reads the ONNX graph, computes
a live range per buffer (first definition .. last consumer), assigns offsets
with reuse and reports the peak the graph pool is sized to.

    python3 infer/tools/mem_plan.py MODEL
    python3 infer/tools/mem_plan.py MODEL --passes all --verbose
    python3 infer/tools/mem_plan.py MODEL --legacy     # the A/B control
    python3 infer/tools/mem_plan.py MODEL --json out.json

``--legacy`` is the estimate the pool used before this pass (the walk that
counted every tensor to the end of the graph), so the two numbers on one line
are the before and after of the same graph.

The plan is what ``vc4infer build`` records in ``plan.memory`` and what sizes a
plain ONNX load's graph pool; ``docs/OPTIMIZER.md`` has the measured table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from vc4infer import coverage                        # noqa: E402
from vc4infer import optimizer as O                  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--passes", default="all",
                    help="pass spec (default: all, the artifact's graph)")
    ap.add_argument("--legacy", action="store_true",
                    help="print the pre-liveness estimate as well")
    ap.add_argument("--verbose", action="store_true",
                    help="print every range, smallest last")
    ap.add_argument("--min-bytes", type=int, default=0,
                    help="hide ranges below this size in --verbose")
    ap.add_argument("--json", help="write the plan as JSON to this path")
    args = ap.parse_args(argv)

    model = onnx.load(args.model)
    if args.passes and args.passes != "none":
        res = O.optimise(model, passes=args.passes)
        if res.changed:
            model = res.model
    shapes = coverage.refine_shapes(model)
    plan = O.plan_memory(model, shapes)

    if args.legacy:
        old = O.legacy_pool_bytes(model, shapes)
        print("before %.2f MiB -> after %.2f MiB (%.0f%% of the old estimate)"
              % (old / 2 ** 20, plan.peak_mb, 100.0 * plan.peak_bytes / old))
    print("pool: %.2f MiB (peak + %.2f MiB headroom)"
          % (plan.pool_bytes / 2 ** 20, plan.headroom_bytes / 2 ** 20))
    print(plan.format())
    if args.verbose:
        rs = sorted(plan.ranges, key=lambda r: (-r.bytes, r.offset))
        print("ranges (%d):" % len(rs))
        for r in rs:
            if r.bytes < args.min_bytes:
                continue
            print("  %10d %8.1f KiB %-10s %s [%d..%d]"
                  % (r.offset, r.bytes / 1024.0, r.kind, r.name, r.first,
                     r.last))
    if args.json:
        doc = plan.as_dict()
        doc["model"] = os.path.abspath(args.model)
        doc["passes"] = args.passes
        if args.legacy:
            doc["legacy_peak_bytes"] = O.legacy_pool_bytes(model, shapes)
        with open(args.json, "w", encoding="ascii") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
