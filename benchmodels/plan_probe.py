#!/usr/bin/env python3
"""Record the first refusal of an ONNX graph exactly as published.

Opens the graph in strict mode and asks for a plan.  Two outcomes matter:

* the constructor refuses before a plan exists (arena / pool floor), or
* the plan refuses at a named node and operator.

No inference is run.  Output is one JSON record per file.
"""
import json
import os
import sys

import onnx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "infer", "python"))
import vc4infer                                                   # noqa: E402


def main():
    model, out = sys.argv[1], sys.argv[2]
    rec = {"model": os.path.basename(model), "file": model}
    m = onnx.load(model)
    rec["nodes"] = len(m.graph.node)
    rec["opset"] = [o.version for o in m.opset_import if o.domain == ""]
    eng = None
    try:
        eng = vc4infer.Engine(model, arena_mb=96)
    except Exception as e:                                      # noqa: BLE001
        rec["refusal"] = "%s: %s" % (type(e).__name__, e)
        rec["stage"] = "engine-open"
    if eng is not None:
        try:
            p = eng.plan()
            rec["plan"] = {"ok": True, "dispatches": p.dispatches,
                           "tiers": p.to_dict()["tiers"]}
        except Exception as e:                                  # noqa: BLE001
            rec["plan"] = {"ok": False, "refusal": "%s: %s"
                           % (type(e).__name__, e)}
        try:
            eng.close()
        except Exception:                                       # noqa: BLE001
            pass
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("%-34s %s" % (rec["model"],
                        rec.get("refusal") or
                        (rec["plan"]["refusal"] if not rec["plan"]["ok"]
                         else "plans (%d dispatches)"
                         % rec["plan"]["dispatches"])))


if __name__ == "__main__":
    main()
