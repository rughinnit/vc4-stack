#!/usr/bin/env python3
"""Run one .tflite model, optionally with the VC4 delegate, and save outputs.

Kept in a separate process so the delegate's stderr partition report can be
captured by the parent test without redirecting file descriptors in-process.
"""
import argparse
import time

import numpy as np

from ai_edge_litert import interpreter as lrt

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--input", required=False)
ap.add_argument("--output", required=True)
ap.add_argument("--delegate", default=None)
ap.add_argument("--mode", default="auto")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--alloc-only", action="store_true")
a = ap.parse_args()

delegates = []
if a.delegate:
    delegates = [lrt.load_delegate(a.delegate, {"mode": a.mode, "verbose": "1"})]

interp = lrt.Interpreter(
    model_path=a.model,
    experimental_delegates=delegates,
    experimental_op_resolver_type=lrt.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
interp.allocate_tensors()
if a.alloc_only:
    print("VC4_ALLOC_OK")
    raise SystemExit(0)
detail = interp.get_input_details()[0]
x = np.load(a.input).astype(detail["dtype"])
interp.set_tensor(detail["index"], x)
for _ in range(a.warmup):
    interp.invoke()
times = []
for _ in range(a.repeat):
    t0 = time.perf_counter()
    interp.invoke()
    times.append(time.perf_counter() - t0)
outs = [interp.get_tensor(d["index"]) for d in interp.get_output_details()]
np.savez(a.output, *outs)
if times:
    print("VC4_RUN_MS %.3f" % (1000.0 * min(times)))
