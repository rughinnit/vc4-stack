#!/usr/bin/env python3
"""Where one YOLOv8n frame's CPU time goes, on the QPU path.

    breakdown.py --size 160 --runs 10 [--out FILE]

The account has four exact layers and one sampled one:

* ``/proc/<pid>/stat`` utime+stime around the timed loop gives client and
  daemon CPU, the same readings the cost page uses;
* the remote backend's phase counters (``VC4INFER_PROF``) give thread CPU
  inside the launch path: dirty-range cleaning, uniform address mask, the
  socket round trip, and the tail after it; plus upload/download cleaning and
  copying, and kernel loads;
* wrapping every C binding method with ``time.thread_time()`` gives the C
  total, so ``C total - remote phases`` is C op setup (kernel choice, uniform
  fill, scratch bookkeeping) that never reaches the socket;
* wrapping the executor and the layout helpers gives per-node Python time,
  split into layout copies (NCHW/NHWC conversion, padding, gathers) and the
  rest of the node body.

The four add up to the client figure; the daemon figure is added unchanged.
"""
import argparse
import ctypes
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "infer", "python"))
sys.path.insert(0, HERE)

import common  # noqa: E402

import numpy as np  # noqa: E402
import vc4infer  # noqa: E402


class Prof(ctypes.Structure):
    _fields_ = [
        ("on", ctypes.c_int),
        ("pre", ctypes.c_uint64), ("mark", ctypes.c_uint64),
        ("clean", ctypes.c_uint64), ("mask", ctypes.c_uint64),
        ("rt", ctypes.c_uint64), ("tail", ctypes.c_uint64),
        ("up_clean", ctypes.c_uint64), ("up_copy", ctypes.c_uint64),
        ("down_clean", ctypes.c_uint64), ("down_copy", ctypes.c_uint64),
        ("load", ctypes.c_uint64), ("flush", ctypes.c_uint64),
        ("pad_zero", ctypes.c_uint64), ("pad_copy", ctypes.c_uint64),
        ("civac_off", ctypes.c_uint64), ("panel_zero", ctypes.c_uint64),
        ("pad_zero_bytes", ctypes.c_uint64),
        ("pad_copy_bytes", ctypes.c_uint64),
        ("civac_off_bytes", ctypes.c_uint64),
        ("n_launch", ctypes.c_uint64), ("n_upload", ctypes.c_uint64),
        ("n_download", ctypes.c_uint64), ("n_load", ctypes.c_uint64),
        ("n_flush", ctypes.c_uint64),
        ("n_batch", ctypes.c_uint64), ("n_batch_launch", ctypes.c_uint64),
        ("clean_bytes", ctypes.c_uint64),
        ("up_bytes", ctypes.c_uint64), ("down_bytes", ctypes.c_uint64),
    ]


LAYOUT = ("to_nhwc", "nchw", "_to_nchw_buf", "_prepad", "_repad_nhwc",
          "_expand_to", "_gather_flat", "_slice_ch", "_compact",
          "download_nhwc", "upload_pitched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=160, choices=(96, 128, 160))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    eng = vc4infer.Engine(common.vc4_model(args.size), arena_mb=64,
                          passes="all")
    x = np.random.default_rng(0).standard_normal(
        (1, 3, args.size, args.size)).astype(np.float32)
    for _ in range(args.warmup):
        eng.run(x)

    bind = eng._eng
    runner = eng._runner

    # --- phase counters from the remote backend
    lib = bind.lib
    lib.vc4_prof_ptr.restype = ctypes.POINTER(Prof)
    lib.vc4_prof_reset.argtypes = []

    # --- C-time wrappers: every method that calls into the library
    c_calls = {"n": 0, "cpu": 0.0, "by": {}}
    wrapped = []
    for name in dir(bind):
        if name.startswith("_") or name in ("close", "stats"):
            continue
        attr = getattr(bind, name)
        if not callable(attr):
            continue
        orig = attr

        def make(orig, name):
            def f(*a, **k):
                t = time.thread_time()
                try:
                    return orig(*a, **k)
                finally:
                    dt = time.thread_time() - t
                    c_calls["n"] += 1
                    c_calls["cpu"] += dt
                    c_calls["by"][name] = c_calls["by"].get(name, 0.0) + dt
            return f
        setattr(bind, name, make(orig, name))
        wrapped.append(name)

    # --- layout wrappers inside the executor
    layout = {"cpu": 0.0, "by": {}}
    for name in LAYOUT:
        if not hasattr(runner, name):
            continue
        orig = getattr(runner, name)

        def make(orig, name):
            def f(*a, **k):
                t = time.thread_time()
                try:
                    return orig(*a, **k)
                finally:
                    dt = time.thread_time() - t
                    layout["cpu"] += dt
                    layout["by"][name] = layout["by"].get(name, 0.0) + dt
            return f
        setattr(runner, name, make(orig, name))

    # --- per-node account
    per_op = {}
    dispatch = runner._dispatch

    def node_dispatch(node):
        c0 = c_calls["cpu"]
        t = time.thread_time()
        try:
            return dispatch(node)
        finally:
            dt = time.thread_time() - t
            dc = c_calls["cpu"] - c0
            e = per_op.setdefault(node.op_type,
                                  {"cpu": 0.0, "c": 0.0, "nodes": 0})
            e["cpu"] += dt
            e["c"] += dc
            e["nodes"] += 1
    runner._dispatch = node_dispatch

    # --- exact C totals per frame; reset the phase counters for the loop
    p = lib.vc4_prof_ptr().contents
    if not p.on:
        raise SystemExit("set VC4INFER_PROF=1 for the remote phase counters")
    lib.vc4_prof_reset()
    c_calls["n"] = 0
    c_calls["cpu"] = 0.0
    c_calls["by"] = {}
    layout["cpu"] = 0.0
    layout["by"] = {}
    per_op.clear()

    dp = common.vc4d_pid()
    c0 = common.proc_cpu_s()
    d0 = common.proc_cpu_s(dp) if dp else 0.0
    t0 = time.thread_time()
    walls = []
    for _ in range(args.runs):
        t = time.perf_counter()
        eng.run(x)
        walls.append((time.perf_counter() - t) * 1e3)
    run_thread = time.thread_time() - t0
    c1 = common.proc_cpu_s()
    d1 = common.proc_cpu_s(dp) if dp else 0.0
    pr = lib.vc4_prof_ptr().contents

    n = float(args.runs)
    per_node_cpu = sum(e["cpu"] for e in per_op.values())
    skeleton = run_thread - per_node_cpu

    res = {
        "size": args.size,
        "runs": args.runs,
        "wall_ms_median": statistics.median(walls),
        "fps": 1000.0 / statistics.median(walls),
        "per_frame": {
            "client_cpu_s": (c1 - c0) / n,
            "daemon_cpu_s": max(0.0, (d1 - d0)) / n,
            "total_cpu_s": ((c1 - c0) + max(0.0, d1 - d0)) / n,
        },
        "client": {
            "c_total_s": c_calls["cpu"] / n,
            "python_s": (run_thread - c_calls["cpu"]) / n,
            "thread_sum_s": run_thread / n,
            "executor_s": skeleton / n,
            "node_python_s": (per_node_cpu - c_calls["cpu"]) / n,
            "layout_s": layout["cpu"] / n,
            "op_host_s": (per_node_cpu - c_calls["cpu"] - layout["cpu"]) / n,
            "c_calls_per_frame": c_calls["n"] / n,
            "by": {k: v / n for k, v in sorted(c_calls["by"].items(),
                                               key=lambda kv: -kv[1])},
            "layout_by": {k: v / n for k, v in sorted(
                layout["by"].items(), key=lambda kv: -kv[1])},
        },
        "remote": {
            "pre_s": pr.pre / 1e9 / n, "mark_s": pr.mark / 1e9 / n,
            "clean_s": pr.clean / 1e9 / n, "mask_s": pr.mask / 1e9 / n,
            "roundtrip_s": pr.rt / 1e9 / n, "tail_s": pr.tail / 1e9 / n,
            "up_clean_s": pr.up_clean / 1e9 / n,
            "up_copy_s": pr.up_copy / 1e9 / n,
            "down_clean_s": pr.down_clean / 1e9 / n,
            "down_copy_s": pr.down_copy / 1e9 / n,
            "load_s": pr.load / 1e9 / n, "flush_s": pr.flush / 1e9 / n,
            "pad_zero_s": pr.pad_zero / 1e9 / n,
            "pad_copy_s": pr.pad_copy / 1e9 / n,
            "civac_off_s": pr.civac_off / 1e9 / n,
            "panel_zero_s": pr.panel_zero / 1e9 / n,
            "pad_zero_bytes": pr.pad_zero_bytes / n,
            "pad_copy_bytes": pr.pad_copy_bytes / n,
            "civac_off_bytes": pr.civac_off_bytes / n,
            "launches_per_frame": pr.n_launch / n,
            "batch_requests_per_frame": pr.n_batch / n,
            "requests_per_frame": (pr.n_batch / n) if pr.n_batch
                                  else (pr.n_launch / n),
            "uploads_per_frame": pr.n_upload / n,
            "downloads_per_frame": pr.n_download / n,
            "loads_per_frame": pr.n_load / n,
            "clean_lines_per_frame": pr.clean_bytes / n,
            "up_bytes_per_frame": pr.up_bytes / n,
            "down_bytes_per_frame": pr.down_bytes / n,
        },
        "nodes": {k: {"nodes_per_frame": v["nodes"] / n,
                      "cpu_s": v["cpu"] / n, "c_s": v["c"] / n,
                      "non_c_s": (v["cpu"] - v["c"]) / n}
                  for k, v in sorted(per_op.items(),
                                     key=lambda kv: -kv[1]["cpu"])},
    }
    remote_sum = (pr.pre + pr.mark + pr.clean + pr.mask + pr.rt + pr.tail +
                  pr.up_clean + pr.up_copy + pr.down_clean + pr.down_copy +
                  pr.load + pr.flush) / 1e9 / n
    res["client"]["remote_s"] = remote_sum
    res["client"]["c_setup_s"] = c_calls["cpu"] / n - remote_sum
    print("BREAKDOWN_JSON " + json.dumps(res, sort_keys=True))
    eng.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
