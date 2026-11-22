#!/usr/bin/env python3
"""layout_bench.py -- wall and CPU per frame for the layout pass, A/B.

One model, one pass specification, one timed loop; the numbers are the frame
time (median of ``--repeat`` runs) and the client plus daemon CPU spent in the
same loop (``/proc/<pid>/stat``, the reading ``docs/CPU-BREAKDOWN.md`` uses).
The two readings answer different questions: the pass removes QPU copy
launches, which is wall time, and daemon round trips, which is CPU.

    # only one client may hold the QPU
    ps aux | grep '[v]c4' ; systemctl is-active vc4d
    tools/thermal-guard.sh --max 75 -- python3 infer/tools/onnx/layout_bench.py \\
        infer/packaging/vc4infer/data/models/yolov8n_160.onnx \\
        --passes all --out $TMPDIR/out_all.npy --json $TMPDIR/all.json

Run the same command with ``--passes all,-layout`` for the before figure and
with ``--passes none`` for the unoptimised graph.  ``np.array_equal`` on the
two ``--out`` files is the bit-identity check; the tool does not open the
device twice, because that would be two clients.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))

import vc4infer                                            # noqa: E402

TICKS = os.sysconf("SC_CLK_TCK")


def _cpu(pid=None):
    path = "/proc/self/stat" if pid is None else "/proc/%d/stat" % pid
    try:
        with open(path) as f:
            fields = f.read().split()
    except OSError:
        return 0.0
    return (int(fields[13]) + int(fields[14])) / float(TICKS)


def _daemon_pid(socket):
    want = socket.encode()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % entry, "rb") as f:
                cmd = f.read().split(b"\0")
        except OSError:
            continue
        if cmd and cmd[0].endswith(b"vc4d") and want in cmd:
            return int(entry)
    return None


def _input_shape(model):
    tt = model.graph.input[0].type.tensor_type
    return [int(d.dim_value) if d.HasField("dim_value") else 1
            for d in tt.shape.dim]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--passes", default="all")
    ap.add_argument("--input", default=None,
                    help="an .npy feed; default rng(0) at the declared shape")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--socket", default="/run/vc4d.sock")
    ap.add_argument("--out", default=None, help="write the output tensor here")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    model = onnx.load(args.model)
    name = model.graph.input[0].name
    if args.input:
        x = np.load(args.input).astype(np.float32)
    else:
        x = np.random.default_rng(0).standard_normal(
            _input_shape(model)).astype(np.float32)
    out_name = model.graph.output[0].name

    eng = vc4infer.Engine(args.model, passes=args.passes)
    for _ in range(max(1, args.warmup)):
        out = eng.run({name: x})[out_name]

    pid = _daemon_pid(args.socket)
    c0, d0 = _cpu(), _cpu(pid)
    walls = []
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        out = eng.run({name: x})[out_name]
        walls.append((time.perf_counter() - t0) * 1e3)
    c1, d1 = _cpu(), _cpu(pid)
    eng.run({name: x}, profile=True)
    launches = int(sum(p["launches"] for p in eng.last_profile))
    nodes = len(eng.last_profile)

    res = {
        "model": os.path.basename(args.model),
        "passes": args.passes,
        "wall_ms": statistics.median(walls),
        "wall_min_ms": min(walls),
        "client_cpu_s": (c1 - c0) / len(walls),
        "daemon_cpu_s": (d1 - d0) / len(walls),
        "total_cpu_s": (c1 - c0 + d1 - d0) / len(walls),
        "launches": launches,
        "nodes": nodes,
        "arena_bytes": int(eng._eng.arena_bytes),
        "daemon_pid": pid,
        "repeat": len(walls),
    }
    print(json.dumps(res, indent=1))
    if args.out:
        np.save(args.out, out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
