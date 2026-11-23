#!/usr/bin/env python3
"""overlap.py - wall, duty and host cost per frame, synchronous vs pipelined.

The number this page exists for is the duty cycle: QPU time divided by wall
time.  A serialised frame (submit, wait, submit, wait) has
wall = QPU + host, so duty = QPU/(QPU+host); a frame whose host work is hidden
has wall ~= QPU and duty -> 1.  This script measures both paths on the same
input and reports them side by side, with the client's and the daemon's CPU
time per frame and the number of daemon requests per frame.

usage:
  tools/overlap.py --sizes 96,128,160 --frames 20
  tools/overlap.py --sizes 160 --check          # byte-identical outputs
  tools/overlap.py --sizes 160 --sync-only

Run it under tools/thermal-guard.sh --max 75, with one GPU client, on a board
whose ARM clock is pinned (see docs/CLOCK-VARIANCE.md).  It refuses to report a
pipelined cell when the daemon does not advertise protocol 4: that would be a
synchronous measurement with a pipelined label.
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "infer", "python"))
sys.path.insert(0, os.path.join(ROOT, "tools", "cost"))

import common                                                   # noqa: E402
from vc4infer.engine import Engine                              # noqa: E402


class Prof(ctypes.Structure):
    """struct vc4_prof (infer/src/vc4_dev_remote.c), in declaration order."""
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
        ("pad_zero_bytes", ctypes.c_uint64), ("pad_copy_bytes", ctypes.c_uint64),
        ("civac_off_bytes", ctypes.c_uint64),
        ("n_launch", ctypes.c_uint64), ("n_upload", ctypes.c_uint64),
        ("n_download", ctypes.c_uint64), ("n_load", ctypes.c_uint64),
        ("n_flush", ctypes.c_uint64),
        ("n_batch", ctypes.c_uint64), ("n_batch_launch", ctypes.c_uint64),
        ("clean_bytes", ctypes.c_uint64), ("up_bytes", ctypes.c_uint64),
        ("down_bytes", ctypes.c_uint64),
        ("wait", ctypes.c_uint64), ("wait_pad", ctypes.c_uint64),
        ("wait_other", ctypes.c_uint64), ("n_reap", ctypes.c_uint64),
        ("n_wait_pad", ctypes.c_uint64), ("wait_kind", ctypes.c_uint64 * 6),
        # dirty-range accounting (infer/src/vc4_dev_remote.c, docs/CIVAC.md)
        ("mk_bytes", ctypes.c_uint64), ("mk_calls", ctypes.c_uint64),
        ("mk_hull", ctypes.c_uint64), ("mk_max", ctypes.c_uint64),
        ("mk_ndirty_max", ctypes.c_uint64),
        ("cl_bytes", ctypes.c_uint64), ("cl_ranges", ctypes.c_uint64),
        ("cl_max", ctypes.c_uint64), ("cl_uniq", ctypes.c_uint64),
        ("mk_op", ctypes.c_uint64), ("mk_uni", ctypes.c_uint64),
        ("mk_scratch", ctypes.c_uint64),
        ("mk_scr_area", ctypes.c_uint64), ("mk_pool_area", ctypes.c_uint64),
        ("cl_frame_bytes", ctypes.c_uint64),
        ("cl_frame_uniq", ctypes.c_uint64),
    ]


def prof_of(eng):
    lib = eng._eng.lib
    lib.vc4_prof_ptr.restype = ctypes.POINTER(Prof)
    return lib.vc4_prof_ptr().contents


def input_for(eng, seed=1234):
    spec = eng.inputs[0]
    n = int(np.prod(spec.shape))
    rng = np.random.default_rng(seed)
    return {spec.name: rng.standard_normal(n).astype(np.float32)
            .reshape(spec.shape)}


def cell(model, frames, warmup, pipeline, check, feeds=None):
    eng = Engine(model)
    try:
        x = feeds if feeds is not None else input_for(eng)
        run = eng.run_pipelined if pipeline else eng.run
        for _ in range(warmup):
            run(x)
        if check and pipeline:
            ref = eng.run(x)
            got = run(x)
            same = all(np.array_equal(ref[k], got[k]) for k in ref)
            print("  outputs %s" % ("byte-identical" if same else "DIFFER"))
            if not same:
                bad = [k for k in ref if not np.array_equal(ref[k], got[k])]
                raise SystemExit("pipelined output differs from the "
                                 "synchronous path on %s: stop here"
                                 % ", ".join(bad))
        # Single-frame latency: after an idle gap, so nothing is in flight.
        time.sleep(0.3)
        t0 = time.monotonic()
        run(x)
        latency = time.monotonic() - t0
        pid_d = common.vc4d_pid()
        walls, qpu, req, launches, cpu_c, cpu_d, temps = [], [], [], [], [], [], []
        wait = []
        padwait = []
        kinds = []
        prof = prof_of(eng) if os.environ.get("VC4INFER_PROF") else None
        if prof is None:
            raise SystemExit("set VC4INFER_PROF=1: requests per frame are read "
                             "from the launch-path profiler")
        for _ in range(frames):
            c0 = common.proc_cpu_s()
            d0 = common.proc_cpu_s(pid_d) if pid_d else 0.0
            b0 = prof.n_batch
            w0 = prof.wait
            wp0 = prof.wait_pad
            wk0 = [prof.wait_kind[i] for i in range(6)]
            t0 = time.monotonic()
            run(x)
            walls.append(time.monotonic() - t0)
            # eng.stats is per run, not cumulative: launches counted and GPU
            # time accounted by the run that just finished.
            qpu.append(eng.stats["gpu_ms"] / 1000.0)
            launches.append(eng.stats["launches"])
            req.append(prof.n_batch - b0)
            wait.append((prof.wait - w0) / 1e9)
            padwait.append((prof.wait_pad - wp0) / 1e9)
            kinds.append([(prof.wait_kind[i] - wk0[i]) / 1e9 for i in range(6)])
            cpu_c.append(common.proc_cpu_s() - c0)
            cpu_d.append((common.proc_cpu_s(pid_d) - d0) if pid_d else 0.0)
            temps.append(common.temp_c())
        wall = statistics.median(walls)
        return {
            "wall_ms": wall * 1000.0,
            "fps": 1.0 / wall,
            "qpu_ms": statistics.median(qpu) * 1000.0,
            "duty": statistics.median(qpu) / wall,
            "wait_ms": statistics.median(wait) * 1000.0,
            "padwait_ms": statistics.median(padwait) * 1000.0,
            "kinds_ms": [statistics.median(k[i] for k in kinds) * 1000.0
                         for i in range(6)],
            "latency_ms": latency * 1000.0,
            "requests": statistics.median(req),
            "launches": statistics.median(launches),
            "client_cpu_ms": statistics.median(cpu_c) * 1000.0,
            "daemon_cpu_ms": statistics.median(cpu_d) * 1000.0,
            "temp": (min(temps), max(temps)),
            "arena_used_mb": eng._eng.arena_bytes / 2 ** 20,
            "pool_mb": eng._runner.pool.size / 2 ** 20,
            "pool_peak_mb": eng._runner.pool.high / 2 ** 20,
        }
    finally:
        eng.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get(
        "VC4INFER_MODELS", os.path.expanduser("~/work/onnx/models")))
    ap.add_argument("--sizes", default="96,128,160")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--pipe-only", action="store_true",
                    help="only the pipelined path (for a daemon-side counter)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.environ["VC4INFER_PROF"] = "1"

    pid = common.vc4d_pid()
    print("client pid %d, vc4d pid %s, arm %.0f MHz, temp %.1f C"
          % (os.getpid(), pid, common.arm_hz() / 1e6, common.temp_c()))
    print("%-6s %-9s %8s %7s %8s %6s %7s %8s %8s %8s %8s %8s %7s"
          % ("size", "path", "wall ms", "FPS", "QPU ms", "duty", "req/f",
             "launch/f", "cli CPU", "dmn CPU", "wait ms", "pad ms", "lat ms"))
    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        model = os.path.join(args.models, "yolov8n_%d.onnx" % size)
        if not os.path.exists(model):
            print("%-6d %s missing" % (size, model))
            continue
        paths = ([False] if args.sync_only
                 else [True] if args.pipe_only else [False, True])
        for pipeline in paths:
            try:
                c = cell(model, args.frames, args.warmup, pipeline, args.check)
            except Exception as e:                  # noqa: BLE001
                print("%-6d %-9s refused: %s" % (size,
                      "pipelined" if pipeline else "sync", e))
                continue
            print("%-6d %-9s %8.1f %7.2f %8.1f %6.2f %7.1f %8.1f %8.1f "
                  "%8.1f %8.1f %8.1f %7.1f  (pool %.1f/%.1f MiB of %.0f, "
                  "%d-%d C)"
                  % (size, "pipelined" if pipeline else "sync",
                     c["wall_ms"], c["fps"], c["qpu_ms"], c["duty"],
                     c["requests"], c["launches"], c["client_cpu_ms"],
                     c["daemon_cpu_ms"], c["wait_ms"], c["padwait_ms"],
                     c["latency_ms"],
                     c["pool_mb"], c["pool_peak_mb"], c["arena_used_mb"],
                     c["temp"][0], c["temp"][1]))
            if pipeline:
                k = c["kinds_ms"]
                print("       blocked at: depth %.1f ms, invalidate %.1f, "
                      "l2 flush %.1f, scratch %.1f, host %.1f, other %.1f"
                      % (k[5], k[2], k[3], k[1], k[4], k[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
