#!/usr/bin/env python3
"""One (runtime, size) cell: wall time, CPU time, RSS and headroom.

    measure.py --runtime vc4|ncnn --size 96|128|160 --runs N [--warmup 3]

Writes one JSON object prefixed ``COST_JSON `` on stdout.  Run it under
``tools/thermal-guard.sh``; the guard's own line reports the peak.

CPU time is read from /proc/<pid>/stat around the timed loop.  For vc4infer it
is reported twice: the client process, and the client plus the vc4d daemon that
actually launches the kernels.  A loop that does everything except inference
gives the harness floor, so the reader can see how much of the number is
Python.
"""
import argparse
import json
import os
import statistics
import sys
import time

import os as _os
import sys as _sys
_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "tools"))
from checkout_guard import require_checkout                # noqa: E402

import common


def cpu_snapshot():
    return {
        "self": common.proc_cpu_s(),
        "daemon": common.proc_cpu_s(common.vc4d_pid())
        if common.vc4d_pid() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=("vc4", "ncnn"), required=True)
    ap.add_argument("--size", type=int, choices=(96, 128, 160), required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4,
                    help="ncnn intra-op threads")
    ap.add_argument("--allow-installed", action="store_true",
                    help="test the installed vc4infer package instead of "
                         "this checkout")
    ap.add_argument("--rate", type=float, default=None,
                    help="deliver frames at this rate (FPS), sleeping "
                         "between runs")
    args = ap.parse_args()
    if args.runtime == "vc4":
        require_checkout(__file__, allow_installed=args.allow_installed)

    # Import before the RSS baseline so the model, not the interpreter, is the
    # delta that the load column reports.
    if args.runtime == "ncnn":
        import ncnn  # noqa: F401
    else:
        import vc4infer  # noqa: F401

    rss0, _ = common.proc_rss_kb()
    if args.runtime == "ncnn":
        h = common.load_ncnn(args.size, args.threads)
    else:
        h = common.load_vc4(args.size)
    run = h["run"]

    for _ in range(args.warmup):
        run()
    rss_loaded, _ = common.proc_rss_kb()

    # Harness floor: the same loop shape with the inference call removed.
    floor_runs = max(args.runs, 20)
    c0 = common.proc_cpu_s()
    t0 = time.perf_counter()
    for _ in range(floor_runs):
        _ = h["input"][0, 0, 0, 0]
        time.perf_counter()
    floor_s = (common.proc_cpu_s() - c0) / floor_runs
    floor_wall = (time.perf_counter() - t0) / floor_runs

    time.sleep(0.2)
    sampler = common.Sampler()
    cpu0 = cpu_snapshot()
    sampler.start()
    runs, elems = [], None
    period = 1.0 / args.rate if args.rate else None
    t_start = time.perf_counter()
    for i in range(args.runs):
        if period is not None:
            target = t_start + i * period
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        t = time.perf_counter()
        info = run()
        runs.append((time.perf_counter() - t) * 1e3)
        elems = info["elems"]
    if period is not None:
        # Complete the last frame period so runs/elapsed is the delivered rate,
        # not the rate up to the end of the final inference.
        tail = t_start + args.runs * period - time.perf_counter()
        if tail > 0:
            time.sleep(tail)
    wall = time.perf_counter() - t_start
    sampler.stop()
    cpu1 = cpu_snapshot()

    client_s = cpu1["self"] - cpu0["self"]
    daemon_s = max(0.0, cpu1["daemon"] - cpu0["daemon"])
    total_s = client_s + daemon_s
    rss_after, hwm = common.proc_rss_kb()

    res = {
        "runtime": args.runtime,
        "size": args.size,
        "runs": args.runs,
        "warmup": args.warmup,
        "threads": args.threads if args.runtime == "ncnn" else None,
        "target_fps": args.rate,
        "delivered_fps": args.runs / wall,
        "wall_ms": {
            "median": statistics.median(runs),
            "mean": statistics.mean(runs),
            "min": min(runs),
            "max": max(runs),
            "runs": [round(v, 3) for v in runs],
        },
        "fps_mean": 1000.0 / statistics.mean(runs),
        "wall_s": wall,
        "cpu": {
            "client_s": client_s,
            "daemon_s": daemon_s,
            "total_s": total_s,
            "per_infer_client_s": client_s / args.runs,
            "per_infer_total_s": total_s / args.runs,
            "share_pct": 100.0 * total_s / wall / common.CORES,
            "client_share_pct": 100.0 * client_s / wall / common.CORES,
            "floor_s": floor_s,
            "floor_wall_ms": floor_wall * 1e3,
            "floor_pct_of_cell": 100.0 * floor_s * args.runs / total_s,
            "headroom_cores": common.CORES - total_s / wall,
        },
        "rss": {
            "before_load_mb": rss0 / 1024.0,
            "loaded_mb": rss_loaded / 1024.0,
            "after_mb": rss_after / 1024.0,
            "hwm_mb": hwm / 1024.0,
            "load_delta_mb": (rss_loaded - rss0) / 1024.0,
            "hwm_delta_mb": (hwm - rss0) / 1024.0,
        },
        "model_bytes": h["model_bytes"],
        "out_elems": elems,
        "out_elems_expected": common.expected_elems(args.size),
        "system": sampler.summary(),
        "clocks": {"core_hz": common.core_hz(),
                   "arm_hz": common.arm_hz(),
                   "throttled": common.throttled(),
                   "lcrit_alarm": common.lcrit_alarm()},
    }
    if "engine" in h:
        res["memory"] = h["engine"].memory()
        res["vc4d_pid"] = common.vc4d_pid()
    if "engine" in h:
        try:
            h["engine"].close()
        except Exception:
            pass
    print("COST_JSON " + json.dumps(res, sort_keys=True))
    sys.stdout.flush()
    # ncnn's Python wheel frees its pool allocator before the Mats that point
    # into it and aborts during interpreter teardown.  The measurement is
    # already on stdout and every run has completed; exit without running the
    # destructors.
    os._exit(0)


if __name__ == "__main__":
    main()
