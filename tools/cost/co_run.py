#!/usr/bin/env python3
"""Co-run cell: a fixed CPU workload while one runtime infers continuously.

    co_run.py --runtime none|ncnn|vc4 --size 160 \
              --burner ./burner --threads 4 --iters K [--timeout 120]

The burner is started first, then inference runs flat out until the burner
exits.  ``--runtime none`` is the alone calibration: the same burner, the same
instruction count, no inference.  Every cell is deterministic (the burner
checksum is printed and compared), so throughput lost is a like-for-like
ratio.

Writes one JSON object prefixed ``COST_JSON `` on stdout.  Run under
``tools/thermal-guard.sh --max 71``.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import common

STOP = False


def on_term(signum, frame):
    global STOP
    STOP = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=("none", "ncnn", "vc4"),
                    required=True)
    ap.add_argument("--size", type=int, choices=(96, 128, 160), default=160)
    ap.add_argument("--burner", required=True)
    ap.add_argument("--threads", type=int, choices=(1, 4), required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--nthreads", type=int, default=4,
                    help="ncnn intra-op threads")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    run = None
    h = None
    if args.runtime != "none":
        if args.runtime == "ncnn":
            h = common.load_ncnn(args.size, args.nthreads)
        else:
            h = common.load_vc4(args.size)
        run = h["run"]
        for _ in range(args.warmup):
            run()

    cmd = [args.burner, "--threads", str(args.threads),
           "--iters", str(args.iters)]
    burner = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)

    sampler = common.Sampler()
    cpu0 = {"self": common.proc_cpu_s(),
            "daemon": common.proc_cpu_s(common.vc4d_pid())
            if common.vc4d_pid() else 0.0}
    sampler.start()
    t0 = time.perf_counter()
    frames = []
    while burner.poll() is None and not STOP:
        if time.perf_counter() - t0 > args.timeout:
            burner.kill()
            break
        if run is None:
            time.sleep(0.05)
            continue
        t = time.perf_counter()
        run()
        frames.append((time.perf_counter() - t) * 1e3)
    t1 = time.perf_counter()
    sampler.stop()
    cpu1 = {"self": common.proc_cpu_s(),
            "daemon": common.proc_cpu_s(common.vc4d_pid())
            if common.vc4d_pid() else 0.0}
    out, err = burner.communicate()
    rc = burner.returncode

    burner_doc = {}
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            burner_doc = json.loads(line)

    client_s = cpu1["self"] - cpu0["self"]
    daemon_s = max(0.0, cpu1["daemon"] - cpu0["daemon"])
    total_s = client_s + daemon_s
    span = t1 - t0
    res = {
        "runtime": args.runtime,
        "size": args.size,
        "nthreads": args.nthreads if args.runtime == "ncnn" else None,
        "burner": {
            "threads": args.threads,
            "iters": args.iters,
            "wall_s": burner_doc.get("wall_s"),
            "cpu_s": burner_doc.get("cpu_s"),
            "checksum": burner_doc.get("checksum"),
            "returncode": rc,
        },
        "inference": {
            "frames": len(frames),
            "span_s": span,
            "fps": len(frames) / span if span else 0.0,
            "wall_ms_median": sorted(frames)[len(frames) // 2]
            if frames else None,
            "client_s": client_s,
            "daemon_s": daemon_s,
            "total_s": total_s,
            "cpu_per_sec": total_s / span if span else 0.0,
            "share_pct": 100.0 * total_s / span / common.CORES
            if span else 0.0,
            "headroom_cores": common.CORES - total_s / span
            if span else None,
        },
        "system": sampler.summary(),
        "clocks": {"core_hz": common.core_hz(),
                   "arm_hz": common.arm_hz()},
        "stopped": STOP,
    }
    print("COST_JSON " + json.dumps(res, sort_keys=True))
    sys.stdout.flush()
    os._exit(99 if (STOP or rc != 0) else 0)


if __name__ == "__main__":
    main()
