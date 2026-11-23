#!/usr/bin/env python3
"""profile_classes.py -- where a graph's QPU time goes, by node class.

The engine's per-node profile already reports ``gpu_ms`` per node, which is
the daemon's own kick-to-idle time for the launches that node made.  That is
QPU time, not host time: a node's wall time includes the socket round trips
and the ARM-side copies and is reported as ``wall_ms`` instead.

This tool adds one thing the per-node rows do not have: the split of a conv
node into its im2col panel pass and its GEMM.  Both go through
``ctx->st.pack_ns`` (the panel) and ``ctx->st.kernel_ns`` (the GEMM), so
wrapping ``_conv_nhwc`` and reading the two counters around each call gives
the split per layer, and summing gives the share of conv QPU time that the
panel materialisation costs.

    TMPDIR=$HOME/scratch/tmp python3 infer/tools/onnx/profile_classes.py \
        infer/packaging/vc4infer/data/models/yolov8n_160.onnx --reps 3

Output is one table per section:

  class      total QPU ms for the op class, its share, launches and nodes
  conv       panel ms vs GEMM ms for the whole graph
  layers     the same split for each conv layer (--conv only)

The GPU must be free: one engine owns one arena.  Run under
``tools/thermal-guard.sh --max 75`` on this board, with the ARM clock capped
where the measurement requires it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import os as _os
import sys as _sys
_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.join(_ROOT, "tools"))
from checkout_guard import require_checkout                # noqa: E402
from collections import OrderedDict

import numpy as np


def _engine(model, cache, passes):
    import vc4infer
    return vc4infer.Engine(model, cache_dir=cache, passes=passes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--input", default=None,
                    help="input .npy (default: <model stem>.in.npy beside it)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--passes", default=None,
                    help="optimiser passes (default: the engine's, $VC4_OPT)")
    ap.add_argument("--cache", default=None, help="kernel cache directory")
    ap.add_argument("--conv", action="store_true",
                    help="print the per-layer conv panel/GEMM split")
    ap.add_argument("--allow-installed", action="store_true",
                    help="test the installed vc4infer package instead of "
                         "this checkout")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    require_checkout(__file__, allow_installed=args.allow_installed)

    inp = args.input
    if inp is None:
        inp = os.path.splitext(args.model)[0] + ".in.npy"
    if not os.path.isfile(inp):
        ap.error("no input array at %s (pass --input)" % inp)
    x = np.load(inp).astype(np.float32)

    eng = _engine(args.model, args.cache, args.passes)
    runner = eng._runner
    ceng = eng._eng

    # Per-layer panel/GEMM split.  The wrapper records one row per conv call
    # in the run; a graph node runs once per inference.
    rows = []
    orig = runner._conv_nhwc

    def wrapped(node, xin, wa, ba, kh, kw, sh, pad, dil=1, groups=1, act=0,
                pads=None):
        p0, k0 = ceng.stats.pack_ns, ceng.stats.kernel_ns
        out = orig(node, xin, wa, ba, kh, kw, sh, pad, dil, groups, act, pads)
        p1, k1 = ceng.stats.pack_ns, ceng.stats.kernel_ns
        rows.append({"name": node.name, "shape": tuple(int(v) for v in xin.shape),
                     "w": tuple(int(v) for v in wa.shape), "k": int(kw),
                     "panel_ms": (p1 - p0) / 1e6, "gemm_ms": (k1 - k0) / 1e6})
        return out

    runner._conv_nhwc = wrapped

    feed = {runner.float_inputs[0][0]: x} if runner.float_inputs else {"x": x}
    eng.run(feed, profile=True)                 # warm-up: JIT is not measured

    acc = OrderedDict()
    walls = []
    t_pack = t_kern = 0.0
    for _ in range(max(1, args.reps)):
        rows.clear()                    # one row per call, last run kept
        t0 = time.perf_counter()
        eng.run(feed, profile=True)
        walls.append((time.perf_counter() - t0) * 1e3)
        for p in eng.last_profile:
            e = acc.setdefault(p["op"], {"gpu": 0.0, "launch": 0, "nodes": 0})
            e["gpu"] += p["gpu_ms"] / max(1, args.reps)
            e["launch"] += p["launches"] / max(1, args.reps)
            e["nodes"] += 1 / max(1, args.reps)
        t_pack = sum(r["panel_ms"] for r in rows)
        t_kern = sum(r["gemm_ms"] for r in rows)

    gpu = sum(e["gpu"] for e in acc.values())
    launches = int(sum(e["launch"] for e in acc.values()))
    walls.sort()
    wall = walls[len(walls) // 2]

    print("model   %s" % args.model)
    print("reps    %d (median wall, one warm-up discarded)" % args.reps)
    print("wall    %.1f ms   QPU %.1f ms   launches %d   QPU duty %.0f%%"
          % (wall, gpu, launches, 100.0 * gpu / wall if wall else 0.0))
    print()
    print("%-12s %9s %7s %8s %6s" % ("class", "QPU ms", "share", "launches",
                                     "nodes"))
    for op, e in sorted(acc.items(), key=lambda kv: -kv[1]["gpu"]):
        print("%-12s %9.2f %6.1f%% %8d %6.1f" %
              (op, e["gpu"], 100.0 * e["gpu"] / gpu if gpu else 0.0,
               e["launch"], e["nodes"]))
    print("%-12s %9.2f %6.1f%% %8d %6.1f" %
          ("total", gpu, 100.0, launches, sum(e["nodes"] for e in acc.values())))

    if rows:
        print()
        print("conv panel (im2col) %.2f ms = %.1f%% of conv QPU; "
              "GEMM %.2f ms = %.1f%%"
              % (t_pack, 100.0 * t_pack / (t_pack + t_kern) if t_pack + t_kern
                 else 0.0, t_kern,
                 100.0 * t_kern / (t_pack + t_kern) if t_pack + t_kern else 0.0))
    if args.conv:
        print("%-52s %5s %5s %7s %8s %8s" %
              ("layer", "Cin", "Cout", "k/p", "panel", "gemm"))
        for r in sorted(rows, key=lambda r: -(r["panel_ms"] + r["gemm_ms"])):
            print("%-52s %5d %5d %5d %8.3f %8.3f" %
                  (r["name"][-52:], r["shape"][1], r["w"][0], r["k"],
                   r["panel_ms"], r["gemm_ms"]))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": args.model, "wall_ms": wall, "gpu_ms": gpu,
                       "launches": launches,
                       "classes": [{"op": op, **e} for op, e in acc.items()],
                       "conv": {"panel_ms": t_pack, "gemm_ms": t_kern,
                                "layers": rows}}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
