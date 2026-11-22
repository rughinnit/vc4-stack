"""vc4infer command line: run / bench / nodes / selftest.

Thin over the engine: every subcommand opens an :class:`vc4infer.Engine` and
calls it; the detection decode is :mod:`vc4infer.postprocess`.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

from . import __version__, _binding, coverage, postprocess
from . import _runtime as runtime
from ._errors import (HostNodeError, HostNotAllowed, NeedsDeviceDispatch,
                      ShapeError, UnsupportedNode)
from .artifact import ArtifactError
from .engine import Engine


# --------------------------------------------------------------------- utils
def _shape_str(shape):
    return "x".join(str(d) for d in shape)


class UsageError(SystemExit):
    """Exit status 2: bad invocation, missing file, model/input mismatch."""

    def __init__(self, msg):
        sys.stderr.write("vc4infer: %s\n" % msg)
        super().__init__(2)


def _die(msg):
    raise UsageError(msg)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def _tier_counts(backs):
    out = {}
    for b in backs:
        out[b.backend] = out.get(b.backend, 0) + 1
    return out


def _tier_line(n, total):
    """'237 vc4, 30 folded, ...' style summary of a backend list."""
    c = _tier_counts(n)
    order = ("vc4", "folded", "host-meta", "host-control", "host-unwritten",
             "unsupported-graph")
    parts = ["%d %s" % (c[t], t) for t in order if c.get(t)]
    return ", ".join(parts) + (" of %d" % total if total else "")


def _engine_info():
    lib = getattr(_binding, "_LIB", "?")
    try:
        import onnx
        onnx_ver = onnx.__version__
    except Exception:
        onnx_ver = "?"
    return lib, onnx_ver, np.__version__


def _backend_str():
    info = runtime.prepare_env()
    if info["daemon"]:
        return ("vc4d (/run/vc4d.sock, %s)" %
                ("running" if runtime.daemon_reachable() else "NOT RUNNING"))
    return "direct /dev/mem (root)"


def _installed_packages():
    """Version of every vc4 package dpkg knows about (empty in a source tree)."""
    names = ("vc4infer", "vc4d", "vc4infer-dev", "vc4-query", "vc4-smi",
             "vc4train", "vc4top", "vc4top-dkms", "vc4-toolkit",
             "vc4-toolkit-fast")
    try:
        run = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package} ${Version}\n"] + list(names),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return []
    have = {}
    for line in run.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            have[parts[0]] = parts[1]
    return [(n, have[n]) for n in names if n in have]


def _system_cache():
    """The read-only prewarmed cache shipped by vc4-toolkit-fast, if any."""
    import glob
    d = "/usr/lib/vc4infer/kcache"
    if not os.path.isdir(d):
        return None
    return d, len(glob.glob(os.path.join(d, "*.bin")))


def cmd_version(args):
    lib, onnx_ver, np_ver = _engine_info()
    info = runtime.prepare_env()
    print("%s %s" % ("vc4infer", __version__))
    print("engine lib:      %s" % lib)
    print("python:          %s" % sys.version.split()[0])
    print("onnx / numpy:    %s / %s" % (onnx_ver, np_ver))
    print("backend:         %s" % _backend_str())
    print("backend choice:  %s  (managed = vc4d, unprivileged, ~0.1 ms/launch;"
          "\n                 direct = root + /dev/mem, no launch overhead, no"
          " validation)" % runtime.backend())
    print("kernel generator:%s" % (info["kgen"] or "MISSING"))
    print("kernel cache:    %s" % info["cache"])
    sc = _system_cache()
    if sc:
        print("system kcache:   %s (%d blobs, prewarmed read-only)" % sc)
    pkgs = _installed_packages()
    if pkgs:
        print("packages:        %s"
              % ", ".join("%s %s" % p for p in pkgs))
    else:
        print("packages:        (not installed from a .deb)")
    return 0


# ------------------------------------------------------------------- inputs
def _input_image(path, shape, imgsz):
    from PIL import Image
    if len(shape) != 4:
        _die("image input needs a 4-D model input (NCHW), model has %s"
             % _shape_str(shape))
    n, c, h, w = shape
    if c != 3:
        _die("image input needs 3 channels, model expects %d" % c)
    if n != 1:
        _die("image input needs batch 1, model expects %d" % n)
    size = imgsz or (h if h > 0 else w)
    if h > 0 and w > 0 and (size != h or size != w):
        _die("model input is %dx%d but --imgsz %d was given; re-export the "
             "model at %d or drop --imgsz" % (h, w, size, size))
    img = Image.open(path).convert("RGB")
    x, meta = postprocess.Letterbox(size)(img)
    return (x.astype(np.float32), meta,
            "%s (%dx%d -> %dx%d)" % (os.path.basename(path), meta["orig_w"],
                                     meta["orig_h"], size, size))


def _input_npy(path, shape):
    a = np.load(path)
    a = np.asarray(a, np.float32)
    if list(a.shape) == list(shape):
        return a, None, "%s %s" % (os.path.basename(path), _shape_str(a.shape))
    if len(shape) > 1 and list(a.shape) == list(shape[1:]):
        a = a[None]
        return a, None, "%s %s (batch added)" % (os.path.basename(path),
                                                _shape_str(a.shape))
    _die("%s has shape %s, model input is %s"
         % (path, _shape_str(a.shape), _shape_str(shape)))


def _input_csv(path, shape):
    try:
        a = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=1)
    except ValueError:
        a = np.loadtxt(path, dtype=np.float32, ndmin=1)
    a = np.asarray(a, np.float32).ravel()
    want = int(np.prod(shape))
    want_nb = int(np.prod(shape[1:])) if len(shape) > 1 else want
    if a.size == want:
        a = a.reshape(shape)
    elif a.size == want_nb:
        a = a.reshape(shape[1:])[None]
    else:
        _die("%s has %d values, model input %s needs %d (or %d without batch)"
             % (path, a.size, _shape_str(shape), want, want_nb))
    return a, None, "%s %d values -> %s" % (os.path.basename(path), a.size,
                                            _shape_str(a.shape))


def load_input(path, shape, imgsz=None):
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")):
        return _input_image(path, shape, imgsz)
    if lower.endswith((".npy", ".npz")):
        return _input_npy(path, shape)
    return _input_csv(path, shape)


# ---------------------------------------------------------------------- run
def _fmt_dets(dets, names=None):
    lines = ["  %-3s %-14s %6s %9s %9s %9s %9s" %
             ("#", "class", "conf", "x1", "y1", "x2", "y2")]
    for i in range(len(dets)):
        x1, y1, x2, y2 = dets.boxes[i]
        lines.append("  %-3d %-14s %6.3f %9.1f %9.1f %9.1f %9.1f"
                     % (i + 1, postprocess.class_name(dets.classes[i], names),
                        dets.scores[i], x1, y1, x2, y2))
    return "\n".join(lines)


def cmd_run(args):
    runtime.prepare_env()
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    if args.input is None:
        _die("run needs --input <image|csv|npy>")
    if not os.path.isfile(args.input):
        _die("no such input: %s" % args.input)

    with Engine(args.model, mode=args.mode, host_policy=args.host_policy,
                cache_dir=args.cache,
                artifact=False if args.no_artifact else None) as eng:
        spec = eng.inputs[0]
        x, meta, src = load_input(args.input, spec.shape, args.imgsz)
        t0 = time.perf_counter()
        outs = eng.run({spec.name: x}, profile=True)
        wall_ms = (time.perf_counter() - t0) * 1e3
        gpu_ms = eng.stats["gpu_ms"]
        launches = eng.stats["launches"]
        backs = eng.node_backends()
        boundary = dict(eng.last_boundary)
        host_compute = eng.host_compute_nodes()
        host_bytes = int(eng._runner.host_bytes)
        host_sites = eng._runner.host_copy_report()
        out_names = [o.name for o in eng.outputs]

    raw = outs[out_names[0]]
    names = postprocess.COCO80
    if args.classes:
        with open(args.classes) as f:
            names = [line.strip() for line in f if line.strip()]

    dets = None
    if postprocess.is_detection_output(raw):
        imgsz = args.imgsz or (spec.shape[2] if len(spec.shape) == 4 else None)
        dets = postprocess.decode_yolo(raw, imgsz=imgsz, conf=args.conf,
                                       iou=args.iou, max_det=args.max_det)
        dets.meta["coordinates"] = ("original image pixels" if meta
                                    else "model input pixels")
        if meta:
            dets.boxes = postprocess.Letterbox.to_original(dets.boxes, meta)

    saved = None
    if dets is not None and meta and not args.no_save:
        out_path = args.out
        if out_path is None:
            stem, _ = os.path.splitext(args.input)
            out_path = stem + "_vc4infer.jpg"
        from PIL import Image
        postprocess.draw(Image.open(args.input).convert("RGB"), dets,
                         names).save(out_path, quality=92)
        saved = out_path

    doc = {
        "model": os.path.abspath(args.model),
        "input": src,
        "mode": args.mode,
        "host_policy": args.host_policy,
        "input_shape": list(spec.shape),
        "outputs": [{"name": n, "shape": list(np.asarray(outs[n]).shape),
                     "min": float(np.min(outs[n])),
                     "max": float(np.max(outs[n]))}
                    for n in out_names],
        "timing": {"wall_ms": wall_ms, "gpu_ms": gpu_ms,
                   "launches": launches},
        "boundary": boundary,
        "backends": _tier_counts(backs),
    }
    if args.mode == "hybrid":
        doc["host_nodes"] = [
            {"name": b.name, "op": b.op, "backend": b.backend,
             "reason": b.reason} for b in backs if b.on_host]
        doc["notice"] = ("hybrid run: host nodes are not VC4 performance")
    if dets is not None:
        doc["detections"] = dets.to_dict(names)
        doc["decode"] = {k: v for k, v in dets.meta.items()}
        if saved:
            doc["annotated"] = os.path.abspath(saved)
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(doc, f, indent=1, default=_json_default)
    if args.json:
        print(json.dumps(doc, indent=1, default=_json_default))
        return 0

    print("model:  %s" % args.model)
    if getattr(eng, "artifact", None) is not None:
        a = eng.artifact
        print("plan:   %s (vc4x, %d prepacked buffers, %d blobs)"
              % (a.keys.get("abi_tag", "artifact"),
                 len(a.manifest.get("weights", {}).get("entries", [])),
                 len(a.manifest.get("kernels", {}).get("entries", []))))
    print("input:  %s  ->  %s" % (src, _shape_str(spec.shape)))
    print("output: %s  %s" % (raw.shape, raw.dtype))
    print("run:    %.1f ms  (%.0f GPU ms, %d launches; %s)"
          % (wall_ms, gpu_ms, launches,
             "first run in a process, so it includes the one-time "
             "weight-panel packs" if getattr(eng, "artifact", None) is None
             else "first run in a process; the panels were prepacked at "
                  "build time"))
    print("mode:   %s%s" % (args.mode,
                            "" if args.mode == "strict"
                            else " (host_policy=%s)" % args.host_policy))
    print("tiers:  %s" % _tier_line(backs, len(backs)))
    print("host_compute_nodes=%d  bytes_copied_by_host=%d"
          % (len(host_compute), host_bytes))
    if host_sites:
        for s in host_sites[:8]:
            print("  host bytes: %-34s %-10s %-14s %8d B x%d"
                  % (s["name"][:34], s["op"], s["site"], s["bytes"],
                     s["count"]))
    if args.mode == "hybrid":
        print("boundary: %d tensor(s) copied across the VC4/host boundary "
              "(%d host->VC4, %d VC4->host)"
              % (boundary.get("tensors", 0), boundary.get("host_to_vc4", 0),
                 boundary.get("vc4_to_host", 0)))
        if any(b.backend != "vc4" for b in backs):
            print("NOTE:   hybrid execution includes host nodes; these "
                  "timings are not VC4 performance")
    if dets is not None:
        print("decode: %d anchors, %d classes, %s; %d above conf %.2f -> %d "
              "after NMS (IoU %.2f); boxes in %s"
              % (dets.meta["anchors"], dets.meta["classes"],
                 dets.meta["layout"], dets.meta["candidates"], args.conf,
                 len(dets), args.iou, dets.meta["coordinates"]))
        print(_fmt_dets(dets, names) if len(dets) else "  (no detections)")
    else:
        a = raw.ravel()
        print("values: min %.4g  max %.4g  mean %.4g"
              % (a.min(), a.max(), a.mean()))
        if a.size <= 64:
            with np.printoptions(precision=4, suppress=True):
                print("        %s" % (a.reshape(-1)[:16],))
        else:
            k = min(5, a.size)
            idx = np.argsort(-a)[:k]
            print("top-%d: %s" % (k, ", ".join("out[%d]=%.4g" % (i, a[i])
                                              for i in idx)))
    if saved:
        print("saved:  %s" % saved)
    if args.save_json:
        print("json:   %s" % args.save_json)
    return 0


# --------------------------------------------------------------------- bench
def cmd_bench(args):
    runtime.prepare_env()
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    with Engine(args.model, mode=args.mode, host_policy=args.host_policy,
                cache_dir=args.cache,
                artifact=False if args.no_artifact else None) as eng:
        feeds = None
        src = None
        if args.input:
            spec = eng.inputs[0]
            x, _meta, src = load_input(args.input, spec.shape, args.imgsz)
            feeds = {spec.name: x}
        elif args.imgsz and len(eng.inputs[0].shape) == 4 \
                and eng.inputs[0].shape[2] == eng.inputs[0].shape[3] \
                and args.imgsz != eng.inputs[0].shape[2]:
            _die("model input is %dx%d, --imgsz %d does not fit"
                 % (eng.inputs[0].shape[2], eng.inputs[0].shape[3],
                    args.imgsz))
        res = eng.profile(reps=args.reps, feeds=feeds)
        if src is None:
            src = "random normal (rng 0), %s" % _shape_str(
                eng.inputs[0].shape)

    if args.json:
        print(json.dumps(res, indent=1, default=_json_default))
        return 0
    print("model:  %s" % args.model)
    print("input:  %s" % src)
    print("mode:   %s%s" % (res["mode"],
                            "" if res["mode"] == "strict"
                            else " (host_policy=%s)" % res["host_policy"]))
    print("warm-up (kernel JIT + first run): %.1f ms" % res["warmup_ms"])
    print("runs:   %s ms" % " / ".join("%.1f" % t for t in res["runs_ms"]))
    print("%.1f ms/image (mean), %.1f ms (median), best %.1f ms"
          % (res["mean_ms"], res["median_ms"], res["min_ms"]))
    print("%.2f FPS (mean), %.2f FPS (best)  over %d image(s)"
          % (res["fps_mean"], res["fps_best"], res["reps"]))
    print("%d launches/image, GPU %.1f ms (%.0f%% of wall), node wall sum %.1f "
          "ms" % (res["launches"], res["gpu_ms"],
                  100 * res["gpu_ms"] / res["mean_ms"], res["wall_sum_ms"]))
    print("tiers:  vc4 %d, folded %d, host-meta %d, host-control %d, "
          "host-unwritten %d, unsupported-graph %d"
          % (res["nodes_vc4"], res.get("nodes_folded", 0),
             res["nodes_host_meta"], res["nodes_host_control"],
             res["nodes_host_unwritten"], res["nodes_unsupported_graph"]))
    print("        %d/%d compute nodes on VC4 (%.1f%%), %.1f%% of node wall "
          "time on host"
          % (res["nodes_vc4"],
             res["nodes_vc4"] + res["nodes_host_control"] +
             res["nodes_host_unwritten"] + res["nodes_unsupported_graph"],
             res["pct_compute_vc4"], res["pct_time_host"]))
    print("util:   host_compute_nodes=%d  bytes_copied_by_host=%d  "
          "arithmetic_on_vc4=%.1f%%  qpu_duty=%.1f%%"
          % (res["host_compute_nodes"], res["bytes_copied_by_host"],
             res["arithmetic_on_vc4"], res["qpu_duty"]))
    print("        launches=%d (%.2f/node), launch path %.1f ms (%.1f%%)"
          % (res["launches"], res["launches_per_node"],
             res["launch_path_ms"], res["launch_path_pct"]))
    if res["mode"] == "hybrid":
        print("boundary: %d tensor(s) across the VC4/host boundary "
              "(%d host->VC4, %d VC4->host)"
              % (res["boundary"].get("tensors", 0),
                 res["boundary"].get("host_to_vc4", 0),
                 res["boundary"].get("vc4_to_host", 0)))
        if res["nodes_host_control"] or res["nodes_host_unwritten"] or \
                res["pct_time_host"] > 0.0:
            print("NOTE:   hybrid execution includes host work; these are not "
                  "VC4-only numbers")
    print("arena %.1f MiB mapped, graph pool %.1f MiB, peak %.1f MiB"
          % (res["arena_mb"], res["pool_mb"], res["peak_mb"]))
    return 0


# --------------------------------------------------------------------- build
def cmd_build(args):
    """Write a .vc4x engine artifact: optimised graph, packed weights, plan."""
    runtime.prepare_env()
    from .artifact import ArtifactError, build
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        out = base + ".vc4x"

    def log(msg):
        if not args.json:
            sys.stderr.write("build: %s\n" % msg)
            sys.stderr.flush()

    try:
        s = build(args.model, out, opt=args.opt, imgsz=args.imgsz,
                  opset=args.opset, embed_graph=not args.no_embed_graph,
                  embed_kernels=not args.no_embed_kernels,
                  cache_dir=args.cache, arena_mb=args.arena_mb, log=log)
    except ArtifactError as e:
        sys.stderr.write("vc4infer: %s\n" % e)
        return 3
    if args.json:
        print(json.dumps(s, indent=1, default=_json_default))
        return 0
    print("model:   %s -> %s" % (s["model"], s["path"]))
    print("artifact %s (%.2f MiB)"
          % (s["artifact_sha256"][:16], (s["file_bytes"] or 0) / 2 ** 20))
    print("graph:   %d nodes, opt=%s%s"
          % (s["nodes"], s["opt"],
             (" (passes: %s)" % ", ".join(s["passes"])) if s["passes"] else ""))
    print("weights: %d prepacked buffers, %.2f MiB"
          % (s["weights"], s["weight_bytes"] / 2 ** 20))
    print("kernels: %d blobs, %.2f MiB%s"
          % (s["kernels"], s["kernel_bytes"] / 2 ** 20,
             "" if s["kernels"] else " (names only)"))
    print("plan:    pool %.2f MiB, scratch %.2f MiB, arena %.2f MiB"
          % (s["pool_bytes"] / 2 ** 20, s["scratch_bytes"] / 2 ** 20,
             (s.get("arena_bytes") or 0) / 2 ** 20))
    t = s["timings"]
    print("build:   %.1f s (optimise %.1f s, panels %.1f s, plan %.1f s)"
          % (t["total_s"], t["optimise_s"], t["panels_s"], t["plan_s"]))
    print("keys:    generator %s, abi %s, board %s"
          % (s["generator_digest"][:16], s["abi_tag"], s["board"]))
    return 0


# -------------------------------------------------------------------- verify
def cmd_verify(args):
    """Check a .vc4x against this engine, board and generator, byte for byte."""
    from .artifact import Artifact, ArtifactError
    runtime.prepare_env()
    try:
        art = Artifact.load(args.artifact, verify=True)
        art.validate(arena=None)
    except ArtifactError as e:
        sys.stderr.write("vc4infer: %s\n" % e)
        return 3
    s = art.summary()
    if args.json:
        print(json.dumps(s, indent=1, default=_json_default))
        return 0
    print("artifact %s" % s["path"])
    print("  format     vc4x layout %s, created %s"
          % (s["layout_version"], s["created"]))
    print("  model      %s (%s nodes, opt=%s)"
          % (s["model"], s["nodes"], s["opt"]))
    print("  weights    %d prepacked buffers, %.2f MiB"
          % (s["weights"], s["weight_bytes"] / 2 ** 20))
    print("  kernels    %d blobs, %.2f MiB" % (s["kernels"],
                                               s["kernel_bytes"] / 2 ** 20))
    print("  plan       pool %.2f MiB, scratch %.2f MiB"
          % (s["pool_bytes"] / 2 ** 20, s["scratch_bytes"] / 2 ** 20))
    print("  generator  %s" % s["generator_digest"])
    print("  engine     %s" % s["engine_digest"])
    print("  board      %s" % s["board"])
    print("all sections verified; every key matches this engine")
    return 0


# --------------------------------------------------------------------- nodes
def cmd_nodes(args):
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    from .artifact import is_artifact
    if is_artifact(args.model):
        _die("nodes reads an ONNX graph; for a .vc4x use `vc4infer verify`, "
             "which checks the artifact's graph, plan and keys")
    rep = coverage.analyse(args.model)
    folded = set()
    if args.mode == "hybrid" and args.host_policy == "device":
        import onnx as _onnx
        folded, _vals = coverage.fold_metadata(_onnx.load(args.model), rep)
    plan = coverage.backend_plan(rep, mode=args.mode,
                                 host_policy=args.host_policy,
                                 folded=folded)
    by_index = {b["index"]: b for b in plan}
    tiers = {t: 0 for t in ("vc4", "folded", "host-meta", "host-control",
                            "host-unwritten", "unsupported-graph")}
    for b in plan:
        tiers[b["backend"]] = tiers.get(b["backend"], 0) + 1
    if args.json:
        doc = dict(rep)
        doc["mode"] = args.mode
        doc["host_policy"] = args.host_policy
        doc["backends"] = plan
        doc["host_debt"] = [b for b in plan
                            if b["backend"] == "host-unwritten"]
        print(json.dumps(doc, indent=1, default=_json_default))
        return 0
    print("model:  %s  (%.1f MiB, opset %s, %s)"
          % (args.model, (rep["bytes"] or 0) / 2 ** 20, rep["opset"],
             rep["producer"] or "unknown producer"))
    for i in rep["inputs"]:
        print("input:  %s  %s %s" % (i["name"], _shape_str(i["shape"]),
                                    i["dtype"]))
    for o in rep["outputs"]:
        print("output: %s  %s %s" % (o["name"], _shape_str(o["shape"]),
                                    o["dtype"]))
    print("%d nodes: %d supported, %d unsupported, %d unverified"
          % (rep["total"], rep["supported"], rep["unsupported"],
             rep["unknown"]))
    print("tiers:  %d vc4, %d folded, %d host-meta, %d host-control, "
          "%d host-unwritten, %d unsupported-graph"
          % (tiers["vc4"], tiers["folded"], tiers["host-meta"],
             tiers["host-control"], tiers["host-unwritten"],
             tiers["unsupported-graph"]))
    print("policy: mode=%s host_policy=%s" % (args.mode, args.host_policy))
    if rep["silu_pairs"]:
        print("rewrites: %d Mul(x,Sigmoid(x)) pairs fuse into one SiLU launch "
              "(%d dispatches)" % (rep["silu_pairs"], rep["dispatches"]))
    print()
    print("  %-22s %6s %6s %6s %6s %6s %6s  %s"
          % ("op", "nodes", "vc4", "fold", "meta", "ctl", "unwr",
             "implementation"))
    tier_by_index = {b["index"]: b["backend"] for b in plan}
    for c in rep["counts"]:
        ops = [n for n in rep["nodes"] if n["op"] == c["op"]]
        n_vc4 = sum(1 for n in ops if tier_by_index[n["index"]] == "vc4")
        n_fold = sum(1 for n in ops
                     if tier_by_index[n["index"]] == "folded")
        n_meta = sum(1 for n in ops
                     if tier_by_index[n["index"]] == "host-meta")
        n_ctl = sum(1 for n in ops
                    if tier_by_index[n["index"]] == "host-control")
        n_unwr = sum(1 for n in ops
                     if tier_by_index[n["index"]] == "host-unwritten")
        print("  %-22s %6d %6d %6d %6d %6d %6d  %s"
              % (c["op"], c["nodes"], n_vc4, n_fold, n_meta, n_ctl, n_unwr,
                 c["note"] or "no VC4 implementation"))
    bad = [n for n in rep["nodes"] if n["status"] == "unsupported"]
    unk = [n for n in rep["nodes"] if n["status"] == "unknown"]
    debt = [b for b in plan if b["backend"] == "host-unwritten"]
    if bad:
        print("\nunsupported by the VC4 executor (%d):" % len(bad))
        for n in bad:
            b = by_index[n["index"]]
            print("  %-40s %-12s %-16s %s"
                  % (n["name"][:40], n["op"], b["backend"], b["reason"]))
    else:
        print("\nunsupported by the VC4 executor: none")
    if unk:
        print("\nunverified nodes (%d, shape-dependent constraint):" % len(unk))
        for n in unk:
            print("  %-40s %-12s %s" % (n["name"][:40], n["op"], n["reason"]))
    host = [b for b in plan if b["backend"] in ("host-meta", "host-control")]
    if host:
        print("\nhost tiers (%d: %d metadata, %d control flow):"
              % (len(host),
                 sum(1 for b in host if b["backend"] == "host-meta"),
                 sum(1 for b in host if b["backend"] == "host-control")))
        for b in host:
            print("  %-40s %-12s %-12s %s"
                  % (b["name"][:40], b["op"], b["backend"], b["reason"]))
    if debt:
        print("\nhost DEBT (%d) - numeric op(s) with no VC4 kernel:" % len(debt))
        for b in debt:
            print("  %-40s %-12s %s" % (b["name"][:40], b["op"], b["reason"]))
        if args.mode == "hybrid" and args.host_policy == "structural":
            print("\nhost_policy=structural refuses these nodes: this model "
                  "is NOT 100% VC4 and will raise HostNotAllowed.")
        elif args.host_policy == "any":
            print("\nhost_policy=any will run these in the numpy reference "
                  "evaluator (compatibility only, not VC4 performance).")
    else:
        print("\nhost debt: none")
    if args.verbose:
        print("\nall nodes (backend = where the node runs in %s mode):"
              % args.mode)
        for n in plan:
            print("  %4d %-44s %-12s %-18s %s"
                  % (n["index"], n["name"][:44], n["op"], n["backend"],
                     n["reason"]))
    print("\nstatic report only; `vc4infer run` is authoritative")
    remaining = [b for b in plan if b["backend"] in
                 ("host-meta", "host-control", "host-unwritten")]
    if remaining and args.mode == "hybrid" and \
            args.host_policy in ("device", "structural"):
        return 4
    return 0


# ---------------------------------------------------------------------- util
def cmd_util(args):
    runtime.prepare_env()
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    with Engine(args.model, mode=args.mode, host_policy=args.host_policy,
                cache_dir=args.cache,
                artifact=False if args.no_artifact else None) as eng:
        feeds = None
        src = None
        if args.input:
            spec = eng.inputs[0]
            x, _meta, src = load_input(args.input, spec.shape, args.imgsz)
            feeds = {spec.name: x}
        res = eng.utilization(reps=args.reps, feeds=feeds)
        if src is None:
            src = "random normal (rng 0), %s" % _shape_str(
                eng.inputs[0].shape)
        backs = eng.node_backends()
        debt = eng.host_debt()
    if args.json:
        doc = dict(res)
        doc["backends"] = list(backs)
        doc["host_debt"] = list(debt)
        print(json.dumps(doc, indent=1, default=_json_default))
        return 0
    print("model:  %s" % args.model)
    print("input:  %s" % src)
    print("mode:   %s (host_policy=%s)" % (res["mode"], res["host_policy"]))
    print("%.1f ms/image (mean), %.2f FPS over %d run(s)"
          % (res["mean_ms"], res["fps_mean"], res["reps"]))
    print()
    print("  %-26s %s" % ("device invariant",
                          "OK" if res["device_ok"] else "VIOLATED"))
    print("  %-26s %d" % ("host_compute_nodes", res["host_compute_nodes"]))
    print("  %-26s %d B" % ("bytes_copied_by_host",
                            res["bytes_copied_by_host"]))
    print("  %-26s %.1f%%  (%d/%d MACs)"
          % ("arithmetic_on_vc4", res["arithmetic_on_vc4"],
             res["macs_vc4"], res["macs_total"]))
    print("  %-26s %.1f%%  (%.1f / %.1f ms)"
          % ("qpu_duty", res["qpu_duty"], res["qpu_busy_ms"], res["mean_ms"]))
    print("  %-26s %d  (%.2f per dispatched node)"
          % ("launches", res["launches"], res["launches_per_node"]))
    print("  %-26s %.1f ms  (%.1f%% of wall)"
          % ("launch path", res["launch_path_ms"], res["launch_path_pct"]))
    print("  %-26s %d vc4, %d folded, %d host-meta, %d host-control, "
          "%d host-unwritten"
          % ("nodes", res["nodes_vc4"], res["nodes_folded"],
             res["nodes_host_meta"], res["nodes_host_control"],
             res["nodes_host_unwritten"]))
    sites = res.get("host_copy_sites") or []
    if sites:
        print("\nhost byte movement (%d site(s), %d B):"
              % (len(sites), res["bytes_copied_by_host"]))
        print("  %-38s %-10s %-16s %10s %6s"
              % ("node", "op", "site", "bytes", "calls"))
        for s in sites[:40]:
            print("  %-38s %-10s %-16s %10d %6d"
                  % (s["name"][:38], s["op"], s["site"], s["bytes"],
                     s["count"]))
        if len(sites) > 40:
            print("  ... %d more" % (len(sites) - 40))
    elif res["bytes_copied_by_host"] == 0:
        print("\nhost byte movement: none")
    if debt:
        print("\nhost debt (numeric, no VC4 kernel):")
        for b in debt:
            print("  %-38s %-10s %s" % (b.name[:38], b.op, b.reason))
    if res["mode"] == "hybrid" and not res["device_ok"]:
        print("\nNOTE: this run does not satisfy the device invariant; the "
              "numbers above are not a VC4-only utilisation figure.")
    return 0


# ---------------------------------------------------------------- model data
# The package ships SHA256SUMS beside the selftest models.  A file replaced
# outside dpkg, a partial copy or a truncated download used to surface as an
# engine error from whichever case read it first -- a non-finite softmax input,
# in the case that occasioned this check.  The digests name the file instead.
MODEL_SUMS = "SHA256SUMS"
MODEL_DATA_MISMATCH = 8        # exit status: the data, not the engine


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_sums(models):
    """The manifest beside the models as (name, sha256) pairs, or None."""
    path = os.path.join(models, MODEL_SUMS)
    if not os.path.isfile(path):
        return None
    entries = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                _die("%s:%d: not a sha256sum line: %r" % (path, n, line))
            name = parts[1]
            if name.startswith("*"):            # sha256sum binary marker
                name = name[1:]
            entries.append((name, parts[0].lower()))
    return entries


def _verify_models(models, entries=None):
    """Problems as (name, expected, actual-or-None); None when no manifest.

    Empty when every listed file is present and carries the recorded digest;
    ``actual`` is None for a file the manifest lists and the directory lacks.
    """
    if entries is None:
        entries = _model_sums(models)
    if entries is None:
        return None
    problems = []
    for name, want in entries:
        path = os.path.join(models, name)
        if not os.path.isfile(path):
            problems.append((name, want, None))
            continue
        got = _sha256_file(path)
        if got != want:
            problems.append((name, want, got))
    return problems


def _report_model_data(models):
    """Print the model-data verdict; None when it is safe to run cases."""
    entries = _model_sums(models)
    problems = _verify_models(models, entries)
    if problems is None:
        print("  model data: no %s in %s; integrity check skipped"
              % (MODEL_SUMS, models))
        return None
    if not problems:
        print("  model data: %d files match %s" % (len(entries), MODEL_SUMS))
        return None
    print("MODEL DATA MISMATCH: %d of %d files in %s are not the packaged ones"
          % (len(problems), len(entries), models))
    for name, want, got in problems:
        print("  %-20s expected %s  actual %s"
              % (name, want, got or "(missing)"))
    print("  This is the model data, not the engine, so no case was run.  "
          "Reinstall the package (apt-get install --reinstall vc4infer) or "
          "regenerate %s with infer/tools/update_model_sums.py."
          % MODEL_SUMS)
    return MODEL_DATA_MISMATCH


# ------------------------------------------------------------------ selftest
CASES = [
    ("modelA", "conv/relu/maxpool/gemm/softmax 32x32"),
    ("modelB", "YOLO-ish block: stride-2, depthwise, pool, concat, resize"),
    ("modelC", "conv/gap/gemm 8x8"),
    ("yolov8n_160", "real YOLOv8n COCO, 1x3x160x160 -> 1x84x525"),
]


def _selftest_case(models, stem, eng):
    onnx_path = os.path.join(models, stem + ".onnx")
    suffix = ".ref.npy" if stem.startswith("yolov8n") else ".out.npy"
    in_path = os.path.join(models, stem + ".in.npy")
    ref_path = os.path.join(models, stem + suffix)
    for p in (onnx_path, in_path, ref_path):
        if not os.path.isfile(p):
            return {"name": stem, "error": "missing %s" % p}
    x = np.load(in_path).astype(np.float32)
    ref = np.load(ref_path).astype(np.float32)
    spec = eng.inputs[0]
    t0 = time.perf_counter()
    eng.run({spec.name: x}, profile=True)          # warm-up for this graph
    warm = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    outs = eng.run({spec.name: x}, profile=True)
    ms = (time.perf_counter() - t0) * 1e3
    got = np.asarray(outs[eng.outputs[0].name], np.float64)
    exp = ref.astype(np.float64)
    if got.shape != exp.shape:
        return {"name": stem, "error": "shape %s != reference %s"
                % (got.shape, exp.shape)}
    d = np.abs(got - exp)
    tol = 5e-4 + 5e-4 * np.maximum(np.abs(exp), 1.0)
    over = int((d > tol).sum())
    return {
        "name": stem, "nodes": len(eng.last_profile),
        "launches": int(sum(p["launches"] for p in eng.last_profile)),
        "ms": ms, "warmup_ms": warm,
        "shape": list(got.shape), "max_abs": float(d.max()),
        "max_rel": float((d / np.maximum(np.abs(exp), 1.0)).max()),
        "elements": int(d.size), "over": over, "pass": over == 0,
    }


def cmd_selftest(args):
    models = args.models or runtime.models_dir()
    if not os.path.isdir(models):
        _die("selftest data not found (%s); set VC4INFER_MODELS or install "
             "the vc4infer package" % models)
    lib, onnx_ver, np_ver = _engine_info()
    print("vc4infer %s selftest" % __version__)
    print("  models:  %s" % models)
    print("  engine:  %s" % lib)
    print("  backend: %s" % _backend_str())
    print("  onnx %s, numpy %s, tolerance atol=rtol=5e-4 (over must be 0)"
          % (onnx_ver, np_ver))
    status = _report_model_data(models)
    if status:
        print()
        print("SELFTEST NOT RUN: the model files are not the packaged ones; "
              "nothing here says anything about the engine")
        return status
    print()
    rows, failed = [], 0
    for stem, desc in CASES:
        try:
            with Engine(os.path.join(models, stem + ".onnx"),
                        cache_dir=args.cache) as eng:
                res = _selftest_case(models, stem, eng)
        except (UnsupportedNode, runtime.Error) as e:
            res = {"name": stem, "error": str(e)}
        rows.append(res)
        if "error" in res:
            failed += 1
            print("  %-12s ERROR  %s" % (stem, res["error"]))
            continue
        ok = res["pass"]
        failed += 0 if ok else 1
        print("  %-12s %-4s %-14s %4d launches %7.1f ms  max abs %8.2e  "
              "max rel %8.2e  %d/%d over  [%s]"
              % (stem, "PASS" if ok else "FAIL", _shape_str(res["shape"]),
                 res["launches"], res["ms"], res["max_abs"], res["max_rel"],
                 res["over"], res["elements"], desc))
    print()
    if failed:
        print("SELFTEST FAILED: %d/%d cases" % (failed, len(rows)))
        return 1
    print("SELFTEST PASSED: %d/%d cases, 0 elements over tolerance"
          % (len(rows), len(rows)))
    return 0


# ---------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(
        prog="vc4infer",
        description="Run fixed-shape fp32 ONNX graphs on the VideoCore IV QPUs.",
        epilog="Every float op runs on the QPUs through libvc4infer/vc4d; "
               "unsupported nodes raise instead of falling back to the CPU. "
               "See /usr/share/doc/vc4infer/README.md.")
    ap.add_argument("--version", action="store_true",
                    help="print version, engine, backend and cache paths")
    ap.add_argument("--backend", choices=runtime.BACKENDS, default=None,
                    help="managed (default, vc4d) or direct (root + /dev/mem, "
                         "faster per launch, no validation)")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    p = sub.add_parser("run", help="run one inference and decode detections")
    p.add_argument("model")
    p.add_argument("--input", required=False,
                   help="image (jpg/png), .npy tensor or .csv values")
    p.add_argument("--imgsz", type=int, default=None,
                   help="letterbox size (must match the model input)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence threshold (default 0.25)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold (default 0.45)")
    p.add_argument("--max-det", type=int, default=300,
                   help="maximum detections kept (default 300)")
    p.add_argument("--classes", default=None,
                   help="file with one class name per line (default COCO-80)")
    p.add_argument("--out", default=None,
                   help="annotated image path (default <input>_vc4infer.jpg)")
    p.add_argument("--no-save", action="store_true",
                   help="do not write an annotated image")
    p.add_argument("--json", action="store_true",
                   help="print one JSON document instead of a table")
    p.add_argument("--save-json", default=None,
                   help="also write the JSON document to this path")
    p.add_argument("--mode", choices=("strict", "hybrid"), default="strict",
                   help="strict (default) raises on nodes without a VC4 "
                        "kernel; hybrid may use allowed host tiers")
    p.add_argument("--host-policy", choices=("device", "structural", "any"),
                   default="device",
                   help="hybrid host policy: device (default) forbids runtime "
                        "host compute, structural allows metadata/control, any "
                        "also runs unwritten numeric ops")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.add_argument("--no-artifact", action="store_true",
                   help="for a .vc4x: plan from the ONNX graph it carries, as "
                        "the engine does for an .onnx (no frozen plan)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("bench", help="time a model end to end")
    p.add_argument("model")
    p.add_argument("--imgsz", type=int, default=None,
                   help="input size (validated against the model)")
    p.add_argument("--reps", type=int, default=3, help="timed runs (default 3)")
    p.add_argument("--input", default=None,
                   help="optional real input (image/npy/csv) instead of rng")
    p.add_argument("--json", action="store_true")
    p.add_argument("--mode", choices=("strict", "hybrid"), default="strict",
                   help="strict (default) or hybrid execution")
    p.add_argument("--host-policy", choices=("device", "structural", "any"),
                   default="device", help="hybrid host policy")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.add_argument("--no-artifact", action="store_true",
                   help="for a .vc4x: plan from its ONNX graph as usual")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser(
        "build", help="build a .vc4x engine artifact from ONNX or .pt")
    p.add_argument("model", help="fixed-shape fp32 .onnx, or a .pt to export")
    p.add_argument("-o", "--out", default=None,
                   help="output artifact (default <model>.vc4x)")
    p.add_argument("--opt", default="all",
                   help="optimiser passes baked into the artifact: all "
                        "(default), none, or a comma-separated list")
    p.add_argument("--imgsz", type=int, default=None,
                   help=".pt export size (default 640); recorded in the "
                        "artifact")
    p.add_argument("--opset", type=int, default=17,
                   help=".pt export opset (default 17); recorded")
    p.add_argument("--no-embed-graph", action="store_true",
                   help="reference <out>.onnx instead of embedding the graph")
    p.add_argument("--no-embed-kernels", action="store_true",
                   help="record kernel blob names and hashes only; the loader "
                        "then needs the cache to hold them")
    p.add_argument("--arena-mb", type=int, default=None,
                   help="client arena to plan against (default: the daemon's)")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser(
        "verify", help="check a .vc4x: every section, every key")
    p.add_argument("artifact")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("nodes", help="static node coverage and backend report")
    p.add_argument("model")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list every node with status, backend and reason")
    p.add_argument("--json", action="store_true")
    p.add_argument("--mode", choices=("strict", "hybrid"), default="hybrid",
                   help="which policy the backend report describes")
    p.add_argument("--host-policy", choices=("device", "structural", "any"),
                   default="device",
                   help="device/structural exit non-zero when host debt exists")
    p.set_defaults(func=cmd_nodes)

    p = sub.add_parser("util", help="device-invariant utilisation accounting")
    p.add_argument("model")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--reps", type=int, default=3, help="timed runs (default 3)")
    p.add_argument("--input", default=None,
                   help="optional real input (image/npy/csv) instead of rng")
    p.add_argument("--mode", choices=("strict", "hybrid"), default="hybrid",
                   help="hybrid (default) or strict")
    p.add_argument("--host-policy", choices=("device", "structural", "any"),
                   default="device", help="hybrid host policy")
    p.add_argument("--json", action="store_true")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.add_argument("--no-artifact", action="store_true",
                   help="for a .vc4x: plan from its ONNX graph as usual")
    p.set_defaults(func=cmd_util)

    p = sub.add_parser("selftest", help="run the shipped reference models")
    p.add_argument("--models", default=None, help="model directory")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.set_defaults(func=cmd_selftest)

    for _name, _sp in sub.choices.items():
        if "--backend" not in _sp._option_string_actions:
            _sp.add_argument("--backend", choices=runtime.BACKENDS,
                             default=argparse.SUPPRESS,
                             help="managed (default) or direct (root only)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "backend", None):
        try:
            runtime.set_backend(args.backend)
        except PermissionError as e:
            sys.stderr.write("vc4infer: %s\n" % e)
            return 7
        except (ValueError, FileNotFoundError) as e:
            sys.stderr.write("vc4infer: %s\n" % e)
            return 2
    if getattr(args, "version", False):
        return cmd_version(args)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 2
    try:
        return args.func(args)
    except NeedsDeviceDispatch as e:
        sys.stderr.write("vc4infer: needs device dispatch: %s\n" % e)
        return 6
    except HostNotAllowed as e:
        sys.stderr.write("vc4infer: host not allowed: %s\n" % e)
        return 4
    except HostNodeError as e:
        sys.stderr.write("vc4infer: host node failed: %s\n" % e)
        return 5
    except UnsupportedNode as e:
        sys.stderr.write("vc4infer: unsupported: %s\n" % e)
        return 3
    except ShapeError as e:
        _die(str(e))
    except ArtifactError as e:
        sys.stderr.write("vc4infer: %s\n" % e)
        return 3
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    sys.exit(main())
