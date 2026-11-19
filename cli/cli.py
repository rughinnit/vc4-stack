"""vc4infer command line: run / bench / nodes / selftest."""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from . import __version__, coverage, decode, runtime


# --------------------------------------------------------------------- utils
def _shape_str(shape):
    return "x".join(str(d) for d in shape)


def _die(msg):
    raise SystemExit("vc4infer: " + msg)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def _engine_info():
    lib = getattr(runtime.vc4infer, "_LIB", "?")
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


def cmd_version(args):
    lib, onnx_ver, np_ver = _engine_info()
    info = runtime.prepare_env()
    print("%s %s" % ("vc4infer", __version__))
    print("engine lib:      %s" % lib)
    print("python:          %s" % sys.version.split()[0])
    print("onnx / numpy:    %s / %s" % (onnx_ver, np_ver))
    print("backend:         %s" % _backend_str())
    print("kernel generator:%s" % (info["kgen"] or "MISSING"))
    print("kernel cache:    %s" % info["cache"])
    return 0


# ------------------------------------------------------------------- inputs
def _input_image(path, dims, imgsz):
    from PIL import Image
    if len(dims) != 4:
        _die("image input needs a 4-D model input (NCHW), model has %s"
             % _shape_str(dims))
    n, c, h, w = dims
    if c != 3:
        _die("image input needs 3 channels, model expects %d" % c)
    if n != 1:
        _die("image input needs batch 1, model expects %d" % n)
    size = imgsz or (h if h > 0 else w)
    if h > 0 and w > 0 and (size != h or size != w):
        _die("model input is %dx%d but --imgsz %d was given; re-export the "
             "model at %d or drop --imgsz" % (h, w, size, size))
    img = Image.open(path).convert("RGB")
    lb = decode.Letterbox(size)
    x, meta = lb(img)
    return x.astype(np.float32), meta, "%s (%dx%d -> %dx%d)" % (
        os.path.basename(path), meta["orig_w"], meta["orig_h"], size, size)


def _input_npy(path, dims):
    a = np.load(path)
    a = np.asarray(a, np.float32)
    if list(a.shape) == list(dims):
        return a, None, "%s %s" % (os.path.basename(path), _shape_str(a.shape))
    if len(dims) > 1 and list(a.shape) == list(dims[1:]):
        a = a[None]
        return a, None, "%s %s (batch added)" % (os.path.basename(path),
                                                _shape_str(a.shape))
    _die("%s has shape %s, model input is %s"
         % (path, _shape_str(a.shape), _shape_str(dims)))


def _input_csv(path, dims):
    try:
        a = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=1)
    except ValueError:
        a = np.loadtxt(path, dtype=np.float32, ndmin=1)
    a = np.asarray(a, np.float32).ravel()
    want = int(np.prod(dims))
    want_nb = int(np.prod(dims[1:])) if len(dims) > 1 else want
    if a.size == want:
        a = a.reshape(dims)
    elif a.size == want_nb:
        a = a.reshape(dims[1:])[None]
    else:
        _die("%s has %d values, model input %s needs %d (or %d without batch)"
             % (path, a.size, _shape_str(dims), want, want_nb))
    return a, None, "%s %d values -> %s" % (os.path.basename(path), a.size,
                                            _shape_str(a.shape))


def load_input(path, dims, imgsz=None):
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")):
        return _input_image(path, dims, imgsz)
    if lower.endswith((".npy", ".npz")):
        return _input_npy(path, dims)
    return _input_csv(path, dims)


# ---------------------------------------------------------------------- run
def _fmt_dets(dets, names=None):
    lines = ["  %-3s %-14s %6s %9s %9s %9s %9s" %
             ("#", "class", "conf", "x1", "y1", "x2", "y2")]
    for i in range(len(dets["scores"])):
        x1, y1, x2, y2 = dets["boxes"][i]
        lines.append("  %-3d %-14s %6.3f %9.1f %9.1f %9.1f %9.1f"
                     % (i + 1, decode.class_name(dets["classes"][i], names),
                        dets["scores"][i], x1, y1, x2, y2))
    return "\n".join(lines)


def cmd_run(args):
    runtime.prepare_env()
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    if args.input is None:
        _die("run needs --input <image|csv|npy>")
    if not os.path.isfile(args.input):
        _die("no such input: %s" % args.input)

    eng = runtime.open_engine(args.cache)
    try:
        r = runtime.Runner(args.model, eng)
        if not r.float_inputs:
            _die("%s has no float input" % args.model)
        in_name, dims = r.float_inputs[0]
        x, meta, src = load_input(args.input, dims, args.imgsz)

        t0 = time.perf_counter()
        outs = r.run({in_name: x}, profile=True)
        wall_ms = (time.perf_counter() - t0) * 1e3
        gpu_ms = sum(p["gpu_ms"] for p in r.profile)
        launches = int(sum(p["launches"] for p in r.profile))
    finally:
        eng.close()

    out = np.asarray(outs[0], np.float32)
    dets = None
    names = decode.COCO80
    if args.classes:
        with open(args.classes) as f:
            names = [line.strip() for line in f if line.strip()]

    if decode.is_detection_output(out):
        dets = decode.decode_head(out, conf=args.conf, iou=args.iou,
                                  max_det=args.max_det, class_names=names)
        if meta:
            dets["boxes"] = decode.Letterbox.to_original(dets["boxes"], meta)

    saved = None
    if dets is not None and not args.no_save:
        out_path = args.out
        if out_path is None:
            stem, _ = os.path.splitext(args.input)
            out_path = stem + "_vc4infer.jpg"
        from PIL import Image
        img = decode.draw(Image.open(args.input).convert("RGB"), dets, names)
        img.save(out_path, quality=92)
        saved = out_path

    if args.json:
        doc = {
            "model": os.path.abspath(args.model),
            "input": src,
            "input_shape": list(dims),
            "outputs": [{"shape": list(np.asarray(o).shape),
                         "min": float(np.min(o)), "max": float(np.max(o))}
                        for o in outs],
            "timing": {"wall_ms": wall_ms, "gpu_ms": gpu_ms,
                       "launches": launches},
        }
        if dets is not None:
            doc["detections"] = [
                {"class_id": int(dets["classes"][i]),
                 "class_name": decode.class_name(dets["classes"][i], names),
                 "confidence": float(dets["scores"][i]),
                 "box": [float(v) for v in dets["boxes"][i]]}
                for i in range(len(dets["scores"]))]
            doc["decode"] = {"conf": args.conf, "iou": args.iou,
                             "candidates": dets["n_raw"],
                             "anchors": dets["n_anchors"],
                             "classes": dets["n_classes"],
                             "layout": dets["layout"],
                             "coordinates": ("original image pixels" if meta
                                             else "model input pixels")}
        if saved:
            doc["annotated"] = os.path.abspath(saved)
        print(json.dumps(doc, indent=1, default=_json_default))
        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(doc, f, indent=1, default=_json_default)
        return 0

    print("model:  %s" % args.model)
    print("input:  %s  ->  %s" % (src, _shape_str(dims)))
    print("output: %s  %s" % (out.shape, "fp32"))
    print("run:    %.1f ms  (%.0f GPU ms, %d launches)"
          % (wall_ms, gpu_ms, launches))
    if dets is not None:
        where = "original image pixels" if meta else "model input pixels"
        print("decode: YOLO head %s, %d anchors, %d classes; "
              "%d above conf %.2f -> %d after NMS (IoU %.2f); boxes in %s"
              % (dets["layout"], dets["n_anchors"], dets["n_classes"],
                 dets["n_raw"], args.conf, len(dets["scores"]), args.iou,
                 where))
        if len(dets["scores"]):
            print(_fmt_dets(dets, names))
        else:
            print("  (no detections)")
    else:
        a = out.ravel()
        print("values: min %.4g  max %.4g  mean %.4g" %
              (a.min(), a.max(), a.mean()))
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
        doc = {"model": os.path.abspath(args.model), "input": src,
               "timing": {"wall_ms": wall_ms, "gpu_ms": gpu_ms,
                          "launches": launches}}
        if dets is not None:
            doc["detections"] = [
                {"class_id": int(dets["classes"][i]),
                 "class_name": decode.class_name(dets["classes"][i], names),
                 "confidence": float(dets["scores"][i]),
                 "box": [float(v) for v in dets["boxes"][i]]}
                for i in range(len(dets["scores"]))]
        with open(args.save_json, "w") as f:
            json.dump(doc, f, indent=1, default=_json_default)
        print("json:   %s" % args.save_json)
    return 0


# --------------------------------------------------------------------- bench
def cmd_bench(args):
    runtime.prepare_env()
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    eng = runtime.open_engine(args.cache)
    try:
        r = runtime.Runner(args.model, eng)
        if not r.float_inputs:
            _die("%s has no float input" % args.model)
        in_name, dims = r.float_inputs[0]
        if args.input:
            x, meta, src = load_input(args.input, dims, args.imgsz)
        else:
            size = [d if d > 0 else 1 for d in dims]
            if args.imgsz and len(dims) == 4 and dims[2] == dims[3]:
                if args.imgsz != dims[2]:
                    _die("model input is %dx%d, --imgsz %d does not fit"
                         % (dims[2], dims[3], args.imgsz))
            x = np.random.default_rng(0).standard_normal(
                size).astype(np.float32)
            src = "random normal (rng 0), %s" % _shape_str(x.shape)
        feed = {in_name: x}

        t0 = time.perf_counter()
        r.run(feed, profile=True)                     # warm-up: kernel JIT
        warm_ms = (time.perf_counter() - t0) * 1e3

        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            r.run(feed, profile=True)
            times.append((time.perf_counter() - t0) * 1e3)
        prof = r.profile
        launches = int(sum(p["launches"] for p in prof))
        gpu_ms = float(sum(p["gpu_ms"] for p in prof))
        wall_sum = float(sum(p["wall_ms"] for p in prof))
        mean = statistics.mean(times)
        res = {
            "model": os.path.abspath(args.model),
            "input": src, "input_shape": list(dims),
            "warmup_ms": warm_ms, "runs_ms": times,
            "mean_ms": mean, "median_ms": statistics.median(times),
            "min_ms": min(times),
            "fps_mean": 1000.0 / mean, "fps_best": 1000.0 / min(times),
            "reps": args.reps,
            "nodes": len(prof), "launches": launches,
            "gpu_ms": gpu_ms, "wall_sum_ms": wall_sum,
            "gpu_fraction": gpu_ms / mean if mean else 0.0,
            "arena_mib": eng.arena_bytes / 2 ** 20,
            "pool_mib": r.pool.size / 2 ** 20,
            "pool_peak_mib": r.pool.used_bytes / 2 ** 20,
        }
    finally:
        eng.close()

    if args.json:
        print(json.dumps(res, indent=1, default=_json_default))
        return 0
    print("model:  %s" % args.model)
    print("input:  %s" % res["input"])
    print("warm-up (kernel JIT + first run): %.1f ms" % res["warmup_ms"])
    print("runs:   %s ms" % " / ".join("%.1f" % t for t in res["runs_ms"]))
    print("%.1f ms/image (mean), %.1f ms (median), best %.1f ms"
          % (res["mean_ms"], res["median_ms"], res["min_ms"]))
    print("%.2f FPS (mean), %.2f FPS (best)  over %d image(s)"
          % (res["fps_mean"], res["fps_best"], args.reps))
    print("%d launches/image, GPU %.1f ms (%.0f%% of wall), node wall sum %.1f ms"
          % (res["launches"], res["gpu_ms"], 100 * res["gpu_fraction"],
             res["wall_sum_ms"]))
    print("arena %.1f MiB mapped, graph pool %.1f MiB, peak %.1f MiB"
          % (res["arena_mib"], res["pool_mib"], res["pool_peak_mib"]))
    return 0


# --------------------------------------------------------------------- nodes
def cmd_nodes(args):
    if not os.path.isfile(args.model):
        _die("no such model: %s" % args.model)
    rep = coverage.analyse(args.model)
    if args.json:
        print(json.dumps(rep, indent=1, default=_json_default))
        return 0
    print("model:  %s  (%.1f MiB, opset %s, %s)"
          % (args.model, rep["bytes"] / 2 ** 20, rep["opset"],
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
    if rep["silu_pairs"]:
        print("rewrites: %d Mul(x,Sigmoid(x)) pairs fuse into one SiLU launch "
              "(%d dispatches)"
              % (rep["silu_pairs"], rep["dispatches"]))
    print()
    print("  %-22s %6s %6s %6s  %s" % ("op", "nodes", "ok", "bad",
                                       "implementation"))
    for c in rep["counts"]:
        if c["op"] not in coverage.OPS:
            continue
        print("  %-22s %6d %6d %6d  %s"
              % (c["op"], c["nodes"], c["supported"], c["unsupported"],
                 c["note"]))
    bad = [n for n in rep["nodes"] if n["status"] == "unsupported"]
    unk = [n for n in rep["nodes"] if n["status"] == "unknown"]
    if bad:
        print("\nunsupported nodes (%d):" % len(bad))
        for n in bad:
            print("  %-40s %-12s %s" % (n["name"][:40], n["op"], n["reason"]))
    else:
        print("\nunsupported nodes: none")
    if unk:
        print("\nunverified nodes (%d, shape-dependent constraint):" % len(unk))
        for n in unk:
            print("  %-40s %-12s %s" % (n["name"][:40], n["op"], n["reason"]))
    if args.verbose:
        print("\nall nodes:")
        for n in rep["nodes"]:
            print("  %4d %-44s %-12s %-11s %s"
                  % (n["index"], n["name"][:44], n["op"], n["status"],
                     n["reason"]))
    print("\nstatic report only; `vc4infer run` is authoritative")
    return 0


# ------------------------------------------------------------------ selftest
CASES = [
    ("modelA", "conv/relu/maxpool/gemm/softmax 32x32"),
    ("modelB", "YOLO-ish block: stride-2, depthwise, pool, concat, resize"),
    ("modelC", "conv/gap/gemm 8x8"),
    ("yolov8n_160", "real YOLOv8n COCO, 1x3x160x160 -> 1x84x525"),
]


def _selftest_case(models, stem, eng):
    onnx_path = os.path.join(models, stem + ".onnx")
    if stem.startswith("yolov8n"):
        in_path = os.path.join(models, stem + ".in.npy")
        ref_path = os.path.join(models, stem + ".ref.npy")
    else:
        in_path = os.path.join(models, stem + ".in.npy")
        ref_path = os.path.join(models, stem + ".out.npy")
    for p in (onnx_path, in_path, ref_path):
        if not os.path.isfile(p):
            return {"name": stem, "error": "missing %s" % p}
    x = np.load(in_path).astype(np.float32)
    ref = np.load(ref_path).astype(np.float32)
    r = runtime.Runner(onnx_path, eng)
    feed = {r.float_inputs[0][0]: x}
    t0 = time.perf_counter()
    outs = r.run(feed, profile=True)               # warm-up for this graph
    warm = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    outs = r.run(feed, profile=True)
    ms = (time.perf_counter() - t0) * 1e3
    got = np.asarray(outs[0], np.float64)
    exp = ref.astype(np.float64)
    if got.shape != exp.shape:
        return {"name": stem, "error": "shape %s != reference %s"
                % (got.shape, exp.shape)}
    d = np.abs(got - exp)
    atol = rtol = 5e-4
    tol = atol + rtol * np.maximum(np.abs(exp), 1.0)
    over = int((d > tol).sum())
    return {
        "name": stem, "nodes": len(r.profile),
        "launches": int(sum(p["launches"] for p in r.profile)),
        "ms": ms, "warmup_ms": warm,
        "shape": list(got.shape), "max_abs": float(d.max()),
        "max_rel": float((d / np.maximum(np.abs(exp), 1.0)).max()),
        "elements": int(d.size), "over": over,
        "pass": over == 0,
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
    print()
    rows = []
    failed = 0
    for stem, desc in CASES:
        eng = runtime.open_engine(args.cache)
        try:
            res = _selftest_case(models, stem, eng)
        except (runtime.UnsupportedNode, runtime.vc4infer.Error) as e:
            res = {"name": stem, "error": str(e)}
        finally:
            eng.close()
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
    ap.add_argument("--version", action="version",
                    version="vc4infer " + __version__)
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
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("bench", help="time a model end to end")
    p.add_argument("model")
    p.add_argument("--imgsz", type=int, default=None,
                   help="input size (validated against the model)")
    p.add_argument("--reps", type=int, default=3, help="timed runs (default 3)")
    p.add_argument("--input", default=None,
                   help="optional real input (image/npy/csv) instead of rng")
    p.add_argument("--json", action="store_true")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("nodes", help="static node coverage report")
    p.add_argument("model")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list every node with its status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_nodes)

    p = sub.add_parser("selftest", help="run the shipped reference models")
    p.add_argument("--models", default=None, help="model directory")
    p.add_argument("--cache", default=None, help="kernel blob cache directory")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 2
    try:
        return args.func(args)
    except runtime.UnsupportedNode as e:
        sys.stderr.write("vc4infer: unsupported: %s\n" % e)
        return 3
    except KeyboardInterrupt:
        return 130
