"""vc4infer command line: run / bench / nodes / selftest.

Thin over the engine: every subcommand opens an :class:`vc4infer.Engine` and
calls it; the detection decode is :mod:`vc4infer.postprocess`.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from . import __version__, _binding, coverage, postprocess
from . import _runtime as runtime
from ._errors import ShapeError, UnsupportedNode
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

    with Engine(args.model, cache_dir=args.cache) as eng:
        spec = eng.inputs[0]
        x, meta, src = load_input(args.input, spec.shape, args.imgsz)
        t0 = time.perf_counter()
        outs = eng.run({spec.name: x}, profile=True)
        wall_ms = (time.perf_counter() - t0) * 1e3
        gpu_ms = eng.stats["gpu_ms"]
        launches = eng.stats["launches"]
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
        "input_shape": list(spec.shape),
        "outputs": [{"name": n, "shape": list(np.asarray(outs[n]).shape),
                     "min": float(np.min(outs[n])),
                     "max": float(np.max(outs[n]))}
                    for n in out_names],
        "timing": {"wall_ms": wall_ms, "gpu_ms": gpu_ms,
                   "launches": launches},
    }
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
    print("input:  %s  ->  %s" % (src, _shape_str(spec.shape)))
    print("output: %s  %s" % (raw.shape, raw.dtype))
    print("run:    %.1f ms  (%.0f GPU ms, %d launches; first run in a "
          "process, so it includes the one-time weight-panel packs)"
          % (wall_ms, gpu_ms, launches))
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
    with Engine(args.model, cache_dir=args.cache) as eng:
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
    print("warm-up (kernel JIT + first run): %.1f ms" % res["warmup_ms"])
    print("runs:   %s ms" % " / ".join("%.1f" % t for t in res["runs_ms"]))
    print("%.1f ms/image (mean), %.1f ms (median), best %.1f ms"
          % (res["mean_ms"], res["median_ms"], res["min_ms"]))
    print("%.2f FPS (mean), %.2f FPS (best)  over %d image(s)"
          % (res["fps_mean"], res["fps_best"], res["reps"]))
    print("%d launches/image, GPU %.1f ms (%.0f%% of wall), node wall sum %.1f "
          "ms" % (res["launches"], res["gpu_ms"],
                  100 * res["gpu_ms"] / res["mean_ms"], res["wall_sum_ms"]))
    print("arena %.1f MiB mapped, graph pool %.1f MiB, peak %.1f MiB"
          % (res["arena_mb"], res["pool_mb"], res["peak_mb"]))
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
              "(%d dispatches)" % (rep["silu_pairs"], rep["dispatches"]))
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
    if getattr(args, "version", False):
        return cmd_version(args)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 2
    try:
        return args.func(args)
    except UnsupportedNode as e:
        sys.stderr.write("vc4infer: unsupported: %s\n" % e)
        return 3
    except ShapeError as e:
        _die(str(e))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
