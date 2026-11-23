#!/usr/bin/env python3
"""resize_matrix.py -- the Resize mode x coordinate_transformation_mode matrix.

``op_matrix.py`` measures the attribute variants its recipes synthesise; this
tool exists because Resize has a two-dimensional attribute space (``mode`` x
``coordinate_transformation_mode`` x ``nearest_mode``) and a regression can hide
in a combination no recipe covers.  Every combination the runner accepts is run
through ``vc4infer`` (hybrid, host_policy=device) and compared with
``onnx.reference``.  It is the verification behind the claim in docs/RELEASE.md
and the fix for the 1.2.2 ``Resize`` mismatch.

Usage
    PYTHONPATH=infer/python TMPDIR=$HOME/scratch/tmp \\
        python3 infer/tools/resize_matrix.py [--sizes]
    python3 infer/tools/resize_matrix.py --skip-gpu     # graph validity only

The GPU phase refuses an engine outside this checkout (``--allow-installed``
tests an installed one on purpose); see tools/checkout_guard.py.

Exit status is 0 only when every accepted combination is within tolerance.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import os as _os
import sys as _sys
_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "tools"))
from checkout_guard import require_checkout                # noqa: E402

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

COORDS = ("half_pixel", "half_pixel_symmetric", "pytorch_half_pixel",
          "asymmetric", "align_corners", "tf_crop_and_resize")
NEAREST_MODES = ("round_prefer_floor", "round_prefer_ceil", "floor", "ceil")
TOL_ABS = 1e-4
TOL_REL = 1e-4

X_SHAPE = (1, 4, 8, 8)
SCALES = (1.0, 1.0, 2.0, 2.0)
SIZES = (1, 4, 13, 11)          # deliberately not a power of two


def build(path, coord, mode, nearest_mode=None, sizes=None):
    use_roi = coord == "tf_crop_and_resize"
    ins = (["X", "roi" if use_roi else "", "", "sizes"]
           if sizes is not None else
           ["X", "roi" if use_roi else "", "scales", ""])
    attrs = {"mode": mode, "coordinate_transformation_mode": coord}
    if nearest_mode is not None:
        attrs["nearest_mode"] = nearest_mode
    node = helper.make_node("Resize", ins, ["Y"], **attrs)
    inits = []
    if use_roi:
        roi = np.array([0.0, 0.0, 0.05, 0.05, 1.0, 1.0, 0.95, 0.95],
                       np.float32)
        inits.append(helper.make_tensor("roi", TensorProto.FLOAT, (8,),
                                        roi.tolist()))
    if sizes is not None:
        inits.append(helper.make_tensor("sizes", TensorProto.INT64,
                                        (4,), [int(v) for v in sizes]))
    else:
        inits.append(helper.make_tensor("scales", TensorProto.FLOAT, (4,),
                                        [float(v) for v in SCALES]))
    graph = helper.make_graph(
        [node], "resize",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(X_SHAPE))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    m.ir_version = 10
    # the checker requires a shape field on every graph output (rank 0 = unknown)
    for vi in m.graph.output:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            tt.shape.SetInParent()
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true",
                    help="only build/validate the graphs (no device)")
    ap.add_argument("--allow-installed", action="store_true",
                    help="test the installed vc4infer package instead of "
                         "this checkout")
    ap.add_argument("--sizes", action="store_true",
                    help="use the sizes form instead of scales")
    args = ap.parse_args()
    if not args.skip_gpu:
        require_checkout(__file__, allow_installed=args.allow_installed)

    x = np.random.RandomState(11).uniform(-1, 1, X_SHAPE).astype(np.float32)
    sizes = SIZES if args.sizes else None

    eng = None
    if not args.skip_gpu:
        from vc4infer import _runtime
        eng = _runtime.open_engine(retry_s=5.0)

    work = tempfile.mkdtemp(prefix="resize-matrix-")
    rows = []
    bad = 0
    for coord in COORDS:
        variants = [("nearest", nm) for nm in NEAREST_MODES]
        variants.append(("linear", None))
        for mode, nm in variants:
            tag = "%s/%s%s" % (coord, mode, "/" + nm if nm else "")
            path = os.path.join(work, "r.onnx")
            try:
                m = build(path, coord, mode, nm, sizes)
            except Exception as e:                      # noqa: BLE001
                rows.append((tag, "invalid-model", None, str(e)))
                bad += 1
                continue
            ref = ReferenceEvaluator(m).run(None, {"X": x})[0]
            if args.skip_gpu:
                rows.append((tag, "graph-ok", None, ""))
                continue
            from vc4infer._errors import (HostNotAllowed,
                                          NeedsDeviceDispatch,
                                          UnsupportedNode)
            from vc4infer import _runtime
            try:
                runner = _runtime.Runner(path, eng, mode="hybrid",
                                         host_policy="device")
                runner.check_policy()
                got = runner.run({"X": x}, profile=True)[0]
            except (UnsupportedNode, HostNotAllowed,
                    NeedsDeviceDispatch) as e:
                msg = str(e).splitlines()[0][:90]
                if any(k in msg for k in ("daemon is dead", "GPU timeout",
                                          "I/O error", "device I/O")):
                    # A dead daemon is not an engine result: stop, restart
                    # vc4d and run this matrix again (docs/RELEASE.md).
                    rows.append((tag, "DAEMON-DEAD", None, msg))
                    print("\nABORT: the device died (%s).  Restart vc4d "
                          "(sudo systemctl restart vc4d) and re-run; the "
                          "remaining combinations were not measured." % msg,
                          file=sys.stderr)
                    return 2
                rows.append((tag, "refused", None, msg))
                continue
            d = np.abs(np.asarray(got, np.float64) -
                       np.asarray(ref, np.float64))
            scale = np.maximum(np.abs(ref), 1.0)
            ok = bool(np.all(d <= TOL_ABS + TOL_REL * scale))
            rows.append((tag, "ok" if ok else "MISMATCH",
                         (float(d.max()),), ""))
            if not ok:
                bad += 1

    w = max(len(r[0]) for r in rows)
    print("%-*s  %-12s  %9s  %s" % (w, "coord/mode", "status", "max abs", ""))
    print("-" * (w + 32))
    for tag, status, stat, note in rows:
        print("%-*s  %-12s  %9s  %s"
              % (w, tag, status,
                 "-" if not stat else "%.3g" % stat[0], note))
    acc = [r for r in rows if r[1] in ("ok", "MISMATCH")]
    print("\n%d combinations: %d accepted and run, %d mismatch, %d refused, "
          "%d invalid" % (len(rows), len(acc), sum(r[1] == "MISMATCH"
                                                    for r in rows),
                          sum(r[1] == "refused" for r in rows),
                          sum(r[1] == "invalid-model" for r in rows)))
    print("graph form: %s" % ("sizes" if sizes is not None else "scales"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
