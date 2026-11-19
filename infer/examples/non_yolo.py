#!/usr/bin/env python3
"""The same Engine on models that are not YOLO, including a multi-input graph.

    python3 infer/examples/make_models.py          # once, writes examples/models/
    python3 infer/examples/non_yolo.py
    python3 infer/examples/non_yolo.py --models /usr/share/vc4infer/models

Runs:
  * modelA: the shipped conv/relu/pool/gemm/softmax classifier (1x3x32x32)
  * mlp:    1x16 -> Gemm -> Relu -> Gemm -> Softmax        (1x16)
  * two_input: a+b -> Add -> Relu -> GAP -> Flatten -> Gemm (two feeds)
"""
import argparse
import os
import sys

import numpy as np

try:
    import vc4infer
except ImportError:                     # in-tree checkout: package in infer/python
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    import vc4infer

HERE = os.path.dirname(os.path.abspath(__file__))


def _check(name, got, ref, tol=5e-4):
    d = np.abs(np.asarray(got, np.float64) - np.asarray(ref, np.float64))
    bad = int((d > tol + tol * np.maximum(np.abs(ref), 1.0)).sum())
    print("  %-10s %-14s max abs %.3e  over tol %d/%d  %s"
          % (name, "x".join(str(v) for v in np.shape(got)), d.max(), bad,
             d.size, "PASS" if bad == 0 else "FAIL"))
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=HERE, help="dir with the demo models")
    ap.add_argument("--shipped", default="/usr/share/vc4infer/models",
                    help="dir with the shipped selftest models")
    a = ap.parse_args()
    models = os.path.join(a.models, "models") if os.path.isdir(
        os.path.join(a.models, "models")) else a.models
    ok = True

    # 1. a classifier from the shipped selftest data
    cls = os.path.join(a.shipped, "modelA.onnx")
    if os.path.isfile(cls):
        with vc4infer.Engine(cls) as eng:
            x = np.load(os.path.join(a.shipped, "modelA.in.npy"))
            ref = np.load(os.path.join(a.shipped, "modelA.out.npy"))
            print("modelA (classifier)", eng.inputs, "->", eng.outputs)
            ok &= _check("modelA", eng.run({"x": x})["y"], ref)

    # 2. tiny MLP with a softmax, one input
    mlp = os.path.join(models, "mlp.onnx")
    if os.path.isfile(mlp):
        with vc4infer.Engine(mlp) as eng:
            x = np.load(os.path.join(models, "mlp.in.npy"))
            ref = np.load(os.path.join(models, "mlp.ref.npy"))
            out = eng.run({"x": x})["probs"]
            print("mlp", eng.inputs, "->", eng.outputs, "argmax",
                  int(out.argmax()))
            print("  plan:", eng.plan().to_dict())
            ok &= _check("mlp", out, ref)
    else:
        print("mlp.onnx missing; run infer/examples/make_models.py")

    # 3. a two-input graph: both feeds in one dict
    two = os.path.join(models, "two_input.onnx")
    if os.path.isfile(two):
        with vc4infer.Engine(two) as eng:
            feeds = {"a": np.load(os.path.join(models, "two_input.a.npy")),
                     "b": np.load(os.path.join(models, "two_input.b.npy"))}
            ref = np.load(os.path.join(models, "two_input.ref.npy"))
            print("two_input", eng.inputs, "->", eng.outputs)
            ok &= _check("two_input", eng.run(feeds)["logits"], ref)
            try:
                eng.run({"a": feeds["a"]})
            except vc4infer.ShapeError as e:
                print("  missing feed rejected:", e)

    # 4. the same graph through the raw executor path (no post-processing)
    if os.path.isfile(mlp):
        with vc4infer.Engine(mlp) as eng:
            prof = eng.profile(reps=2)
            print("profile: %.2f ms/image, %d launches, GPU %.2f ms"
                  % (prof["mean_ms"], prof["launches"], prof["gpu_ms"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
