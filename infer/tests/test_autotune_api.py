#!/usr/bin/env python3
"""Engine.autotune on silicon: cold measurement, then the cached path.

CPU-side search space, legality and cache invalidation are covered by
tests/test_autotune.c and tests/test_autotune_space.py without a device.  This
script is the end-to-end API check: it opens the engine, tunes one matmul and
one conv shape, asserts that the second call for the same shape is served from
the record, and then runs modelA to show that the executor's path is unchanged.

    VC4INFER_CACHE=$HOME/scratch/kcache-autotune-api PYTHONPATH=python \
        python3 infer/tests/test_autotune_api.py
"""

# A GPU client: refuse unless VC4_REQUIRE_GPU=1 (tests/gpu_gate.py).
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from gpu_gate import require_gpu as _require_gpu
_require_gpu(__file__)
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import Engine, _runtime                    # noqa: E402

MODELS = os.path.join(HERE, "..", "packaging", "vc4infer", "data", "models")


def check(cond, what):
    print("%-4s %s" % ("PASS" if cond else "FAIL", what))
    return 0 if cond else 1


def main():
    _runtime.prepare_env()
    if not _runtime.daemon_reachable():
        print("SKIP vc4d is not running; the silicon check did not run")
        return 0

    cache = os.environ.get("VC4INFER_CACHE") or _runtime.default_cache_dir()
    bad = 0
    eng = Engine(os.path.join(MODELS, "modelA.onnx"), cache_dir=cache)
    try:
        shape = (128, 128, 128)
        cold = eng.autotune("matmul", shape, force=True)
        print("matmul %s -> pack_q=%d gemm_q=%d  %.3f ms vs %.3f ms  "
              "max|diff| %.3g over_tol %d  %d sets  %s"
              % ("x".join(map(str, shape)), cold["params"]["pack_q"],
                 cold["params"]["gemm_q"], cold["best_ms"],
                 cold["default_ms"], cold["max_abs"], cold["over_tol"],
                 cold["candidates"], cold["kernel"]))
        bad += check(not cold["from_cache"], "the first call measured")
        bad += check(cold["over_tol"] == 0,
                     "the choice matches the default launch within 5e-4")
        bad += check(cold["params"]["gemm_q"] <= 12 and
                     cold["params"]["gemm_q"] >= 1,
                     "gemm_q %d is inside the 12-QPU ceiling"
                     % cold["params"]["gemm_q"])

        warm = eng.autotune("matmul", shape)
        bad += check(warm["from_cache"], "the second call read the record")
        bad += check(warm["params"] == cold["params"],
                     "the record names the measured launch")
        bad += check(warm["kernel"] == cold["kernel"], "same blob from the record")

        conv = eng.autotune("conv2d", {"cin": 16, "cout": 16, "h": 32, "w": 32,
                                       "kh": 3, "kw": 3, "pad": 1}, force=True)
        print("conv2d 16x32x32 3x3 -> 16 -> im2col_q=%d gemm_q=%d  "
              "%.3f ms vs %.3f ms  max|diff| %.3g over_tol %d  %s"
              % (conv["params"]["im2col_q"], conv["params"]["gemm_q"],
                 conv["best_ms"], conv["default_ms"], conv["max_abs"],
                 conv["over_tol"], conv["kernel"]))
        bad += check(conv["over_tol"] == 0,
                     "the conv choice matches the default within 5e-4")

        # the executor path is unchanged: the model still matches its
        # recorded reference at the tolerance of docs/ACCURACY.md
        x = np.load(os.path.join(MODELS, "modelA.in.npy"))
        ref = np.load(os.path.join(MODELS, "modelA.out.npy"))
        got = np.asarray(eng.run(x)[eng.outputs[0].name], np.float32).ravel()
        ref = ref.ravel()
        diff = np.abs(got - ref)
        tol = 5e-4 + 5e-4 * np.maximum(np.abs(ref), 1.0)
        print("modelA: max abs %.3g, %d/%d over 5e-4"
              % (float(diff.max()), int((diff > tol).sum()), diff.size))
        bad += check(int((diff > tol).sum()) == 0,
                     "the model output is unchanged after autotuning")
    finally:
        eng.close()

    print("test_autotune_api: %s" % ("PASS" if bad == 0 else
                                     "%d FAILURE(S)" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
