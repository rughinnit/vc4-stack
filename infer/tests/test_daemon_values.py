#!/usr/bin/env python3
"""test_daemon_values.py -- G4: parameters <= -2.0 through vc4d.

Every fp32 value <= -2.0 has a bit pattern >= 0xC0000000, the GPU bus window.
vc4d used to reject any uniform word that looked like an address, and the
engine refused these calls in ``vc4_guard.c`` so the user saw the real reason
instead of the daemon's "uniform is a bus/arena/MMIO address".  Before the fix
this script printed:

    REFUSED Clip(-3,3)      Clip: unary: ... parameter -3 has bit pattern
                            0xc0400000, which vc4d refuses ...
    REFUSED LeakyRelu(-2)   ... parameter -2 has bit pattern 0xc0000000 ...

and now every case must run on the QPU and agree with the ONNX reference.
``Pow(x, -2)`` with a constant exponent lowers to reciprocal(x*x) and never
carried -2 as a uniform, so it ran before too; it is kept because the bug
report named it and because it must stay correct through the new launch path.

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 python3 tests/test_daemon_values.py
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
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from vc4infer import _binding as vc4infer                       # noqa: E402
from vc4infer._onnxrun import Runner                            # noqa: E402
from vc4infer._errors import UnsupportedNode                    # noqa: E402

F32 = TensorProto.FLOAT
TOL = 1e-4


def build(op, attrs=None, inits=(), n_in=1):
    inputs = [helper.make_tensor_value_info("x", F32, None)]
    names = ["x"]
    for i in range(1, n_in):
        names.append("c%d" % i)
        inputs.append(helper.make_tensor_value_info(names[-1], F32, []))
    node = helper.make_node(op, names, ["y"], **(attrs or {}))
    graph = helper.make_graph(
        [node], op, inputs,
        [helper.make_tensor_value_info("y", F32, None)], list(inits))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    return model


CASES = []

x = np.array([[-7.0, -4.0, -3.0, -2.0, -0.5, 0.0, 0.5, 2.0, 3.0, 6.0]],
             np.float32)
CASES.append(("LeakyRelu(-2)", build("LeakyRelu", {"alpha": -2.0}), {"x": x},
              np.where(x > 0, x, np.float32(-2.0) * x)))
CASES.append(("Clip(-3,3)", build("Clip", {"min": -3.0, "max": 3.0}), {"x": x},
              np.clip(x, np.float32(-3.0), np.float32(3.0))))

xp = np.array([[0.25, 0.5, 1.0, 2.0, 4.0]], np.float32)
CASES.append(("Pow(x,-2)", build(
    "Pow", None,
    [numpy_helper.from_array(np.array(-2.0, np.float32), "c1")], n_in=2),
    {"x": xp}, np.power(xp, np.float32(-2.0))))


def main():
    work = os.path.join(os.environ.get("TMPDIR", "/tmp"), "vc4-values")
    os.makedirs(work, exist_ok=True)
    eng = vc4infer.Engine(cache_dir=os.environ.get("VC4INFER_CACHE") or None)
    bad = 0
    try:
        for name, model, feeds, ref in CASES:
            path = os.path.join(work, name.replace("/", "_").replace(
                "(", "_").replace(")", "").replace(",", "_") + ".onnx")
            onnx.save(model, path)
            try:
                runner = Runner(path, eng, mode="hybrid",
                                host_policy="device")
                out = runner.run(feeds)
                got = out["y"] if isinstance(out, dict) else out[0]
                err = float(np.max(np.abs(np.asarray(got, np.float32) - ref)))
                ok = err <= TOL
                print("%s %-18s max|err|=%.3g" % ("ok  " if ok else "FAIL",
                                                  name, err))
                bad += 0 if ok else 1
            except UnsupportedNode as e:
                bad += 1
                print("FAIL %-18s refused: %s" % (name, str(e).splitlines()[0]))
            except Exception as e:                       # noqa: BLE001
                bad += 1
                print("FAIL %-18s %s: %s" % (name, type(e).__name__, e))
    finally:
        eng.close()
    print("%d cases, %d failed" % (len(CASES), bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
