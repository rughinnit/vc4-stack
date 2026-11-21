#!/usr/bin/env python3
"""Register the C execution-provider plugin with the installed ONNX Runtime.

    PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
        infer/onnxruntime_ep/tests/test_ep_register.py

This is a CPU-side test: the plugin claims no nodes, so no work is dispatched.
It builds ``c_ep/libvc4_ep.so`` if needed, registers it with
``onnxruntime.register_execution_provider_library``, and checks:

* the library loads and its ``CreateEpFactories`` / ``ReleaseEpFactory`` entry
  points are used (the plugin prints its registration line);
* the VideoCore IV EP device appears in ``onnxruntime.get_ep_devices()``;
* a session can be created with that device and its output is *identical* to
  the CPU EP, because the plugin claimed nothing;
* the exported counters agree: queries > 0, claimed == 0.

A plugin that cannot execute must not pretend to. If a future execute path is
added, ``vc4_ep_nodes_claimed()`` becomes non-zero and this test fails: that
is the intended signal to replace it with a numeric acceptance test.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EP_DIR = os.path.join(ROOT, "c_ep")
SO = os.path.join(EP_DIR, "libvc4_ep.so")


def build():
    subprocess.run(["make", "-s", "-C", EP_DIR], check=True)
    if not os.path.isfile(SO):
        raise SystemExit("build did not produce %s" % SO)
    print("built %s (%d bytes)" % (SO, os.path.getsize(SO)))


def make_model(path):
    """Conv -> Relu -> Gemm, small enough to compare element by element."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal((4, 3, 3, 3), dtype=np.float32) * 0.1
    b = rng.standard_normal((4,), dtype=np.float32) * 0.1
    g = rng.standard_normal((4 * 8 * 8, 6), dtype=np.float32) * 0.1
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "w", "b"], ["c"],
                          name="conv", pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["c"], ["r"], name="relu"),
         helper.make_node("Flatten", ["r"], ["flat"], name="flatten"),
         helper.make_node("Gemm", ["flat", "g"], ["output"], name="gemm")],
        "ep_register",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b"),
         numpy_helper.from_array(g, "g")])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def main():
    build()
    # Load with RTLD_GLOBAL first so the plugin's exported counters resolve to
    # the same instance ORT dlopen()s below.
    lib = ctypes.CDLL(SO, mode=os.RTLD_GLOBAL)
    for name in ("vc4_ep_capability_calls", "vc4_ep_nodes_examined",
                 "vc4_ep_nodes_claimed"):
        getattr(lib, name).restype = ctypes.c_ulong
    lib.vc4_ep_status.restype = ctypes.c_char_p
    print("plugin says: %s" % lib.vc4_ep_status().decode())

    ort.register_execution_provider_library("vc4", SO)
    providers = ort.get_available_providers()
    devices = ort.get_ep_devices()
    print("available providers: %s" % providers)
    mine = [d for d in devices if d.ep_name == "vc4"]
    for d in devices:
        print("  device: ep=%s vendor=%s type=%s meta=%s"
              % (d.ep_name, d.ep_vendor, d.device.type, dict(d.ep_metadata)))
    if not mine:
        raise SystemExit("registered library produced no vc4 EP device")

    work = os.path.join(os.path.expanduser("~"), "scratch", "tmp", "vc4ort")
    os.makedirs(work, exist_ok=True)
    path = make_model(os.path.join(work, "ep_register.onnx"))
    x = np.random.default_rng(5).standard_normal(
        (1, 3, 8, 8)).astype(np.float32)

    ref = ort.InferenceSession(path,
                               providers=["CPUExecutionProvider"]).run(
                                   None, {"input": x})[0]

    opts = ort.SessionOptions()
    opts.add_provider_for_devices([mine[0]], {})
    sess = ort.InferenceSession(path, sess_options=opts)
    got = sess.run(None, {"input": x})[0]
    print("session providers: %s" % sess.get_providers())
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    print("output vs CPU EP: %d/%d over 1e-6, max abs %.3g"
          % (int((d > 1e-6).sum()), d.size, float(d.max())))

    calls = lib.vc4_ep_capability_calls()
    examined = lib.vc4_ep_nodes_examined()
    claimed = lib.vc4_ep_nodes_claimed()
    print("plugin counters: %d capability call(s), %d node(s) examined, "
          "%d claimed" % (calls, examined, claimed))

    assert d.max() <= 1e-6, "CC EP changed the result: max abs %g" % d.max()
    assert calls >= 1, "GetCapability was never called"
    assert examined >= 1, "no node was examined"
    assert claimed == 0, ("this build is capability-only but claimed %d node(s)"
                          % claimed)
    assert "vc4" in providers, providers

    ort.unregister_execution_provider_library("vc4")
    print("\nOK: plugin registered, advertised one device, claimed no nodes, "
          "output identical to the CPU EP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
