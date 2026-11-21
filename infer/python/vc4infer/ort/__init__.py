"""ONNX Runtime with the VideoCore IV QPUs as an execution provider.

    import vc4infer.ort
    sess = vc4infer.ort.InferenceSession("yolov8n_160.onnx",
                                         providers=["vc4"])
    out = sess.run(None, {"images": x})
    sess.report()          # vc4 | vc4-folded | ort-cpu + %nodes/%MACs/ms

The provider claims a node only when the engine can execute it exactly (the
static report in :mod:`vc4infer.coverage` is authoritative); everything else
runs on ORT's CPU EP and is named in the report.  The claim decision is also
available without a session or a daemon::

    plan = vc4infer.ort.plan("yolov8n_160.onnx")
    plan.report(per_node=True)     # same table the session prints

``vc4infer.ort`` is a graph-partitioning execution provider driven from the
public Python API.  A native EP plugin is separate: ``infer/onnxruntime_ep/
c_ep`` builds ``libvc4_ep.so`` against vendored ORT 1.32.0 plugin headers, and
that library currently registers and claims nothing (it has no execute path).
See ``docs/ONNXRUNTIME.md``.
"""
from ._session import (InferenceSession, PROVIDER, available_providers,
                       get_available_providers, is_available)
from ._report import rows as node_backends, summary
from ._partition import Plan, plan

__all__ = [
    "InferenceSession", "get_available_providers", "available_providers",
    "is_available", "node_backends", "summary", "PROVIDER", "Plan", "plan",
]
