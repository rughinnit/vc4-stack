"""ONNX Runtime with the VideoCore IV QPUs as an execution provider.

    import vc4infer.ort
    sess = vc4infer.ort.InferenceSession("yolov8n_160.onnx",
                                         providers=["vc4"])
    out = sess.run(None, {"images": x})
    sess.report()          # vc4 | vc4-folded | ort-cpu + %nodes/%MACs/ms

The provider claims a node only when the engine can execute it exactly (the
static report in :mod:`vc4infer.coverage` is authoritative); everything else
runs on ORT's CPU EP and is named in the report.  ``vc4infer.ort`` is a
graph-partitioning execution provider, not an in-process ORT plugin: the
installed onnxruntime wheel ships no headers, so the EP plugin ABI is not
available to build against.  See ``docs/ONNXRUNTIME.md``.
"""
from ._session import (InferenceSession, PROVIDER, available_providers,
                       get_available_providers, is_available)
from ._report import rows as node_backends, summary

__all__ = [
    "InferenceSession", "get_available_providers", "available_providers",
    "is_available", "node_backends", "summary", "PROVIDER",
]
