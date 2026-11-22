"""vc4infer -- ONNX inference on the VideoCore IV QPUs.

    import vc4infer

    vc4infer.devices()                        # ['vc4']
    dev = vc4infer.Device("vc4")              # .name .memory .queue_depth .supports(op)
    eng = vc4infer.Engine("yolov8n.onnx", dev)
    eng.inputs, eng.outputs                   # TensorSpec(name, shape, dtype)
    out = eng.run({"images": x})              # {'output0': np.ndarray}, raw
    eng.plan()                                # strict pre-flight check
    eng.node_backends()                       # [(name, op, backend, reason)]
    eng.host_debt()                           # numeric ops with no VC4 kernel
    eng.profile(reps=3)                       # per-node + total ms, launches
    eng.memory()                              # arena / pool / peak

Post-processing is a separate, replaceable module:

    from vc4infer.postprocess import decode_yolo, nms
    det = decode_yolo(out["output0"], imgsz=160, conf=0.25, iou=0.45)

YOLO is a thin adapter on top:

    det = vc4infer.yolo.Detector(eng)         # or Detector("yolov8n.onnx")
    r = det.predict("bus.jpg", imgsz=160)     # Detection: boxes/scores/classes

Strict mode is the default: every float operation of the graph runs on the
QPUs through ``libvc4infer`` and an unsupported node raises
:class:`UnsupportedNode` naming the node and op; there is no CPU arithmetic
fallback.  ``Engine(path, mode="hybrid")`` is the opt-in exception:
``host_policy="structural"`` lets metadata (Shape/Constant) and control flow
(If/Loop/Scan) use the numpy reference evaluator and raises
:class:`HostNotAllowed` for numeric operators that still need a VC4 kernel;
``host_policy="any"`` runs those too, for compatibility only.
"""

__version__ = "1.2.10"

from ._binding import (                                        # noqa: F401
    Engine as _LibraryEngine, Buf, Stats, OpInfo, Error, Unsupported, VC4_OK,
    ERR_UNSUPPORTED, OP_MATMUL, OP_CONV2D, OP_ADD, OP_MUL, OP_RELU,
    OP_SIGMOID, OP_SILU, LAYOUT_ROW_MAJOR, ACT_NONE, ACT_RELU,
)
from ._errors import (                                          # noqa: F401
    Vc4InferError, DeviceError, ShapeError, UnsupportedNode,
    HostNotAllowed, HostNodeError, NeedsDeviceDispatch,
)
from . import coverage                                          # noqa: F401
from .device import Device, devices                             # noqa: F401
from .engine import Engine, Plan, TensorSpec, NodeInfo, NodeBackend  # noqa: F401
from . import postprocess                                       # noqa: F401
from . import yolo                                              # noqa: F401

__all__ = [
    "__version__",
    "devices", "Device",
    "Engine", "Plan", "TensorSpec", "NodeInfo", "NodeBackend",
    "postprocess", "yolo", "coverage", "ort",
    "Vc4InferError", "DeviceError", "ShapeError", "UnsupportedNode",
    "HostNotAllowed", "HostNodeError", "NeedsDeviceDispatch",
    "Error", "Unsupported",
]


def default_cache_dir():
    """Kernel blob cache: ``$VC4INFER_CACHE`` or ``~/.cache/vc4infer/kcache``."""
    from ._runtime import default_cache_dir as _d
    return _d()


from . import ort  # noqa: E402,F401  (ORT shim; onnxruntime is optional)
