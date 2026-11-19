"""vc4infer -- ONNX inference on the VideoCore IV QPUs.

    import vc4infer

    vc4infer.devices()                        # ['vc4']
    dev = vc4infer.Device("vc4")              # .name .memory .queue_depth .supports(op)
    eng = vc4infer.Engine("yolov8n.onnx", dev)
    eng.inputs, eng.outputs                   # TensorSpec(name, shape, dtype)
    out = eng.run({"images": x})              # {'output0': np.ndarray}, raw
    eng.plan()                                # strict pre-flight check
    eng.profile(reps=3)                       # per-node + total ms, launches
    eng.memory()                              # arena / pool / peak

Post-processing is a separate, replaceable module:

    from vc4infer.postprocess import decode_yolo, nms
    det = decode_yolo(out["output0"], imgsz=160, conf=0.25, iou=0.45)

YOLO is a thin adapter on top:

    det = vc4infer.yolo.Detector(eng)         # or Detector("yolov8n.onnx")
    r = det.predict("bus.jpg", imgsz=160)     # Detection: boxes/scores/classes

Every float operation of the graph runs on the QPUs through ``libvc4infer``.
Unsupported nodes raise :class:`UnsupportedNode` naming the node and op; there
is no CPU arithmetic fallback.
"""

__version__ = "1.1.1"

from ._binding import (                                        # noqa: F401
    Engine as _LibraryEngine, Buf, Stats, OpInfo, Error, Unsupported, VC4_OK,
    ERR_UNSUPPORTED, OP_MATMUL, OP_CONV2D, OP_ADD, OP_MUL, OP_RELU,
    OP_SIGMOID, OP_SILU, LAYOUT_ROW_MAJOR,
)
from ._errors import (                                          # noqa: F401
    Vc4InferError, DeviceError, ShapeError, UnsupportedNode,
)
from . import coverage                                          # noqa: F401
from .device import Device, devices                             # noqa: F401
from .engine import Engine, Plan, TensorSpec, NodeInfo          # noqa: F401
from . import postprocess                                       # noqa: F401
from . import yolo                                              # noqa: F401

__all__ = [
    "__version__",
    "devices", "Device",
    "Engine", "Plan", "TensorSpec", "NodeInfo",
    "postprocess", "yolo", "coverage",
    "Vc4InferError", "DeviceError", "ShapeError", "UnsupportedNode",
    "Error", "Unsupported",
]


def default_cache_dir():
    """Kernel blob cache: ``$VC4INFER_CACHE`` or ``~/.cache/vc4infer/kcache``."""
    from ._runtime import default_cache_dir as _d
    return _d()
