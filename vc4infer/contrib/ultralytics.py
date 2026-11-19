"""Use the VC4 engine as an Ultralytics inference backend.

Ultralytics 8.4 dispatches model formats through
``ultralytics.nn.autobackend.AutoBackend._BACKEND_MAP``, a class dict keyed by
format name, and resolves ``.onnx`` files to the key ``"onnx"``.  There is no
entry-point plugin API in 8.4.215 (checked: ``AutoBackend`` imports the backend
classes directly and has no ``entry_points`` lookup), so selecting vc4 is one
documented dict entry:

    import vc4infer.contrib.ultralytics as vu
    vu.register()                       # AutoBackend._BACKEND_MAP["onnx"] = VC4Backend

    from ultralytics import YOLO
    model = YOLO("yolov8n_160.onnx")    # unchanged ultralytics workflow
    results = model.predict("bus.jpg", imgsz=160, conf=0.25)
    results[0].boxes.xyxy               # ultralytics Results, as usual

``register()`` returns the previous backend class; ``unregister(previous)``
restores it.  Only the ``onnx`` format is patched -- ``.pt``, ``.engine`` and
the rest keep their normal backends.

The obstacle in one sentence: the format is derived from the file suffix
(``guess_model_type``) before the backend map is consulted, so a vc4-specific
selector (``format="vc4"``) would need a fork of ``guess_model_type``/
``export_formats``; patching the map keeps the stock ultralytics code path.

The backend is glue: torch tensor in, numpy through ``Engine.run``, numpy out.
Ultralytics' own preprocessing (letterbox) and post-processing
(``non_max_suppression``) stay in charge, so this is the *stock* ultralytics
pipeline running its convolutions on the QPUs.
"""
import numpy as np

from .._errors import DeviceError
from ..device import Device
from ..engine import Engine

try:                                            # optional dependency
    from ultralytics.nn.backends.base import BaseBackend
    _IMPORT_ERROR = None
except Exception as exc:                        # pragma: no cover
    BaseBackend = None
    _IMPORT_ERROR = exc


if BaseBackend is not None:

    class VC4Backend(BaseBackend):
        """``BaseBackend`` implementation over :class:`vc4infer.Engine`."""

        def __init__(self, weight, device, fp16=False, **kwargs):
            if fp16:
                raise DeviceError("vc4infer is fp32 only; fp16 is not supported")
            self.engine = None
            super().__init__(weight, device, fp16)

        def load_model(self, weight):
            self.apply_metadata(self.read_metadata(weight))
            self.engine = Engine(str(weight), Device("vc4"))
            spec = self.engine.inputs[0]
            if len(spec.shape) == 4:
                n, c, h, w = spec.shape
                if h > 0 and w > 0:
                    self.imgsz = (h, w)
                if c > 0:
                    self.channels = c
                if n > 0:
                    self.batch = n
                self.dynamic = min(spec.shape) < 0
            self.model = self.engine

        def forward(self, im, **kwargs):
            x = im.detach().cpu().numpy()
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            out = self.engine.run({self.engine.inputs[0].name: x})
            return out[self.engine.outputs[0].name]

        def close(self):
            if self.engine is not None:
                self.engine.close()
                self.engine = None


_ORIGINAL_WARMUP = None


def _cpu_warmup(self, imgsz=(1, 3, 640, 640), im=None):
    """Ultralytics' warmup without the unconditional ``import torchvision``.

    ``AutoBackend.warmup`` imports torchvision on every non-end2end model so
    that the NMS path is bound to ``torchvision.ops.nms``.  This board's
    torchvision 0.29.2 registers no operators against its custom torch 2.15
    build (``RuntimeError: operator torchvision::nms does not exist``), so the
    stock warmup aborts before inference.  On CPU the stock warmup does nothing
    else, so this replacement runs the same NMS warmup with ultralytics' own
    pure-torch implementation.
    """
    import torch
    from ultralytics.utils.nms import non_max_suppression

    boxes = torch.rand(1, 84, 16, device=self.device)
    boxes[:, :4] *= imgsz[-1]
    non_max_suppression(boxes)


def register(force=True, patch_warmup=True):
    """Route Ultralytics' ``.onnx`` models through the VC4 engine.

    Returns the backend class that was replaced (pass it to
    :func:`unregister`).  Raises ImportError when ultralytics is not installed.
    """
    if BaseBackend is None:
        raise ImportError(
            "ultralytics is not importable (%s); install it with "
            "`pip install ultralytics` to use the vc4 backend" % _IMPORT_ERROR)
    global _ORIGINAL_WARMUP
    from ultralytics.nn.autobackend import AutoBackend
    previous = AutoBackend._BACKEND_MAP.get("onnx")
    if previous is VC4Backend and not force:
        return previous
    AutoBackend._BACKEND_MAP["onnx"] = VC4Backend
    if patch_warmup and _ORIGINAL_WARMUP is None:
        _ORIGINAL_WARMUP = AutoBackend.warmup
        AutoBackend.warmup = _cpu_warmup
    return previous


def unregister(previous=None):
    """Restore the stock ``.onnx`` backend and warmup (default: ONNX Runtime)."""
    global _ORIGINAL_WARMUP
    from ultralytics.nn.autobackend import AutoBackend
    if _ORIGINAL_WARMUP is not None:
        AutoBackend.warmup = _ORIGINAL_WARMUP
        _ORIGINAL_WARMUP = None
    if previous is None:
        from ultralytics.nn.backends.onnx import ONNXBackend
        previous = ONNXBackend
    AutoBackend._BACKEND_MAP["onnx"] = previous
    return previous


def is_registered():
    """True when ``.onnx`` already resolves to the vc4 backend."""
    try:
        from ultralytics.nn.autobackend import AutoBackend
    except Exception:
        return False
    return AutoBackend._BACKEND_MAP.get("onnx") is VC4Backend
