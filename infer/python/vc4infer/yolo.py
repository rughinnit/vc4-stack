"""A thin YOLO adapter over :class:`vc4infer.Engine`.

    import vc4infer
    det = vc4infer.yolo.Detector("yolov8n.onnx")
    r = det.predict("bus.jpg", imgsz=160, conf=0.25, iou=0.45)
    r.boxes, r.scores, r.classes        # numpy, original image pixels

This module is glue only: load an image, letterbox it, call
``Engine.run``, call :func:`vc4infer.postprocess.decode_yolo` (or the
``postprocess=`` callable you pass), map the boxes back.  It does not
reimplement the engine, the decode or NMS.
"""
import os
from dataclasses import dataclass, field

import numpy as np

from . import postprocess as _pp
from .engine import Engine


@dataclass
class Detection:
    """Detection results as numpy arrays (xyxy boxes in original pixels)."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    orig_shape: tuple
    imgsz: int
    speed_ms: dict = field(default_factory=dict)
    path: str = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.boxes.shape[0])

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, np.float32).reshape(-1)
        self.classes = np.asarray(self.classes, np.int64).reshape(-1)

    def to_dict(self, names=None):
        return {"path": self.path, "orig_shape": list(self.orig_shape),
                "imgsz": self.imgsz, "speed_ms": self.speed_ms,
                "detections": _pp.Detections(
                    self.boxes, self.scores, self.classes).to_dict(names)}

    def plot(self, image=None):
        """PIL RGB image with the boxes; draws on ``image`` or ``self.path``."""
        from PIL import Image
        if image is None:
            if not self.path:
                raise ValueError("no source image; pass one to plot(image)")
            image = Image.open(self.path)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return _pp.draw(image.convert("RGB"), _pp.Detections(
            self.boxes, self.scores, self.classes))


class Detector:
    """YOLO detections on top of one :class:`Engine`.

    ``engine_or_path`` may be an already-open ``Engine`` (shared with other
    consumers) or a model path, in which case the detector opens one.
    ``names`` defaults to COCO-80; pass your own list for other checkpoints.
    """

    def __init__(self, engine_or_path, names=None, *, device="vc4",
                 cache_dir=None):
        self._owns_engine = not isinstance(engine_or_path, Engine)
        self.engine = (Engine(engine_or_path, device, cache_dir=cache_dir)
                       if self._owns_engine else engine_or_path)
        self.names = list(names) if names else list(_pp.COCO80)
        shape = self.engine.inputs[0].shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError("detector needs a 1x3xHxW model input, got %s"
                             % (shape,))
        self.imgsz = int(shape[2])

    def predict(self, image, imgsz=None, conf=0.25, iou=0.45, max_det=300,
                agnostic=False, postprocess=_pp.decode_yolo):
        """Run one image and decode the head.

        ``postprocess`` is any callable
        ``fn(raw_output, imgsz=..., conf=..., iou=..., max_det=...,
        agnostic=...)`` returning a ``Detections`` (or a dict with ``boxes``,
        ``scores`` and ``classes``).  It receives model-pixel boxes; this
        method maps them back to the original image.
        """
        import time
        from PIL import Image

        size = int(imgsz or self.imgsz)
        if size != self.imgsz:
            raise ValueError("model input is %dx%d; re-export at %d instead of "
                             "resizing" % (self.imgsz, self.imgsz, size))
        t0 = time.perf_counter()
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8)) \
                if image.ndim == 3 and image.shape[2] == 3 else image
        elif isinstance(image, (str, os.PathLike)):
            img = Image.open(os.fspath(image)).convert("RGB")
        else:
            img = image.convert("RGB")
        x, meta = _pp.Letterbox(size)(img)
        t1 = time.perf_counter()
        raw = self.engine.run({self.engine.inputs[0].name: x})[
            self.engine.outputs[0].name]
        t2 = time.perf_counter()
        dets = postprocess(raw, imgsz=size, conf=float(conf), iou=float(iou),
                           max_det=int(max_det), agnostic=bool(agnostic))
        if not isinstance(dets, _pp.Detections):
            dets = _pp.Detections(dets["boxes"], dets["scores"],
                                          dets["classes"], {"raw": dets})
        boxes = _pp.Letterbox.to_original(dets.boxes, meta)
        t3 = time.perf_counter()
        return Detection(boxes, dets.scores, dets.classes,
                         orig_shape=(meta["orig_h"], meta["orig_w"]),
                         imgsz=size, meta=dets.meta,
                         path=os.fspath(image) if isinstance(
                             image, (str, os.PathLike)) else None,
                         speed_ms={"preprocess": (t1 - t0) * 1e3,
                                   "inference": (t2 - t1) * 1e3,
                                   "postprocess": (t3 - t2) * 1e3})

    def close(self):
        if self._owns_engine:
            self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
