"""Host-side post-processing: YOLO head decode and NMS, pure numpy.

This module is deliberately separate from the engine.  ``Engine.run`` returns
the raw graph output; nothing here is called unless you ask for it.  The YOLO
adapter (:class:`vc4infer.yolo.Detector`) is one consumer; a caller with a
different head can ignore it entirely and write their own:

    raw = engine.run({"images": x})["output0"]
    my = my_decode(raw, conf=0.4)          # any numpy you like
    boxes = my["boxes"]

The reference decode is small enough to copy and modify:

    decode_yolo(output, imgsz, conf, iou)  -> Detections
    nms(boxes, scores, iou_thr)            -> keep indices

``output`` is the Ultralytics YOLOv8/YOLO11 head ``[1, 4+nc, A]`` (or the
transposed ``[1, A, 4+nc]``, and the YOLOv5 ``5+nc`` layout with an objectness
column).  Boxes are cx,cy,w,h in model-input pixels.
"""
from dataclasses import dataclass, field

import numpy as np

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

PAD_VALUE = 114          # Ultralytics' letterbox fill


@dataclass
class Detections:
    """Detections: numpy boxes in ``boxes``, scores and class ids.

    ``meta`` carries what the decode saw (anchors, classes, coordinate frame)
    so a caller can decide how to map or filter.  ``boxes`` are xyxy.
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.boxes.shape[0])

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)

    def to_dict(self, names=None):
        names = names or COCO80
        return [{"class_id": int(self.classes[i]),
                 "class_name": class_name(self.classes[i], names),
                 "confidence": float(self.scores[i]),
                 "box": [float(v) for v in self.boxes[i]]}
                for i in range(len(self))]

    def top(self, k=5):
        """The ``k`` highest-scoring detections, as a new Detections."""
        idx = np.argsort(-self.scores)[:k]
        return Detections(self.boxes[idx], self.scores[idx],
                          self.classes[idx], dict(self.meta))


def class_name(cid, names=None):
    names = names or COCO80
    cid = int(cid)
    return names[cid] if 0 <= cid < len(names) else "class_%d" % cid


def is_detection_output(arr):
    """True for a YOLO detection head, False for a plain classifier output."""
    a = np.asarray(arr)
    return a.ndim == 3 and a.shape[0] == 1 and min(a.shape[1], a.shape[2]) >= 5


def nms(boxes, scores, iou_thr):
    """Greedy NMS for one class.  ``boxes`` xyxy, ``scores`` 1-D.

    Returns the kept indices, highest score first.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, np.int64)


def decode_yolo(output, imgsz=None, conf=0.25, iou=0.45, max_det=300,
                agnostic=False):
    """Decode one YOLO head into :class:`Detections` (model input pixels).

    Parameters
    ----------
    output : ndarray
        ``[1, 4+nc, A]`` / ``[1, A, 4+nc]``, or YOLOv5's ``5+nc`` layout.
    imgsz : int, optional
        The model input size; boxes are clamped to ``[0, imgsz]`` and recorded
        in ``meta``.  Not needed for decoding.
    conf, iou : float
        Confidence threshold and NMS IoU threshold.
    max_det : int
        Maximum number of detections kept.
    agnostic : bool
        Class-agnostic NMS (otherwise suppression is per class).
    """
    a = np.asarray(output, np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError("detection head: expected 2-D after batch, got %r"
                         % (np.asarray(output).shape,))
    if a.shape[0] > a.shape[1]:
        a = a.T                      # (anchors, channels) -> (channels, anchors)
        layout = "anchors_first"
    else:
        layout = "channels_first"
    C, A = a.shape
    if C < 5:
        raise ValueError("detection head: %d channels is not a YOLO head" % C)

    if C == 85:                      # YOLOv5: 80 classes + objectness
        scores_all = a[5:] * a[4][None, :]
    else:
        scores_all = a[4:]

    boxes = np.empty((A, 4), np.float32)
    boxes[:, 0] = a[0] - a[2] * 0.5
    boxes[:, 1] = a[1] - a[3] * 0.5
    boxes[:, 2] = a[0] + a[2] * 0.5
    boxes[:, 3] = a[1] + a[3] * 0.5
    if imgsz:
        np.clip(boxes, 0, float(imgsz), out=boxes)

    best = scores_all.max(axis=0)
    cls_id = scores_all.argmax(axis=0).astype(np.int64)
    m = best >= conf
    n_raw = int(m.sum())
    boxes, best, cls_id = boxes[m], best[m], cls_id[m]
    meta = {"layout": layout, "anchors": int(A),
            "classes": int(scores_all.shape[0]), "candidates": n_raw,
            "conf": float(conf), "iou": float(iou),
            "imgsz": int(imgsz) if imgsz else None,
            "coordinates": "model input pixels",
            "nms": "agnostic" if agnostic else "per-class"}
    if n_raw == 0:
        return Detections(np.zeros((0, 4)), np.zeros(0),
                          np.zeros(0, np.int64), meta)

    if agnostic:
        keep = nms(boxes, best, iou)
    else:
        ids = np.unique(cls_id)
        keep = np.concatenate([np.flatnonzero(cls_id == c)[
            nms(boxes[cls_id == c], best[cls_id == c], iou)] for c in ids]) \
            if ids.size else np.zeros(0, np.int64)
    keep = keep[np.argsort(-best[keep])][:int(max_det)]
    return Detections(boxes[keep], best[keep], cls_id[keep], meta)


class Letterbox:
    """Aspect-preserving resize to ``imgsz`` square with centred padding.

    The preprocessing half of the pair; :meth:`to_original` is the inverse
    mapping used after the decode.
    """

    def __init__(self, imgsz, pad_value=PAD_VALUE):
        self.imgsz = int(imgsz)
        self.pad_value = pad_value

    def __call__(self, img):
        """PIL.Image -> (float32 1x3xHxW array, meta dict)."""
        from PIL import Image
        w, h = img.size
        s = min(self.imgsz / float(w), self.imgsz / float(h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz),
                           (self.pad_value,) * 3)
        pad_x = (self.imgsz - nw) // 2
        pad_y = (self.imgsz - nh) // 2
        canvas.paste(resized, (pad_x, pad_y))
        a = np.asarray(canvas, dtype=np.float32) / 255.0
        meta = {"orig_w": w, "orig_h": h, "scale": s,
                "pad_x": float(pad_x), "pad_y": float(pad_y),
                "imgsz": self.imgsz}
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None]), meta

    @staticmethod
    def to_original(boxes, meta):
        """Map xyxy boxes from model pixels back to the original image."""
        b = np.asarray(boxes, np.float64).copy()
        if not meta or "scale" not in meta:
            return b.astype(np.float32)
        b[:, [0, 2]] = (b[:, [0, 2]] - meta["pad_x"]) / meta["scale"]
        b[:, [1, 3]] = (b[:, [1, 3]] - meta["pad_y"]) / meta["scale"]
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, meta["orig_w"])
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, meta["orig_h"])
        return b.astype(np.float32)


def draw(image, detections, names=None):
    """Annotate a PIL image with :class:`Detections`; returns a new RGB image."""
    from PIL import ImageDraw
    names = names or COCO80
    img = image.convert("RGB").copy()
    dr = ImageDraw.Draw(img)
    for i in range(len(detections)):
        x1, y1, x2, y2 = [float(v) for v in detections.boxes[i]]
        cid = int(detections.classes[i])
        score = float(detections.scores[i])
        col = ((37 * cid + 53) % 200 + 55,
               (91 * cid + 17) % 200 + 55,
               (151 * cid + 101) % 200 + 55)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
        label = "%s %.2f" % (class_name(cid, names), score)
        tw = dr.textlength(label)
        ty = max(0.0, y1 - 12)
        dr.rectangle([x1, ty, x1 + tw + 4, ty + 12], fill=col)
        dr.text((x1 + 2, ty + 1), label, fill=(0, 0, 0))
    return img
