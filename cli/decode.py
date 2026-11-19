"""YOLO head decode, host NMS and image preprocessing / annotation.

The engine computes the graph; the *decode* of the detection head is deliberately
host code because YOLOv8's head is a set of elementwise affine combinations plus
an argmax, not convolutions, and the engine has no gather/sort kernels.  No float
arithmetic of the network itself happens here: this module only turns the raw
``[1, 4+nc, anchors]`` tensor into boxes, thresholds them (a comparison), sorts
and suppresses overlaps.

Supported head layouts (as exported by Ultralytics):

    YOLOv8 / YOLO11   [1, 4+nc, A]  or  [1, A, 4+nc]   (no objectness)
    YOLOv5            [1, 5+nc, A]  or  [1, A, 5+nc]   (objectness at 4)

Boxes are cx,cy,w,h in *model input pixels*, exactly what the ONNX export emits;
the caller maps them back to the original image with the letterbox metadata.
"""
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

# Ultralytics' letterbox fill value
PAD_VALUE = 114


class Letterbox:
    """Aspect-preserving resize to imgsz square with centred padding."""

    def __init__(self, imgsz, pad_value=PAD_VALUE):
        self.imgsz = int(imgsz)
        self.pad_value = pad_value

    def __call__(self, img):
        """img: PIL.Image RGB -> (float32 CHW array, meta dict)."""
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
        chw = np.ascontiguousarray(a.transpose(2, 0, 1)[None])
        meta = {"orig_w": w, "orig_h": h, "scale": s,
                "pad_x": float(pad_x), "pad_y": float(pad_y),
                "letterbox": self.imgsz}
        return chw, meta

    @staticmethod
    def to_original(boxes, meta):
        """boxes xyxy in model pixels -> xyxy in the original image."""
        if not meta or "scale" not in meta:
            return boxes
        b = np.asarray(boxes, np.float64).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - meta["pad_x"]) / meta["scale"]
        b[:, [1, 3]] = (b[:, [1, 3]] - meta["pad_y"]) / meta["scale"]
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, meta["orig_w"])
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, meta["orig_h"])
        return b


def nms(boxes, scores, iou_thr):
    """Greedy NMS for one class.  boxes: xyxy, scores: 1-D."""
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


def is_detection_output(arr):
    """True for a YOLO detection head, False for a plain classifier output."""
    a = np.asarray(arr)
    return a.ndim == 3 and a.shape[0] == 1 and min(a.shape[1], a.shape[2]) >= 5


def decode_head(out, conf=0.25, iou=0.45, agnostic=False, max_det=300,
                class_names=None):
    """Decode one YOLO head tensor into detections (model input pixels).

    Returns a dict with parallel arrays boxes (N,4 xyxy), scores, classes, plus
    ``n_raw`` (candidates over threshold before NMS) and ``layout``.
    """
    a = np.asarray(out, np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError("detection head: expected a 2-D tensor, got %r"
                         % (np.asarray(out).shape,))
    if a.shape[0] > a.shape[1]:
        a = a.T                      # -> (channels, anchors)
        layout = "anchors_last"
    else:
        layout = "channels_last"
    C, A = a.shape
    if C < 5:
        raise ValueError("detection head: %d channels is not a YOLO head" % C)

    # YOLOv5 exports carry an objectness column; YOLOv8/11 do not.
    ch = C - 4
    if ch == 85:                     # 80 classes + objectness
        obj = a[4]
        cls = a[5:]
        scores_all = cls * obj[None, :]
    else:
        cls = a[4:]
        scores_all = cls

    boxes = np.empty((A, 4), np.float32)
    boxes[:, 0] = a[0] - a[2] * 0.5
    boxes[:, 1] = a[1] - a[3] * 0.5
    boxes[:, 2] = a[0] + a[2] * 0.5
    boxes[:, 3] = a[1] + a[3] * 0.5

    best = scores_all.max(axis=0)
    cls_id = scores_all.argmax(axis=0).astype(np.int64)
    m = best >= conf
    n_raw = int(m.sum())
    boxes, best, cls_id = boxes[m], best[m], cls_id[m]

    if n_raw == 0:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                "classes": np.zeros(0, np.int64), "n_raw": 0,
                "layout": layout, "n_anchors": int(A),
                "n_classes": int(cls.shape[0])}

    if agnostic:
        keep = nms(boxes, best, iou)
    else:
        keep = [nms(boxes[cls_id == c], best[cls_id == c], iou)
                for c in np.unique(cls_id)]
        sel = []
        for c, k in zip(np.unique(cls_id), keep):
            idx = np.flatnonzero(cls_id == c)
            sel.append(idx[k])
        keep = np.concatenate(sel) if sel else np.zeros(0, np.int64)
    keep = keep[np.argsort(-best[keep])][:max_det]
    return {"boxes": boxes[keep], "scores": best[keep],
            "classes": cls_id[keep], "n_raw": n_raw, "layout": layout,
            "n_anchors": int(A), "n_classes": int(cls.shape[0])}


def class_name(cid, names=None):
    names = names or COCO80
    cid = int(cid)
    if 0 <= cid < len(names):
        return names[cid]
    return "class_%d" % cid


def draw(image, dets, names=None):
    """Annotate a PIL image with decoded boxes; returns a new RGB image."""
    from PIL import ImageDraw
    img = image.convert("RGB").copy()
    dr = ImageDraw.Draw(img)
    for i in range(len(dets["scores"])):
        x1, y1, x2, y2 = [float(v) for v in dets["boxes"][i]]
        cid = int(dets["classes"][i])
        score = float(dets["scores"][i])
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
