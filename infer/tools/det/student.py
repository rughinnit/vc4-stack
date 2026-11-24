#!/usr/bin/env python3
"""The vc4det student as a trainable PyTorch module, and its geometry.

`infer/tools/det/vc4det-n.yaml` is the specification: Ultralytics instantiates
it, `build.py` exports it, and the ONNX is what runs on the board.  This module
re-implements that graph by hand so it can be trained here, and nothing else:

    0  Conv 3->16  3x3 s2  96->48  + SiLU   stem
    1  Conv 16->32 3x3 s2  48->24  + SiLU
    2  Conv 32->32 3x3 s1  24      + SiLU
    3  Conv 32->64 3x3 s2  24->12  + SiLU
    4  Conv 64->64 3x3 s1  12      + SiLU
    5  Conv 64->32 1x1     12      + SiLU
    6  Conv 32->32 3x3 s1  12      + SiLU
    7  Conv 32->8  1x1     12      logits, no activation

Layer 7 is `[32, 6, 1]` in the shipping YAML (one class: objectness, cx, cy, w,
h, class).  This module defaults to 8 logits -- objectness, cx, cy, w, h and
three class scores -- for the three-class BCCD set.  The engine pads the logits
conv's Cout to 16 either way, so the two graphs issue the same MACs and the same
launch count; `vc4det-n3.yaml` is the matching Ultralytics YAML that `build.py`
audits, and `docs/CASE-BCCD.md` records the measured equality.

BatchNorm is used during training and folded into the conv at export
(:func:`fuse`), which is exactly what `DetectionModel.fuse()` does before
`build.py` writes the ONNX: no BatchNormalization node survives.

Head convention (the decode, applied identically in training, in validation and
on the board by `student_detect.py`):

    p    = sigmoid(raw)                     raw: (N, 8, 12, 12) at stride 8
    obj  = p[0]
    cx   = (col + p[1]) * 8                 pixels in the 96x96 canvas
    cy   = (row + p[2]) * 8
    w    = p[3] * 96
    h    = p[4] * 96
    cls  = p[5:8]                           independent sigmoids, 3 classes
    score= obj * max(cls)

Boxes are xyxy in the letterboxed canvas, i.e. the same frame ultralytics'
predictions are in after its own letterbox, so the two can be compared box for
box.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMGSZ = 96
STRIDE = 8
PAD_VALUE = 114          # ultralytics' letterbox fill, and vc4infer's
EPS = 1e-9


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class ConvBNAct(nn.Module):
    """Conv2d(bias=False) -> BatchNorm2d -> SiLU, the ultralytics Conv block."""

    def __init__(self, cin, cout, k, s):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Student(nn.Module):
    """8 convs, 22 compute nodes, 0.046 GFLOP at 96x96 (engine MACs)."""

    def __init__(self, nc=3):
        super().__init__()
        self.nc = int(nc)
        self.stem = ConvBNAct(3, 16, 3, 2)      # 96 -> 48
        self.s1a = ConvBNAct(16, 32, 3, 2)      # 48 -> 24
        self.s1b = ConvBNAct(32, 32, 3, 1)
        self.s2a = ConvBNAct(32, 64, 3, 2)      # 24 -> 12
        self.s2b = ConvBNAct(64, 64, 3, 1)
        self.p3a = ConvBNAct(64, 32, 1, 1)
        self.p3b = ConvBNAct(32, 32, 3, 1)
        self.head = nn.Conv2d(32, 5 + self.nc, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1a(x)
        x = self.s1b(x)
        x = self.s2a(x)
        x = self.s2b(x)
        x = self.p3a(x)
        x = self.p3b(x)
        return self.head(x)


class FusedConv(nn.Module):
    """Conv2d with the BatchNorm folded in, plus the activation."""

    def __init__(self, conv, act):
        super().__init__()
        self.conv = conv
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x))


def _fold(block: ConvBNAct) -> FusedConv:
    conv, bn = block.conv, block.bn
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(
        conv.out_channels)
    s = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, conv.dilation, 1, bias=True)
    fused.requires_grad_(False)
    with torch.no_grad():
        fused.weight.copy_((w * s[:, None, None, None]).contiguous())
        fused.bias.copy_((b - bn.running_mean.detach()) * s + bn.bias.detach())
    fused = fused.to(w.device).eval()
    act = nn.SiLU()
    return FusedConv(fused, act)


def fuse(model: Student) -> nn.Module:
    """Return an inference-only copy with every BN folded into its conv.

    The result is what goes to ONNX: 8 Conv, 7 Sigmoid, 7 Mul, no
    BatchNormalization, same as `build.py`'s `model.fuse()`.
    """
    m = Student(model.nc)
    m.load_state_dict(model.state_dict())
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    for name in ("stem", "s1a", "s1b", "s2a", "s2b", "p3a", "p3b"):
        setattr(m, name, _fold(getattr(m, name)))
    head = nn.Conv2d(32, 5 + m.nc, 1, bias=True)
    head.requires_grad_(False)
    with torch.no_grad():
        head.weight.copy_(model.head.weight.detach())
        head.bias.copy_(model.head.bias.detach())
    m.head = FusedConv(head, nn.Identity())
    return m.eval()


# --------------------------------------------------------------------------
# geometry: letterbox, as vc4infer.postprocess.Letterbox does it
# --------------------------------------------------------------------------
def letterbox_size(w, h, imgsz=IMGSZ):
    s = min(imgsz / float(w), imgsz / float(h))
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return s, nw, nh, (imgsz - nw) // 2, (imgsz - nh) // 2


def letterbox(img, boxes, imgsz=IMGSZ, pad_value=PAD_VALUE):
    """PIL image + xyxy pixels -> network input and boxes in the canvas.

    ``boxes`` are (N,4) xyxy in the input image's pixels.  Returns
    (float32 (3,imgsz,imgsz), (N,4) xyxy in canvas pixels, meta).
    """
    from PIL import Image
    w, h = img.size
    s, nw, nh, px, py = letterbox_size(w, h, imgsz)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (pad_value,) * 3)
    canvas.paste(resized, (px, py))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    a = np.ascontiguousarray(a.transpose(2, 0, 1))
    b = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * s + px
    b[:, [1, 3]] = b[:, [1, 3]] * s + py
    return a, b, {"scale": s, "pad": (px, py), "size": (nw, nh)}


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------
def decode(raw, nc=3, imgsz=IMGSZ, stride=STRIDE):
    """(B,5+nc,H,W) logits -> boxes xyxy pixels, scores, class ids.

    Returns (boxes (B,H*W,4), score (B,H*W), cls (B,H*W)).
    """
    b, c, h, w = raw.shape
    p = torch.sigmoid(raw)
    dev = raw.device
    gy, gx = torch.meshgrid(torch.arange(h, device=dev),
                            torch.arange(w, device=dev), indexing="ij")
    cx = (p[:, 1] + gx[None]) * stride
    cy = (p[:, 2] + gy[None]) * stride
    bw = p[:, 3] * imgsz
    bh = p[:, 4] * imgsz
    boxes = torch.stack([cx - bw * 0.5, cy - bh * 0.5,
                         cx + bw * 0.5, cy + bh * 0.5], dim=-1)
    boxes = boxes.reshape(b, h * w, 4)
    obj = p[:, 0].reshape(b, -1)
    cls_p = p[:, 5:5 + nc].reshape(b, nc, -1).permute(0, 2, 1)
    score, cls = cls_p.max(dim=-1)
    return boxes, obj * score, cls


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def ciou_pair(a, b):
    """Complete IoU between matched pairs, (N,4) against (N,4) -> (N,)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    iou = inter / (area_a + area_b - inter).clamp(min=EPS)
    cx = (a[:, 0] + a[:, 2] - b[:, 0] - b[:, 2]) / 2
    cy = (a[:, 1] + a[:, 3] - b[:, 1] - b[:, 3]) / 2
    lt2 = torch.min(a[:, :2], b[:, :2])
    rb2 = torch.max(a[:, 2:], b[:, 2:])
    diag = ((rb2 - lt2) ** 2).sum(-1).clamp(min=EPS)
    wa = (a[:, 2] - a[:, 0]).clamp(min=EPS)
    ha = (a[:, 3] - a[:, 1]).clamp(min=EPS)
    wb = (b[:, 2] - b[:, 0]).clamp(min=EPS)
    hb = (b[:, 3] - b[:, 1]).clamp(min=EPS)
    v = (4 / math.pi ** 2) * (torch.atan(wa / ha) - torch.atan(wb / hb)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + EPS)
    return iou - ((cx ** 2 + cy ** 2) / diag + alpha * v)


def ciou_grid(a, b):
    """Complete IoU between (B,N,4) and (B,M,4) -> (B,N,M)."""
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * \
             (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * \
             (b[..., 3] - b[..., 1]).clamp(min=0)
    lt = torch.max(a[:, :, None, :2], b[:, None, :, :2])
    rb = torch.min(a[:, :, None, 2:], b[:, None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, :, None] + area_b[:, None, :] - inter
    iou = inter / union.clamp(min=EPS)
    ca = (a[:, :, None, :2] + a[:, :, None, 2:]) / 2
    cb = (b[:, None, :, :2] + b[:, None, :, 2:]) / 2
    rho = ((ca - cb) ** 2).sum(-1)
    lt2 = torch.min(a[:, :, None, :2], b[:, None, :, :2])
    rb2 = torch.max(a[:, :, None, 2:], b[:, None, :, 2:])
    diag = ((rb2 - lt2) ** 2).sum(-1).clamp(min=EPS)
    wa = (a[:, :, 2] - a[:, :, 0]).clamp(min=EPS)[:, :, None]
    ha = (a[:, :, 3] - a[:, :, 1]).clamp(min=EPS)[:, :, None]
    wb = (b[:, :, 2] - b[:, :, 0]).clamp(min=EPS)[:, None, :]
    hb = (b[:, :, 3] - b[:, :, 1]).clamp(min=EPS)[:, None, :]
    v = (4 / math.pi ** 2) * (torch.atan(wa / ha) - torch.atan(wb / hb)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + EPS)
    return iou - (rho / diag + alpha * v)


def assign_batch(pred_boxes, cls_prob, gt_boxes, gt_mask, gt_cls, topk=2,
                 alpha=0.5, beta=6.0, iou_min=0.05):
    """Task-aligned assignment for a batch, one level, one target per cell.

    ``pred_boxes`` (B,A,4), ``cls_prob`` (B,A,nc), ``gt_boxes`` (B,G,4),
    ``gt_mask`` (B,G) bool, ``gt_cls`` (B,G) long.

    Returns (img, cell, gt, align): flat matched pairs, at most one per
    (image, cell), plus the per-image normaliser of the alignment metric.
    """
    b, a, _ = pred_boxes.shape
    g = gt_boxes.shape[1]
    dev = pred_boxes.device
    empty = (torch.zeros(0, dtype=torch.long, device=dev),
             torch.zeros(0, dtype=torch.long, device=dev),
             torch.zeros(0, dtype=torch.long, device=dev),
             torch.zeros(0, device=dev))
    if g == 0:
        return empty
    u = ciou_grid(pred_boxes.detach(), gt_boxes).clamp(min=0)
    s = cls_prob.detach().gather(2, gt_cls.clamp(min=0)[:, None, :].expand(
        b, a, g)).clamp(min=EPS)
    metric = (s ** alpha) * (u ** beta)
    metric = torch.where((u > iou_min) & gt_mask[:, None, :], metric,
                         torch.zeros_like(metric))
    k = min(topk, a)
    vals, idx = metric.topk(k, dim=1)                       # (B,k,G)
    al = vals.permute(0, 2, 1).reshape(b, g * k)            # (B,G*k)
    cell = idx.permute(0, 2, 1).reshape(b, g * k)
    gtj = torch.arange(g, device=dev)[None, :, None].expand(b, g, k).reshape(
        b, g * k)
    img = torch.arange(b, device=dev)[:, None].expand(b, g * k)
    keep = al > 0
    img, cell, gtj, al = img[keep], cell[keep], gtj[keep], al[keep]
    if cell.numel() == 0:
        return empty
    gid = img * a + cell
    order = torch.argsort(gid.to(torch.float64) * 1000.0 -
                          al.to(torch.float64))
    img, cell, gtj, al, gid = img[order], cell[order], gtj[order], al[order], \
        gid[order]
    first = torch.ones_like(cell, dtype=torch.bool)
    first[1:] = gid[1:] != gid[:-1]
    return img[first], cell[first], gtj[first], al[first]


def student_loss(raw, targets, nc=3, imgsz=IMGSZ, stride=STRIDE,
                 w_box=7.5, w_cls=0.5, w_obj=1.0, topk=2):
    """YOLO-style loss on the raw head for a batch.

    ``targets``: dict of padded tensors
      boxes  (B,G,4) xyxy canvas pixels
      mask   (B,G) bool
      cls    (B,G) long
      soft   (B,G,nc) float in [0,1]  (class targets, teacher probs or one-hot)
      weight (B,G) float
    Returns (total, {"box","cls","obj"}).
    """
    b, a = raw.shape[0], raw.shape[2] * raw.shape[3]
    dev = raw.device
    boxes, score, _ = decode(raw, nc=nc, imgsz=imgsz, stride=stride)
    obj_logit = raw[:, 0].reshape(b, a)
    cls_logit = raw[:, 5:5 + nc].reshape(b, nc, a).permute(0, 2, 1)
    clsp = torch.sigmoid(cls_logit)
    gt_boxes = targets["boxes"].to(dev)
    gt_mask = targets["mask"].to(dev)
    gt_cls = targets["cls"].to(dev).long()
    soft = targets["soft"].to(dev)
    weight = targets["weight"].to(dev)

    img, cell, gtj, al = assign_batch(boxes.detach(), clsp.detach(), gt_boxes,
                                      gt_mask, gt_cls, topk=topk)
    loss_obj_all = F.binary_cross_entropy_with_logits(
        obj_logit, torch.zeros_like(obj_logit), reduction="none")
    zero = raw.sum() * 0.0
    if cell.numel() == 0:
        return w_obj * loss_obj_all.mean(), {"box": zero, "cls": zero,
                                             "obj": loss_obj_all.mean()}

    # per-image normaliser: the strongest alignment in that image
    al_max = torch.zeros(b, device=dev).index_reduce_(
        0, img, al, "amax", include_self=True).clamp(min=EPS)
    alw = (al / al_max[img]) * weight[img, gtj]
    denom = torch.zeros(b, device=dev).index_add_(0, img, alw).clamp(min=1.0)
    pb = boxes[img, cell]
    gb = gt_boxes[img, gtj]
    iou = ciou_pair(pb, gb)
    loss_box = (((1.0 - iou) * alw) / denom[img]).sum() / b
    loss_cls = (F.binary_cross_entropy_with_logits(
        cls_logit[img, cell], soft[img, gtj], reduction="none").sum(-1) *
        alw / denom[img]).sum() / b
    t_obj = torch.zeros_like(obj_logit)
    t_obj[img, cell] = (al / al_max[img]) * weight[img, gtj]
    loss_obj = F.binary_cross_entropy_with_logits(
        obj_logit, t_obj.detach(), reduction="none").mean()
    total = w_box * loss_box + w_cls * loss_cls + w_obj * loss_obj
    return total, {"box": loss_box, "cls": loss_cls, "obj": loss_obj}


# --------------------------------------------------------------------------
# validation: mAP50, mAP50-95
# --------------------------------------------------------------------------


