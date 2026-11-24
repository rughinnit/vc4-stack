#!/usr/bin/env python3
"""Train the vc4det student on a YOLO dataset, with and without a teacher.

    /mnt/Ext/venv-tsl2/bin/python infer/tools/det/train_student.py \
        --data ~/bccd/datasets/merged --pseudo /mnt/Ext/bccd-student/pseudo/train \
        --epochs 400 --batch 64 --imgsz 96 --device cuda:0 \
        --out /mnt/Ext/bccd-student/runs/s1

The student is :mod:`student` (8 convs, 12x12 cells at stride 8, 8 logits:
objectness, cx, cy, w, h, three class scores).  The dataset is YOLO txt, ground
truth in ``<data>/labels/<split>`` plus optional teacher pseudo-labels in
``--pseudo`` (see :mod:`pseudo`), merged per image: a pseudo box that overlaps
a ground-truth box is dropped, so the teacher only contributes objects the
labels do not have, and its confidence rides along as the soft class target.

fp32 only: this gfx803 part has no usable AMP path, and the model is 81 k
parameters, so there is nothing to gain from it and a NaN to lose.

Modes
    --eval-only CKPT          validate a checkpoint and stop
    --export-only CKPT        fuse, export ONNX, verify it, and stop
    --teacher-eval PT         score an ultralytics checkpoint on the same val
                              split with the same letterbox, decode and AP code
                              (this is the number the student is compared with)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import det_post as P  # noqa: E402
import student as S  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def read_yolo(path, w, h, nc):
    """YOLO txt -> dict.  5 columns = ground truth, 9 = pseudo (conf+p...)."""
    boxes, cls, soft, weight, is_ps = [], [], [], [], []
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            cls.append(c)
            if len(p) >= 9:
                prob = np.asarray([float(v) for v in p[6:9]], np.float32)
                if prob.sum() <= 0:
                    prob = np.zeros(nc, np.float32)
                    prob[c] = 1.0
                soft.append(prob)
                weight.append(1.0)
                is_ps.append(True)
            else:
                prob = np.zeros(nc, np.float32)
                prob[c] = 1.0
                soft.append(prob)
                weight.append(1.0)
                is_ps.append(False)
    if not boxes:
        return None
    return {"boxes": np.asarray(boxes, np.float32),
            "cls": np.asarray(cls, np.int64),
            "soft": np.stack(soft).astype(np.float32),
            "weight": np.asarray(weight, np.float32),
            "pseudo": np.asarray(is_ps, bool)}


def merge_targets(gt, ps, nc, dedup=0.5):
    """Ground truth plus pseudo boxes that do not duplicate it."""
    if gt is None:
        gt = {"boxes": np.zeros((0, 4), np.float32),
              "cls": np.zeros(0, np.int64),
              "soft": np.zeros((0, nc), np.float32),
              "weight": np.zeros(0, np.float32)}
    if ps is None or len(ps["boxes"]) == 0:
        return gt
    if dedup > 0 and len(gt["boxes"]):
        iou = P.iou_matrix(ps["boxes"], gt["boxes"])
        keep = iou.max(axis=1) < dedup
    else:
        keep = np.ones(len(ps["boxes"]), bool)
    out = {k: np.concatenate([gt[k], ps[k][keep]], axis=0)
           for k in ("boxes", "cls", "soft", "weight")}
    return out


def rot90_boxes(boxes, w, h):
    """Boxes after a 90 degree CCW rotation of a w x h image."""
    b = boxes.copy()
    b[:, [0, 2]] = boxes[:, [1, 3]]
    b[:, [1, 3]] = w - boxes[:, [2, 0]]
    return b


class CellDataset(torch.utils.data.Dataset):
    """YOLO images -> letterboxed 96x96 tensors and per-image targets."""

    def __init__(self, data, split, imgsz=96, nc=3, train=False,
                 pseudo_dir=None, dedup=0.5, cache=False, seed=0):
        self.img_dir = os.path.join(data, "images", split)
        self.gt_dir = os.path.join(data, "labels", split)
        self.pseudo_dir = pseudo_dir
        self.imgsz = imgsz
        self.nc = nc
        self.train = train
        self.dedup = dedup
        names = sorted(n for n in os.listdir(self.img_dir)
                       if n.lower().endswith(IMG_EXT))
        self.paths = [os.path.join(self.img_dir, n) for n in names]
        self.stems = [os.path.splitext(n)[0] for n in names]
        self._gt = P.index_labels(self.gt_dir)
        self._ps = P.index_labels(pseudo_dir) if pseudo_dir else {}
        self.missing = [i for i in range(len(self.paths))
                        if self.label_path(i, self._gt) is None]
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.cache = None
        if cache:
            self.cache = []
            for p in self.paths:
                with Image.open(p) as im:
                    self.cache.append(np.asarray(im.convert("RGB")))

    def label_path(self, i, index):
        path = index.get(self.stems[i])
        if path is None:
            path = index.get(P.norm_stem(self.stems[i]))
        return path

    def __len__(self):
        return len(self.paths)

    def raw(self, i):
        if self.cache is not None:
            return Image.fromarray(self.cache[i])
        return Image.open(self.paths[i]).convert("RGB")

    def targets(self, i, w, h):
        gt = read_yolo(self.label_path(i, self._gt), w, h, self.nc)
        ps = (read_yolo(self.label_path(i, self._ps), w, h, self.nc)
              if self._ps else None)
        return merge_targets(gt, ps, self.nc, self.dedup)

    def __getitem__(self, i):
        img = self.raw(i)
        w, h = img.size
        t = self.targets(i, w, h)
        boxes = t["boxes"].copy()
        cls, soft, weight = t["cls"], t["soft"], t["weight"]

        if self.train:
            r = self.py_rng
            if r.random() < 0.75:                       # 0/90/180/270
                for _ in range(r.randint(1, 3)):
                    img = img.rotate(90, expand=True)   # counter-clockwise
                    boxes = rot90_boxes(boxes, w, h)    # a w x h canvas -> h x w
                    w, h = h, w
            if r.random() < 0.5:                        # mirror
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x1 = boxes[:, 0].copy()
                boxes[:, 0] = w - boxes[:, 2]
                boxes[:, 2] = w - x1
            if r.random() < 0.6:                        # scale jitter
                z = r.uniform(0.9, 1.2)
                nw, nh = max(8, int(round(w * z))), max(8, int(round(h * z)))
                img = img.resize((nw, nh), Image.BILINEAR)
                boxes = boxes * z
                w, h = nw, nh
            if r.random() < 0.5:                        # photometric
                img = ImageEnhance.Brightness(img).enhance(r.uniform(0.85, 1.15))
                img = ImageEnhance.Contrast(img).enhance(r.uniform(0.85, 1.15))

        x, b96, meta = S.letterbox(img, boxes, self.imgsz)
        # drop boxes the canvas/crop cut away
        if len(b96):
            orig_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * \
                np.maximum(boxes[:, 3] - boxes[:, 1], 0)
            c = np.clip(b96, 0, self.imgsz)
            area = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
            ok = (area > 1.0) & (area > 0.3 * orig_area * meta["scale"] ** 2)
            b96, cls, soft, weight = b96[ok], cls[ok], soft[ok], weight[ok]
        return (torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(b96)),
                torch.from_numpy(np.ascontiguousarray(cls)),
                torch.from_numpy(np.ascontiguousarray(soft)),
                torch.from_numpy(np.ascontiguousarray(weight)))


def collate(batch):
    """Stack images, pad the per-image targets to (B,Gmax,...) with a mask."""
    x = torch.stack([b[0] for b in batch])
    gmax = max(1, max(int(b[1].shape[0]) for b in batch))
    b_n, nc = len(batch), batch[0][3].shape[1]
    boxes = torch.zeros(b_n, gmax, 4)
    mask = torch.zeros(b_n, gmax, dtype=torch.bool)
    cls = torch.zeros(b_n, gmax, dtype=torch.long)
    soft = torch.zeros(b_n, gmax, nc)
    weight = torch.zeros(b_n, gmax)
    for i, (_, bx, cl, sf, wt) in enumerate(batch):
        n = int(bx.shape[0])
        if not n:
            continue
        boxes[i, :n] = bx
        mask[i, :n] = True
        cls[i, :n] = cl
        soft[i, :n] = sf
        weight[i, :n] = wt
    return x, {"boxes": boxes, "mask": mask, "cls": cls, "soft": soft,
               "weight": weight}


def val_collate(batch):
    x = torch.stack([b[0] for b in batch])
    return x, [b[1] for b in batch], [b[2] for b in batch]


@torch.no_grad()
def validate(model, loader, nc=3, imgsz=96, device="cpu", conf=0.001, iou=0.6):
    model.eval()
    preds, gts = [], []
    for x, boxes, cls in loader:
        raw = model(x.to(device))
        b, s, c = S.decode(raw, nc=nc, imgsz=imgsz)
        for i in range(x.shape[0]):
            pb, ps, pc = P.postprocess(b[i].cpu().numpy(), s[i].cpu().numpy(),
                                      c[i].cpu().numpy(), conf=conf, iou=iou,
                                      max_det=100, nc=nc)
            preds.append((pb, ps, pc))
            gts.append((boxes[i].numpy(), cls[i].numpy()))
    res = P.map_metric(preds, gts, nc=nc)
    ndet = [len(p[1]) for p in preds]
    res["det_per_image"] = float(np.mean(ndet)) if ndet else 0.0
    return res, preds, gts


# --------------------------------------------------------------------------
# teacher, scored with the same protocol
# --------------------------------------------------------------------------
@torch.no_grad()
def teacher_eval(weights, data, split, imgsz, nc, device):
    from ultralytics import YOLO
    m = YOLO(weights)
    net = m.model.to(device).eval()
    ds = CellDataset(data, split, imgsz=imgsz, nc=nc, train=False)
    preds, gts = [], []
    for i in range(len(ds)):
        img = ds.raw(i)
        w, h = img.size
        t = ds.targets(i, w, h)
        x, gt96, _ = S.letterbox(img, t["boxes"], imgsz)
        out = net(torch.from_numpy(x)[None].to(device))
        if isinstance(out, (list, tuple)):
            out = out[0]                     # (1, 4+nc, anchors), decoded
        a = out[0].float().cpu().numpy()     # (4+nc, anchors)
        if a.shape[0] < a.shape[1]:
            a = a.T                          # -> (anchors, 4+nc)
        # ultralytics >= 8.2 returns cx, cy, w, h in the head; the DFL decode is
        # dist2bbox(..., xywh=True), xyxy only appears inside the NMS op
        cx, cy, bw, bh = (a[:, 0], a[:, 1], a[:, 2], a[:, 3])
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                        axis=1).astype(np.float32)
        pb, ps, pc = P.postprocess(xyxy, a[:, 4:].max(1),
                                   a[:, 4:].argmax(1).astype(np.int64),
                                   conf=0.001, iou=0.6, max_det=100, nc=nc)
        preds.append((pb, ps.astype(np.float32), pc))
        gts.append((gt96, t["cls"]))
    res = P.map_metric(preds, gts, nc=nc)
    res["weights"] = os.path.abspath(weights)
    res["imgsz"] = imgsz
    res["protocol"] = "same letterbox + own AP code as the student"
    res["det_per_image"] = float(np.mean([len(p[1]) for p in preds]))
    return res


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_onnx(model, path, imgsz=96, nc=3):
    import onnx
    m = S.fuse(model).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(m, dummy, path, opset_version=13,
                      input_names=["images"], output_names=["output0"],
                      do_constant_folding=True, verbose=False,
                      training=torch.onnx.TrainingMode.EVAL)
    onnx.checker.check_model(path)
    g = onnx.shape_inference.infer_shapes(onnx.load(path))
    hist = {}
    for n in g.graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    # numeric check torch (fused) against onnxruntime, fp32
    import onnxruntime as ort
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    # nominally the same MACs as build.py's engine count: stem Cin and logits
    # Cout are padded to 16 by the runner
    shapes = []
    for node in g.graph.node:
        if node.op_type != "Conv":
            continue
        w = [a for a in g.graph.initializer if a.name == node.input[1]][0]
        co, ci, kh, kw = w.dims
        out_shape = None
        for vi in g.graph.value_info:
            if vi.name == node.output[0]:
                out_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes.append({"cin": int(ci), "cout": int(co), "k": int(kh * kw),
                       "out": out_shape})
    out_dims = [d.dim_value for d in g.graph.output[0].type.tensor_type.shape.dim]
    macs = 0
    for i, s in enumerate(shapes):
        if s["out"]:
            hw = s["out"][2] * s["out"][3]
        elif len(out_dims) == 4:
            hw = out_dims[2] * out_dims[3]
        else:
            hw = 0
        cin = 16 if i == 0 else s["cin"]
        cout = 16 if s["cout"] % 16 else s["cout"]
        macs += cin * s["k"] * cout * hw
    return {"path": os.path.abspath(path), "opset": 13, "imgsz": imgsz,
            "nodes": len(g.graph.node), "ops": dict(sorted(hist.items())),
            "params": int(sum(p.numel() for p in model.parameters())),
            "macs_engine": int(macs), "gflop_engine": round(macs * 2 / 1e9, 4),
            "kib": round(os.path.getsize(path) / 1024.0, 1),
            "ort_max_abs_diff": diff, "convs": shapes}


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/bccd/datasets/merged"))
    ap.add_argument("--pseudo", default=None, help="pseudo-label dir for train")
    ap.add_argument("--out", default=os.path.expanduser("~/scratch/bccd-student/run"))
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--w-box", type=float, default=7.5)
    ap.add_argument("--w-cls", type=float, default=0.5)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--dedup-iou", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--eval-only", default=None)
    ap.add_argument("--export-only", default=None)
    ap.add_argument("--teacher-eval", default=None)
    ap.add_argument("--teacher-imgsz", type=int, default=None)
    ap.add_argument("--split", default="val")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    if args.teacher_eval:
        imgsz = args.teacher_imgsz or args.imgsz
        res = teacher_eval(args.teacher_eval, args.data, args.split, imgsz,
                           args.nc, dev)
        print(json.dumps(res, indent=1))
        with open(os.path.join(args.out, "teacher_%d.json" % imgsz), "w") as f:
            json.dump(res, f, indent=1)
        return 0

    val_ds = CellDataset(args.data, args.split, imgsz=args.imgsz, nc=args.nc,
                         train=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=1,
        collate_fn=val_collate)

    if args.eval_only:
        ck = torch.load(args.eval_only, map_location=dev)
        model = S.Student(nc=args.nc).to(dev)
        model.load_state_dict(ck.get("ema", ck.get("model", ck)))
        res, _, _ = validate(model, val_loader, nc=args.nc, imgsz=args.imgsz,
                            device=dev)
        print(json.dumps(res, indent=1))
        return 0

    if args.export_only:
        ck = torch.load(args.export_only, map_location="cpu")
        model = S.Student(nc=args.nc)
        model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        path = os.path.join(args.out, "vc4det-n3_%d.onnx" % args.imgsz)
        info = export_onnx(model, path, imgsz=args.imgsz, nc=args.nc)
        print(json.dumps(info, indent=1))
        with open(os.path.join(args.out, "export.json"), "w") as f:
            json.dump(info, f, indent=1)
        return 0

    train_ds = CellDataset(args.data, "train", imgsz=args.imgsz, nc=args.nc,
                           train=True, pseudo_dir=args.pseudo,
                           dedup=args.dedup_iou, cache=args.cache,
                           seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate, drop_last=True,
        persistent_workers=args.workers > 0)

    model = S.Student(nc=args.nc).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.999))
    steps_per_epoch = max(1, len(train_ds) // args.batch)
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch

    log("dataset train=%d (background %d) val=%d batch=%d steps/epoch=%d "
        "epochs=%d pseudo=%s"
        % (len(train_ds), len(train_ds.missing), len(val_ds), args.batch,
           steps_per_epoch, args.epochs, args.pseudo or "none"))
    log("model params=%d device=%s torch=%s"
        % (sum(p.numel() for p in model.parameters()), dev, torch.__version__))

    it = 0
    best = -1.0
    hist = []
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        agg = {"box": 0.0, "cls": 0.0, "obj": 0.0, "n": 0, "nan": 0}
        loss = None
        for x, targets in train_loader:
            for g in opt.param_groups:
                g["lr"] = (args.lr * (it + 1) / max(1, warm) if it < warm else
                           0.5 * args.lr * (1 + math.cos(
                               math.pi * (it - warm) / max(1, total - warm))))
            x = x.to(dev, non_blocking=True)
            raw = model(x)
            loss, comps = S.student_loss(raw, targets, nc=args.nc,
                                         imgsz=args.imgsz, w_box=args.w_box,
                                         w_cls=args.w_cls, w_obj=args.w_obj,
                                         topk=args.topk)
            if not torch.isfinite(loss):
                agg["nan"] += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                log("epoch %d step %d: non-finite loss, step skipped"
                    % (ep + 1, it))
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            it += 1
            d = args.ema if it > warm else 0.9
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(),
                                  model.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    else:
                        pe.copy_(pm)
            for k in comps:
                agg[k] = agg[k] + comps[k]        # already a per-batch mean
            agg["n"] += 1
        nb = max(1, agg["n"])
        msg = ("epoch %d/%d loss %.4f (box %.4f cls %.4f obj %.4f) lr %.2e "
               "%.1fs%s" % (
                   ep + 1, args.epochs,
                   float(loss) if loss is not None else float("nan"),
                   float(agg["box"]) / nb, float(agg["cls"]) / nb,
                   float(agg["obj"]) / nb, opt.param_groups[0]["lr"],
                   time.time() - t_start,
                   "" if not agg["nan"] else " nan=%d" % agg["nan"]))
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            res, _, _ = validate(ema, val_loader, nc=args.nc, imgsz=args.imgsz,
                                 device=dev)
            msg += " | val mAP50 %.4f mAP50-95 %.4f det/img %.1f" % (
                res["map50"], res["map"], res["det_per_image"])
            hist.append({"epoch": ep + 1, "loss": float(loss),
                         "map50": res["map50"], "map": res["map"]})
            if res["map"] > best:
                best = res["map"]
                torch.save({"ema": ema.state_dict(),
                            "model": model.state_dict(), "epoch": ep + 1,
                            "args": vars(args), "val": res},
                           os.path.join(args.out, "best.pt"))
        log(msg)
        torch.save({"ema": ema.state_dict(), "model": model.state_dict(),
                    "epoch": ep + 1, "args": vars(args)},
                   os.path.join(args.out, "last.pt"))

    path = os.path.join(args.out, "vc4det-n3_%d.onnx" % args.imgsz)
    info = export_onnx(ema, path, imgsz=args.imgsz, nc=args.nc)
    log("export %s" % json.dumps(info))
    res, _, _ = validate(ema, val_loader, nc=args.nc, imgsz=args.imgsz,
                         device=dev)
    report = {"args": vars(args), "val": res, "export": info,
              "best_map": best, "history": hist,
              "seconds": round(time.time() - t_start, 1)}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("final val mAP50 %.4f mAP50-95 %.4f" % (res["map50"], res["map"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
