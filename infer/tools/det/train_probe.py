#!/usr/bin/env python3
"""Time one forward+backward of a vc4det graph on the board's four A53 cores.

    ~/work/torch-vc4/.venv/bin/python infer/tools/det/train_probe.py vc4det-n --imgsz 96

The graph has no Ultralytics loss head (no Split, no DFL), so `yolo train`
cannot consume it: the 6-channel per-cell logits need their own assignment and
loss.  This probe measures the compute a training step would cost here, so the
"train it on the board" question can be answered with a number instead of a
guess.  Five steps, batch 4, threads 4, median reported.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TVSHIM = os.path.join(os.path.dirname(HERE), "onnx", "tvshim")


def ensure_torchvision():
    try:
        import torchvision
        torchvision.ops.nms
    except Exception:
        for k in list(sys.modules):
            if k == "torchvision" or k.startswith("torchvision."):
                del sys.modules[k]
        sys.path.insert(0, TVSHIM)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args(argv)

    ensure_torchvision()
    import torch
    from ultralytics.nn.tasks import DetectionModel

    torch.set_num_threads(4)
    model = DetectionModel(os.path.join(HERE, args.variant + ".yaml"),
                           nc=1, verbose=False)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(args.batch, 3, args.imgsz, args.imgsz)

    times = []
    for i in range(args.steps + 1):
        t0 = time.perf_counter()
        out = model(x)
        loss = out.float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.set_num_threads(4)
        times.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(times[1:])
    steps_per_epoch = 118287 // args.batch          # COCO train2017
    report = {
        "variant": args.variant, "imgsz": args.imgsz, "batch": args.batch,
        "threads": 4, "step_ms": round(med, 1),
        "steps": [round(t, 1) for t in times],
        "coco_steps_per_epoch": steps_per_epoch,
        "coco_epoch_hours": round(steps_per_epoch * med / 3.6e6, 2),
        "torch": torch.__version__,
    }
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
