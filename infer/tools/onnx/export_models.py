#!/usr/bin/env python3
"""Export the two small ONNX models and their PyTorch CPU references.

The graphs are deliberately fixed-shape (batch 1) and use only operators the
VC4 runner implements.  Model B is a YOLO-ish block: stride-2 conv, depthwise
conv, 5x5 max pool, channel concat, nearest Resize.

Outputs (in models/):
    modelA.onnx  modelA.in.npy  modelA.out.npy
    modelB.onnx  modelB.in.npy  modelB.out.npy
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models")
os.makedirs(OUT, exist_ok=True)


class NetA(nn.Module):
    """2x Conv(3x3)+ReLU -> MaxPool 2x2 -> Flatten -> Linear -> Softmax."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.softmax(x, dim=1)


class NetC(nn.Module):
    """Conv+ReLU -> GlobalAveragePool -> Flatten -> Linear (tests GAP)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)


class YoloBlock(nn.Module):
    """stride-2 conv, depthwise conv, 5x5 max pool, concat, 2x nearest resize."""

    def __init__(self):
        super().__init__()
        self.conv_s2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.head = nn.Conv2d(64, 16, 1)

    def forward(self, x):
        c = F.relu(self.conv_s2(x))            # 32 x 16 x 16
        d = F.relu(self.dw(c))                 # 32 x 16 x 16
        p = self.pool5(d)                      # 32 x 16 x 16
        cat = torch.cat([p, c], dim=1)         # 48 x 16 x 16
        up = F.interpolate(cat, scale_factor=2, mode="nearest")  # 48 x 32 x 32
        return self.head(up)


def export(name, model, shape, seed):
    torch.manual_seed(seed)
    model.eval()
    x = torch.randn(*shape, dtype=torch.float32)
    with torch.inference_mode():
        y = model(x)
    onnx_path = os.path.join(OUT, name + ".onnx")
    torch.onnx.export(
        model, x, onnx_path,
        opset_version=13,
        input_names=["x"], output_names=["y"],
        do_constant_folding=True,
        dynamo=False,
    )
    np.save(os.path.join(OUT, name + ".in.npy"), x.numpy())
    np.save(os.path.join(OUT, name + ".out.npy"), y.numpy())
    print("%s: input %s -> output %s  (%.1f KiB onnx)" %
          (name, tuple(x.shape), tuple(y.shape),
           os.path.getsize(onnx_path) / 1024.0))


def main():
    export("modelA", NetA(), (1, 3, 32, 32), 1234)
    export("modelB", YoloBlock(), (1, 16, 32, 32), 5678)
    export("modelC", NetC(), (1, 16, 8, 8), 9012)


if __name__ == "__main__":
    main()
