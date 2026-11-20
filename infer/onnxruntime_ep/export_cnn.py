#!/usr/bin/env python3
"""Export the small CNN used by the ORT-provider test with torch.onnx.export.

    /path/to/venv-with-torch/bin/python export_cnn.py [out.onnx]

The graph deliberately stays inside the engine's declared op set: Conv 3x3/1x1,
Relu, MaxPool, GlobalAveragePool, Flatten, Gemm.  Batch 1, fixed 1x3x32x32, so
the VC4 provider must claim every node (the test asserts that, so a torch
version that emits something unexpected fails loudly instead of silently
splitting).
"""
import os
import sys

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), "scratch", "tmp", "vc4ort", "smallcnn.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.manual_seed(0)
    model = SmallCNN().eval()
    x = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model, x, out,
        input_names=["input"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("wrote %s (%d bytes), torch %s" % (out, os.path.getsize(out),
                                             torch.__version__))


if __name__ == "__main__":
    main()
