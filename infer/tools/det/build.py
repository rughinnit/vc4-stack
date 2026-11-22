#!/usr/bin/env python3
"""Build and export a vc4det variant from its Ultralytics YAML.

    ~/work/torch-vc4/.venv/bin/python infer/tools/det/build.py vc4det-n --imgsz 96

The YAML is instantiated by Ultralytics itself (``DetectionModel``), so the
width/depth scaling is the framework's, not a hand-rolled copy.  This tool then
audits what came out before anything reaches the GPU:

* every internal conv has Cin and Cout a multiple of 16 (the engine's NHWC conv
  constraint; the 3-channel stem and the 6-channel logits are the documented
  exceptions the runner pads);
* no BatchNormalization survives the fuse, no op outside the engine's table;
* MACs are counted from the real shapes, with the engine's white-box costs:
  the stem runs as 16 channels, the logits conv as 16.

Exports are batch-1, static, fp32, opset 17, simplifier off.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVSHIM = os.path.join(os.path.dirname(HERE), "onnx", "tvshim")

# Ops the engine dispatches (docs/ONNX.md sec. 3).  A graph outside this set is
# refused by the executor, so the export fails here first, with the node name.
SUPPORTED = {
    "Conv", "Relu", "Sigmoid", "Add", "Mul", "Sub", "Div", "Split", "Slice",
    "Transpose", "Softmax", "Shape", "Gather", "Squeeze", "Unsqueeze", "Cast",
    "Flatten", "MaxPool", "Concat", "Resize", "MatMul", "Gemm",
    "GlobalAveragePool", "Reshape", "Identity", "Constant", "ConstantOfShape",
    "Clip", "Pad",
}


def ensure_torchvision():
    """Ultralytics imports torchvision at module load; the wheel here does not
    register against this torch build.  Same shim as infer/tools/onnx."""
    try:
        import torchvision
        torchvision.ops.nms
        return "real"
    except Exception:
        for k in list(sys.modules):
            if k == "torchvision" or k.startswith("torchvision."):
                del sys.modules[k]
        sys.path.insert(0, TVSHIM)
        return "shim"


def shape_trace(model, imgsz):
    """Forward once, recording each Conv2d's in/out shape and MACs."""
    import torch
    rows = []
    handles = []

    def hook(mod, inp, out):
        cin = mod.in_channels
        cout = mod.out_channels
        k = mod.kernel_size[0] * mod.kernel_size[1]
        g = mod.groups
        macs = (cin // g) * k * cout * out.shape[-2] * out.shape[-1]
        rows.append({
            "cin": cin, "cout": cout, "k": k, "groups": g,
            "out": [int(v) for v in out.shape[-2:]],
            "macs": int(macs),
        })

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
    with torch.no_grad():
        y = model(torch.zeros(1, 3, imgsz, imgsz))
    for h in handles:
        h.remove()
    return rows, y


def engine_macs(rows):
    """MACs the engine actually issues, not the nominal ones.

    * stem: 3 input channels are zero-padded to 16, so the first conv multiplies
      16, not 3;
    * logits conv: Cout 6 is zero-padded to 16;
    * depthwise: the block-diagonal expansion is Cin x the MACs (none here).
    """
    total = 0
    detail = []
    for i, r in enumerate(rows):
        cin, cout, k = r["cin"], r["cout"], r["k"]
        hw = r["out"][0] * r["out"][1]
        cin_eff = 16 if i == 0 else cin
        cout_eff = 16 if cout % 16 else cout
        m = cin_eff * k * cout_eff * hw
        total += m
        detail.append({
            "layer": i, "nominal": [cin, cout, k, r["out"]],
            "engine_cin": cin_eff, "engine_cout": cout_eff,
            "macs": int(m),
        })
    return total, detail


def audit_channels(rows):
    bad = []
    for i, r in enumerate(rows):
        if i == 0:
            continue                       # stem Cin=3 is padded by the runner
        if r["cin"] % 16 or r["cout"] % 16:
            bad.append({"layer": i, "cin": r["cin"], "cout": r["cout"]})
    # last conv's Cout is the logits width and is padded by the runner
    return [b for b in bad if b["layer"] != len(rows) - 1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="yaml stem, e.g. vc4det-n")
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--scale", default=None)
    args = ap.parse_args(argv)

    tv = ensure_torchvision()
    import torch
    import onnx
    from ultralytics.nn.tasks import DetectionModel

    yaml = os.path.join(HERE, args.variant + ".yaml")
    scale = args.scale or args.variant.rsplit("-", 1)[-1]
    os.makedirs(args.out_dir, exist_ok=True)

    model = DetectionModel(yaml, nc=1, verbose=False)
    model.eval()
    model.fuse()                            # Conv+BN -> Conv, so no BatchNorm

    rows, out = shape_trace(model, args.imgsz)
    bad = audit_channels(rows)
    if bad:
        raise SystemExit("channels not multiples of 16: %s" % bad)

    macs, detail = engine_macs(rows)
    onnx_path = os.path.join(
        args.out_dir, "%s_%d.onnx" % (args.variant, args.imgsz))
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=17, dynamo=False,
        input_names=["images"], output_names=["output0"],
        do_constant_folding=True, verbose=False)
    onnx.checker.check_model(onnx_path)

    m = onnx.load(onnx_path)
    hist = {}
    for n in m.graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    unsupported = sorted(set(hist) - SUPPORTED)
    if unsupported:
        raise SystemExit("unsupported ops for the engine: %s" % unsupported)

    report = {
        "variant": args.variant,
        "yaml": args.variant + ".yaml",
        "scale": scale,
        "imgsz": args.imgsz,
        "torch": torch.__version__,
        "torchvision": tv,
        "onnx_ops": dict(sorted(hist.items())),
        "onnx_nodes": len(m.graph.node),
        "onnx_kib": round(os.path.getsize(onnx_path) / 1024.0, 1),
        "conv_layers": len(rows),
        "params": sum(p.numel() for p in model.parameters()),
        "macs_nominal": int(sum(r["macs"] for r in rows)),
        "macs_engine": int(macs),
        "gflop_engine": round(macs * 2 / 1e9, 4),
        "output_shape": [int(v) for v in out.shape],
        "layers": detail,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "layers"},
                     indent=1))
    print("MACs (engine) %.2f M  = %.4f GFLOP" %
          (macs / 1e6, macs * 2 / 1e9))
    return report


if __name__ == "__main__":
    main()
