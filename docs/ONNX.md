# ONNX on the VideoCore IV engine

`libvc4infer` is an **operator library**, not a graph runtime.  This document
describes the graph path added on top of it: a small ONNX exporter/runner in
`onnx/`, the two engine ops it needed (`vc4_maxpooln`, `vc4_softmax_fwd`), the
measured end-to-end numbers, and the honest coverage gap for YOLOv8n.

Everything that *computes* runs on the QPU.  The host only moves bytes
(padding, channel gathers, concat, nearest resize, boundary compaction) — the
same class of work the engine already does for its im2col layout.  An op or
attribute the engine cannot dispatch raises an error naming the node and its
ONNX op type; there is no CPU math fallback.

Hardware: Raspberry Pi 3B, BCM2837, VideoCore IV @ **stock** clocks
(core 400 MHz / V3D 300 MHz), kernel 6.18, engine through **vc4d**
(`VC4INFER_DAEMON=1`, unprivileged, one GPU user at a time).

Test date: 2026-11-19.  Engine tree: `engine/` at this commit.

## 1. Files

```
onnx/export_models.py   define + torch.onnx.export the fixed-shape models,
                        save models/<name>.onnx, .in.npy and the torch CPU
                        reference .out.npy
onnx/onnxrun.py         the ONNX executor (onnx package + ctypes binding)
onnx/verify.py          3 runs per model, compares with the torch reference,
                        prints per-node wall/GPU time and launch counts
onnx/export_yolo.py     real pretrained YOLOv8n -> ONNX (ultralytics)
onnx/yolo_report.py     static op/attribute/shape coverage + a dynamic run
onnx/tvshim/            name-only torchvision stand-in for the YOLO export
```

Generated artifacts live in `onnx/models/` (not committed; regenerate with
`export_models.py`).  The engine additions are:

```
src/vc4_convbwd.c   vc4_maxpooln          generic NxN / stride-S max pool
src/vc4_train.c     vc4_softmax_fwd       forward softmax (fused kernel 'p' probe)
tools/kgen.py       build_pooln           the generic pool kernel
tests/test_onnxops.c CPU-oracle check of both
python/vc4infer.py  binding for both ops, vc4_get_stats, and a conv2d argtypes fix
```

## 2. Running a model

```sh
cd vc4-stack
source /home/raspi/work/torch-vc4/.venv/bin/activate   # torch CPU + numpy + onnx

python onnx/export_models.py
VC4INFER_DAEMON=1 python onnx/verify.py                # all three models
VC4INFER_DAEMON=1 python onnx/onnxrun.py onnx/models/modelB.onnx \
        --input onnx/models/modelB.in.npy --profile --repeat 3

# optional real YOLOv8n export + coverage report (downloads ~6 MB of weights)
python onnx/export_yolo.py                             # YOLO_IMGSZ=32 for 32x32
VC4INFER_DAEMON=1 python onnx/yolo_report.py onnx/models/yolov8n_640.onnx
```

The runner takes the GPU lock through vc4d; do **not** hold
`/tmp/vc4-gpu.lock` in the calling shell in daemon mode (vc4d takes it per
launch, and a second holder makes every launch time out).

`onnxrun.py` accepts fixed-shape fp32 models with batch 1 and one float input.
It evaluates the graph in ONNX topological order and keeps every activation in
the single 64 MiB arena.

## 3. Supported nodes

| ONNX node | implementation | constraints |
|---|---|---|
| `Conv` 3x3 / 1x1, stride 1, group 1 | `vc4_conv2d` (QPU im2col + QPU GEMM) | pads symmetric or pre-padded on host; `(H+2p)*align16(W+2p)*4 <= 8191` |
| `Conv` 3x3 stride 2 pad 1, group 1 | 4 phase images + one stride-1 `vc4_conv2d` | graph-level decomposition, 4x the MACs (see §6) |
| `Conv` 1x1 stride 2 pad 0 | phase (0,0) + `vc4_conv2d` 1x1 | 4x fewer outputs; no extra MACs |
| `Conv` depthwise (groups = Cin = Cout), stride 1 | block-diagonal dense weight + `vc4_conv2d` | Cout = Cin, stride 1; Cin x the MACs (see §6) |
| `Relu`, `Add`, `Mul`, `Sigmoid` | `vc4_elementwise` | elementwise; `Add`/`Mul` require equal shapes |
| `MaxPool` 2x2 stride 2 | `vc4_maxpooln` | native path is the general kernel |
| `MaxPool` NxN, any stride/pad | `vc4_maxpooln` (new QPU kernel) | border filled with `-inf` on host (bytes); output compacted |
| `Concat` | host byte concat | channel axis of NCHW only |
| `Resize` nearest | host byte gather | `mode=nearest`, `asymmetric` or `half_pixel`, integer or float scale |
| `MatMul` | `vc4_matmul` | 2-D only |
| `Gemm` | `vc4_matmul` + GPU `Add` for bias | `transA=0`, `alpha=1`, `beta∈{0,1}`; C of `[N]`, `[1,N]` or `[M,N]` |
| `Softmax` | `vc4_softmax_fwd` (new) | last axis, ≤16 classes, rows padded to 8 |
| `Flatten`, `Reshape`, `Identity` | view (no copy) | element count must match |
| `GlobalAveragePool` | `vc4_matmul` with a `1/HW` column | NCHW, batch 1 |

Everything else fails loudly, e.g.:

```
onnxrun.UnsupportedNode: node '/model.9/Split' (Split): unsupported op Split
```

Inputs/outputs must be fp32; batch > 1, dynamic shapes, quantization, arbitrary
permutations, `Slice`, `Shape`/`Gather`/`Expand`, `BatchNormalization`, `Pow`,
`Div` and `LRN` are not implemented.

## 4. Measured results

Torch CPU reference, tolerance `|got-ref| <= 1e-4 + 1e-4*max(|ref|,1)`.
A warm-up run compiles the QPU kernels (JIT, not counted); the three measured
runs follow.  "gpu" is the engine's pack+kernel time; "wall" includes the
daemon round trips, the ARM store and the byte-only layout work.

### modelA — 2x Conv3x3+ReLU, MaxPool2x2, Flatten, Linear, Softmax
input `1x3x32x32` → output `1x10`; 8 nodes, 38 launches.

| node | op | wall ms | gpu ms | launches |
|---|---|---:|---:|---:|
| /conv1/Conv | Conv 3→16 3x3 s1 | 6.11 | 2.25 | 3 |
| /Relu | Relu | 1.36 | 0.43 | 6 |
| /conv2/Conv | Conv 16→32 3x3 s1 | 6.66 | 2.94 | 3 |
| /Relu_1 | Relu | 2.09 | 0.79 | 11 |
| /pool/MaxPool | MaxPool 2x2 s2 | 2.85 | 0.78 | 11 |
| /Flatten | Flatten | 0.26 | 0 | 0 |
| /fc/Gemm | Gemm 8192→10 | 16.52 | 6.59 | 3 |
| /Softmax | Softmax 10 | 0.91 | 0.08 | 1 |

3 runs: **38.6 / 38.2 / 38.0 ms**; GPU 13.85 ms; max abs **4.84e-06**,
max rel (den=max(|ref|,1)) **4.84e-06**, **0 / 10 mismatches**.
First cold run after an empty kernel cache: 2.4 s (kgen JIT).

### modelB — YOLO-ish block
Conv 3x3 s2 → depthwise Conv 3x3 → MaxPool 5x5 s1 p2 → Concat → Resize 2x
nearest → Conv 1x1.  input `1x16x32x32` → output `1x16x32x32`; 9 nodes,
26 launches.

| node | op | wall ms | gpu ms | launches |
|---|---|---:|---:|---:|
| /conv_s2/Conv | Conv 16→32 3x3 s2 p1 (4 phases) | 9.86 | 3.86 | 3 |
| /Relu | Relu | 0.87 | 0.22 | 3 |
| /dw/Conv | depthwise 32→32 3x3 (dense expansion) | 4.69 | 2.04 | 3 |
| /Relu_1 | Relu | 0.88 | 0.24 | 3 |
| /pool5/MaxPool | MaxPool 5x5 s1 p2 | 2.57 | 0.78 | 11 |
| /Concat | channel concat 32+32 | 0.93 | 0 | 0 |
| /Constant | Constant | 0.42 | 0 | 0 |
| /Resize | nearest 2x | 5.05 | 0 | 0 |
| /head/Conv | Conv 1x1 64→16 | 4.95 | 0.86 | 3 |

3 runs: **35.8 / 33.4 / 32.3 ms**; GPU 7.99 ms; max abs **4.29e-06**,
max rel (den=max(|ref|,1)) **3.98e-06**, **0 / 16384 mismatches**.
First cold run: 1.9 s.

### modelC — Conv+ReLU, GlobalAveragePool, Flatten, Linear
input `1x16x8x8` → output `1x4`; 5 nodes, 9 launches.
3 runs **6.0 / 5.9 / 6.0 ms**; GPU 0.85 ms; max abs **4.06e-07**,
**0 / 4 mismatches**.

Per-node in `results.json`.  The wall/GPU gap is the per-launch daemon round
trip (~0.2–0.5 ms) plus the ARM-side layout copies; the GEMM in modelA is
dominated by the 64-column pad of a 10-wide output.

## 5. Engine additions

* `vc4_maxpooln(ctx, in, C,H,W, N,stride,pad, out,&Ho,&Wo)` — a straight-line
  4-output-row x 16-column QPU task; the host builds the `-inf` border
  (`tests/test_onnxops.c` checks 2x2/5x5/3x3 stride-2 cases exactly).
* `vc4_softmax_fwd(ctx, logits, rows, cols, out)` — rows % 8, cols ≤ 16.
  Uses the fused training kernel's `p` probe (same degree-5 `2^f` polynomial
  and 3x Newton reciprocal); the host pads to 16 classes with `-1e30` and
  compacts the result.  Measured error ≤ 3.4e-07.
* `python/vc4infer.py` gained `maxpooln`, `softmax_fwd`, `stats`, and a
  corrected `vc4_conv2d` ctypes signature (the old one had the wrong number
  and order of argument types, so `conv2d` could not be called from Python).
* The op table now lists `maxpool/NxNsS` as supported.

## 6. Why stride 2 and depthwise are decomposed

`vc4_conv2d` remains a **stride-1, group-1** kernel: its im2col VDR loads 16
consecutive pixels, so decimating on the QPU would need a different gather.
Rather than approximate or fall back to the CPU, the runner rewrites the node:

* **stride 2, 3x3, pad 1.**  Split the input into four parity phase images
  `P[a,b][i,j] = X[2i+a, 2j+b]` (a byte gather).  Each tap of the original
  conv belongs to exactly one phase and lands at a `dr,dq ∈ {-1,0}` offset, so
  a stride-1 3x3 conv over the concatenated 4*Cin-channel phase tensor with a
  phase-masked weight reproduces the result exactly.  The zeros waste 4x the
  MACs; every MAC still runs on the QPU.  modelB validates it to 4.3e-06.
* **depthwise.**  Expand to a dense `[Cout][Cin][3][3]` block-diagonal weight
  (byte copy + zero fill) and use the ordinary `vc4_conv2d`.  This costs Cin
  times the real MACs.  It is honest but not the right long-term kernel; a
  per-channel schedule is the obvious next step.

The alternative — a CPU convolution or a "close enough" substitute — is not
used anywhere.

## 7. YOLOv8n

A real Ultralytics YOLOv8n was exported and fed to the runner.  Weights are
the official `yolov8n.pt` COCO checkpoint (6.2 MB, 3,151,904 parameters,
8.7 GFLOPs); nothing is synthetic.

Export status: **feasible**, ONNX opset 13, 12.2 MiB.  Two shapes were
exported: the standard `1x3x640x640` and `1x3x32x32` (to separate the engine's
shape limit from the op gap).  The export ran on the same Pi with the venv's
torch 2.15 CPU build.  Caveat: the venv's torchvision 0.29.2 wheel registers
no ops against this custom torch build (`operator torchvision::nms does not
exist`), so `onnx/export_yolo.py` substitutes the name-only shim in
`onnx/tvshim/` when the real import fails.  A plain detection ONNX export
uses no torchvision code; the shim raises if a transform or NMS is actually
called.  The exported graph and weights are the real model.

Graph: **263 nodes**, 127 initializers, one input `images`, one output
`output0` (640: `1x84x8400`, 32: `1x84x21`).

| op | nodes | runner op? |
|---|---:|---|
| Conv | 64 | yes (subject to the pitch limit) |
| Mul | 60 | yes: 59 are SiLU `x*sigmoid(x)`, 1 broadcasts |
| Sigmoid | 58 | yes |
| Constant | 24 | yes |
| Concat | 17 | 13 channel-axis NCHW; 4 are head concats on 3-D tensors (axis -1 / 1) |
| Add | 9 | 8 equal-shape residuals; 1 broadcasts |
| Split | 8 | **no** |
| Reshape | 8 | yes |
| MaxPool | 3 | yes (SPPF 5x5) |
| Resize | 2 | yes (nearest 2x) |
| Div | 2 | **no** |
| Slice | 2 | **no** |
| Sub | 2 | **no** |
| Transpose | 1 | **no** |
| Softmax | 1 | op exists, but this one is the DFL softmax over axis 1, not the last axis |
| Shape | 1 | **no** |
| Gather | 1 | **no** |

**640x640.**  Static check: 224 / 263 nodes pass the runner's op/attribute/
shape tests.  The dynamic run stops at the first node:

```
FAILED: Conv /model.0/conv/Conv: padded plane 322x336 = 432768 B > 8191 B
        (engine im2col VDR pitch limit)
```

15 of the 64 convolutions exceed the pitch limit at 640 (the first layer's
padded plane is 432 KB; the engine's VDR pitch field allows 8191 B).  This is
a hard shape blocker independent of the op gap.

**32x32.**  Every conv now fits, the runner JIT-compiles and executes the
first layers, and then stops at the first real op gap:

```
FAILED: node '/model.2/Split': unsupported op Split
```

Static check: 239 / 263 nodes pass; 24 fail on op type or attribute.
The remaining gap, in order of node count:

| gap | nodes | what it needs |
|---|---:|---|
| `Split` (channel halves of a Conv) | 8 | Slice/Split on axis 1 — a byte gather, the cheapest next step |
| head `Concat` on 3-D tensors (axis -1 or 1) | 4 | Concat on rank-3 tensors (the runner only handles NCHW channel concat) |
| `Div`, `Sub`, `Slice` (anchor/grid decode) | 6 | elementwise Sub/Div and Slice; decode is arithmetic, must be QPU |
| broadcast `Add` / `Mul` | 2 | elementwise broadcast (or a host-side expand + GPU op) |
| DFL `Softmax` on axis 2 | 1 | softmax over a non-last axis |
| `Transpose` (DFL) | 1 | arbitrary 4-D permute (the engine has only 2-D dense transpose) |
| `Shape` + `Gather` | 2 | constant-fold the static shapes (no GPU work needed) |
| Conv pitch at 640 | 15 | channel tiling or an NHWC repack (engine change) |

So the honest verdict is: **YOLOv8n does not run end-to-end on the VC4
today.**  At 32x32 the graph is stopped by `Split` after the first few layers;
at the standard 640x640 it is stopped earlier by the im2col pitch limit.  The
24-node op gap is small and mostly byte-level or simple elementwise work, but
the pitch limit is the structural blocker: no 640x640 convolution fits the
current `(H+2p)*align16(W+2p)*4 <= 8191` constraint.

## 8. Limitations that matter for real models

* **Plane pitch limit.**  `(H+2p)*align16(W+2p)*4 <= 8191` bytes.  At
  640x640 the first YOLO conv needs 1.68 MB; at 64x64 it still needs 21 KB.
  Only feature maps up to roughly 32x32 fit the current im2col VDR pitch
  field.  Removing this is the single biggest blocker for real CNNs
  (channel tiling or an NHWC repack).
* **fp32 only**, no quantization, no dynamic shapes, batch 1.
* **Softmax** is limited to 16 classes by the fused kernel's 16-lane rows.
* **Concat/Resize** are host byte copies, so they cost a round trip but no
  arithmetic.  A real runner would do them in the arena (a memcpy kernel).
* **Launch overhead.**  Small layers are launch-bound: the elementwise path
  issues one launch per 12 groups of 256 floats, so a 32x32x32 ReLU is 11
  launches (~0.8 ms GPU, ~2 ms wall).  Fusing or batching more groups per
  launch is the fix.
* The runner re-executes the graph from scratch each call; weights are
  uploaded once per process but there is no graph-level memory planner.
