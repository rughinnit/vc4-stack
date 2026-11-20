# vc4infer — ONNX inference on the VideoCore IV QPUs

`vc4infer` runs fixed-shape, batch-1, fp32 ONNX graphs on the QPUs of a
Raspberry Pi 1/2/3, Zero or CM1/CM3.  It is the packaged form of the
`libvc4infer` engine and the ONNX executor described in
[ONNX.md](ONNX.md): every float operation — convolutions, GEMMs,
elementwise activations, pooling, softmax — is dispatched to the QPU.  The ARM
cores only move bytes (padding, channel gathers, concat, nearest resize),
allocate, and do integer shape bookkeeping.  An operator, attribute or shape the
engine cannot dispatch raises an error naming the node and its ONNX op type;
there is no CPU arithmetic fallback.

The package contains the CLI (`/usr/bin/vc4infer`), the engine shared library
and header, the ctypes binding, the ONNX executor, the QPU kernel generator that
JITs the kernels into a per-user cache, and the reference models used by
`vc4infer selftest`.

Measured on a Raspberry Pi 3B, BCM2837, Debian 13 arm64, kernel 6.18,
**stock clocks** (core 400 MHz / V3D 300 MHz, `get_throttled=0x0`), engine
through **vc4d** (`VC4INFER_DAEMON=1`, unprivileged, one GPU user at a time).

## Install

The vc4infer package is published in the same signed apt repository as vc4top:

```sh
# one-time, if the repository is not configured yet
curl -fsSL https://rughinnit.github.io/vc4-apt/key.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/vc4top-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/vc4top-archive-keyring.gpg] \
https://rughinnit.github.io/vc4-apt stable main" \
  | sudo tee /etc/apt/sources.list.d/vc4-apt.list

sudo apt update
sudo apt install vc4infer
```

Dependencies (installed automatically): `python3`, `python3-numpy`,
`python3-onnx`, `python3-pil`.

The same Python package is also on PyPI-style install from the same host; the
wheel is pure Python and uses the engine library the .deb installs:

```sh
sudo apt install vc4infer
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ vc4infer
# (the older https://rughinnit.github.io/vc4top apt/pip location is deprecated)
```

See [PYTHON-API.md](PYTHON-API.md) for the `Engine` / `Detector` API.

`vc4infer` speaks to the unprivileged helper daemon **vc4d** over
`/run/vc4d.sock`; the CLI defaults `VC4INFER_DAEMON=1`.  If vc4d is not running
the command says so instead of failing with a permission error:

```
vc4infer: vc4d is not running (no /run/vc4d.sock).
  start it:  sudo systemctl start vc4d
  or run as root with the direct backend: VC4INFER_DAEMON=0
```

The client never opens `/dev/mem`.  That descriptor belongs to vc4d, which
drops to `nobody` with an empty capability set before serving requests; a
client run only has its socket and the dma-buf mapping.  (vc4d itself is not
part of this package — it is the board's privileged helper; see
[DAEMON.md](DAEMON.md).)

## Commands

```
vc4infer --version
vc4infer run  <model.onnx> --input <image|csv|npy> [--imgsz N] [--conf 0.25] [--iou 0.45]
              [--mode strict|hybrid] [--host-policy structural|any] [--json]
vc4infer bench <model.onnx> [--imgsz N] [--reps N] [--input <image|npy|csv>]
              [--mode strict|hybrid] [--host-policy structural|any] [--json]
vc4infer nodes <model.onnx> [-v] [--mode strict|hybrid]
              [--host-policy device|structural|any] [--json]
vc4infer util <model.onnx> [--imgsz N] [--reps N]
              [--mode strict|hybrid] [--host-policy device|structural|any] [--json]
vc4infer selftest [--models DIR]
```

### `run`

Runs one inference.  Images (JPEG/PNG/BMP/WebP) are letterboxed to the model
input size, converted to RGB, scaled to `[0,1]` and uploaded as NCHW fp32.
A YOLOv8/YOLO11 detection head (`[1, 4+nc, A]` or `[1, A, 4+nc]`, and the
YOLOv5 `5+nc` layout with an objectness column) is decoded on the host: boxes
from the `cx,cy,w,h` rows, argmax class, confidence threshold, class-aware
greedy NMS.  Boxes are mapped back through the letterbox to original image
pixels.  An annotated image is written next to the input unless `--no-save` is
given; `--json` prints one JSON document instead of the table.

```
$ vc4infer run yolov8n.onnx --input bus.jpg --imgsz 160
model:  yolov8n.onnx
input:  bus.jpg (810x1080 -> 160x160)  ->  1x3x160x160
output: (1, 84, 525)  fp32
run:    970.7 ms  (271 GPU ms, 742 launches; first run in a process, so it includes the one-time weight-panel packs)
decode: YOLO head channels_first, 525 anchors, 80 classes; 38 above conf 0.25 -> 4 after NMS (IoU 0.45); boxes in original image pixels
  #   class            conf        x1        y1        x2        y2
  1   person          0.778      39.0     409.2     228.7     901.7
  2   bus             0.703       1.8     226.9     800.6     758.8
  3   person          0.619     678.5     424.5     810.0     865.8
  4   person          0.597     206.2     391.5     353.2     863.2
saved:  bus_vc4infer.jpg
```

The decode and the NMS are host code by design: the head is an affine
combination plus an argmax, and the engine has no sort/gather kernels.  No
network arithmetic happens on the ARM — only comparisons, sorting and
thresholding of the network's own output.

`--input` also accepts a `.npy` tensor (exact model input shape, or that shape
without the batch axis) and a `.csv` of floats that fills the input in NCHW
order.  Non-detection models (classifiers) print the output statistics and the
top-5 values instead of a detection table.

`--mode strict` is the default and refuses any node without a VC4 kernel.
`--mode hybrid` is opt-in and defaults to `--host-policy device`: the host does
only plan-time constant/shape folding and orchestration, every runtime node
must be VC4, a numeric op with no kernel raises `HostNotAllowed` (exit 4), and
`If`/`Loop`/`Scan` raise `NeedsDeviceDispatch` (exit 6).  `--host-policy
structural` allows metadata and control flow on the host at runtime;
`--host-policy any` additionally runs unwritten numeric ops in the numpy
reference evaluator (compatibility only).  The `tiers:` line reports the split,
`host_compute_nodes` and `bytes_copied_by_host` are printed explicitly, and
hybrid timings are *not* VC4-only numbers (see [ONNX.md](ONNX.md) section 10).

### `bench`

One warm-up run performs the kernel JIT (not counted), then `--reps` timed runs.
The `tiers:` line splits nodes into `vc4`, `host-meta`, `host-control` and
`host-unwritten`; the next line gives the compute-node VC4 share and the wall
time attributed to the host.  A hybrid timing is never a VC4-only figure.

```
$ vc4infer bench yolov8n.onnx --reps 3
model:  yolov8n.onnx
input:  random normal (rng 0), 1x3x160x160
warm-up (kernel JIT + first run): 973.4 ms
runs:   724.6 / 743.1 / 738.7 / 726.9 / 745.8 ms
735.8 ms/image (mean), 738.7 ms (median), best 724.6 ms
1.36 FPS (mean), 1.38 FPS (best)  over 5 image(s)
678 launches/image, GPU 250.1 ms (34% of wall), node wall sum 690.6 ms
arena 64.0 MiB mapped, graph pool 40.0 MiB, peak 26.6 MiB
```

"GPU" is the engine's pack+kernel time as reported by the library; "wall" is
the whole `run()`, including the daemon round trips, the ARM-side store and the
byte-only layout work.  The gap is launch overhead: at 160x160 the graph issues
678 launches.

### `util`

Prints the device-invariant utilisation table: `host_compute_nodes`,
`bytes_copied_by_host` (with the per-node/per-site byte debt), the share of
graph MACs on the VC4 (`arithmetic_on_vc4`), QPU duty cycle (busy / wall), and
the launch-path share.  `--json` includes the full site list.

```
$ vc4infer util yolov8n_160.onnx --reps 3
...
  device invariant           VIOLATED
  host_compute_nodes         0
  bytes_copied_by_host       17077040 B
  arithmetic_on_vc4          100.0%  (273249600/273249600 MACs)
  qpu_duty                   48.7%  (302.9 / 622.4 ms)
  launches                   678  (3.85 per dispatched node)
  launch path                319.5 ms  (51.3% of wall)
```

`device invariant` is `OK` only when both runtime host compute and host byte
movement are zero.  The byte debt is listed by node, site and size.

### `nodes`

A static coverage and backend report — node counts per op, a verdict per node
with the reason, the execution tier of every node, host debt, and the number of
`Mul(x, Sigmoid(x))` pairs that the executor fuses into one `Silu` launch.  It
reads the graph, attributes, initialiser shapes and shape-inferred value_info;
it does not run the model.  `-v` lists every node with its backend
(`vc4`, `host-meta`, `host-control`, `host-unwritten`, `unsupported-graph`).
With `--mode hybrid --host-policy device` a non-empty host-debt list (or any
unfolded metadata/control node) makes the command exit 4, so a CI check cannot
silently accept a model that is not 100% compute-on-VC4.

```
$ vc4infer nodes yolov8n.onnx
model:  yolov8n.onnx  (12.1 MiB, opset 13, pytorch 2.15.0)
input:  images  1x3x160x160 FLOAT
output: output0  1x84x525 FLOAT
263 nodes: 263 supported, 0 unsupported, 0 unverified
rewrites: 57 Mul(x,Sigmoid(x)) pairs fuse into one SiLU launch (206 dispatches)

tiers:  233 vc4, 30 host-meta, 0 host-control, 0 host-unwritten, 0 unsupported-graph
policy: mode=hybrid host_policy=structural

  op                      nodes    vc4   meta    ctl   unwr  implementation
  Conv                       64     64      0      0      0  QPU conv: NHWC 1x1/3x3 s1/s2, ...
  Mul                        60     58      2      0      0  QPU elementwise
  Sigmoid                    58     58      0      0      0  QPU elementwise
  Constant                   24      0     24      0      0  initialiser
  ...
unsupported by the VC4 executor: none

host tiers (30: 30 metadata, 0 control flow):
  /model.22/Shape     Shape    host-meta   metadata: Shape (no activation arithmetic)
  ...
host debt: none

static report only; `vc4infer run` is authoritative
```

### `selftest`

Runs the shipped reference models and compares every output element with its
stored CPU reference at `atol = rtol = 5e-4`:

| case | model | shape |
|---|---|---|
| `modelA` | 2x Conv3x3+ReLU, MaxPool2x2, Flatten, Linear, Softmax | 1x3x32x32 -> 1x10 |
| `modelB` | stride-2 conv, depthwise conv, MaxPool 5x5, Concat, Resize 2x, 1x1 conv | 1x16x32x32 -> 1x16x32x32 |
| `modelC` | Conv+ReLU, GlobalAveragePool, Flatten, Linear | 1x16x8x8 -> 1x4 |
| `yolov8n_160` | the real Ultralytics YOLOv8n COCO checkpoint at 160x160 | 1x3x160x160 -> 1x84x525 |

```
$ vc4infer selftest
vc4infer 1.2.10 selftest
  models:  /usr/share/vc4infer/models
  engine:  /usr/lib/aarch64-linux-gnu/libvc4infer.so.1
  backend: vc4d (/run/vc4d.sock, running)
  onnx 1.17.0, numpy 2.2.4, tolerance atol=rtol=5e-4 (over must be 0)

  modelA       PASS 1x10             36 launches    43.2 ms  max abs 4.84e-06  max rel 4.84e-06  0/10 over  [...]
  modelB       PASS 1x16x32x32       13 launches    33.6 ms  max abs 4.41e-06  max rel 4.09e-06  0/16384 over  [...]
  modelC       PASS 1x4               8 launches     9.2 ms  max abs 4.06e-07  max rel 4.06e-07  0/4 over  [...]
  yolov8n_160  PASS 1x84x525        678 launches   806.6 ms  max abs 2.04e-02  max rel 2.23e-04  0/44100 over  [...]
  ...
SELFTEST PASSED: 4/4 cases, 0 elements over tolerance
```

Exit status 0 on success, 1 when any element is outside tolerance or a model
fails to run.

## Python API

The same engine is importable — numpy in, numpy out, no framework types:

```python
import vc4infer

vc4infer.devices()                         # ['vc4']
dev = vc4infer.Device("vc4")               # .name .memory .queue_depth .supports(op)
eng = vc4infer.Engine("yolov8n.onnx", dev)
eng.inputs, eng.outputs                    # TensorSpec(name, shape, dtype)
out = eng.run({"images": x})               # {'output0': np.ndarray}  raw head
eng.plan()                                 # strict: raises on an unsupported node
eng.profile(reps=3)                        # per-node + total ms, launches, GPU/wall
eng.memory()                               # arena / pool / peak

from vc4infer.postprocess import decode_yolo
det = decode_yolo(out["output0"], imgsz=160, conf=0.25, iou=0.45)

det = vc4infer.yolo.Detector(eng)          # thin YOLO adapter (or Detector(path))
r = det.predict("bus.jpg", imgsz=160)      # Detection: .boxes .scores .classes
```

The engine knows nothing about YOLO.  Decoding and NMS live in
`vc4infer/postprocess.py` as pure numpy and are replaceable with the
`postprocess=` argument; `vc4infer.yolo.Detector` is ~130 lines of glue that
loads an image, calls `Engine.run`, calls the decoder and maps boxes back.

There is also an Ultralytics integration (stock `YOLO(...)` workflow, vc4
backend registered with one dict entry):

```python
import vc4infer.contrib.ultralytics as vu
vu.register()
from ultralytics import YOLO
model = YOLO("yolov8n_160.onnx")
r = model.predict("bus.jpg", imgsz=160, conf=0.25)[0]
```

The full reference, the threading model, the error table, the exact
Ultralytics patch and the board-specific torchvision caveat are in
[PYTHON-API.md](PYTHON-API.md).

Three runnable examples live in `examples/` and are installed to
`/usr/share/doc/vc4infer/`:

```sh
python3 examples/non_yolo.py                 # Engine on 3 non-YOLO models
python3 examples/run_yolo.py --model yolov8n.onnx --image bus.jpg
python3 examples/raw_engine.py --model modelC.onnx
python3 examples/bench.py --model yolov8n.onnx --reps 3
python3 examples/ultralytics_predict.py --model yolov8n.onnx --image bus.jpg
```

## Supported nodes

Everything that computes runs on the QPUs; the "host" entries are byte movement
or integer bookkeeping only, and an unsupported operator raises
`UnsupportedNode` naming the node.

| ONNX node | implementation | constraints |
|---|---|---|
| `Conv` (NHWC, default) | `vc4_conv_nhwc` / `_dil` / `_group` + `pack_w` | KH,KW in {1,3,5}, stride 1/2, dilation 1..8, symmetric pads; `K=Cin*KH*KW %16`; grouped needs 16-aligned per-group blocks; `stride*Cin*4 <= 8191` B; batch N; no plane-size limit |
| `Conv` depthwise | block-diagonal dense weight + `vc4_conv_nhwc` | `Cout == Cin`, square kernel; costs `Cin` x the MACs |
| `Conv` (NCHW fallback) | `vc4_conv2d` | non-square kernels/strides; stride 1, group 1, batch 1, dilation 1; `(H+2p)*align16(W+2p)*4 <= 8191` B |
| `Relu`, `Sigmoid`, `Silu` | `vc4_elementwise` | any layout |
| `Add`, `Mul` | `vc4_elementwise` | equal shapes or constant broadcast |
| `Sub`, `Div` | GPU `Mul(-1)`+`Add`; `Div` by a graph constant | see [ONNX.md §7.1](ONNX.md) |
| `MaxPool` | `vc4_maxpooln` / `vc4_maxpool_nhwc` | square window, no `ceil_mode`, no dilation |
| `Concat` | host byte concat | channel axis of a 4-D tensor (or axis 1/-1 of 3-D) |
| `Split` | host byte gather | channel axis of a 4-D tensor |
| `Slice` | host gather | any axis, int64 starts/ends/axes/steps |
| `Transpose` | host permute | any permutation |
| `Resize` | host gather (nearest) / GPU `Mul`+`Add` (linear) | `mode=nearest` or `linear`, `asymmetric` or `half_pixel` |
| `Softmax` | `vc4_softmax_fwd` | <= 16 classes on the softmax axis |
| `MatMul`, `Gemm` | `vc4_matmul` (+ GPU `Add` for the bias) | 2-D; `transA=0`, `alpha=1`, `beta in {0,1}` |
| `GlobalAveragePool` | `vc4_matmul` with a `1/HW` column | NCHW, batch 1 |
| `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Identity`, `Cast` | view / type change | fp32 targets, element count must match |
| `Constant` | initialiser | |
| `Shape`, `Gather` | host int64 | static shapes |

fp32 and static shapes only.  Batch N is supported on the NHWC conv path,
pooling, upsample, concat, elementwise and softmax; the NCHW fallback, `Gemm`
and `GlobalAveragePool` are batch 1 and raise otherwise.

## Measured performance

Stock clocks, `VC4INFER_DAEMON=1`, one image per run, warm kernel cache, same
boot.  The board is shared; a run measured while a concurrent build was
running came out slower, so the range is honest rather than a best case.

| model | input | runs | mean | FPS | launches/img | GPU (pack+kernel) | over 5e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| yolov8n 160 | 1x3x160x160 -> 1x84x525 | 615-628 ms (quiet), 725-746 ms under a concurrent build | **638.3 ms** | **1.57** | 678 | 243.4 ms (38 %) | 0/44100 |
| yolov8n 320 | 1x3x320x320 -> 1x84x2100 | 1593.8 / 1631.9 / 1482.7 ms | **1569.5 ms** | **0.64** | 1824 | 616.4 ms (39 %) | 1/176400 (max rel 5.07e-4) |

The 160 row is the paired A/B session of [ONNX.md §7.3](ONNX.md) (619.2 /
622.4 / 623.8 / 628.6 ms, mean 623.2 ms = 1.61 FPS) rounded up to the 638 ms /
1.57 FPS measured through the packaged CLI on a busier boot.  The 320 row is the
same graph with the 150 MiB arena of the next section ([ONNX.md §8.3](ONNX.md)).  Wall time on this board drifts
between sessions ([CLOCK-VARIANCE.md](CLOCK-VARIANCE.md)); the launch count and
the GPU time are stable, the wall is not.

Verified with `vc4infer selftest` on this boot (second run, warm cache):

| case | launches | ms | max abs | max rel | over 5e-4 |
|---|---:|---:|---:|---:|---:|
| modelA | 36 | 43.2 | 4.84e-06 | 4.84e-06 | 0/10 |
| modelB | 13 | 33.6 | 4.41e-06 | 4.09e-06 | 0/16384 |
| modelC | 8 | 9.2 | 4.06e-07 | 4.06e-07 | 0/4 |
| yolov8n_160 | 678 | 806.6 | 2.04e-02 | 2.23e-04 | 0/44100 |

`vc4infer bench yolov8n_160.onnx --reps 5` in the same window: 724.6 / 743.1 /
738.7 / 726.9 / 745.8 ms, mean 735.8 ms = 1.36 FPS, 678 launches, GPU 250.1 ms
(34 % of wall); the Python `Engine.profile` measured 634.5 ms = 1.58 FPS in a
quieter minute, i.e. the API and the CLI agree once the session variance is
accounted for.

Per-op wall time at 160x160 (from [ONNX.md §7.3](ONNX.md)): Conv 413.8 ms
(201.9 GPU) over 64 nodes, the 60 fused SiLU `Mul` nodes 82.1 ms (33.5 GPU) over
468 launches, Concat 28.8 ms (host), Split 12.0 ms (host), everything else
below 10 ms.  The 678 launches are 128 conv launches (im2col + GEMM) and 468
SiLU launches; the elementwise path is one 256-float group per QPU per launch
on this silicon (see below).

The 320x320 numbers need a larger arena; see the next section.

## 320x320: the arena recipe

A 320x320 export needs a padded 6.6 MB stem plane, a 16.4 MB im2col panel and
about 4x the 160x160 activation working set (~68 MiB pool + >= 23 MiB of
per-conv scratch).  That does not fit the default 72 MiB daemon arena; it fits
with a 150 MiB daemon arena and a 145 MiB client region, no kernel change:

```sh
sudo systemctl stop vc4d
sudo /usr/local/sbin/vc4d --socket /run/vc4d.sock --arena-mb 150 \
     --group vc4 --drop-user nobody --max-timeout-ms 10000 &
sleep 1
VC4INFER_ARENA_MB=145 VC4_POOL_RESERVE_MB=32 \
  vc4infer bench yolov8n_320.onnx --reps 3
```

Measured with that recipe: 113 MiB graph pool, 33.4 MiB peak, 1569.5 ms/image
(0.64 FPS), 1824 launches, 1/176400 elements over 5e-4.  The client region is
what the engine can see, so `VC4INFER_ARENA_MB` must match what the daemon
serves; the engine defaults to 64 MiB and the daemon's systemd unit on this
board uses `--arena-mb 72`.

The 320 export is produced off-board or on-board with ultralytics:

```sh
YOLO_IMGSZ=320 python3 onnx/export_yolo.py    # writes models/yolov8n_320.onnx
```

## How it works

* `libvc4infer` is an **operator library**, not a graph runtime.  It owns one
  dma-buf arena (64 MiB by default), JIT-compiles QPU programs with
  `kgen.py` into a per-user cache, uploads them at distinct code offsets, and
  launches them through vc4d.
* The executor (`vc4onnxrun.py`) evaluates the graph in ONNX topological order,
  keeps every activation in one graph pool carved out of the arena by liveness,
  and rewrites `Mul(x, Sigmoid(x))` into one fused `Silu` launch.
* The CLI (`vc4infer_cli`) adds input preprocessing, the YOLO decode and NMS,
  annotation, JSON output, and the static coverage report.

## Known limits

* **fp32 only**, **static shapes**.  No quantisation, no dynamic dims.  Batch N
  runs on the NHWC path (conv, pool, upsample, concat, elementwise, softmax);
  the NCHW fallback, `Gemm` and `GlobalAveragePool` are batch 1.  The model
  input size cannot be changed at run time; an export at the wanted size is the
  only way (`--imgsz` must match).
* **No GPU NMS / no GPU decode.**  The detection head decode and NMS run on the
  ARM (comparisons, sort, thresholding of the network's own output).  This is a
  deliberate boundary, not a missing kernel: the head is not convolution work.
* **Unsupported nodes raise**, they never fall back.  `Expand`,
  `BatchNormalization`, `Pow`, `LRN`, arbitrary `ConvTranspose`, conv kernels
  outside 1x1/3x3/5x5 (or the NCHW fallback's 1x1/3x3), non-square pools and
  `Softmax` with more than 16 classes are not implemented.
* **Detection quality depends on the input size.**  YOLOv8n was trained at 640;
  at 160x160 the same weights see a 4x smaller image, so small and distant
  objects are missed and boxes are coarse.  The export, the letterbox and the
  decode are correct (the graph matches the CPU reference to 5e-4); the
  *accuracy* of the detections is limited by the 160-pixel input, not by the
  QPU arithmetic.  Use 320x320 (0.70 FPS) for better recall, and treat 160 as
  the throughput mode.
* **Depthwise conv** is expanded to a dense block-diagonal weight, costing
  `Cin` times the real MACs; **stride 2** is decomposed into 4 phase images
  (4x the MACs for 3x3).  Both are exact, not approximate.
* **Launch overhead** dominates small layers: one elementwise group per QPU per
  launch is the only shape this board's VPM/DMA path survives
  ([ONNX.md §7.6](ONNX.md)), and a 263-node graph is 678 launches.
* The legacy NCHW conv path still has the 8191-byte im2col pitch limit; the
  NHWC path used by everything YOLO-shaped does not.
* Model weights are uploaded once per `Runner`; there is no cross-process
  weight cache and no graph-level memory planner beyond liveness recycling.
* The selftest's YOLOv8n checkpoint is the Ultralytics COCO model (AGPL-3.0);
  see the package copyright file.

## Package layout

```
/usr/bin/vc4infer                               CLI
/usr/lib/aarch64-linux-gnu/libvc4infer.so.1     engine (SONAME libvc4infer.so.1)
/usr/include/vc4infer.h                         C API
/usr/lib/python3/dist-packages/vc4infer/        the Python package (same code as the wheel)
    __init__.py  engine.py  device.py  postprocess.py  yolo.py  cli.py
    coverage.py  _binding.py  _onnxrun.py  _runtime.py  _errors.py
    contrib/ultralytics.py  kgen/
/usr/lib/vc4infer/kgen/                         QPU kernel generator for C clients
/usr/share/vc4infer/models/                     selftest models, inputs, refs
/usr/share/doc/vc4infer/                        this file, PYTHON-API.md, examples/, copyright
/usr/share/man/man1/vc4infer.1.gz
```

The wheel installs the same `vc4infer/` package inside the interpreter's
`site-packages` and the `vc4infer` CLI entry point is the .deb's
`/usr/bin/vc4infer` (the wheel's `python3 -m vc4infer.cli` works too).

The engine kernels are JIT-compiled on first use into
`$XDG_CACHE_HOME/vc4infer/kcache` (about a second for a 100-kernel graph,
then reused).  `VC4INFER_CACHE` overrides the directory.

## Building and publishing

From a `vc4-stack` checkout on the board:

```sh
make -C infer build/libvc4infer.so           # the engine
infer/packaging/vc4infer/build_deb.sh        # -> dist/vc4infer_<ver>_arm64.deb
infer/packaging/build_wheel.sh               # -> dist/vc4infer-<ver>-py3-none-any.whl
packaging/publish_apt.sh                     # signed apt repo (vc4top archive key)
```

`publish_apt.sh` adds `pool/main/v/vc4infer/` and `pip/` (the wheel plus a
PEP 503 simple index) to the `main` branch of `rughinnit/vc4-apt`, regenerates `Packages`,
`Release`, `Release.gpg` and `InRelease` with the existing archive key, and
pushes.  The same keyring (`vc4top-archive-keyring.gpg`) verifies both
packages.

`python3 -m vc4infer.cli <command>` is the module form of the CLI, useful
inside a venv where /usr/bin/vc4infer may not be on `PATH` first.
