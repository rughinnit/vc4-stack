# vc4infer Python API

`vc4infer` is an inference **engine** for ONNX models on the VideoCore IV QPUs:
numpy in, numpy out, no framework types.  YOLO is one consumer of it; neither
the engine nor the API knows what a detection head is.

```python
import vc4infer

vc4infer.devices()                       # ['vc4']
eng = vc4infer.Engine("yolov8n.onnx")    # device=Device("vc4")
eng.inputs                               # (TensorSpec('images', 1x3x160x160, float32),)
out = eng.run({"images": x})             # {'output0': np.ndarray}  raw
eng.plan()                               # strict pre-flight check
eng.profile(reps=3)                      # per-node + total ms, launches
eng.memory()                             # arena / pool / peak
```

Post-processing is a separate module you can replace:

```python
from vc4infer.postprocess import decode_yolo, nms
det = decode_yolo(out["output0"], imgsz=160, conf=0.25, iou=0.45)
```

And YOLO is a thin adapter:

```python
det = vc4infer.yolo.Detector(eng)        # or Detector("yolov8n.onnx")
r = det.predict("bus.jpg", imgsz=160, conf=0.25, iou=0.45)
r.boxes, r.scores, r.classes             # numpy, original image pixels
```

## Install

Two pieces, on purpose:

| piece | contains | install |
|---|---|---|
| `vc4infer` .deb | `libvc4infer.so.1`, headers, CLI, kernel JIT, selftest models, the same Python package system-wide | `sudo apt install vc4infer` |
| `vc4infer` wheel | the pure-Python package (engine API, CLI module, executor, ctypes binding, kernel generator, examples) — `py3-none-any` | `pip install vc4infer-1.1.2-py3-none-any.whl` |

The wheel is pure Python, so it cannot carry the architecture-specific engine
library; the .deb provides it.  In a venv:

```sh
sudo apt install vc4infer
python3 -m venv ~/venvs/vc4 && . ~/venvs/vc4/bin/activate
pip install https://rughinnit.github.io/vc4-apt/pip/vc4infer-1.1.2-py3-none-any.whl
# or from the PEP 503 index (--index-url works too):
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ vc4infer
```

pip dependencies: `numpy`, `onnx>=1.14`, `pillow`.  `vc4d` must be running
(`systemctl start vc4d`); the package defaults `VC4INFER_DAEMON=1` and the
client never opens `/dev/mem`.

## API reference

### Module

| name | meaning |
|---|---|
| `vc4infer.__version__` | `"1.1.2"` |
| `vc4infer.devices()` | `["vc4"]` |
| `vc4infer.Device("vc4")` | the device object: `.name`, `.memory`, `.queue_depth`, `.ops`, `.supports(op)`, `.describe()` |
| `vc4infer.Engine(model_path, device=None, *, mode="strict", host_policy="device", arena_mb=None, cache_dir=None)` | the engine |
| `vc4infer.TensorSpec`, `Plan`, `NodeInfo`, `NodeBackend` | metadata dataclasses |
| `vc4infer.postprocess` | `decode_yolo`, `nms`, `Detections`, `Letterbox`, `draw`, `COCO80` |
| `vc4infer.yolo.Detector(engine_or_path, names=None, *, device="vc4")` | the YOLO adapter |
| `vc4infer.contrib.ultralytics` | `register()`, `unregister()`, `is_registered()`, `VC4Backend` |
| `vc4infer.UnsupportedNode(node, op, detail)` | unsupported operator/attribute/shape (strict) |
| `vc4infer.HostNotAllowed(node, op, reason)` | hybrid `device`/`structural`: a numeric op with no VC4 kernel (host debt) |
| `vc4infer.NeedsDeviceDispatch(node, op, feature)` | hybrid `device`: If/Loop/Scan need a persistent kernel + device-side queue |
| `vc4infer.HostNodeError(node, op, detail)` | hybrid: the numpy reference evaluator failed on a host node |
| `vc4infer.DeviceError` | device/engine cannot be opened (daemon, arena busy) |
| `vc4infer.ShapeError` | feed name/shape/dtype mismatch (a `ValueError`) |
| `vc4infer.Error`, `vc4infer.Unsupported` | raw `libvc4infer` errors |

### `Device`

| member | value |
|---|---|
| `.name` | `"vc4"` |
| `.memory` | dict: `{"kind": "dma-buf arena", "arena_mb": 64, "shared": True, "backend": "vc4d", "socket": "/run/vc4d.sock", ...}` |
| `.queue_depth` | `1` — vc4d keeps one launch in flight per client and serialises clients |
| `.ops` | the ONNX operators with a QPU implementation |
| `.supports(op)` | `True`/`False` for one ONNX op type |
| `.describe()` | `Device(vc4, arena=64 MiB via vc4d, queue_depth=1, ops=27)` |

`Device("anything-else")` raises `DeviceError` listing `devices()`.

### `Engine`

| member | returns / does |
|---|---|
| `Engine(path, device, *, mode="strict", host_policy="device", arena_mb=, cache_dir=)` | loads the graph, opens one arena, uploads weights |
| `.mode`, `.host_policy` | the execution policy in force |
| `.device` | the `Device` |
| `.inputs`, `.outputs` | tuples of `TensorSpec(name, shape, dtype)`, resolved from the ONNX graph |
| `.run(feeds, profile=False)` | `{output_name: np.ndarray}` — raw outputs, no post-processing |
| `.nodes()` | list of `NodeInfo(name, op, supported, constraint)`; `supported` is `True`, `False`, or `None` for shape-dependent ("unverified") |
| `.unsupported()` | the nodes with `supported is False` |
| `.plan()` | strict: raises `UnsupportedNode` on the first unsupported node. structural hybrid: raises `HostNotAllowed` on the first numeric host-debt node. Else returns a `Plan` |
| `.node_backends()` | `[(name, op, backend, reason)]` for every node: `vc4`, `host-meta`, `host-control`, `host-unwritten`, `unsupported-graph` (actual verdict after a hybrid run) |
| `.host_debt()` | only the `host-unwritten` nodes: numeric ops that still need a VC4 kernel |
| `.host_compute_nodes()` | nodes that performed runtime CPU compute; `[]` under `device` |
| `.utilization(reps=1, feeds=None)` | MAC share on VC4, host bytes (per site), QPU duty, launches/node and launch-path share |
| `.profile(reps=1, feeds=None)` | total mean/median/min ms, FPS, launches, GPU vs wall, per-node backend/reason breakdown, tier split (`nodes_vc4`, `host_meta`, `host_control`, `host_unwritten`), boundary copies, memory |
| `.memory()` | `{arena_bytes, arena_mb, pool_bytes, pool_mb, peak_bytes, peak_mb}` |
| `.last_profile`, `.last_boundary`, `.stats` | per-node timings/backend, `{tensors, host_to_vc4, vc4_to_host}`, and `{launches, gpu_ms}` of the last run |
| `.close()` / `with` | release the arena |

`feeds` is `{input_name: ndarray}`; for a single-input model a bare ndarray is
accepted too.  Every input must be present, named exactly, shaped exactly, and
convertible to fp32; otherwise `ShapeError` names the input and both shapes.
Outputs are the graph's own outputs in ONNX order — for YOLOv8n that is the raw
`[1, 84, 525]` head.

Static vs runtime: `nodes()`/`plan()` read the graph, attributes, initialiser
shapes and shape inference; they catch an unsupported op before the GPU is
touched.  `run()` is authoritative for a concrete input (a shape-dependent
constraint can still raise there).

Hybrid mode is opt-in and strict remains the default.  Hybrid defaults to
`host_policy="device"`: the host only folds metadata at plan time and does
orchestration; every runtime node must be VC4.  A numeric op without a kernel
raises `HostNotAllowed`, and If/Loop/Scan raise `NeedsDeviceDispatch`.
`host_policy="structural"` allows runtime metadata and control flow on the
host; `host_policy="any"` additionally runs unwritten numeric ops in the numpy
reference evaluator.  `utilization()` reports the MAC share on VC4, host byte
movement and QPU duty; a hybrid number is never a VC4-only number.  See
[ONNX.md](ONNX.md) section 10.

### `postprocess`

| function | meaning |
|---|---|
| `decode_yolo(output, imgsz=None, conf=0.25, iou=0.45, max_det=300, agnostic=False)` | decode `[1,4+nc,A]` (or `[1,A,4+nc]`, or YOLOv5's `5+nc`) into `Detections` |
| `nms(boxes, scores, iou_thr)` | greedy NMS for one class; returns kept indices |
| `Detections(boxes, scores, classes, meta)` | `.to_dict(names)`, `.top(k)`, `len()` |
| `Letterbox(imgsz)` | PIL -> `(1x3xSxS float32, meta)`; `.to_original(boxes, meta)` |
| `draw(image, detections, names)` | PIL RGB with boxes |
| `COCO80` | the COCO class names |

### `yolo.Detector`

| member | meaning |
|---|---|
| `Detector(engine_or_path, names=None)` | wraps an existing `Engine` (shared) or opens one |
| `.predict(image, imgsz=None, conf=0.25, iou=0.45, max_det=300, agnostic=False, postprocess=decode_yolo)` | run + decode one image |
| `.engine`, `.names`, `.imgsz` | the underlying engine and metadata |
| `.close()` | closes only an engine the detector opened |

`predict` accepts a path, `PIL.Image`, or HWC `uint8` array.  It returns a
`Detection` dataclass of numpy arrays:

| field | shape |
|---|---|
| `.boxes` | `(N, 4)` xyxy in original image pixels |
| `.scores` | `(N,)` |
| `.classes` | `(N,)` int |
| `.orig_shape` | `(h, w)` |
| `.imgsz`, `.meta`, `.speed_ms`, `.path` | model input size, decode metadata, timings |
| `.to_dict(names)`, `.plot(image=None)` | JSON-able dict, PIL image |

## Replacing post-processing

The adapter calls ``postprocess=`` with

```python
postprocess(raw_output, imgsz=..., conf=..., iou=..., max_det=..., agnostic=...)
```

and accepts either a `Detections` or a dict with `boxes` (N,4, model-input
pixels), `scores`, `classes`.  The adapter then maps the boxes to the original
image; everything else is yours.

```python
from vc4infer.postprocess import Detections, decode_yolo

def big_boxes(raw, imgsz, conf, iou, max_det=300, agnostic=False):
    d = decode_yolo(raw, imgsz=imgsz, conf=conf, iou=iou)
    area = (d.boxes[:, 2] - d.boxes[:, 0]) * (d.boxes[:, 3] - d.boxes[:, 1])
    keep = area > 0.02 * imgsz * imgsz
    return Detections(d.boxes[keep], d.scores[keep], d.classes[keep], d.meta)

r = det.predict("bus.jpg", postprocess=big_boxes)
```

Or skip the adapter entirely:

```python
raw = eng.run({"images": x})["output0"]
boxes = raw[0, :4].T                      # cx, cy, w, h -- your arithmetic
```

## Ultralytics integration

`vc4infer.contrib.ultralytics` plugs the engine into Ultralytics 8.4's backend
registry:

```python
import vc4infer.contrib.ultralytics as vu
vu.register()                     # AutoBackend._BACKEND_MAP["onnx"] = VC4Backend

from ultralytics import YOLO
model = YOLO("yolov8n_160.onnx")  # unchanged workflow
r = model.predict("bus.jpg", imgsz=160, conf=0.25)[0]
r.boxes.xyxy, r.boxes.conf, r.boxes.cls
r.save("bus_ultralytics_vc4.jpg")
```

Ultralytics' own letterbox, `non_max_suppression` and `Results` stay in charge;
only the convolutions move to the QPUs.  Verified on this board (ultralytics
8.4.215, torch 2.15.0+cpu):

```
registered vc4 for .onnx (replaced ONNXBackend)
backend: VC4Backend
detections: 5
  person       0.767  [44.0, 411.3, 212.5, 895.3]
  person       0.613  [220.8, 399.3, 350.0, 862.2]
  bus          0.554  [13.0, 234.7, 794.1, 761.6]
  person       0.491  [681.1, 436.0, 810.0, 860.1]
  truck        0.391  [19.5, 223.1, 797.2, 756.0]
```

**Why not `format="vc4"`?**  Ultralytics derives the format from the file
suffix (`guess_model_type`) *before* it consults `AutoBackend._BACKEND_MAP`, and
`export_formats()` has no vc4 row; a `format="vc4"` selector would need a fork
of those two functions.  Registering the existing `"onnx"` key is the minimal
patch (one dict entry, returned so `unregister(previous)` restores ONNX
Runtime).

**The board-specific second patch.**  `AutoBackend.warmup()` unconditionally
does `import torchvision` so that NMS binds to `torchvision.ops.nms`.  This
board's torchvision 0.29.2 registers no operators against its custom
torch 2.15.0+cpu build (`RuntimeError: operator torchvision::nms does not
exist`), so the stock warmup aborts before any inference.  `register()` also
swaps in a warmup that runs the same NMS warmup with Ultralytics' own pure-torch
implementation (on CPU the stock warmup does nothing else).  `unregister()`
restores both.  On a normal x86/ARM torch build `patch_warmup=False` is not
needed; the flag exists so the integration never silently changes behaviour.

If you would rather not patch anything, the supported fallback is the plain
engine path: `Engine("yolov8n.onnx").run({"images": x})` plus
`postprocess.decode_yolo` — the same QPU work, no ultralytics objects.

## Verified examples

`examples/non_yolo.py` — the engine on three non-YOLO graphs, including a
two-input model (`models/` generated by `examples/make_models.py`):

```
modelA (classifier) (TensorSpec('x', 1x3x32x32, float32),) -> (TensorSpec('y', 1x10, float32),)
  modelA     1x10           max abs 4.835e-06  over tol 0/10  PASS
mlp (TensorSpec('x', 1x16, float32),) -> (TensorSpec('probs', 1x4, float32),) argmax 3
  plan: {'nodes': 4, 'dispatches': 4, 'unverified': []}
  mlp        1x4            max abs 3.129e-07  over tol 0/4  PASS
two_input (TensorSpec('a', 1x3x8x8, float32), TensorSpec('b', 1x3x8x8, float32)) -> (TensorSpec('logits', 1x5, float32),)
  two_input  1x5            max abs 7.153e-07  over tol 0/5  PASS
  missing feed rejected: missing input(s) ['b']; the graph expects ['a', 'b']
profile: 5.82 ms/image, 8 launches, GPU 0.61 ms
```

`examples/run_yolo.py` — the adapter, the thresholds and a replacement decode:

```
Detection(boxes=[[ 39.0 409.2 228.7 901.7] [  1.8 226.9 800.6 758.8]
                 [678.5 424.5 810.0 865.8] [206.2 391.5 353.2 863.2]],
          scores=[0.778 0.703 0.619 0.597], classes=[0 5 0 0],
          orig_shape=(1080, 810), imgsz=160)
conf 0.60 / iou 0.30 -> 3 detections [0.778, 0.703, 0.619]
custom postprocess (bus only) -> 1 [0.703]
raw head: (1, 84, 525)
```

`examples/raw_engine.py` — inputs/outputs/plan/profile/memory and the raw head;
`examples/bench.py` — the same numbers as `vc4infer bench`;
`examples/ultralytics_predict.py` — the stock Ultralytics pipeline above.

## Performance and parity

The Python layer adds no measurable overhead: `Engine.run` is one
`Runner.run` call (the executor makes one ctypes call per node), buffers and
weight panels live for the engine's lifetime, and the ONNX file is parsed once
in the constructor.

```
Engine.profile(reps=3):      mean 634.5 ms   1.58 FPS   678 launches   GPU 242.7 ms
vc4infer bench --reps 5:     mean 735.8 ms   1.36 FPS   678 launches   GPU 250.1 ms
```

Both lines are the same graph and the same code; the difference is the board's
session-to-session wall-time variance under a concurrent build
([CLOCK-VARIANCE.md](CLOCK-VARIANCE.md)).  Launch count and GPU time are stable.

## Threading and lifetime

* One engine at a time.  An engine owns a 64 MiB arena from vc4d; a second
  concurrent `Engine` cannot map another one and raises `DeviceError` (the
  message says the arena is busy).
* `Engine.run` serialises calls with a lock; a single engine can be shared by
  threads (they queue).
* `close()` (or `with`) frees the arena before the next engine is created.

## Limits

fp32, batch 1, static shapes; no GPU NMS/decode; `Softmax` at most 16 classes;
square 1x1/3x3 convolutions; 320x320 needs a 192 MiB daemon arena
(`VC4_POOL_RESERVE_MB=40 VC4INFER_ARENA_MB=189`, see
[VC4INFER.md](VC4INFER.md#320x320-the-arena-recipe)).
