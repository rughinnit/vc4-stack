# API reference: `vc4infer` and `torch_vc4`

Two public surfaces share one engine and one privileged helper (`vc4d`):

| package | what it is | version on the reference board (2026-11-20) |
|---|---|---|
| `vc4infer` | ONNX graph executor, YOLO adapter, CLI, ctypes binding | **1.2.1** (.deb + wheel in the vc4-apt archive) |
| `torch_vc4` | a declared subset PyTorch `PrivateUse1` backend | **0.3.1** in the source tree (`setup.py`, `torch_vc4.__version__`) and in the vc4-apt pip index as `torch_vc4-0.3.1-cp313-cp313-linux_aarch64.whl` (797,409 B). An editable install made before the bumps still reports **0.2.0** in its metadata. |

The measured performance numbers in TUTORIAL.md section 7 are from the 0.2.0
surface, and the `torch_vc4` table in Part 2 describes that surface.  The
source tree's `setup.py` says 0.3.1; its authoritative per-op table (with
accuracy and timing at three sizes) is `COMPLETENESS.md`.  Do not assume a row
from this document that the table does not carry.

Neither package ever falls back to CPU arithmetic.  An operator, attribute or
shape outside the table raises and names the node/op.

---

# Part 1 - `vc4infer` 1.2.1

## 1.1 Module surface

```python
import vc4infer

vc4infer.__version__          # "1.2.1"
vc4infer.devices()            # ["vc4"]
vc4infer.Device("vc4")
vc4infer.Engine(path, device=None, *, mode="strict", host_policy="device",
                arena_mb=None, cache_dir=None, passes=None)
vc4infer.TensorSpec           # (name, shape, dtype)
vc4infer.Plan                 # validated static plan
vc4infer.NodeInfo             # (name, op, supported, constraint)
vc4infer.NodeBackend          # (name, op, backend, reason)
vc4infer.postprocess          # decode_yolo, nms, Detections, Letterbox, draw, COCO80
vc4infer.yolo.Detector        # thin YOLO adapter
vc4infer.coverage             # the static coverage/backend table
vc4infer.contrib.ultralytics  # register(), unregister(), is_registered(), VC4Backend
```

`vc4infer.default_cache_dir()` returns `$VC4INFER_CACHE` or
`~/.cache/vc4infer/kcache`.

Errors are exported at module level: `Vc4InferError`, `DeviceError`,
`ShapeError`, `UnsupportedNode`, `HostNotAllowed`, `HostNodeError`,
`NeedsDeviceDispatch`, plus the raw library `Error` and `Unsupported`.

## 1.2 `Device`

Only `"vc4"` exists.  `Device("cpu")` raises `DeviceError` listing
`devices()`.

| member | value / meaning |
|---|---|
| `.name` | `"vc4"` |
| `.memory` | dict: `{"kind": "dma-buf arena", "arena_mb": 64, "shared": True, "backend": "vc4d", "socket": "/run/vc4d.sock", "note": ...}`.  `arena_mb` comes from `VC4INFER_ARENA_MB` (default 64); it is a request, not a guarantee. |
| `.queue_depth` | `1`.  `vc4d` keeps one launch in flight per client and serialises clients. |
| `.ops` | tuple of ONNX operator names with a QPU implementation. |
| `.supports(op)` | `True`/`False` for one ONNX op name. |
| `.describe()` | `Device(vc4, arena=64 MiB via vc4d, queue_depth=1, ops=N)` |

`Device.__eq__` accepts a `Device` or the string `"vc4"`.

## 1.3 `Engine`

### Constructor

```python
Engine(model_path, device=None, *, mode="strict", host_policy="device",
       arena_mb=None, cache_dir=None, passes=None)
```

| parameter | meaning |
|---|---|
| `model_path` | fixed-shape, batch-1, fp32 ONNX file.  Parsed once; the file must exist. |
| `device` | `Device("vc4")` or `"vc4"`; `None` means `Device("vc4")`. |
| `mode` | `"strict"` (default) or `"hybrid"`. |
| `host_policy` | hybrid only: `"device"` (default), `"structural"`, `"any"`. |
| `arena_mb` | arena size to request from `vc4d` (default: the daemon's, normally 64 MiB). |
| `cache_dir` | kernel blob cache (default `~/.cache/vc4infer/kcache`). |
| `passes` | optimiser passes (`docs/OPTIMIZER.md`): `None` reads `$VC4_OPT`, the empty string runs none.  Grammar: `canonicalise`, `ew`, `epilogue`, `all`, and `-name` to disable one after `all`.  Every pass is off by default. |

The constructor opens one arena and uploads the weights.  Only one engine
should be open at a time; a second one cannot map a second 64 MiB arena and
raises `DeviceError` (`arena is busy`).  A graph with no fp32 input raises
`UnsupportedNode`.

Example:

```python
eng = vc4infer.Engine("yolov8n_160.onnx")
eng.inputs    # (TensorSpec('images', 1x3x160x160, float32),)
eng.outputs   # (TensorSpec('output0', 1x84x525, float32),)
```

### Execution modes and host policy

| mode / policy | runtime arithmetic allowed on host | raises |
|---|---|---|
| `mode="strict"` (default) | none | `UnsupportedNode` on the first node without a VC4 kernel |
| `mode="hybrid", host_policy="device"` | none; plan-time constant/shape folding and orchestration only | `HostNotAllowed` for a numeric `host-unwritten` node; `NeedsDeviceDispatch` for `If`/`Loop`/`Scan`; `HostNotAllowed` for a metadata node that could not be folded |
| `mode="hybrid", host_policy="structural"` | runtime metadata and control flow | `HostNotAllowed` for `host-unwritten` numeric nodes |
| `mode="hybrid", host_policy="any"` | metadata, control flow, and `host-unwritten` numeric ops in the numpy reference evaluator | `unsupported-graph` still raises |

A hybrid timing is **never** a VC4-only number; `profile()`/`util` print a
`hybrid_note` and split the time by tier.  `host_policy="any"` is a
compatibility escape, not a performance mode.

### Methods

| method | returns / does |
|---|---|
| `.run(feeds, profile=False)` | `{output_name: np.ndarray}`.  `feeds` is `{input_name: ndarray}` or one ndarray for a single-input model.  Inputs are checked for name, shape and dtype and converted to contiguous fp32.  Outputs are the graph's own outputs in ONNX order; no decode, threshold or NMS.  `profile=True` fills `last_profile`. |
| `.profile(reps=1, feeds=None)` | dict; a warm-up run (JIT) is **not** counted, then `reps` timed runs.  Keys: `warmup_ms`, `runs_ms`, `mean_ms`, `median_ms`, `min_ms`, `fps_mean`, `fps_best`, `launches`, `gpu_ms`, `wall_sum_ms`, `nodes` (per-node list), `nodes_total`, `nodes_vc4`, `nodes_folded`, `nodes_host_meta`, `nodes_host_control`, `nodes_host_unwritten`, `nodes_unsupported_graph`, `pct_nodes_vc4`, `pct_compute_vc4`, `host_compute_nodes`, `host_compute_node_names`, `arithmetic_on_vc4`, `macs_total`, `macs_vc4`, `bytes_copied_by_host`, `host_copy_sites`, `qpu_duty`, `qpu_busy_ms`, `qpu_idle_ms`, `launches_per_node`, `launch_path_ms`, `launch_path_pct`, `time_ms`, `pct_time_host`, `boundary`, `hybrid_note`, `summary`, plus `memory()`. |
| `.utilization(reps=1, feeds=None)` | `profile()` plus `device_ok` = (`host_compute_nodes == 0` and `bytes_copied_by_host == 0`). |
| `.memory()` | `{arena_bytes, arena_mb, pool_bytes, pool_mb, peak_bytes, peak_mb, device}`. |
| `.nodes()` | list of `NodeInfo(name, op, supported, constraint)`.  `supported` is `True`, `False`, or `None` for shape-dependent ("unverified").  `.status` is `supported`/`unsupported`/`unverified`. |
| `.unsupported()` | nodes with `supported is False`. |
| `.node_backends()` | list of `NodeBackend(name, op, backend, reason)`.  Backends: `vc4`, `host-meta`, `host-control`, `host-unwritten`, `unsupported-graph`, plus `folded` in reports.  After a hybrid run this is the actual verdict; before it, the static partition.  `.on_host` is true for the `host-*` values. |
| `.host_debt()` | only `host-unwritten` nodes: numeric ops with no VC4 kernel. |
| `.host_compute_nodes()` | nodes that performed runtime host compute; empty under the device policy. |
| `.plan()` | validates and returns a `Plan`; strict raises `UnsupportedNode` on the first unsupported node; hybrid/device also raises `HostNotAllowed`/`NeedsDeviceDispatch`; hybrid/structural raises `HostNotAllowed` on `host-unwritten`. |
| `.close()` / `with` | releases the arena.  `__del__` also closes. |
| `.last_profile`, `.last_boundary`, `.stats` | per-node timings/backends; `{tensors, host_to_vc4, vc4_to_host}`; `{launches, gpu_ms}` of the last run. |

`Plan` fields: `model`, `device`, `inputs`, `outputs`, `nodes`, `dispatches`,
`unverified`, `mode`, `host_policy`, `backends`, `host_debt`;
`Plan.to_dict()` returns a JSON-able summary.

`NodeInfo` is `(name, op, supported, constraint)`.  `NodeBackend` is
`(name, op, backend, reason)`.

### Static vs runtime

`nodes()`/`plan()` read the graph, attributes, initialiser shapes and
shape-inferred `value_info`; they catch an unsupported op before touching the
GPU.  `run()` is authoritative for a concrete input: a shape-dependent
constraint can still raise there.  `vc4infer nodes` is the CLI form and prints
the same verdicts.

## 1.4 Error types

Every deliberate error is one of these.  Raw `libvc4infer` failures keep
`vc4infer.Error` (or `vc4infer.Unsupported`) and carry the library's message.

| class | meaning | exact message shape (verified) |
|---|---|---|
| `UnsupportedNode(node, op, detail)` | strict mode: node/attribute/shape has no QPU implementation.  Attributes `.node`, `.op`, `.detail`. | `UnsupportedNode: node 'Sin_0' (Sin): no implementation` |
| `HostNotAllowed(node, op, reason, policy)` | hybrid: a numeric op with no VC4 kernel (`host-unwritten`) is refused by the policy.  Attributes `.node`, `.op`, `.reason`, `.policy`. | `HostNotAllowed: node 'Sin_0' (Sin) may not run on the host under host_policy='device': numeric op with no VC4 kernel: no implementation` (then a second line: `host_policy='any' runs it in the numpy reference evaluator; the operator still needs a VC4 kernel`) |
| `NeedsDeviceDispatch(node, op, feature)` | hybrid/device: `If`/`Loop`/`Scan` need branch/iteration decisions on the device.  Attributes `.node`, `.op`, `.feature`. | `NeedsDeviceDispatch: node 'If_1' (If) needs device-side dispatch: If/Loop/Scan branch execution on the device` (then: `persistent QPU kernel + device-side work queue; ...`) |
| `HostNodeError(node, op, detail)` | hybrid: the numpy reference evaluator itself failed on a host node. | `node '...' (Op): host reference evaluator failed: ...` |
| `DeviceError` | no such device, or the engine/daemon/arena could not be opened. | `device 'cpu' is not available; vc4infer only runs on the VideoCore IV QPU (device='vc4'). Available: ['vc4']` |
| `ShapeError` | a feed's name/shape/dtype does not match the graph input; subclasses `ValueError`. | `input 'images' has shape (1, 3, 320, 320), the graph expects 1x3x160x160` |
| `Vc4InferError` | base class for the package's own errors. | - |

The three policy errors are by design.  There is no silent CPU fallback; the
choice of policy is explicit, and `host_policy="any"` is the only way to run a
numeric op on the host (in the numpy reference evaluator), with the debt still
reported.

## 1.5 `vc4infer.postprocess`

Pure numpy; separate from the engine so you can replace it.

| name | meaning |
|---|---|
| `decode_yolo(output, imgsz=None, conf=0.25, iou=0.45, max_det=300, agnostic=False)` | decode `[1, 4+nc, A]`, `[1, A, 4+nc]`, or YOLOv5 `5+nc` into `Detections`; model-input pixels.  Handles per-class or agnostic greedy NMS. |
| `nms(boxes, scores, iou_thr)` | greedy NMS for one class; returns kept indices, highest score first. |
| `Detections(boxes, scores, classes, meta)` | `.boxes` (N,4) xyxy, `.scores` (N,), `.classes` (N,) int, `.meta`; `.to_dict(names)`, `.top(k)`, `len()`. |
| `Letterbox(imgsz)` | callable `PIL -> (1x3xSxS float32, meta)`; `.to_original(boxes, meta)` maps model-input boxes to original pixels.  `PAD_VALUE = 114`. |
| `draw(image, detections, names)` | PIL RGB with boxes. |
| `COCO80`, `class_name(cid, names)` | the default class names. |
| `is_detection_output(arr)` | true for a `[1, 4+nc, A]`-shaped head. |

`detect` layouts and coordinate frames are recorded in `Detections.meta`:
`layout`, `anchors`, `classes`, `candidates`, `conf`, `iou`, `imgsz`,
`coordinates` ("model input pixels"), `nms` ("agnostic"/"per-class").

## 1.6 `vc4infer.yolo.Detector`

```python
Detector(engine_or_path, names=None, *, device="vc4", cache_dir=None)
```

Wraps an existing `Engine` (shared; the detector does not close it) or opens
one from a path (the detector owns it).  `names` defaults to `COCO80`.
The model must have a `1x3xHxW` input; `self.imgsz = H`.

| member | meaning |
|---|---|
| `.predict(image, imgsz=None, conf=0.25, iou=0.45, max_det=300, agnostic=False, postprocess=decode_yolo)` | load/letterbox, `Engine.run`, decode, map boxes back.  `image` is a path, `PIL.Image`, or HWC `uint8` array.  Passing an `imgsz` different from the model input raises `ValueError` ("re-export at N"); the model input size is fixed. |
| `.engine`, `.names`, `.imgsz` | the underlying engine and metadata. |
| `.close()` | closes only an engine the detector opened. |

`postprocess=` receives
`(raw_output, imgsz=, conf=, iou=, max_det=, agnostic=)` and returns a
`Detections` (or a dict with `boxes`, `scores`, `classes`).  The adapter maps
the boxes from model-input pixels to the original image; everything else is
yours.

`Detection` fields: `.boxes` (N,4) xyxy in original pixels, `.scores` (N,),
`.classes` (N,) int, `.orig_shape`, `.imgsz`, `.speed_ms`
(`preprocess`/`inference`/`postprocess`), `.path`, `.meta`; `.to_dict(names)`,
`.plot(image=None)`, `len()`.

Verified example:

```python
det = vc4infer.yolo.Detector("/usr/share/vc4infer/models/yolov8n_160.onnx")
r = det.predict("bus.jpg", imgsz=160, conf=0.60, iou=0.30)
# len(r) == 3, scores [0.778, 0.703, 0.619]
```

## 1.7 `vc4infer.contrib.ultralytics`

```python
import vc4infer.contrib.ultralytics as vu

previous = vu.register(force=True, patch_warmup=True)
vu.is_registered()          # True
vu.unregister(previous)     # restore the previous .onnx backend
```

| function | meaning |
|---|---|
| `register(force=True, patch_warmup=True)` | sets `AutoBackend._BACKEND_MAP["onnx"] = VC4Backend`; returns the previous class.  Raises `ImportError` if ultralytics is not importable.  `force=False` is a no-op if already registered. |
| `unregister(previous=None)` | restores the previous `.onnx` backend (default: `ONNXBackend`) and the original `AutoBackend.warmup`. |
| `is_registered()` | true when `.onnx` currently resolves to `VC4Backend`. |
| `VC4Backend` | `BaseBackend` implementation: `load_model`, `forward`, `close`.  fp16 raises `DeviceError` (the engine is fp32 only).  It reads the model input shape for `imgsz`/`channels`/`batch`; a negative dimension marks `dynamic`. |

The two patches, and why:

1. **Backend map.**  Ultralytics 8.4 derives the format from the file suffix
   (`guess_model_type`) before consulting `_BACKEND_MAP`; there is no
   entry-point plugin API in 8.4.215.  Patching the existing `"onnx"` key is
   the minimal change.  `.pt`, `.engine` and other formats keep their normal
   backends.
2. **Warmup shim.**  `AutoBackend.warmup()` imports `torchvision`
   unconditionally.  This board's torchvision 0.29.2 registers no operators
   against torch 2.15.0+cpu (`RuntimeError: operator torchvision::nms does not
   exist`), so stock warmup aborts.  `patch_warmup=True` replaces it with the
   same NMS check using Ultralytics' pure-torch implementation.  On a normal
   torch/torchvision build, pass **`patch_warmup=False`**; the flag exists so
   the integration never silently changes behaviour.

The backend is glue: torch tensor in, numpy through `Engine.run`, numpy out.
Ultralytics' own letterbox, `non_max_suppression` and `Results` stay in charge.

## 1.8 Environment variables

| variable | effect |
|---|---|
| `VC4INFER_DAEMON` | `1` (default in the packages) uses `vc4d`; `0` is the direct-root `/dev/mem` path used only by the low-level test suite. |
| `VC4INFER_ARENA_MB` | client arena size to request (default 64).  Must be <= what `vc4d` serves. |
| `VC4_POOL_RESERVE_MB` | overrides the arena bytes kept free for the engine's per-op scratch.  The graph pool itself is sized from the model (weights + live activations + conv panels) since 1.2.5; set this only when a model needs more conv scratch than the estimate (the 320 recipe uses 32). |
| `VC4INFER_CACHE` | kernel blob cache directory (default `~/.cache/vc4infer/kcache`). |
| `VC4INFER_KGEN` | explicit path to `kgen.py`; overrides the baked/installed resolution. |
| `VC4INFER_TMO_MS` | launch timeout in ms (default 20000 in the test harness). |
| `VC4INFER_NO_JIT` | refuse to JIT; fail if a blob is not cached. |
| `VC4INFER_DEBUG` | library debug output. |
| `VC4INFER_IGNORE_ERRSTAT` | do not fail on a non-zero `ERRSTAT` (diagnostics only). |
| `VC4INFER_LIB` | explicit `libvc4infer` path for the ctypes binding. |
| `VC4INFER_ENGINE` / `VC4INFER_MODELS` | engine tree / model directory overrides. |
| `ASM3_DIR` | where `kgen.py` finds `qenc.py`/`gen.py`/`vpmwords.py` in a source-tree build.  `torch_vc4` sets it automatically if unset; a source-tree engine build needs it for uncached shapes. |
| `TORCH_VC4_ARENA_MB` | `torch_vc4` tensor region size (default 24 MiB). |

## 1.9 Threading and lifetime

* One engine at a time.  An engine owns one dma-buf arena; a second concurrent
  `Engine` raises `DeviceError` (the message says the arena is busy).
* `Engine.run` serialises calls with a lock; one engine can be shared by
  threads, and calls queue.
* `close()` (or `with`) frees the arena before the next engine is created.
* Every launch is synchronous (`LAUNCH_WAIT`); there are no streams and no
  asynchronous queue.

## 1.10 Performance, as the API reports it

`Engine.profile` and `vc4infer util` report the same counters.  On YOLOv8n
160x160, quiet board, 2026-11-20:

```
mean 478.7 ms   2.09 FPS   784 launches   GPU 259 ms   qpu_duty 54%
host_compute_nodes 0   bytes_copied_by_host 0 B
arithmetic_on_vc4 100.0% (273249600/273249600 MACs)
```

The API and the CLI agree once the session variance is accounted for.  The
launch count and the MAC share are stable; wall time is not.

---

# Part 2 - `torch_vc4`

## 2.1 Install and version

The backend is a compiled `cp313`/`aarch64` extension (one translation unit,
statically linked against `libvc4infer.a` and `libvc4client.a`).  Build it on
the Pi:

```sh
cd ~/work/torch-vc4
python3 -m venv .venv
. .venv/bin/activate
pip install --no-cache-dir --no-deps \
  /path/to/torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl
pip install --no-cache-dir typing_extensions filelock sympy networkx jinja2 \
  fsspec setuptools
ulimit -v 1800000
MAX_JOBS=1 pip install --no-build-isolation -e .
```

The wheel is in the pip index:
`pip install --extra-index-url
https://rughinnit.github.io/vc4-apt/pip/simple/ torch_vc4` installs 0.3.1.
It is a cp313/aarch64 build (the backend is a compiled extension), so it
needs the `+cpu` torch wheel on the board.  The source tree's `setup.py` says
0.3.1; an editable install made before the bumps still says 0.2.0.

Prerequisites: `vc4d` active, user in `vc4`, an engine tree with
`build/libvc4infer.a` and `tools/kgen.py`, and the `+cpu` torch wheel (the
PyPI aarch64 wheel is the CUDA build and does not import).

## 2.2 Entry points

```python
import torch, torch_vc4 as tv

tv.device()                 # torch.device("privateuseone", 0), reports vc4:0
tv.is_available()           # True if the engine arena came up (vc4d reachable)
tv.stats()                  # {"launches", "pack_ms", "kernel_ms", "gpu_cycles",
                            #  "flops", "gflops_kernel"}
tv.reset_stats()
tv.arena()                  # {"region_off", "region_bytes", "arena_bytes",
                            #  "arena_used"}
tv.synchronize()            # no-op: every launch is synchronous
tv.mse_loss(pred, target)   # MSE that stays on the device
tv.fold_bn_into_conv(weight, bias, mean, var, gamma, beta, eps=1e-5)
tv.adaptive_matrix(H, W, oh, ow)   # the averaging matrix for adaptive pool
tv.SUPPORTED_OPS            # the tuple of intercepted aten ops
```

`torch_vc4` renames `PrivateUse1` to `vc4` at import and registers a
`torch.vc4` module for RNG bookkeeping.  Importing prints torch's one-time
"Overriding a previously registered kernel" `UserWarning`; it refers to the
`PrivateUse1` stub/batching kernels this backend deliberately replaces
(`aten::add`, `convolution_overrideable`, `max_pool2d`, ...).  It is expected.

### Device string caveat

`torch.device("vc4")` raises `RuntimeError: Invalid device string: 'vc4'` in
torch 2.15: the parser is populated during `import torch`, before the rename.
Use `tv.device()` or `torch.device("privateuseone")`.  Tensors still report
`device(type='vc4')`.

## 2.3 Op table

The table below is the **0.2.0** surface: it is what the measured numbers and
the tutorial use, and what the `torch_vc4` README described then.  The current
tree is 0.3.1, with grouped conv, bilinear resize and the routed unary bodies
on top of this table; its authoritative per-op table is `COMPLETENESS.md`.
Do not assume a row from memory; run the op or read that table.

Status meanings:

* **device** - the arithmetic runs on the QPU; the host may move bytes
  (padding, permutes, concat, nearest resize) but does not multiply-add.
* **composite** - several engine kernels, or a host-built constant matrix plus
  a QPU kernel; still no host MACs.
* **raise** - not implemented; the error names the op.  There is no CPU
  fallback.

### Memory, views, transfer

| op | status | notes |
|---|---|---|
| `empty.memory_format`, `empty_strided` | device | f32 (plus int64 storage for batch-norm counters); contiguous strides; unit dims may have any stride |
| `copy_` (`.to`, `.cpu`, `.clone`) | device | f32, contiguous |
| `fill_`, `zero_` | device | f32 |
| `view`, `flatten`, `reshape`, `as_strided` | device (metadata) | alias, no copy; compute ops require contiguity and reject a non-contiguous view rather than silently copying |
| `contiguous`, `clone` | composite | host strided gather (bytes only) |
| `_local_scalar_dense` (`item`, `float(t)`) | device | downloads one element |
| `normal_`, `uniform_` | device | host `std::mt19937` generates, upload is a byte copy; `uniform_` is what module constructors call |

### GEMM / linear

| op | status | constraints |
|---|---|---|
| `mm`, `matmul` | device | 2-D, contiguous, f32; engine pack + asm10 GEMM |
| `linear` | device | `x @ W^T + bias`; 2-D or batched-2-D |

### Convolution

| op | status | constraints |
|---|---|---|
| `convolution_overrideable` (conv2d) | device | NCHW, batch 1, f32; kernel 3x3 or 1x1; stride 1 or 2; symmetric `pad <= k/2`; `groups=1` or depthwise (multiplier 1); dilation 1; the NCHW fallback has `(H+2p)*align16(W+2p)*4 <= 8191` |
| `convolution_backward_overrideable` | device | dX/dW for stride 1/2 and depthwise; the standard `F.conv2d`/`nn.Conv2d` autograd path |
| stride-2 3x3 | composite | four parity phase images + one stride-1 conv: exact, 4x the MACs |
| depthwise | composite | block-diagonal dense weight: exact, `Cin` x the MACs |
| 5x5, dilated, `conv3d`, `conv_transpose`, non-square stride/pad | raise | names the reason |

### Pooling / resize / softmax

| op | status | constraints |
|---|---|---|
| `max_pool2d` | device | 2x2/2 engine kernel (needs `H%8==0, W%16==0`) or generic NxN/stride/pad; square, dilation 1, `ceil_mode=False` |
| `_adaptive_avg_pool2d` / `adaptive_avg_pool2d` | composite | host-built averaging matrix + one QPU GEMM; any output size, batch 1 |
| `upsample_nearest2d` | composite | host byte gather (PyTorch's nearest index rule); any scale; not bilinear |
| `_softmax` | device | last dim only, <= 16 classes (engine lane width) |

### Elementwise / unary

| op | status | notes |
|---|---|---|
| `add.Tensor`, `mul.Tensor`, `sub.Tensor` | device | identical shapes; a Python scalar / CPU 0-dim operand becomes a GPU scalar op; broadcasting raises |
| `add/mul/div/sub.Scalar`, `neg`, `pow.Tensor_Scalar(2)` | device | `pow` only exponent 2 (x*x) |
| `div.Tensor` | device | `x / s` is one GPU multiply by `1/s`; `x / y` of two device tensors raises (no reciprocal-divide kernel) |
| `relu`, `silu`, `sigmoid` | device | verified engine bodies |
| `sin`, `cos`, `exp`, `log`, `tanh`, `sqrt`, `abs`, `sign`, `clip`, `leaky_relu`, `hardsigmoid`, `hardswish`, `gelu` | raise | no kernel in 0.2.0; the message names `aten::sin` etc. |

### Reductions / indexing / shape

| op | status | notes |
|---|---|---|
| `sum`, `mean` (full) | device | ones-vector GEMM, 4096-element chunks |
| `sum`/`mean` over a dim, `amax`/`amin`, `argmax`, `gather`, `index_select`, `scatter`, `sort`, `topk` | raise | not in 0.2.0; names the op or the precise reason |
| `var`, `std` | raise | not in 0.2.0 |
| `cat` | composite | host byte concat on the channel axis of NCHW (or dim 0 of a 3-D tensor) |
| `native_batch_norm`, `batch_norm` | device | inference mode only; `fold_bn_into_conv()` folds a BN into conv weights offline |
| everything else | raise | catch-all `PrivateUse1` fallback: `torch_vc4: 'aten::...' is not implemented for device vc4` |

### Explicitly absent (raises)

`sin`/`cos`/`exp`/`log`, `var`/`std`, `max`/`argmax` reductions,
`topk`/`sort`, `bmm`, broadcasting, `div` of two device tensors (no reciprocal
kernel), `conv3d`, `conv_transpose`, dilated convs, `ceil_mode` pooling,
`nn.LSTM`, `torch.compile`, int64/float64 tensors, batch > 1,
`adaptive_max_pool2d`, bilinear resize, `softmax` over a non-last dim or > 16
classes, and image `print()`/`repr()`.

## 2.4 Autograd coverage

Registered at `AutogradPrivateUse1` (`torch_vc4/autograd.py`), 0.2.0:

| forward | backward |
|---|---|
| `mm`, `matmul`, `linear` | `vc4_matmul_bt`, `vc4_transpose` + `vc4_matmul` |
| `conv2d` (stride 1/2, depthwise) | `vc4_conv2d_backward` (im2col_r + gemm3 + col2im) |
| `max_pool2d` (2x2/2) | `vc4_maxpool2x2_backward` (argmax mask) |
| `adaptive_avg_pool2d` | GEMM/transpose composition |
| `relu`, `silu`, `sigmoid`, `softmax` | `vc4_relu_backward`, `vc4_sigmoid`/`vc4_elementwise` compositions |
| `add`, `sub`, `mul`, scalar ops, `pow(2)` | elementwise kernels |
| `sum`, `mean` (full) | ones-vector GEMM |
| `view`/`as_strided` (contiguous only) | metadata + `contiguous` gather |

`max_pool2d` backward only exists for 2x2/2 (the engine's argmax-mask
kernel); the generic NxN pool is forward-only and its backward raises.
`batch_norm` is inference-only; there is no training-mode BN backward.
`torch_vc4.mse_loss` is the device MSE; `F.mse_loss` decomposes into ops this
backend does not have.

## 2.5 Accuracy notes (measured, fp32 CPU reference)

Forward parity, 0.2.0:

| op | shape | max abs | max rel |
|---|---|---:|---:|
| `mm` 256³ | - | 5.04e-04 | 1.78e-04 |
| conv 3→5 k3 s1 p1 | 1x5x8x8 | 1.05e-05 | 7.39e-06 |
| conv 16→16 k3 s1 p1 | 1x16x12x12 | 2.02e-04 | 5.28e-05 |
| conv 5→7 k3 s2 p1 | 1x7x5x5 | 1.53e-05 | 5.19e-06 |
| conv depthwise 6→6 k3 | 1x6x8x8 | 3.82e-06 | 1.22e-06 |
| max_pool2d | 9x11 and friends | **exact** | 0 |
| adaptive_avg_pool2d | 1x1, (3,5), (7,7) | <= 8.05e-07 | <= 8.05e-07 |
| softmax 4/10/16 | 3-D | <= 2.09e-07 | <= 2.09e-07 |
| sigmoid, silu | 6x5 | 5.66e-07, 3.58e-07 | same |
| relu, add, mul, neg, pow(2) | 6x5 | **exact** | 0 |
| cat (channel) | 1x7x5x5 | **exact** | 0 |
| upsample nearest | 2x/3x/(7,5) | **exact** | 0 |
| batch_norm eval | 1x5x6x6 | 2.38e-07 | 1.16e-07 |
| fold_bn_into_conv | 1x3x6x6 | 1.14e-05 | 1.46e-06 |

Gradients (0.2.0, 46 checks): all within **5.4e-06 norm-relative**, well inside
the 1e-3 acceptance bar.  The per-element `maxrel` peaks at 6.35e-03 for the
16→16 conv weight gradient - an accumulation over 144 products whose smallest
entries are ~1e-6; the absolute error is 5e-05 on a gradient whose max is
0.35.  The 16→16 conv's 2.0e-04 abs error is the expected fp32 accumulation
error over K=144.

The QPU FPU truncates (round toward zero) and flushes denormals; see
[NUMERICS.md](NUMERICS.md).  Elementwise `mul` is correctly rounded via a
Veltkamp split; `add`/`relu` are bit-exact only for exactly representable
sums.  Verify with a documented tolerance, not bit equality, wherever
truncation can bite.

## 2.6 Performance (0.2.0, stock clocks, 2026-11-20)

| op | vc4 | CPU | ratio |
|---|---:|---:|---:|
| `mm` 256³ | 4.98 ms | 2.891 ms | 1.7x slower |
| conv 16→16 3x3 32x32 | 5.20 ms | 2.30 ms | 2.3x slower |
| `mm` 32³ | 0.41 ms | 0.040 ms | 10.2x |
| `mm` 128³ | 1.05 ms | 0.342 ms | 3.1x |
| linear 1x1024x4 | 2.77 ms | 0.405 ms | 6.8x |
| conv 3→16 3x3 8x8 | 1.62 ms | 0.082 ms | 19.7x |
| conv 16→16 1x1 32x32 | 2.08 ms | 0.230 ms | 9.0x |
| conv depthwise 16 3x3 32x32 | 4.78 ms | 1.45 ms | 3.3x |
| max_pool2d 2x2 16x16 | 1.44 ms | 0.076 ms | 19.0x |
| relu 1024 | 0.24 ms | 0.019 ms | 12.2x |
| softmax 64x10 | 0.91 ms | 0.034 ms | 26.6x |
| batch_norm eval 16x16x16 | 3.87 ms | 0.076 ms | 51.3x |
| sum 4096 | 5.49 ms | 0.029 ms | 187x |

Every vc4 launch is a synchronous `LAUNCH_WAIT` round trip.  Small ops are
launch- and round-trip-bound; a whole-tensor Python dispatch adds 50-100 µs.
At 256³ the QPU GEMM is still 1.7x slower than torch's four-core CPU GEMM.
Eager PyTorch is not the speed path on this board; the graph engine is.  Full
tables and the per-op harness are `RESULTS.md` and `tests/bench_ops.py` in the
`torch-vc4` checkout.

## 2.7 Limits and operational notes

* **One GPU process at a time.**  Restart `vc4d` between long-lived GPU
  processes on this build; a departed client's arena region is not reclaimed
  promptly and the next one can fail with `out of arena memory`.
* **Never kill a client mid-launch.**  A `SIGPIPE` from `| head` or a
  `systemctl restart vc4d` while a launch is in flight wedges the V3D; only a
  reboot clears it.  See TROUBLESHOOTING.md section 3.
* **The arena is 24 MiB for tensors** inside the engine's 64 MiB client
  region.  Running out raises `torch_vc4: out of arena ...`; it does not spill
  to the CPU.
* **`torch.device("vc4")` does not parse**; use `tv.device()`.
* **No `torch.compile`, no streams, no multi-device, no fp16/bf16, no int8
  compute, no float64 compute.**  This is not CUDA parity.
* **BatchNorm training mode** is not in the 0.2.0 surface; use `.eval()` or
  `fold_bn_into_conv`.  The 0.3.x trees add it; see `COMPLETENESS.md`.
* **Importing prints a kernel-override warning**; it is expected and refers to
  the `PrivateUse1` stubs the backend replaces.

For the complete, measured per-op table with the exact raise reason for every
entry, see `COMPLETENESS.md` (the $1 work's document) and the
`torch-vc4` checkout's `RESULTS.md`.
