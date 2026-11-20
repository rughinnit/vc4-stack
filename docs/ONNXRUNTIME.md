# ONNX Runtime with the VC4 execution provider

`vc4infer.ort` puts the VideoCore IV QPUs behind the ONNX Runtime API, so an
ORT user keeps their session and only changes the provider list.

```python
import vc4infer.ort
sess = vc4infer.ort.InferenceSession("yolov8n_160.onnx", providers=["vc4"])
out = sess.run(None, {"images": x})          # list of ndarray, as in ORT
```

`get_inputs()`, `get_outputs()`, `get_providers()`, `run(output_names, feed)`
and the constructor signature follow `onnxruntime.InferenceSession`; the extra
methods (`report`, `node_backends`) are the honesty layer described below.

## Why this is a graph-partitioning provider

The first question is not "how do we implement an EP?" but "what does the
installed ONNX Runtime allow?". The answer on this board, onnxruntime 1.32.0,
cp313, manylinux_2_28_aarch64, is: **no headers are installed**.

```
$V=.../site-packages/onnxruntime
$ ls $V/capi
libonnxruntime.so.1.32.0  libonnxruntime_providers_shared.so
onnxruntime_pybind11_state.cpython-313-aarch64-linux-gnu.so  ...
$ find $V -name '*.h' | wc -l
0
$ nm -D --defined-only $V/capi/libonnxruntime.so.1.32.0 | grep OrtGetApiBase
0000000000126a80 T OrtGetApiBase@@VERS_1.32.0
$ strings $V/capi/libonnxruntime.so.1.32.0 | grep -E 'CreateEpFactories|RegisterExecutionProviderLibrary'
CreateEpFactories
ReleaseEpFactory
RegisterExecutionProviderLibrary
UnregisterExecutionProviderLibrary
$ python -c "import onnxruntime as o; print(o.register_execution_provider_library.__doc__)"
Register an execution provider library with ONNX Runtime.
```

So the runtime *does* support out-of-tree EP plugins: `OrtGetApiBase` is
exported, the plugin entry points (`CreateEpFactories`, `ReleaseEpFactory`)
are present as strings, and the Python binding exposes
`register_execution_provider_library(name, path)`, `get_ep_devices()`,
`OrtEpDevice` and `SessionOptions.add_provider_for_devices(devices, opts)`.
The missing piece is the ABI itself: a plugin must be compiled against
`onnxruntime_ep_c_api.h` at exactly this version, and that header is not in
the wheel, is not in an sdist (there is no onnxruntime sdist on PyPI), and the
source tree is not reachable from the board (`raw.githubusercontent.com`
returns 404 for every onnxruntime path). Compiling against a guessed struct
layout is how you get a segfault in someone else's inference server, so this
is not attempted.

What ships instead is the provider ORT users see anyway: a graph partitioner
on the public Python API. The static node-claiming logic lives in one module
(`vc4infer/ort/_partition.py`) and is exactly the input a native EP's
`GetCapability` would consume, so the work is not wasted if a future image
ships the header.

## What is claimed

A node is claimed only when the strict engine can execute it exactly, using
the same static report as `vc4infer nodes`:

| tier | meaning | backend |
|---|---|---|
| `vc4` | a QPU kernel exists and the static constraints hold | vc4 |
| `host-meta` | Constant/Shape and integer shape arithmetic resolved at plan time | vc4-folded |
| `unknown` | an implementation exists but a shape could not be proven | ort-cpu |
| `host-unwritten` | a numeric op with no QPU kernel (Einsum, TopK, ...) | ort-cpu |
| `host-control` | If/Loop/Scan | ort-cpu |

A whole connected component is downgraded when any boundary tensor is not
fp32 or has a dynamic dimension, and models containing control-flow
subgraphs are not split at all. There is no CPU compute inside a claimed
partition: the sub-model is executed by `mode="strict"`, which raises rather
than fall back.

The report is per node, with measured time where it exists:

```
vc4infer.ort 1.2.2 -- yolov8n_160.onnx (263 nodes, 1 partition)
  providers requested: vc4
  backend       nodes   %nodes        MACs    %MACs
  vc4             233    88.6%      273.2M   100.0%
  vc4-folded       30    11.4%         0    0.0%
  ort-cpu           0     0.0%         0    0.0%
  total           263   100.0%      273.2M
  last run:  vc4 496.47 ms | ort-cpu 0.00 ms | transfer 0.01 ms
  work share: vc4 100.0% of nodes, 100.0% of MACs
  fully claimed: every node runs on the QPUs
```

`vc4-folded` nodes are Constant/Shape nodes: the strict executor resolves
them at plan time with no launch, so they belong to the VC4 side but are not
runtime compute. `%MACs` is the number that matters; a provider claiming 90%
of nodes and 2% of MACs is not accelerating anything, and the table says so.

`node_backends()` returns the same data as rows; `run(..., profile=True)` (or
`InferenceSession(..., profile=True)`) adds the engine's per-node wall/GPU
time to the VC4 rows. ORT nodes only have a partition-level measurement, so
their per-node time stays blank instead of being invented.

### Deliberately unsupported node

`infer/onnxruntime_ep/tests/test_vc4_ep.py` builds Conv → Relu → **Einsum** →
Conv → Relu. vc4infer has no Einsum kernel, so the graph must split in three:

```
  backend       nodes   %nodes        MACs    %MACs
  vc4               4    80.0%         ...
  ort-cpu           1    20.0%         ...
  NOT fully claimed: 1 node(s) run on the ORT CPU EP
```

and the output still matches ORT CPU to `1e-4 + 1e-4*max(|ref|,1)` on every
element. `providers=[("vc4", {"fallback": "error"})]` refuses the model at
session creation instead, naming the node.

## Installation

The provider is part of the `vc4infer` package; only ONNX Runtime is extra:

```sh
sudo apt install ./vc4infer_1.2.2_arm64.deb      # engine + vc4infer package
sudo systemctl enable --now vc4d                 # unprivileged GPU access
pip install onnxruntime                          # 1.32.0 on cp313/aarch64
```

or, in a venv that already has the system package:

```sh
python3 -m venv --system-site-packages ~/ortenv
~/ortenv/bin/pip install onnxruntime
PYTHONPATH=/usr/lib/python3/dist-packages ~/ortenv/bin/python ...
```

Check it:

```python
import vc4infer.ort
print(vc4infer.ort.get_available_providers())    # ['vc4', ...]
print(vc4infer.ort.is_available())               # True with vc4d running
```

## Threading, arena and lifetime

* **One VC4 session per process.** `vc4d` serves one client arena (64 MiB by
  default, 72 MiB on this board). A second live session in the same process
  raises `DeviceError` with that sentence instead of failing obscurely in the
  daemon. Create one session and call `run()` repeatedly; `close()` (or the
  context manager) before opening another. ORT-only sessions
  (`providers=["CPUExecutionProvider"]`) do not take the arena and may coexist.
* `run()` is serialised by an `RLock`: one inference at a time per session.
  Do not call a session concurrently from several threads and expect GPU
  parallelism; the engine is one queue.
* The engine is opened lazily on the first `run()`, not at import or session
  construction, so importing `vc4infer.ort` costs nothing and a missing daemon
  is reported when it matters.
* **Weights** are converted once: the VC4 sub-model (nodes, initialisers,
  boundary types) is written to `~/.cache/vc4infer/ort/<model-sha>/pN.onnx`
  and reused across processes. The engine uploads the packed panels into the
  arena once per session for a single-partition model; a multi-partition model
  has to re-upload when the scheduler switches partitions (the engine has one
  pool per arena), which is stated here because it costs milliseconds.
  Set `VC4INFER_ORT_CACHE` to move the cache.
* The arena is shared by every vc4d client. If another process holds most of
  it, `vc4_init` succeeds with a smaller arena and a large model fails during
  weight upload. `min_arena_mb` (or `VC4INFER_ORT_MIN_ARENA_MB`) makes the
  provider wait for a quiet GPU instead, up to `startup_timeout_s`
  (`VC4INFER_ORT_STARTUP_S`, default 3 s; set 120+ when the board is busy).
  Two clients sharing the QPUs concurrently can hang the GPU, so serialising
  whole runs is the safe pattern.

Provider options (tuple form, e.g.
`providers=[("vc4", {"min_arena_mb": 60})]`):

| option | default | meaning |
|---|---|---|
| `fallback` | `"cpu"` | `"cpu"` leaves unclaimed nodes to ORT; `"error"` refuses |
| `min_arena_mb` | `0` | wait for at least this much arena before running |
| `startup_timeout_s` | `3` | how long to wait for the arena |
| `arena_mb` | daemon default | request a specific arena size |
| `cache_dir` | `~/.cache/vc4infer/kcache` | kernel blob cache |
| `profile` | `False` | record per-node VC4 timings on every run |

## Numerics

fp32 throughout; no precision change is offered or implied. A claimed node
runs the same kernels as `vc4infer run`, so the native engine's per-op
evidence applies. ORT nodes use ORT's fp32 CPU kernels. Both test cases below
check every output element, and YOLO is checked against the stored CPU
reference at the same 5e-4 tolerance as `vc4infer selftest`.

| case | graph | tolerance | result |
|---|---|---|---|
| small CNN (torch.onnx.export) | 12 nodes, all vc4 | 1e-4 + 1e-4·max(\|ref\|,1) | **0/10**, max abs 2.53e-07 |
| CNN + Einsum split | 5 nodes, 4 vc4 + 1 ort | 1e-4 + 1e-4·max(\|ref\|,1) | **0/2,048**, max abs 3.1e-06 |
| YOLOv8n 160 | 263 nodes, all vc4 | 5e-4 + 5e-4·max(\|ref\|,1) | **0/44,100**, max abs 0.0204 (max scaled 0.43) |

## Performance

Stock clocks, no overclock, one session per process, `reps=5` after a warm-up:

| backend | ms/inference | FPS | note |
|---|---:|---:|---|
| ORT + vc4 provider | 518–699 | 1.4–1.9 | all 263 nodes claimed |
| vc4infer native (`Engine`) | 882 | 1.13 | same kernels, no ORT layer |
| ORT CPU | 260 | 3.85 | baseline; fp32 CPU kernels on 4x A53 |

The provider adds Python-level partition dispatch and boundary copies, not
compute: one partition means one `Runner.run` per inference, the same call
`Engine.run` makes. The 518 ms figure is the acceptance run (`reps=2`); the
699 ms figure is the median of 5 runs in the three-way benchmark, taken while
concurrent jobs were using the board. The provider and the native runner are the
same kernels, and repeated measurements on this shared Pi 3 varied by 25%
between processes, so do not read a provider-vs-native difference into the
table.

**ORT's CPU EP is faster than the VC4 at 160x160** (260 ms vs ~520-700 ms).
That is the honest result: the QPUs win on CPU offload and on models whose
compute per byte is high enough to amortise the engine's launch overhead, not
on this small YOLO input. The provider's job is to make the QPUs available to
ORT users, not to claim a speedup that is not there.

## Reproducing

```sh
# export the torch CNN (any venv with torch)
python infer/onnxruntime_ep/export_cnn.py /tmp/smallcnn.onnx

# all acceptance cases (needs vc4d and a free GPU)
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_vc4_ep.py --reps 3

# three-way YOLO benchmark, each backend in its own process
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/bench_yolo160_ort.py --all --reps 5
```

## Limits

* Fixed shapes, fp32, batch 1 for the NCHW fallback (the engine's own limits;
  the NHWC path supports batches, but the ONNX runner's Gemm/GlobalAveragePool
  are 2-D single-image).
* Control flow, custom domains and any numeric operator without a VC4 kernel
  are executed by ORT and reported. This is ORT's contract, made visible.
* The provider does not implement ORT's IOBinding, CUDA-style streams or
  device-memory `OrtValue`s; inputs and outputs are numpy arrays.
* A native in-process EP would remove the Python boundary copies. It needs
  `onnxruntime_ep_c_api.h` for the installed version; see the evidence above.
