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
The ABI is the missing piece: a plugin must be compiled against
`onnxruntime_ep_c_api.h` at exactly this version. The wheel does not ship it,
PyPI publishes no onnxruntime sdist, and the source tree is not reachable from
the board (`raw.githubusercontent.com` answers 404 for every onnxruntime
path). It was obtained at the exact version instead, from the conda-forge
`onnxruntime-cpp` 1.32.0 package (a build of the `v1.32.0` tag), and vendored
in `infer/onnxruntime_ep/third_party/onnxruntime-1.32.0/` with the tarball
hash and package hash in `PROVENANCE.md`.

So there are two separate things in this tree, and they should not be
confused:

* **`vc4infer.ort`** — a graph-partitioning provider on the public Python API.
  It is what actually runs models today, it claims a node only when the strict
  engine can execute it exactly, and the claim table below is measured with
  it.
* **`infer/onnxruntime_ep/c_ep/libvc4_ep.so`** — a native plugin built against
  the vendored ABI. It registers with ORT 1.32.0, advertises one VideoCore IV
  EP device and claims **zero** nodes: no compute path is wired into it. It is
  evidence that the ABI is usable, not an acceleration path. A user adding it
  to a session gets CPU results with one extra capability query per graph.

## What is claimed

A node is claimed only when the strict engine can execute it exactly, using
the same static report as `vc4infer nodes`:

| tier | meaning | backend |
|---|---|---|
| `vc4` | a QPU kernel exists and the static constraints hold | vc4 |
| `host-meta` | Constant/Shape and integer shape arithmetic resolved at plan time | vc4-folded |
| `unknown` | an implementation exists but a shape could not be proven | ort-cpu |
| `host-unwritten` | a numeric op or attribute form with no QPU kernel; measured forms are `Compress` with `axis=None`, `Resize` with `mode="cubic"`, `MaxPool` with `dilations`, and `Conv` with dilation above 8 | ort-cpu |
| `host-control` | If/Loop/Scan | ort-cpu |

A whole connected component is downgraded when any boundary tensor is not
fp32 or has a dynamic dimension. That is not a per-node penalty: a
`Compress(axis=None)` in the middle of a CNN costs the node that produced the
dynamic tensor *and* the island downstream of it, because the partitioner
cannot prove that boundary either (measured below: 4 of 6 nodes fall back).
Models containing control-flow subgraphs are not split at all, and they are
handed to ORT verbatim: an `If`/`Loop` body can capture an outer-scope
initialiser that is not a boundary tensor, so rebuilding a sub-model would
drop it. There is no CPU compute inside a claimed partition: the sub-model is
executed by `mode="strict"`, which raises rather than fall back.

The report is per node, with measured time where it exists:

```
vc4infer.ort 1.2.10 -- yolov8n_160.onnx (263 nodes, 1 partition)
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

### Measured claim table (CPU side)

`infer/onnxruntime_ep/tests/test_ort_claims_cpu.py` exercises the claim
decision through `vc4infer.ort.plan()`. It needs no daemon, opens no engine and
dispatches nothing, so it is safe to run while the GPU is busy. Every plan is
then executed on ORT's CPU EP partition by partition and compared with the
whole model; that isolates the graph surgery (sub-model nodes, initialisers,
boundary order) from the QPU kernels, and a difference there would be a
partitioner bug rather than kernel approximation.

| model | nodes | vc4 | vc4-folded | ort-cpu | partitions | MACs on vc4 | plan on CPU vs whole model |
|---|---:|---:|---:|---:|---:|---:|---|
| small CNN (`torch.onnx.export`) | 12 | 12 | 0 | 0 | 1 | 100% | 0/10 over tol, max abs 0 |
| CNN + `Resize(mode="cubic")` | 5 | 4 | 0 | 1 | 2 vc4 + 1 ORT | 100% (Resize reports 0 MACs) | 0/4096, max abs 0 |
| CNN + `Compress(axis=None)` | 6 | 2 | 0 | 4 | 1 + 1 | 87.1% | 0/1024, max abs 0 |
| `If` | 1 | 0 | 0 | 1 | 0 + 1 | n/a (no MACs) | 0/4, max abs 0 |
| YOLOv8n-160 | 263 | 233 | 30 | 0 | 1 | 100% of 273.2M | 0/44100 over 5e-4, max abs 0 |

`plan()` is the same decision `InferenceSession(providers=["vc4"])` makes:
`plan.report()` prints the table above per node, and
`plan(path, fallback="error")` raises for the first node that would fall back.
Two negative results are part of the measurement:

* with `vc4d` absent, `InferenceSession(providers=["vc4"])` raises
  `DeviceError` naming the missing socket instead of running the graph on ORT
  and calling it acceleration;
* a cubic `Resize` is refused with
  `node 'resize_cubic' (Resize): fallback='error': 1 node(s) cannot run on
  VC4`, and the `If` model with `node 'if' (If)`.

### What the fallback costs

The split case shows the contract ORT defines: the claimed islands are still
VC4 work, the one unclaimed node runs on ORT's CPU EP, and the report names it.
What it does *not* do is hide the boundary: each ORT island is a separate
`onnxruntime.InferenceSession`, so the tensors crossing it are copied into and
out of ORT. For the cubic-Resize split that is 3 dispatches per inference.

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

## The C plugin: registration without pretense

`infer/onnxruntime_ep/c_ep` builds `libvc4_ep.so` from `vc4_ep.cc` against the
vendored headers:

```sh
make -C infer/onnxruntime_ep/c_ep     # ~72 KiB, exports CreateEpFactories/ReleaseEpFactory
```

`CreateEpFactories` returns one factory; `GetSupportedDevices` creates a GPU
hardware device for the VideoCore IV and one `OrtEpDevice` carrying
`vc4.mode=capability-only`; `CreateEp` returns an `OrtEp` whose `GetCapability`
adds nothing to the graph support info. `tests/test_ep_register.py` builds,
registers and runs it on the board:

```
available providers: ['AzureExecutionProvider', 'CPUExecutionProvider', 'vc4']
device: ep=vc4 vendor=vc4infer type=OrtHardwareDeviceType.GPU
        meta={'vc4.mode': 'capability-only',
              'vc4.execution': 'none: claims 0 nodes',
              'vc4.driver': 'vc4d', 'library_path': '.../libvc4_ep.so'}
vc4_ep: GetCapability on 4 node(s), claimed 0 (capability-only build)
session providers: ['vc4', 'CPUExecutionProvider']
output vs CPU EP: 0/6 over 1e-6, max abs 0
plugin counters: 1 capability call(s), 4 node(s) examined, 0 claimed
```

Two ABI facts were established by building it, not by reading:

* `ORT_API_VERSION` is 32 in both the vendored header and the installed
  library. A plugin compiled against another version gets `NULL` from
  `OrtApiBase::GetApi()` inside `CreateEpFactories` and registers no factory,
  so it cannot run with a mismatched struct layout.
* `OrtEpFactory::CreateDataTransfer` is documented as optional, but 1.32 calls
  it during `register_execution_provider_library` without checking for `NULL`.
  A factory that leaves it `NULL` segfaults the interpreter at registration,
  before a session exists. The plugin returns an empty transfer, which is
  accurate: it never owns device memory.

Making it execute would need three things that are not written: `GetCapability`
building the claim list (`_partition.plan()` already produces it), `Compile`
returning an `OrtNodeComputeInfo` that feeds the boundary tensors through
`libvc4infer`'s strict runner, and an allocator/memory description for the EP
device. Until those exist, `vc4_ep_nodes_claimed()` stays zero and this section
says so; the Python provider remains the only path that runs a node.

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
| CNN + `Resize(cubic)` split | 5 nodes, 4 vc4 + 1 ort, 3 partitions | 1e-4 + 1e-4·max(\|ref\|,1) | plan on CPU: **0/4096**, max abs 0 (the device row is re-measured by the acceptance run; the fixture replaced Einsum, which is now a VC4 op) |
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
the board was busy with concurrent work. The provider and the native runner are the
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
# export the torch CNN (any venv with torch); nothing goes under /tmp, which
# is tmpfs on this board
python infer/onnxruntime_ep/export_cnn.py ~/scratch/tmp/vc4ort/smallcnn.onnx

# CPU-side claim table and partition equivalence: no daemon, no GPU
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_ort_claims_cpu.py \
    --cnn ~/scratch/tmp/vc4ort/smallcnn.onnx \
    --json ~/scratch/tmp/vc4ort/claims.json
# expects "OK: 6 CPU-side case(s) passed": cnn 12/12; split 4 vc4 + 1 ort and
# 0/4096; poison 2 vc4 + 4 ort; If 0 claimed; YOLO 233 + 30 + 0 and 0/44100;
# and max abs 0 for every plan against the whole model on ORT CPU

# the C plugin: build, register, compare with the CPU EP (claims no nodes)
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_ep_register.py
# expects: 'vc4' in available providers, one GPU device, 4 nodes examined,
# 0 claimed, output max abs 0

# device acceptance: needs vc4d and the GPU to itself
sudo systemctl restart vc4d          # after any client that timed out
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_vc4_ep.py \
    --data /usr/share/vc4infer/models --work ~/scratch/tmp/vc4ort --reps 3
# expects: cnn fully claimed, 0/10 over tol; split 3 partitions, 4 vc4 + 1 ort,
# 0/4096, refusal names resize_cubic; yolo 263/263 claimed, 0/44100

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
* A model with control flow is never split, even partially: the whole model is
  handed to ORT, because an `If`/`Loop` body can capture an outer-scope
  initialiser that the boundary list does not contain.
* A data-dependent boundary downgrades the island downstream of it as well as
  the node that produced it. `Compress(axis=None)` in a 6-node CNN moves 4
  nodes to ORT.
* The claim decision is static. A boundary whose shape is not in the graph
  (no shape inference result and no `value_info`) is not claimed, even when a
  run would have a fixed shape.
* The native plugin `libvc4_ep.so` claims no nodes and executes nothing. Adding
  it to a session costs a capability query per graph and returns CPU results;
  it is evidence about the plugin ABI, not an acceleration path.
* The provider does not implement ORT's IOBinding, CUDA-style streams or
  device-memory `OrtValue`s; inputs and outputs are numpy arrays.
