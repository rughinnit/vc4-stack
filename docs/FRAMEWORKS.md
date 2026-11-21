# Framework support on VC4 — what plugs in, what it costs, what it cannot do

Board: Raspberry Pi 3B (BCM2837, VideoCore IV/V3D), Debian 13 arm64, CPython
3.13.5, 905 MiB RAM, CMA 256 MiB, stock clocks (core 400 MHz / V3D 300 MHz).
Engine: `libvc4infer` v1.2.1 + `vc4d` (72 MiB arena, 64 MiB client region).
Every number below is either measured on this board or copied from the
project's own run logs; the evidence for each row is in §8.

The short version: **"support every framework" is not N bespoke backends.**
It is one strict op table (§2) behind a small number of *interchange formats*
(ONNX, StableHLO/Relax, ggml) and *plugin interfaces* (PyTorch `PrivateUse1`,
ORT execution provider, PJRT, LiteRT delegate, TensorFlow PluggableDevice,
OpenVINO plugin, `ggml-backend`).  A new framework is a lowering plus a
partition, not a new engine.

---

## 1. Effort classes used here

| class | meaning |
|---|---|
| **available now** | shipped in this project; a user can run it today |
| **small (days)** | ≤1 week; one shim, no new kernel semantics |
| **medium (weeks)** | 2–6 weeks; a real adapter with partitioning and some new kernels |
| **large (months)** | >2 months; needs a compiler backend or a new runtime layer |
| **not viable** | the framework's own ABI or packaging blocks it; a fork would be required |

---

## 2. The engine surface every adapter lowers to

`infer/src/vc4_api.c` is the single source of truth.  `infer/README.md`
lists 22 of 34 entries supported: `matmul`, `conv2d/3x3`, `conv2d/1x1`,
`add`, `mul`, `relu`, `sigmoid`, `silu`, `relu_bwd`, `rsqrt`, `transpose`,
`matmul_bt`, `softmax_ce_bwd`, `softmax` (≤16 classes), `im2col`, `col2im`,
`conv2d/bwd_dx`, `conv2d/bwd_dw`, `maxpool/2x2s2`, `maxpool/NxNsS`,
`maxpool_bwd`.  The ONNX executor composes these into stride-2,
dilated, grouped and depthwise convs, and the graph-level ops (Concat,
Split, Slice, Reshape, Transpose, Resize) are device-side `vc4_copy2d`
gathers/stores.  On YOLOv8n 160×160 that is 233 VC4 nodes + 30 plan-time
folded nodes, `host_compute_nodes = 0`, `bytes_copied_by_host = 0`,
`arithmetic_on_vc4 = 100.0%` (273,249,600 / 273,249,600 MACs).

Any adapter that claims a node without an engine op **must raise**.  That is
the invariant that makes the coverage tables honest.

---

## 3. Ranked support table (inference and "enable it today")

Ranked by value delivered per unit of effort on *this* board.

| # | framework | extension point (the API a third party implements) | effort | testable here | status in this project | what a user writes to enable it | what this buys |
|---|---|---|---|---|---|---|---|
| 1 | **ONNX** | none needed — native executor over the strict op table (`infer/python/vc4infer/_onnxrun.py`) | **available now** — shipped executor, 263-node YOLOv8n run end to end | yes — `onnx` 1.23.2 (cp312-abi3) installs and imports on cp313 | **implemented** — YOLOv8n 160/320 end-to-end, 0 host compute, 100% MACs | `vc4infer run m.onnx --input x.npy`<br>`Engine("m.onnx").run({"x": a})` | the widest model zoo with no framework installed; the export target every other framework can reach |
| 2 | **PyTorch** | `PrivateUse1` + `PrivateUse1HooksInterface` + `AutogradPrivateUse1` (`TORCH_LIBRARY_IMPL(aten, PrivateUse1, …)`) | **available now** — built backend, 46 grad checks, subset raises | yes — torch 2.15.0+cpu wheel; `torch_vc4` built for cp313 | **implemented** — `~/work/torch-vc4` @ 6496a83, 46 gradient checks, CNN trains on-device | `import torch_vc4; d = torch_vc4.device()`<br>`model.to(d); y = model(x.to(d))`<br>`y.sum().backward()` | the only path where the framework's own autograd drives QPU backward kernels; users keep torch code |
| 3 | **ONNX Runtime** | EP C API: `OrtEpFactory`/`OrtEp`/`OrtNodeComputeInfo`, entry points `CreateEpFactories`/`ReleaseEpFactory`; Python `register_execution_provider_library` | **medium (weeks)** for a real EP; the Python provider was **small (days)** | ORT 1.32.0 cp313/aarch64 installs; no VC4 EP exists; wheel ships **no headers** | **in progress** — `vc4infer.ort` graph-partitioning provider + `infer/onnxruntime_ep/tests` | today: `import vc4infer.ort`<br>`InferenceSession(m, providers=["vc4"]).run(None, {"x": a})`<br>EP: `register_execution_provider_library("vc4", "libvc4_ep.so")` | the widest inference ecosystem: every ORT consumer (Python/C/C++/Java/JS) and its quantisation tooling |
| 4 | **ggml / gguf** | `ggml_backend_i` + `ggml_backend_reg_i` + `GGML_BACKEND_DL_IMPL`/`ggml_backend_score`; loaded by `ggml_backend_load_all` | **medium (weeks)** | yes: ggml built from the vendored tree, backend loads both ways (`docs/GGML.md`) | **backend shipped (fp32 subset)** | drop `libggml-vc4.so` in the backend search path;<br>`ggml_backend_load_all()` then lists the device;<br>llama.cpp runs with it | LLM/transformer graphs with the smallest plugin ABI (no compiler), plus `ggml-opt` training |
| 5 | **TFLite / LiteRT** | external delegate `.so`: `tflite_plugin_create_delegate` / `tflite_plugin_destroy_delegate`; internally `TfLiteDelegate::Prepare` + `ReplaceNodeSubsetsWithDelegateKernels` (the dispatch API is exported too, but its header is `litert/c/internal/`) | **medium (weeks)** | yes — `ai-edge-litert` 2.3.0 cp313/aarch64; `Interpreter` + `load_delegate` | **runs on the QPUs** — `litert/`; 34 checks, 0 failures on silicon, max |d| 1.71e-07 against the LiteRT CPU reference | `i.load_delegate("litert/build/libvc4_delegate.so")`<br>`i.allocate_tensors(); i.invoke()` | the edge model zoo; the claimed subset runs on the QPUs and every other node stays in LiteRT |
| 6 | **Apache TVM** | Relax BYOC: pattern table + `FuseOpsByPattern` + `relax.ext.<name>` codegen + JSON runtime. **the 0.27 wheel has no `tvm/relay` package, so `relay.ext` is not available.** | **large (months)** — a codegen is a compiler backend | yes — `apache-tvm` 0.27.0.post1 aarch64 imports; `relax.ext.example_npu` is a template, not built in | **not started** | nothing today;<br>when done: `relax.ext.vc4` codegen + PassContext config | one codegen for TF/Keras/MXNet/PyTorch frontends via Relax importers; `relax.training` autodiff |
| 7 | **JAX** | PJRT C API: `PJRT_Client_Create`, `PJRT_Api` table, `PJRT_LoadedExecutable_Execute`, `PJRT_Buffer_ToHostBuffer`; discovery via `jax_plugins` namespace/entry point → `register_plugin(name, priority=, library_path=, options=)` | **large (months)** for general coverage — PJRT is a compiler backend in disguise (a StableHLO executor); the shipped plugin executes an fp32 subset | yes — jax/jaxlib 0.12.1 cp313/aarch64; plugin executes the subset and `jax.grad` on the QPUs | **plugin shipped (fp32 subset)** — 25 VHLO ops including slice, reverse, pad, iota, gather, any-axis reduce, batched dot and convolution; `grad` runs on the device; unsupported ops refused by name; `docs/PJRT.md` | `JAX_PLATFORMS=vc4` plus a `jax_plugins.vc4` package that calls `register_plugin("vc4", library_path="libpjrt_vc4.so")` (or `PJRT_NAMES_AND_LIBRARY_PATHS`) | the only real JAX path; `grad`/`vmap`/`jit` come free once the primitives execute |
| 8 | **TensorFlow** | PluggableDevice via StreamExecutor C API: `SE_InitPlugin`, `SE_PlatformRegistrationParams`, device/stream/allocator/event fns, plus kernel registration | **large (months)** — a platform plus every kernel | wheel yes (2.21.0 cp313/aarch64, ships `stream_executor.h` and `pjrt_c_api.h`); **runtime import unverified** (§8.4) | **not started** | `tf.load_pluggable_device_library("libvc4_plugin.so")`<br>`with tf.device("/device:VC4:0"): …` | the Keras/TF ecosystem and its host autodiff; the only in-graph TF path |
| 9 | **OpenVINO** | device plugin loaded with `Core::register_plugin(path, "VC4")`; remote tensors via `RemoteContext::create_tensor`; **the plugin ABI header is not in the wheel** | **large (months)** — needs the OpenVINO source tree for the plugin ABI | yes — 2026.4.1 cp313/aarch64, `available_devices == ["CPU"]` | **not started** | `core.register_plugin("libvc4_plugin.so", "VC4")`<br>`core.compile_model(m, "VC4")` | model-server ecosystem; its ONNX/Paddle/PyTorch/TF/TFLite/JAX/GGUF frontends are already in the wheel |
| 10 | **MXNet** | `MXCustomOpRegister` / `mxnet::ext::CustomOp` (operator only); `Storage` has no registration API and `DeviceType` is a closed enum (kCPU/kGPU/kCPUPinned/kCPUShared) | **not viable** — would need an MXNet fork | no — 1.9.1 aarch64 wheel installs but `import mxnet` fails: `libarmpl_lp64_mp.so` missing | **not started** | `MXCustomOpRegister("MyOp", creator)` for ops only; no new device | legacy only; everything it reaches is already reachable through ONNX or TVM |

---

## 4. Training dimension (every row)

Autodiff is not a property of the accelerator; it is a property of the
framework.  Three families:

* **host autodiff** (PyTorch, TF, JAX, MXNet): the framework records/traces the
  forward graph and *emits primitives* for the backward pass.  Training works
  as soon as the device executes those primitives — no gradient maths in the
  device plugin.
* **compiler autodiff** (TVM Relax): a compiler pass (`relax.transform.Gradient`)
  rewrites the module into forward+backward; `relax.training.Trainer` supplies
  the optimiser.  The BYOC codegen must lower the backward primitives.
* **device/graph autodiff** (ggml-opt): the trainer builds a backward graph with
  `ggml_opt_*` and asks the backend to execute it; optimizer state lives in
  backend buffers.
* **none** (ONNX as a format, LiteRT, OpenVINO): inference compilers/runtimes
  with no gradient machinery to plug into.

| framework | where autodiff lives | what training requires from us | training feasibility here | memory (Adam = 16 B/param) |
|---|---|---|---|---|
| ONNX | none in the format | an external trainer (PyTorch/JAX) that exports a new graph; our executor is inference-only | **inference only (no autodiff to plug into)** | as inference (26.6 MiB pool at YOLOv8n-160) |
| PyTorch | host tape (`AutogradPrivateUse1` Functions) | backward kernels (shipped: `matmul_bt`, conv dX/dW, `relu_bwd`, `maxpool_bwd`, `softmax_ce_bwd`) + device optimiser updates (hand-written SGD/Adam) | **training available now** for the op table — tiny CNN 39 ms/step, 46 gradient checks ≤5.4e-6 norm-rel; MLP and conv net 100% train/test | ~0.25–1 M params at 32×32/batch 8 in 64 MiB (§5) |
| ONNX Runtime | ORT Training (separate package) | a *training* EP plus ORT's optimiser/checkpoint machinery; `onnxruntime-training` has no aarch64/cp313 wheel (1.19.2 is x86_64, cp38–cp311) | **not testable here** (no wheel); would be inference-only even with an EP | n/a |
| ggml | device/graph (`ggml-opt`) | backend must execute the backward graph (`ggml_opt_epoch`) and hold grads + optimizer tensors in the same buffers | **training after N backward ops** (ggml-opt is already the trainer; the backend just needs the ops) | 16 B/param in arena; same ceiling |
| TFLite/LiteRT | none | nothing to plug into | **inference only (no autodiff to plug into)** | int8 shrinks footprint only |
| TVM | compiler (`Gradient` pass + `relax.training.Trainer`, SGD/Adam in `relax/training/optimizer.py`) | the `relax.ext.vc4` codegen must lower the gradient ops as well as the forward ones | **training after N backward ops** (and after the codegen exists) | 16 B/param in arena |
| JAX | host trace (`jax.grad` → StableHLO) | a PJRT device that executes the primitives `jax.grad` emits; no gradient maths in the plugin | **training after the PJRT executor** (then training is free) | 16 B/param in arena |
| TensorFlow | host `GradientTape` (graph functions) | PluggableDevice + registered kernels for every op the gradient functions use | **training after the PluggableDevice + N kernels** | 16 B/param in arena |
| OpenVINO | none (OTX trains in PyTorch and converts) | nothing to plug into | **inference only (no autodiff to plug into)** | n/a |
| MXNet | host `mx.autograd` | a working device first; the wheel does not even import | **not feasible on this board** (no runnable MXNet) | n/a |

---

## 5. Memory reality: the largest trainable model

Training state is `params + grads + optimizer`:

| optimizer | bytes/param | 0.25 M params | 1 M params | 3.15 M (YOLOv8n) |
|---|---:|---:|---:|---:|
| SGD | 8 | 1.9 MiB | 7.6 MiB | 24.0 MiB |
| SGD + momentum | 12 | 2.9 MiB | 11.4 MiB | 36.1 MiB |
| Adam | 16 | 3.8 MiB | 15.3 MiB | 48.1 MiB |

Add activations.  One fp32 activation tensor is `batch × C × H × W × 4`
bytes: at 32×32, batch 8, 64 channels that is **2.0 MiB**; at 64×64 it is
**8.0 MiB**.  Backward keeps roughly one to three times the forward set.

Worked budgets against the real arena sizes:

* **Default client (64 MiB, daemon `--arena-mb 72`).**  A 0.25 M-parameter
  CNN with Adam = 3.8 MiB of state, leaving ~60 MiB for ~10–20 saved
  activation tensors at 32×32/batch 8 (2 MiB each).  This fits.  A 1 M-param
  model (15.3 MiB state) also fits at 32×32/batch 8, with ~48 MiB left.
  At 64×64 the activation set (8 MiB/tensor) eats the budget at ~1 M params.
* **Raised arena (`--arena-mb 150`, client `VC4INFER_ARENA_MB=145`, graph
  pool 113 MiB — the configuration measured for 320×320 inference).**  A
  ~1 M-param CNN at 32×32/batch 16, or ~0.25 M at 64×64/batch 8, fits with
  room; a 3.15 M-param YOLOv8n's state alone is 48.1 MiB, and its measured
  forward activation peak is 26.6 MiB at 160×160 *batch 1* — backward needs
  more, and batch 8 multiplies it by eight.
* **Verdict.**  The largest class that trains here is a **small CNN of
  ~10^5–10^6 parameters at ≤64×64, batch 8–16, fp32, Adam or SGD**.  That is
  the same class the project has actually trained: the conv toy net
  (`conv 1→8 → relu → pool → fc`, 640 train/160 test) reaches 100% at
  7.56 steps/s / 121 samples/s, and the torch CNN trains at 39 ms/step.

YOLO-scale training is **not a software gap**.  It is memory and throughput:
forward is 543.5 ms/image at 160×160 (1.84 FPS) and 1323.1 ms at 320×320
(0.76 FPS), with `qpu_duty` 58.9% / 71.9% — the QPU is idle 41% / 28% of the
wall waiting for the ARM/daemon launch path.  Using the project's own
"backward ≈ 2–3× forward" figure, one 160×160 step at batch 8 is
≈ 8 × 543.5 ms × 3 ≈ **13 s**; one COCO epoch (118,287 images / 8 = 14,786
steps) is ≈ **53 hours**, before optimizer or validation.  At 320×320 it is
about four times worse.  No plugin can fix that arithmetic.

---

## 6. What cannot be supported on this hardware, whatever the framework

These are properties of the V3D and the board, not of an adapter.

| capability | why it is impossible here | measured reason |
|---|---|---|
| **fp16 / bf16 arithmetic** | the QPU FPU is fp32; there is no half-precision datapath or instruction | the engine's dtype surface is fp32 only; `matmul/f16` is an explicitly unsupported op; torch rejects `Double` (`only float32 tensors are supported`) and has no half kernels |
| **int8 speed** | the ISA has no packed byte MAC; `v8muld` is a normalised `round(a*b/255)` | measured int8 and fp32 MAC rates agree to **within 0.5%** (`docs/int8-RESULTS.md`); int8 buys footprint, never MAC/s |
| **fp64 at speed** | no fp64 ALU; the FPU is fp32 and truncates (round toward zero) | `docs/NUMERICS.md` §1: both `fadd` and `fmul` round toward zero; there is no fp64 path to measure — it would have to be emulated on fp32/integer pairs |
| **dynamic shapes at scale** | kernels are JIT-compiled per shape, the VDR pitch is 8191 B, and the plan resolves shapes at plan time; `If`/`Loop`/`Scan` are `host-control` and are refused under the device invariant | 263-node YOLOv8n needs ~100 distinct QPU blobs; `Shape`/`Gather` are folded at plan time; `If` raises `NeedsDeviceDispatch` under `host_policy="device"` |
| **working sets >72 MiB** | one dma-buf arena (72 MiB daemon / 64 MiB client); CMA is 256 MiB **shared** with the display and other clients | 320×320 needed `--arena-mb 150` + 145 MiB client + 113 MiB pool (33.4 MiB peak); 640×640 is ~4× that and does not fit CMA |
| **multi-device** | one V3D with 12 QPUs; all clients share the `vc4` group and one arena | CMA total 256 MiB; one device in `vc4d`; `torch_vc4` exposes exactly one `vc4:0` |
| **streams / async concurrency** | `vc4d` allows one launch in flight (`E_BUSY` otherwise) and takes `flock` per launch; the kernel model is one group per QPU per launch | launch path is **41.1%** of wall at 160×160 (223.1 / 543.5 ms); two concurrent workloads can hang the V3D |
| **tensor cores / FMA** | no fused multiply-add instruction; add and mul are separate ALU ops | `README.md`: "No fp16/bf16, no FMA"; dual issue is one add + one mul per word, not an FMA |
| **YOLO-scale training** | state (48.1 MiB for 3.15 M params with Adam) + activations + batch exceed the arena, and a step is seconds | §5: ≈53 h per COCO epoch at 160×160 batch 8; 0.76 FPS forward at 320×320 |

The same list applies to every framework below: a better plugin cannot create
an fp16 datapath, a second GPU, or 10× the memory bandwidth.

---

## 7. Training on VC4: what it is and is not

**It is** fp32 training of small models, with the forward *and* backward
arithmetic on the QPUs and no CPU compute fallback.  There are two working
demonstrations:

* the hand-written trainer (`training/`): MLP 64-32-10 at 100% train/test,
  103.5 steps/s SGD / 90.5 steps/s Adam; conv toy net at 100%,
  7.56 steps/s / 121 samples/s at batch 16; gradient checks 1e-7–1e-5;
* `torch_vc4`: a real `PrivateUse1` backend with `AutogradPrivateUse1`
  Functions, 46 gradient checks within 5.4e-6 norm-relative, and a tiny CNN
  trained end-to-end at 39 ms/step (77 launches/step).

**It is not** YOLO-scale training, mixed precision, large batches, data
parallelism, or a general autograd library.  The engine has no fp16/bf16, no
checkpointing, no gradient accumulation, and one arena.  At 160×160 the
forward alone is 1.84 FPS.

**Inference-only frameworks cannot train, regardless of effort.**  LiteRT and
OpenVINO have no autodiff to plug into; ONNX as a format carries no gradients;
ORT Training is a separate runtime that has no aarch64/cp313 wheel.  Adding a
VC4 delegate to any of them yields faster inference, never a trainer.

**Host-autodiff frameworks make training a kernel-coverage problem, not a new
project.**  PyTorch, JAX and TF derive the backward graph themselves; the work
is to execute the primitives they emit.  That is why the PyTorch path was
able to train as soon as the engine had `matmul_bt`, `relu_bwd`,
`conv2d_backward`, `maxpool_bwd` and `softmax_ce_bwd`.

---

## 8. How a new framework gets added

The same four steps every time; the ONNX and PyTorch paths are the worked
examples.

1. **Lower to the engine op table.**  Map the framework's ops to
   `infer/src/vc4_api.c`.  Anything without a kernel is *claimed by nobody*
   and must raise.  Use `vc4infer nodes -v model` (ONNX) or the adapter's
   partition report; the authoritative list is `coverage.py`.
2. **Verify each op against the framework's own reference.**  Same input,
   same graph, compare per element with `|got-ref| <= atol + rtol*max(|ref|,1)`
   (the project uses 1e-4/1e-4 for op tests, 5e-4/5e-4 for YOLO).  The
   patterns to copy are `infer/tools/onnx/verify_extra.py` (ONNX graphs),
   `infer/tests/genops_probe.py` (single ops) and `torch-vc4/tests/test_ops.py`
   (aten ops).  A pass is a mismatch count, not a boolean.
3. **Publish the coverage table.**  Claimed nodes, refused nodes with the
   reason, `%` of graph MACs on the QPUs, host bytes moved, launches.
   `vc4infer nodes -v` and `vc4infer util --json` already produce this for
   ONNX; every adapter should emit the same shape.
4. **Raise on the rest.**  `UnsupportedNode(node, op)` /
   `HostNotAllowed(node, op)` / `NeedsDeviceDispatch(node, op)`, with the node
   name in the message.  No silent CPU fallback: under
   `host_policy="device"`, `host_compute_nodes` must be `[]`.

Only after those four does an adapter get a row in §3.  The engine stays
strict; the adapter absorbs the framework's quirks.

---

## 9. Why "supports all frameworks" is formats + plugins, not N backends

Every framework in §3 reaches the same 22-op engine through one of five
doors:

| door | frameworks | what crosses it |
|---|---|---|
| **ONNX** (native executor) | PyTorch (`torch.onnx.export`), TF/Keras (`tf2onnx`), JAX (`jax2onnx`), sklearn, … | a static fp32 graph; the executor partitions and runs it |
| **StableHLO / Relax** (compiler front ends) | JAX (StableHLO), TF (via XLA/Relax importer), MXNet (Relax importer) | a compiler IR; needs a codegen, not a shim |
| **ggml** (LLM graphs) | llama.cpp and anything built on ggml/gguf | a graph of ggml tensors and ops |
| **ORT** (runtime) | anything that produces an `.onnx` and runs under ORT | an EP-claimed subgraph |
| **per-framework device plugins** | PyTorch `PrivateUse1`, TF PluggableDevice, JAX PJRT, LiteRT delegate, OpenVINO plugin | the framework's own device ABI |

The last row is the expensive one, and it is only justified when the
framework's *runtime* (not just its exporter) must own the device.  PyTorch
is the proof that it pays off: the same engine that runs an ONNX graph also
trains a CNN because `PrivateUse1` lets torch's autograd call the engine's
backward kernels directly.  JAX is the counter-example: PJRT looks like a
device plugin but requires a StableHLO executor, i.e. a compiler backend, so
it is `large (months)` even though the ABI is clean.  `pjrt/` is that
executor for an fp32 subset (`docs/PJRT.md`); general JAX coverage still means
a compiler backend.

---

## 10. Excluded from the ranking

| package | why | evidence |
|---|---|---|
| `paddlepaddle` 3.3.1 | no Linux aarch64 wheel (macOS arm64 only) | PyPI file list |
| `mindspore` 2.11.0 | no aarch64 wheel (macOS arm64 only for some cp versions) | PyPI file list |
| `tflite-runtime` 2.14.0 | superseded by `ai-edge-litert`; newest wheel is cp311, no cp313 | PyPI file list |
| `onnxruntime-training` 1.19.2 | x86_64 only, cp38–cp311 | PyPI file list |
| `gguf` 0.19.0 | pure-Python *file format* reader; contains no compute backend (ggml is the backend) | wheel is `py3-none-any` |

---

## 11. Evidence: what was actually checked on 2026-11-20

All package checks were run on the board (throwaway venvs under
`~/scratch/tmp/fw`); PyPI metadata was read from `https://pypi.org/pypi/<pkg>/json`.
No GPU workload was started by this survey; the GPU was busy.

### 11.1 Platform

```
uname -m                      aarch64
python3 -V                    Python 3.13.5
cat /etc/os-release           Debian GNU/Linux 13 (trixie)
free -m                       Mem: 905 MiB total; Swap: 2952 MiB
df -h /tmp                    /dev/mmcblk0p2  58G  (disk, not tmpfs)
nproc                         4
/proc/device-tree/model       Raspberry Pi 3 Model B Rev 1.2
ps -eo cmd | grep vc4d        /usr/local/sbin/vc4d --arena-mb 72 --group vc4 --drop-user nobody
grep Cma /proc/meminfo        CmaTotal: 262144 kB
```

### 11.2 Wheels and imports (board)

| package | wheel used | result |
|---|---|---|
| `onnxruntime` 1.32.0 | `cp313-manylinux_2_28_aarch64` | installs; `get_available_providers() == ['AzureExecutionProvider', 'CPUExecutionProvider']`; `register_execution_provider_library` present |
| `apache-tvm` 0.27.0.post1 | `py3-none-manylinux_2_27_aarch64` | imports; **no `tvm.relay`**; `tvm.relax` present; `relax.transform.Gradient` present |
| `ai-edge-litert` 2.3.0 | `cp313-manylinux_2_27_aarch64` | imports; `Interpreter`, `load_delegate` import |
| `tensorflow` 2.21.0 | `cp313-manylinux_2_27_aarch64` (282,213,003 B) | wheel downloaded, ships `stream_executor.h` and `c/pjrt_c_api.h`; **install aborted** (1.3 GB unpack on a 905 MiB board shared with three concurrent builds) → runtime `unverified` |
| `jax` 0.12.1 + `jaxlib` 0.12.1 | `cp313-manylinux_2_27_aarch64` | imports; `jax.jit(x*2)(3.) == 6.0`; `jax.grad(x*x)(3.) == 6.0` on CPU |
| `openvino` 2026.4.1 | `cp313-manylinux_2_35_aarch64` | imports; `Core().available_devices == ['CPU']` |
| `mxnet` 1.9.1 | `py3-none-manylinux2014_aarch64` | installs; `import mxnet` fails: `OSError: libarmpl_lp64_mp.so: cannot open shared object file` |
| `onnx` 1.23.2 | `cp312-abi3-manylinux_2_26_aarch64` (ok on cp313) | imports |
| `torch` 2.15.0 | `cp313-manylinux_2_28_aarch64` | the PyPI wheel is the CUDA build and fails to import (`libcudart.so.13`); the `+cpu` wheel is the usable one (project `RESULTS.md`) |

### 11.3 API names read from wheels/binaries (not from memory)

* **PyTorch** — `torch/utils/backend_registration.py:76 rename_privateuse1_backend`;
  `torch_vc4/csrc/torch_vc4.cpp:23 #include <ATen/detail/PrivateUse1HooksInterface.h>`,
  `:2216 struct Vc4Hooks : at::PrivateUse1HooksInterface`, `:2253 at::RegisterPrivateUse1HooksInterface`,
  `:2265 TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`, `:2293 C10_REGISTER_GUARD_IMPL(PrivateUse1, …)`;
  `torch_vc4/autograd.py:478 AutogradPrivateUse1`.
* **ORT** — `onnxruntime/capi/libonnxruntime.so.1.32.0` strings:
  `CreateEpFactories`, `ReleaseEpFactory`, `OrtEpFactory`, `OrtEpDevice`,
  `OrtNodeComputeInfo::CreateState()`, `OrtEp::GetCapability()`,
  `OrtDataTransferImpl`; `onnxruntime/__init__.py:63` exports
  `register_execution_provider_library`.
* **TVM** — `tvm/relax/backend/contrib/example_npu/README.md` (BYOC flow:
  `FuseOpsByPattern`, `relax.ext.example_npu`, JSON runtime); `tvm/relax/training/`
  (`Trainer`, `SetupTrainer`, `optimizer.SGD/Adam`); `tvm/relax/transform/__init__.py:50 Gradient`.
  `tvm.relay` is absent from the wheel (directory listing; import fails).
* **LiteRT** — `ai_edge_litert/interpreter.py:60-123` uses
  `tflite_plugin_create_delegate` / `tflite_plugin_destroy_delegate` and names
  `tensorflow/lite/delegates/external/external_delegate_interface.h`;
  `libLiteRt.so` exports `LiteRtSetDelegateFunction`, `LiteRtWrapDelegate`,
  `LiteRtUnwrapDelegate`, `LiteRtDispatchDelegateGetHooks`.
* **TensorFlow** — `tensorflow/include/tensorflow/c/experimental/stream_executor/stream_executor.h`:
  `SE_InitPlugin` (line 530), `SE_PlatformRegistrationParams` (509),
  `SE_CreateStreamExecutorParams` (420), `SE_CreateDeviceParams` (187),
  `SE_CreateDeviceFnsParams` (225); `tensorflow/python/framework/load_library.py:160
  load_pluggable_device_library`.
* **PJRT** — `tensorflow/include/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h`:
  `PJRT_Client_Create` (486), `PJRT_Device_GetDescription` (1344),
  `PJRT_LoadedExecutable_Execute` (1917), `PJRT_Buffer_ToHostBuffer` (2266),
  and the `PJRT_Api` field table (`_PJRT_API_STRUCT_FIELD`, 2779+).
* **JAX** — `jax/_src/xla_bridge.py:459 discover_pjrt_plugins()`: `jax_plugins`
  namespace + entry point, `initialize()`, `register_plugin(name, priority=,
  library_path=, options=)`.
* **OpenVINO** — `openvino/include/openvino/runtime/core.hpp:860 register_plugin`,
  `:883 unload_plugin`; `remote_context.hpp:149 create_tensor`;
  `remote_tensor.hpp:23 class RemoteTensor`.  No `IPlugin`/plugin-ABI header anywhere
  in the wheel's include tree (grep).
* **ggml** — `ggml-backend.h`: `ggml_backend_reg_t`, `ggml_backend_register` (248),
  `ggml_backend_load` (272), `ggml_backend_load_all` (276);
  `ggml-backend-impl.h`: `ggml_backend_i` (130), `ggml_backend_device_i` (185),
  `ggml_backend_reg_i` (239, `get_proc_address` 248), `GGML_BACKEND_DL_IMPL` (267),
  `ggml_backend_score` (263); `ggml-opt.h`: `ggml_opt_init` (138),
  `ggml_opt_epoch` (219), `ggml_opt_dataset_init` (39), `ggml_opt_result_*` (164+).
  Read from llama.cpp vendored in `llama-cpp-python` 0.3.37 sdist.
* **MXNet** — `mxnet/include/mxnet/c_api.h:3236 MXCustomOpRegister`;
  `lib_api.h:750 class CustomOp` (`setForward`, `setBackward`, `setCreateOpState`);
  `storage.h:35 class Storage` (virtual `Alloc`/`Free`, `static Storage* Get()`,
  no registration); `base.h:103 enum DeviceType { kCPU, kGPU, kCPUPinned, kCPUShared }`.

### 11.4 Project numbers used above

* `README.md`: peak 28.8 GFLOP/s structural; ALU-only 28.36; coalesced
  16.42; GEMM 9.48; compiled VC4CL 3.09; 76 MiB training arena.
* `docs/ONNX.md`: YOLOv8n-160 543.5 ms / 1.84 FPS / 58.9% duty / 784 launches /
  41.1% launch path / 0 host bytes / 100% MACs; 320×320 1323.1 ms / 0.76 FPS /
  71.9% duty / 1936 launches; 26.6 MiB pool at 160, 33.4 MiB at 320 with a
  145 MiB client and 113 MiB pool.
* `docs/CONV-RESULTS.md`: conv toy net 7.56 steps/s, 121 samples/s, 100%.
* `docs/TRAINING-RESULTS.md`: MLP 103.5 steps/s SGD, 90.5 Adam, 100%,
  gradient checks 1e-7–1e-5.
* `torch-vc4/RESULTS.md`: 46 gradient checks ≤5.4e-6 norm-rel; tiny CNN
  39 ms/step, 77 launches/step, 3080 launches total.
* `docs/int8-RESULTS.md`: int8 vs fp32 MAC rate within 0.5%; no packed byte MAC.
* `docs/NUMERICS.md` §1: FPU truncates; fp32 only.

### 11.5 What is `unverified`, and what blocked it

| item | why |
|---|---|
| TensorFlow 2.21 import/runtime on this board | the install unpacks 1.3 GB onto an SD card while three concurrent builds compile and run; pip was still writing after ~20 min at 235 MB. Aborted to keep the board alive. The **wheel, its aarch64/cp313 tag and its PluggableDevice/PJRT headers are verified**; the import is not. |
| ggml backend on the board | no aarch64 llama.cpp build was attempted; API verified from the vendored sdist. Building llama.cpp on this 905 MiB board is a separate (large) job. |
| ORT EP end-to-end | the wheel ships no headers and `raw.githubusercontent.com` / `api.github.com` returned 404 for this repo from this host, so the EP ABI could not be compiled here; verified from the shipped `.so` strings and Python API instead. |
| OpenVINO plugin build | the plugin ABI header (`openvino/runtime/iplugin.hpp`) is not in the wheel; the OpenVINO source tree is not obtainable on the board (GitHub unreachable, no sdist). |
| fp64 throughput | no fp64 path exists to measure. |
| TF/PJRT header provenance | `pjrt_c_api.h` and `stream_executor.h` were read from the TensorFlow 2.21 wheel (the same XLA/PJRT API JAX speaks), not from JAX's own wheel, which ships only the XLA FFI headers. |

Nothing above was inferred from a blog post or a version guess; where a number
is arithmetic on measured values (the 53 h/epoch figure, the memory budgets in
§5) it says so.
