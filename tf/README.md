# TensorFlow PluggableDevice for the VideoCore IV QPUs

`tf/build/libtf_vc4_plugin.so` is a TensorFlow PluggableDevice: a
StreamExecutor platform that registers one `VC` device whose memory is the
`libvc4infer` dma-buf arena, plus the op kernels that run on the QPUs. It is
the in-graph TensorFlow path from `docs/FRAMEWORKS.md` row 8.

State of this slice (measured on the board, 2026-11-21):

| piece | state |
|---|---|
| `SE_InitPlugin`: platform, device, device fns, stream executor | **registers** — `/physical_device:VCIV:0`, `/device:VCIV:0` |
| allocator over the engine arena, H2D/D2H/D2D copies | **works** (round-trip exact on the engine arena; needs 8 MiB of free daemon arena) |
| `TF_InitKernel`: 18 kernels, fp32 | **registered**; eager dispatches every one to its VCIV kernel |
| refusals (dtype, shape, rank, transposed MatMul, NCHW bias) | **checked** — the CPU stage reaches each kernel and reads the named error |
| arithmetic on the QPU | **wired, not yet measured**: the serialized silicon pass below is the only step left |
| broadcast, `Const`, `Identity`, batched matmul, autodiff | not registered — see "What the next op costs" |

## TensorFlow and API versions

* Targets the `tensorflow` 2.21.0 `cp313-manylinux_2_27_aarch64` wheel.
* StreamExecutor C API **0.0.1** (`SE_MAJOR 0`, `SE_MINOR 0`, `SE_PATCH 1`
  from `tensorflow/c/experimental/stream_executor/stream_executor.h`).
  TensorFlow 2.21 fills these three fields before calling `SE_InitPlugin` but
  never compares them; what it does enforce is every `struct_size`, in both
  directions (`ValidateSEPlatformRegistrationParams`, `ValidateSPPlatform`,
  `ValidateSPPlatformFns`, `ValidateSPStreamExecutor`), so the plugin must set
  `struct_size` on every struct it fills and must not shrink one.
* Kernel registration is the op/kernel C API (`tensorflow/c/kernels.h`):
  `TF_NewKernelBuilder`, `TF_KernelBuilder_TypeConstraint`,
  `TF_RegisterKernelBuilder`, and the plugin's `TF_InitKernel` symbol.
* The plugin links `libtensorflow_framework.so.2` from the wheel for the C API
  symbols and `libvc4infer` for the engine. At load time the framework is
  already in the process, so no `LD_LIBRARY_PATH` is needed.

Two details of 2.21 that cost time:

* **`tf.load_pluggable_device_library` is not exported.** The function exists
  (`tensorflow/python/framework/load_library.py`) but has no `@tf_export`, so
  it must be called as
  `tensorflow.python.framework.load_library.load_pluggable_device_library(path)`.
  `tf.load_library` is a different entry point and does not register a
  StreamExecutor platform.
* **A plugin error aborts the process.** `TF_LoadPluggableDeviceLibrary`
  CHECK-fails on any error the plugin reports (`c_api_experimental.cc:746`),
  so a malformed `SP_Platform` kills the interpreter rather than raising. The
  plugin therefore validates its own struct sizes and pointers.

## The device type is `VCIV`, not `VC4`

TensorFlow 2.21 validates a StreamExecutor platform name and type with
`tensorflow::device_utils::ValidateDeviceType`, whose regex is
`[A-Z][A-Z_]*` — no digits. A platform named `VC4` aborts the load:

```
F ... Check failed: (...) Device name/type 'VC4' must match [A-Z][A-Z_]*.
```

`device_type` is taken from `SP_Platform::type`, and TensorFlow names the
device `/device:<type>:0`, so with a stock wheel the device can only be
`/device:VCIV:0`. The build takes the name as a variable for a TensorFlow whose
regex allows digits:

```
make DEVICE_TYPE=VC4        # only against a patched tensorflow
```

`tf.device("/device:VC4:0")` parses as a `DeviceSpec` in 2.21 but cannot be
satisfied, because no device of that type can be registered.

## What is implemented

StreamExecutor callbacks, and what each one does:

| callback | implementation |
|---|---|
| `get_device_count` | 1 |
| `create_device` / `destroy_device` | `SP_Device` with `hardware_name = "VideoCore IV QPU"` |
| `create_device_fns` / `destroy_device_fns` | struct sizes only; no NUMA/GFLOPS hints |
| `create_stream_executor` / `destroy_stream_executor` | fills every callback TensorFlow requires |
| `allocate` / `deallocate` | blocks from `vc4_alloc()` in the engine arena (host mode: `posix_memalign`) |
| `get_allocator_stats` / `device_memory_usage` | bytes limit/usage of the tensor share of the arena |
| `create_stream` / `destroy_stream` | a stream id; the engine is synchronous |
| `create_stream_dependency`, `get_stream_status` | no-ops (nothing is ever queued) |
| events (`create`/`record`/`wait`/`status`/`block_host_for_event`) | complete at enqueue |
| timers | `CLOCK_MONOTONIC` start/stop |
| `memcpy_dtoh` / `sync_memcpy_dtoh` | `vc4_download()` (cache maintenance included) |
| `memcpy_htod` / `sync_memcpy_htod` | `vc4_upload()` |
| `memcpy_dtod` / `sync_memcpy_dtod` | download to a host bounce buffer, then upload |
| `mem_zero`, `memset`, `memset32` | 4 KiB pattern buffer through `vc4_upload()` |
| `host_callback` | runs the callback inline |
| `synchronize_all_activity` | no-op |

The first 3 MiB of the arena are the QPU code slots and uniform blocks
(`vc4d`'s `A_DATA_BASE`, and the start every engine context is checked against
in `vc4_init()`), so the data heap is `arena - 3 MiB`. The allocator hands
TensorFlow 3/4 of that heap and leaves 1/4 for the engine's per-op scratch
(im2col panels, the rounded-up tail of an elementwise op). `vc4_alloc()` is a
bump allocator with no free, so the plugin keeps a block table and reuses a
freed block before bumping. `use_bfc_allocator` is set, so TensorFlow
suballocates tensors inside one or two of those blocks. This kernel work did
not change that contract: the allocator callbacks above are untouched, and no
kernel allocates for itself.

`VC4_TF_ARENA_MB` (default 64, clamp 8–96) sets the arena request, and
`VC4D_SOCKET` picks the daemon (default `/run/vc4d.sock`). A TensorFlow
process is normally unprivileged, so the plugin defaults the engine to the
`vc4d` backend: if `VC4INFER_DAEMON` is unset it is set to `1` before the
engine starts. `VC4INFER_DAEMON=0` still selects the direct root `/dev/mem`
backend.

`VC4_TF_ARENA=host` replaces the engine arena with 64 B aligned `malloc`
blocks for the copy path only; it exists so the plugin wiring can be checked
when no daemon arena is available. Every kernel refuses to run in that mode.

## The kernel set

Eighteen ops, all `T = float32`. Every one hands its arithmetic to a
`libvc4infer` QPU kernel; none of them computes on the host.

| TensorFlow op | engine call | QPU kernel | accuracy (measured, docs/ACCURACY.md) |
|---|---|---|---|
| `Add`, `AddV2` | `vc4_elementwise(ADD)` | `ew` 256 floats/group | ≤1 ulp (FPU truncates) |
| `Mul` | `vc4_elementwise(MUL)` | `ew` | ≤1 ulp |
| `Sub` | `vc4_elementwise(SUB)` | `ewb sub` (one FSUB) | ≤1 ulp |
| `Div`, `RealDiv` | `vc4_elementwise(DIV)` | `ewb div` (reciprocal of `\|b\|`, sign restored, multiply) | rel 3e-7 for any normal divisor; `\|b\| < 2^-126` saturates instead of returning ±inf |
| `Maximum` | `vc4_elementwise(MAX)` | `ewb max` (one FMAX) | bit-exact selection |
| `Minimum` | `vc4_elementwise(MIN)` | `ewb min` (one FMIN) | bit-exact selection |
| `Relu` | `vc4_elementwise(RELU)` | `ew` | bit-exact |
| `Sigmoid` | `vc4_elementwise(SIGMOID)` | `ew` | abs 1e-6 |
| `Relu6` | `vc4_unary(CLIP, 0, 6)` | `uop clip` | bit-exact |
| `Tanh` | `vc4_unary(TANH)` | `uop tanh` | abs 1.2e-7 |
| `Neg` | `vc4_unary(NEG)` | `uop neg` | bit-exact |
| `Exp` | `vc4_unary(EXP)` | `uop exp` | rel 5.7e-6 |
| `Log` | `vc4_unary(LOG)` | `uop log` | rel 4.8e-7 |
| `Sqrt` | `vc4_unary(SQRT)` | `uop sqrt` | rel 1.1e-7 |
| `MatMul` | `vc4_matmul(A, B, C, M, N, K, ROW_MAJOR)` | pack + GEMM | rel 1e-4 (the engine's own bound) |
| `BiasAdd` | `vc4_copy2d_n` (bias expand) + `vc4_elementwise(ADD)` | copy2d + `ew` | ≤1 ulp |

`RealDiv` is registered because `tf.math.divide` lowers to it; `Div` stays for
graphs built with the raw op. `AddV2` is what `tf.add` uses in 2.21.

`Sub`, `Div`, `Maximum` and `Minimum` are the `kops.EWB` bodies (`FSUB`,
`FMAX`, `FMIN`, reciprocal+multiply) that the engine already carries for its
backward and ONNX paths. `vc4_elementwise` now routes those four selectors to
them, which is what its registry row already advertised; before that change it
refused them as unimplemented, and the LiteRT delegate's `SUB`/`DIV` path
would have failed on silicon for the same reason. **The plugin therefore needs
a `libvc4infer` built from this tree for those four selectors.** Against an
older library they return `VC4_ERR_UNSUPPORTED`, and the node reports it:
`tf-vc4 Sub: sub failed on 1024 elements: not implemented on the GPU (this
selector needs the libvc4infer from this tree)`. Add, Mul, the unary family,
MatMul and BiasAdd work against 1.2.7.

Shape rules:

* binary ops: identical shapes. Broadcasting is refused by name (the
  delegate does the same, with `VC4_CLAIM_BROADCAST`).
* unary ops: any shape with rank ≤ 8; the element count is rounded up to the
  engine's group internally, so no length has to be a multiple of 256.
* `MatMul`: rank 2, `A[M][K] * B[K][N]`, no transpose (`transpose_a` /
  `transpose_b` are refused at kernel construction). `M`, `N` and `K` may be
  any non-zero size: `vc4_matmul` pads `M` to 8 and `N` to 64 in its own
  scratch and never reads the padding back.
* `BiasAdd`: `data_format="NHWC"` (bias on the last axis), any rank ≥ 1, bias
  rank 1 with `bias.shape[0] == x.shape[-1]`. The engine's add is same-shape,
  so the bias row is first expanded over the rows with `vc4_copy2d_n` whose
  source row stride is zero and the add then runs in place — the output holds
  the bias plane until it holds `bias + x`. That is 2 launches, or 3 when the
  row is not a multiple of 16 floats (a tail copy), and 1 when the tensor is a
  single row.

## Launch counts

The elementwise and unary kernels run 256 floats per group and at most 12
groups per launch (`vc4__ew_run` batches one group per QPU per launch), so any
`n ≤ 3072` is a single launch. `vc4_matmul` is always one pack launch plus
one GEMM launch: the packer spreads its 8-row units over at most 4 QPUs and
the GEMM grid-strides its 8-row blocks over at most 12 QPUs, looping its
64-column tiles inside the kernel.

| workload | launches |
|---|---|
| any elementwise/unary op, `n = 1024` or `n = 1000` (padded to 1024) | 1 |
| `MatMul` 32×32×32 | 2 (one pack, one GEMM) |
| `MatMul` 30×34×26 | 2 (M, N and K padded inside scratch) |
| `BiasAdd` `[1][1024]` | 1 (single row: plain add) |
| `BiasAdd` `[7][1000]` | 5 (copy 62 vectors, copy the 8-float tail, add 28 groups = 3 launches) |
| `matmul → bias → relu` on `[8][32] · [32][16]` | 5 (pack, GEMM, bias copy, bias add, relu) |

## Refusals

Every kernel refuses rather than compute on the host, and says which op it is:

```
tf-vc4 Mul needs the engine arena; VC4_TF_ARENA=host is a copy-path diagnostic only
tf-vc4 AddV2: inputs are not the same shape ([4,4] and [4]); broadcasting is not implemented
tf-vc4 MatMul: transpose_a is not supported; vc4_matmul takes A[M][K] and B[K][N] row-major
tf-vc4 BiasAdd: data_format="NCHW" is not supported; only NHWC (bias on the last axis) is
tf-vc4 Sqrt: a tensor pointer is outside the arena (was it placed on the host?)
```

A dtype other than float32 never reaches a kernel: `TF_KernelBuilder_TypeConstraint`
keeps it out of the registry, so eager places e.g. a float64 `Mul` on the CPU.
The compute path checks the dtype again because the registry is not the only
caller of the kernel entry point.

## Host memory types

No kernel calls `TF_KernelBuilder_HostMemory`: every argument of these ops is
arithmetic data and belongs in the arena. TensorFlow therefore allocates all
inputs and outputs through `SP_StreamExecutor::allocate`, whose blocks are
arena ranges; `find_range_locked()` maps a tensor pointer back to its offset
within such a block. `host_memory_allocate` / `host_memory_deallocate` exist
only for the copy path's staging buffers. If an op were ever to pin an
argument to host memory, its pointer would not resolve and the kernel would
refuse, which is the intended failure: the QPU cannot read a malloc'd buffer.

## Build and CPU-side checks

```
cd ~/work/vc4-tf
make -C tf PYTHON=$HOME/scratch/tfvenv/bin/python
TF_CPP_MIN_LOG_LEVEL=1 $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py
```

Expected (no GPU work, no device memory):

```
physical devices after load:  ['/physical_device:CPU:0', '/physical_device:VCIV:0']
logical devices after load:   ['/device:CPU:0', '/device:VCIV:0']
PASS  platform registers /physical_device:VCIV:0
PASS  device /device:VCIV:0 is visible
PASS  kernel Add registered on VCIV: dtypes=[1]
... (one line per op, Add through BiasAdd)
registered kernels: 18 of 18
plugin: arena 64 MiB, 61 MiB of data heap, 45 MiB for tensors, 0 MiB in use, engine arena; engine not started

all checks passed
```

Copy path and per-kernel dispatch, without a QPU launch (host arena):

```
VC4_TF_ARENA=host $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy --dispatch
```

Expected:

```
PASS  constant landed on /device:VCIV:0
PASS  H2D/D2H round-trip is exact: max|d|=0
PASS  Add dispatched to the VCIV kernel: ... tf-vc4 Add needs the engine arena; ...
... (one line per op, Add through BiasAdd)
PASS  broadcast Add is refused by name: ... tf-vc4 AddV2: inputs are not the same shape ([4,4] and [4]); ...
PASS  an empty Mul needs no engine: (0,)
PASS  MatMul refuses K == 0: ... tf-vc4 MatMul: K == 0 is not supported (no zero-fill path)
PASS  BiasAdd refuses a bias of the wrong length: ... the bias has 8 elements but the input's last dimension is 16
PASS  float64 Mul falls back to the CPU (no VC kernel): /job:localhost/.../device:CPU:0
PASS  MatMul refuses transpose_a: ... tf-vc4 MatMul: transpose_a is not supported; ...
PASS  BiasAdd refuses data_format=NCHW: ... tf-vc4 BiasAdd: data_format="NCHW" is not supported; ...

all checks passed
```

That stage is the point: eager placed all 18 ops on `/device:VCIV:0`, called
this plugin's kernels, and each one refused because there is no engine behind
the host arena. The graph stage under the same arena shows the placement of a
multi-op graph:

```
VC4_TF_ARENA=host $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --graph
PASS  graph lands on /device:VCIV:0: ... tf-vc4 MatMul needs the engine arena; ...
```

The same copy stage against the engine arena, no host mode and no QPU
launch, with an 8 MiB arena on the shared daemon:

```
VC4_TF_ARENA_MB=8 $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy
```

```
plugin: arena 8 MiB, 5 MiB of data heap, 3 MiB for tensors, 0 MiB in use, engine arena; engine not started
PASS  constant landed on /device:VCIV:0
PASS  H2D/D2H round-trip is exact: max|d|=0
```

`tf-vc4: engine up, arena 8 MiB, 3 MiB for tensors, device type VCIV` is
printed on stderr the first time device memory is touched, which is when the
allocator maps the real dma-buf.

## The serialized silicon pass

Run only when the board is otherwise idle (the engine serializes launches
through `vc4d`, but the measured numbers need an idle GPU), and with the
engine arena, not the host one.

Build the engine from this tree first — `Sub`, `Div`, `Maximum` and `Minimum`
need this tree's `vc4_elementwise` routing — and link the plugin against it so
the measurement does not overwrite the installed library:

```
cd ~/work/vc4-tf-ops
make -C vc4d build/libvc4client.a
make -C infer build/libvc4infer.so.1
make -C tf PYTHON=$HOME/scratch/tfvenv/bin/python \
  VC4_LIBS="-L$PWD/infer/build -lvc4infer -Wl,-rpath,$PWD/infer/build"
```

(`readelf -d tf/build/libtf_vc4_plugin.so | grep RUNPATH` confirms the
build-tree engine is the one that will load.)

Then, with an idle board:

```
VC4_TF_ARENA_MB=8 VC4_TF_ALLOW_QPU=1 \
  $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py \
  --copy --kernels --graph --add
```

Needs a reachable `vc4d` socket with 8 MiB of free arena. If the shared
daemon's arena is exhausted, give the run its own daemon (it listens on
`/run/vc4d.sock` by default; the engine follows `VC4D_SOCKET`):

```
sudo /usr/sbin/vc4d --socket /run/vc4d-tf.sock --arena-mb 32 \
    --group vc4 --drop-user raspi --max-timeout-ms 20000 &
VC4D_SOCKET=/run/vc4d-tf.sock VC4_TF_ARENA_MB=8 VC4_TF_ALLOW_QPU=1 \
  $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy --kernels --graph
```

Expected, per op and length (the test prints `max|d|` and the count over
tolerance; tolerance is `atol + rtol*|want|`):

| op | n = 1024 | n = 1000 | tolerance |
|---|---|---|---|
| `Add`, `AddV2`, `Mul`, `Sub` | `max|d| ≤ 4.8e-07`, 0 over | same | 1e-6 abs + 1e-6 rel |
| `Div`, `RealDiv` | `max|d| ≤ 3e-07` relative, 0 over | same | 1e-6 abs + 1e-6 rel |
| `Maximum`, `Minimum` | `max|d| = 0`, 0 over | same | exact |
| `Relu`, `Relu6`, `Neg` | `max|d| = 0`, 0 over | same | exact |
| `Sigmoid` | `max|d| ≤ 1e-06`, 0 over | same | 1e-6 abs + 1e-6 rel |
| `Tanh` | `max|d| ≤ 2e-06`, 0 over | same | 2e-6 abs + 1e-6 rel |
| `Exp` | relative ≤ 6e-06, 0 over | same | 1e-6 abs + 2e-5 rel |
| `Log` | relative ≤ 5e-07, 0 over | same | 1e-6 abs + 1e-6 rel |
| `Sqrt` | relative ≤ 1.2e-07, 0 over | same | 1e-6 abs + 1e-6 rel |

`n = 1000` is padded to 1024 in engine scratch, so its numbers match `n = 1024`
apart from the generator seed; the point of the case is that the tail path runs.

| shape | expected |
|---|---|
| `MatMul` 32×32×32 | 0 over 1e-4 abs + 1e-4 rel; `max|d|` in the 1e-05 range (32 truncating FPU adds over partial sums of magnitude ~6) |
| `MatMul` 30×34×26 | 0 over the same bound; exercises M→8, N→64 and K→16 padding |
| `BiasAdd` `[1][1024]` | `max|d| ≤ 4.8e-07`, 0 over 1e-6 abs + 1e-6 rel (single add) |
| `BiasAdd` `[7][1000]` | same bound; exercises the row broadcast plus the 8-float tail copy |
| `matmul → bias → relu` `8×32 · 32×16` | 0 over 1e-4 abs + 1e-4 rel, and the three tensors report `/device:VCIV:0` |

The first launch of each kernel JITs its blob with the kgen the engine was
built with (`/usr/lib/vc4infer/kgen` for the package, `infer/tools/kgen.py`
for this tree) into `~/.cache/vc4infer/kcache` (`ew_0_q4.bin`,
`ewb_sub_q4.bin`, `ewb_max_q4.bin`, `uop_exp_v16.bin`, …). The `ewb
sub/max/min` bodies generate to 3320 bytes (415 64-bit instructions) and `ewb
div` to 13656 bytes (1707 instructions — a three-step Newton reciprocal of
`|b|` with the sign put back, then the multiply), both inside the
2048-instruction code slot; `kgen.py ewb <out> --op max` produces them
without touching the GPU.

Launch counts for the pass: 16 kernels per length, of which 12 take one
launch and four take two (40 launches over the two lengths), 2 per MatMul
shape, 1 and 5 for the two BiasAdd shapes, 5 for the graph and 2 for the two
`--add` lengths — 57 in total. The test reads `vc4_tf_stats()` around every
op and reports the delta next to the accuracy, so each line says
`launches=1 (expected 1)` or `launches=2 (expected 2)` and a mismatch fails
the check; the closing line prints the run's total launches, pack and kernel
milliseconds and FLOPs.

The four two-launch ops are the bodies `kops.HEAVY` generates at 4 vectors per
group — 64 floats per QPU instead of 256 — because their 16-vector copies
measure 2945 (`Sqrt`), 3537 (`Tanh`) and 3641 (`Log`) instructions against a
2048-instruction code slot. At that group size 12 QPUs cover 768 floats per
launch, so 1024 elements is 16 groups and runs as 12 + 4; 1000 pads to the same
1024. `Exp` would fit at 16 vectors (1815 words), but the engine's unary table
keeps it on the same verified 4-vector path as the other three. This is
group-size arithmetic, not a numerical defect: the four ops were 0 over
tolerance on the same run in which the count was wrong, and only the
expectation was corrected.

Two process-level numbers are worth recording with the result: the engine
prints `tf-vc4: engine up, arena <N> MiB, <M> MiB for tensors` once, and the
allocator printout in the registration stage shows the tensor share the run
used.

## What the next op costs

The kernel plumbing is now a template: register the op name, get the inputs,
allocate the output, translate pointers, call the engine under the mutex.

| next op | work | cost |
|---|---|---|
| broadcast `AddV2`/`Mul`/`Sub`/`Div` | the engine's add is same-shape; a broadcast needs an expand (`vc4_copy2d` with a zero row stride, as `BiasAdd` does, or `vc4_gather`) or a broadcast-capable kernel | ~1 day, plus a per-op scratch budget |
| `Const`, `Identity`, `Reshape`, `Cast` | no arithmetic; `Const` is already handled by eager's `_EagerConst` copy, `Identity`/`Reshape` are layout aliases | hours, but every graph needs them |
| batched `MatMul` (`BatchMatMulV2`) | N GEMMs; TensorFlow's rank-2 `MatMul` is registered, the batched form is not | a loop over `vc4_matmul` plus a stride plan |
| `Softmax`, `Mean`, `ReduceSum` | the engine has `vc4_softmax`, `vc4_reduce`; the kernels need axis handling | 1–2 days |
| autodiff | TensorFlow records gradients on the host, so each forward op needs its backward kernel (`ReluGrad`, `SoftmaxCrossEntropyWithLogits`, …). The engine has the backward entry points (`vc4_bwop_run`, `vc4_softmax_ce_bwd`, `vc4_batchnorm_bwd`) | weeks; see `docs/TRAINING.md` |

The honest summary: a small real graph — matmul, bias, relu — now runs
entirely on `/device:VCIV:0`, with every value produced by the QPUs and every
refusal named. What remains expensive is the kernel table, one entry per op
TensorFlow's eager executor or gradient functions use.
