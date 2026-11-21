# TensorFlow PluggableDevice for the VideoCore IV QPUs

`tf/build/libtf_vc4_plugin.so` is a TensorFlow PluggableDevice: a
StreamExecutor platform that registers one `VC` device whose memory is the
`libvc4infer` dma-buf arena, plus an op kernel that runs on the QPUs. It is
the in-graph TensorFlow path from `docs/FRAMEWORKS.md` row 8.

State of this slice (measured on the board, 2026-11-21):

| piece | state |
|---|---|
| `SE_InitPlugin`: platform, device, device fns, stream executor | **registers** — `/physical_device:VC:0`, `/device:VC:0` |
| allocator over the engine arena, H2D/D2H/D2D copies | **works** (round-trip exact on the engine arena; needs 8 MiB of free daemon arena) |
| `TF_InitKernel`: `Add` / `AddV2`, fp32 | **registers**; eager dispatches `AddV2` to it on `/device:VC:0` |
| arithmetic on the QPU | **wired, not yet measured**: the serialized silicon pass below is the only step left |
| broadcast, `Mul`, `Const`, `Identity`, … | not registered — see "What the next op costs" |

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

## The device type is `VC`, not `VC4`

TensorFlow 2.21 validates a StreamExecutor platform name and type with
`tensorflow::device_utils::ValidateDeviceType`, whose regex is
`[A-Z][A-Z_]*` — no digits. A platform named `VC4` aborts the load:

```
F ... Check failed: (...) Device name/type 'VC4' must match [A-Z][A-Z_]*.
```

`device_type` is taken from `SP_Platform::type`, and TensorFlow names the
device `/device:<type>:0`, so with a stock wheel the device can only be
`/device:VC:0`. The build takes the name as a variable for a TensorFlow whose
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
suballocates tensors inside one or two of those blocks.

`VC4_TF_ARENA_MB` (default 64, clamp 8–96) sets the arena request, and
`VC4D_SOCKET` picks the daemon (default `/run/vc4d.sock`). A TensorFlow
process is normally unprivileged, so the plugin defaults the engine to the
`vc4d` backend: if `VC4INFER_DAEMON` is unset it is set to `1` before the
engine starts. `VC4INFER_DAEMON=0` still selects the direct root `/dev/mem`
backend.

`VC4_TF_ARENA=host` replaces the engine arena with 64 B aligned `malloc`
blocks for the copy path only; it exists so the plugin wiring can be checked
when no daemon arena is available. The Add kernel refuses to run in that
mode.

## The op

`Add` and `AddV2`, `T = float32`, same-shape inputs only:

```
out = vc4_elementwise(ctx, VC4_OP_ADD, a, b, out, n)
```

The QPU elementwise kernel does exact IEEE fp32 additions, 256 elements per
group over all QPUs, and pads a tail to the group size in engine scratch, so
`n` does not have to be a multiple of 256. Broadcasting is refused
(`INVALID_ARGUMENT`, the element counts differ), as is any dtype other than
float32 (the kernel builder's `TF_KernelBuilder_TypeConstraint` keeps those out
of the registry in the first place).

The kernel gets the two tensors through `TF_GetInput`, allocates the output
with `TF_AllocateOutput`, and turns `TF_TensorData` pointers into arena
offsets by finding the containing allocator block. No tensor data crosses the
bus twice: the arena mapping *is* the device memory.

Everything engine-side runs under one mutex: a `vc4_ctx` is not thread safe,
and the plugin is the only engine user in the process.

## Build and CPU-side checks

```
cd ~/work/vc4-tf
make -C tf PYTHON=$HOME/scratch/tfvenv/bin/python
TF_CPP_MIN_LOG_LEVEL=1 $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py
```

Expected (no GPU work, no device memory):

```
physical devices after load:  ['/physical_device:CPU:0', '/physical_device:VC:0']
logical devices after load:   ['/device:CPU:0', '/device:VC:0']
PASS  platform registers /physical_device:VC:0
PASS  device /device:VC:0 is visible
PASS  kernel Add registered on VC: dtypes=[1] (registry has ['CPU', 'DEFAULT', 'VC'])
PASS  kernel AddV2 registered on VC: dtypes=[1] (registry has ['CPU', 'DEFAULT', 'VC'])
plugin: arena 64 MiB, 61 MiB of data heap, 45 MiB for tensors, 0 MiB in use, engine arena; engine not started

all checks passed
```

Copy path and eager dispatch, without a QPU launch (host arena):

```
VC4_TF_ARENA=host $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy --dispatch
```

```
PASS  constant landed on /device:VC:0
PASS  H2D/D2H round-trip is exact: max|d|=0
PASS  Add dispatched to the VC kernel: ... tf-vc4 Add needs the engine arena
```

The same copy stage against the engine arena, no host mode and no QPU
launch, with an 8 MiB arena on the shared daemon:

```
VC4_TF_ARENA_MB=8 $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy
```

```
plugin: arena 8 MiB, 5 MiB of data heap, 3 MiB for tensors, 0 MiB in use, engine arena; engine not started
PASS  constant landed on /device:VC:0
PASS  H2D/D2H round-trip is exact: max|d|=0
```

`tf-vc4: engine up, arena 8 MiB, 3 MiB for tensors, device type VC` is
printed on stderr the first time device memory is touched, which is when the
allocator maps the real dma-buf.

That last line is the point: eager placed `AddV2` on `/device:VC:0` and called
this plugin's kernel. The refusal is the plugin refusing to compute without
the engine.

## The serialized silicon pass

Run only when the board is otherwise idle (the engine serializes launches
through `vc4d`, but the measured numbers need an idle GPU), and with the
engine arena, not the host one:

```
cd ~/work/vc4-tf
make -C tf PYTHON=$HOME/scratch/tfvenv/bin/python
VC4_TF_ARENA_MB=8 VC4_TF_ALLOW_QPU=1 \
  $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy --add
```

Needs a reachable `vc4d` socket with 8 MiB of free arena. If the shared
daemon's arena is exhausted, give the run its own daemon (it listens on
`/run/vc4d.sock` by default; the engine follows `VC4D_SOCKET`):

```
sudo /usr/sbin/vc4d --socket /run/vc4d-tf.sock --arena-mb 32 \
    --group vc4 --drop-user raspi --max-timeout-ms 20000 &
VC4D_SOCKET=/run/vc4d-tf.sock VC4_TF_ARENA_MB=8 VC4_TF_ALLOW_QPU=1 \
  $HOME/scratch/tfvenv/bin/python tf/tests/test_plugin_cpu.py --copy --add
```

Expected:

| check | expected |
|---|---|
| constant round-trip, 1024 floats | `max|d| = 0` |
| `tf.add` n=1024 (4 groups, `ew_0_q4` blob already cached) | `max|d| = 0` — exact IEEE fp32 |
| `tf.add` n=1000 (padded to 1024 in engine scratch) | `max|d| = 0` |
| first launch | uses `~/.cache/vc4infer/kcache/ew_0_q4.bin`; if absent, the engine JITs it via `/usr/lib/vc4infer/kgen` |

Two process-level numbers are worth recording with the result: the engine
prints `tf-vc4: engine up, arena <N> MiB, <M> MiB for tensors` once, and
`vc4_get_stats()` shows one launch per 12 QPUs' worth of groups (a 1024-element
add is one launch of 4 QPUs).

## What the next op costs

The kernel plumbing is now a template: register the op name, get the inputs,
allocate the output, translate pointers, call the engine under the mutex.

| next op | work | cost |
|---|---|---|
| `Mul`, `Sub`, `Div` | engine already has the ops (`VC4_OP_MUL`, `VC4_OP_SUB`, `VC4_OP_DIV`); one op selector in the kernel and one registration line each | ~30 lines, an afternoon with the silicon pass |
| `Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log` | unary engine entries (`vc4_relu`, `vc4_sigmoid`, `vc4_unary`); the kernel must take one input and no second tensor | ~60 lines for the family |
| `MatMul` / `Dense` | `vc4_matmul()` needs M, N, K and a layout; the kernel must read shapes from the op context, check alignment (M, N, K are rounded to 16 by the engine), and choose the arena blocks. Plus a gradient-side `BatchedMatMul` decision | 1–2 days |
| broadcast `AddV2` | the engine's add is same-shape; a broadcast needs an expand (gather/copy2d) or a separate broadcast-capable kernel | 1 day, and a per-op scratch budget |
| `Const`, `Identity`, `Reshape`, `Cast` | no arithmetic; `Const` is already handled by eager's `_EagerConst` copy, `Identity`/`Reshape` are layout aliases | hours, but every graph needs them |
| autodiff | TensorFlow records gradients on the host, so each forward op needs its backward kernel (`ReluGrad`, `SoftmaxCrossEntropyWithLogits`, …). The engine has the backward entry points (`vc4_bwop_run`, `vc4_softmax_ce_bwd`, `vc4_batchnorm_bwd`) | weeks; see `docs/TRAINING.md` |

The honest summary: one arithmetic op now runs end to end through the
StreamExecutor API, and the expensive part of TensorFlow support is no longer
the platform — it is the kernel table, one entry per op TensorFlow's eager
executor or gradient functions use.
