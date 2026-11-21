# litert — a LiteRT external delegate for the QPUs

`libvc4_delegate.so` lets LiteRT run the subgraphs the VC4 engine can express
on the VideoCore IV QPUs and leaves every other node to LiteRT's own kernels.
It is an external delegate: `Interpreter.load_delegate()` loads it, the
delegate partitions the graph, and a claimed node is either executed by
`libvc4infer` or the delegate refuses the partition.  There is no host
fallback inside a claimed partition.

Board: Raspberry Pi 3B (BCM2837, V3D), Debian 13 arm64, CPython 3.13.5,
905 MiB RAM.  Engine: the `libvc4infer` built from this checkout
(`infer/build/libvc4infer.so.1`, 1.2.3).

## 1. API version and the ABI that was chosen

Runtime: **ai-edge-litert 2.3.0** (cp313, manylinux_2_27 aarch64), built from
LiteRT **v2.3.0**.  The wheel ships `libLiteRt.so` plus the classic
`ai_edge_litert.interpreter.Interpreter`, and it does **not** ship any TFLite
headers, so the headers this delegate compiles against are vendored from the
matching tag (see `third_party/litert/README.md`).

The delegate implements the classic external-delegate plugin ABI:

```c
TfLiteDelegate *tflite_plugin_create_delegate(
    const char *const *keys, const char *const *values, size_t num_options,
    void (*report_error)(const char *));
void tflite_plugin_destroy_delegate(TfLiteDelegate *);
```

and inside the runtime uses:

* `TfLiteDelegate::Prepare` to claim nodes and call
  `TfLiteContext::ReplaceNodeSubsetsWithDelegateKernels` with one
  `TfLiteRegistration` whose `init`/`prepare`/`invoke` run a partition;
* `TfLiteDelegateParams` (`nodes_to_replace`, `input_tensors`,
  `output_tensors`) as the partition description.

Why this and not the dispatch API (`LiteRtDispatchDelegateGetHooks`,
`LiteRtSetDelegateFunction`)?  The same `libLiteRt.so` exports both, but:

* the dispatch header (`litert/c/internal/litert_dispatch_delegate.h`) is
  under `internal/` in v2.3.0;
* the dispatch model expects a vendor bytecode executable, tensor-buffer
  requirements and an invocation context per partition — a compiler-style
  contract;
* the external plugin ABI hands the runtime a `TfLiteDelegate` anyway, and
  the classic kernel path already reaches every engine entry point the claim
  table uses, including plan-time weight packing and per-invoke tensor
  upload/download.

The dispatch API becomes the right choice if/when a compiled executable
format for the QPUs exists.  It is not needed to run the current op table.

## 2. Partitioning

`Prepare` walks the execution plan in order and asks `vc4_plan_node()` about
each node.  Maximal runs of claimed nodes become delegate kernel nodes; a run
never spans an unclaimed node, so LiteRT still sees its own ops in their
original order.  `Prepare` runs in delegate context, where
`GetNodeAndRegistration` is legal; the accepted per-node plans are cached by
node index and looked up from the kernel hooks, where the runtime forbids that
call (`tflite/core/subgraph.cc: SwitchToKernelContext`).

Before replacing anything, the delegate estimates the persistent and scratch
arena the partition set would need.  If the estimate exceeds
`arena_mb - 1 MiB`, every node is declined and the whole graph stays with
LiteRT; a plan that would abort halfway through `Invoke` is never installed.
The estimate is conservative (it does not share buffers between tensors) and
the README's limitations section states the consequences.

Modes (the `mode` option):

| mode | claims | replaces | touches the QPU | what it is for |
|---|---|---|---|---|
| `auto` (default) | yes | yes | at `allocate_tensors` and `invoke` | normal use |
| `plan` | yes (reported) | no | never | inspect the claim table on a real model |
| `dry` | yes | yes | never | check replacement + kernel prepare; `invoke` fails loudly |
| `off` | no | no | never | bit-identical control run |

`dry` exists because a claimed node must not silently fall back: in dry mode
the kernel really does prepare (with host shadows) and `Invoke` returns
`kTfLiteError` rather than computing on the ARM.

## 3. Claimed ops

A node is claimed only when all of the following hold: every tensor is fp32
with `quantization.type == kTfLiteNoQuantization`, every extent is a positive
constant, and `bytes == 4 * numel` (dense row-major).  Anything else is
refused by name.

| TFLite op | engine call | constraints |
|---|---|---|
| `CONV_2D` | `vc4_conv_nhwc` | NHWC fp32; `Cin % 16 == 0`, `Cout % 16 == 0`; `KH, KW` in {1,3,5} (mixed allowed when one padding value fits both axes); dilation 1; stride 1 or 2; VALID, or SAME with even total padding (SAME/stride 1 always; SAME/stride 2 only on odd extents); `stride * Cin * 4 <= 8191`; filter and bias read-only; fused activation NONE/RELU/RELU6 |
| `DEPTHWISE_CONV_2D` | `vc4_depthwise_nhwc` | NHWC fp32; 3x3, SAME, stride 1, dilation 1, `depth_multiplier == 1`, `C % 16 == 0`; bias absent or all zero (the engine's depthwise panel has no bias slot); fused activation NONE/RELU/RELU6 |
| `FULLY_CONNECTED` | `vc4_conv_nhwc` as a 1x1 conv | rank-2 input `[M,K]`, weights `[N,K]`, `K % 16 == 0`, `N % 16 == 0`, `keep_num_dims == false`, bias optional, fused activation NONE/RELU/RELU6 |
| `ADD`, `MUL`, `SUB`, `DIV` | `vc4_elementwise` | both inputs exactly the output shape (no broadcasting); no fused activation |
| `RELU`, `RELU6`, `RELU_N1_TO_1`, `LOGISTIC`, `TANH` | `vc4_relu` / `vc4_unary` / `vc4_sigmoid` | same shape in/out |
| `SOFTMAX` | `vc4_softmax` | `beta == 1`; last axis `<= 16` classes and `rows % 8 == 0` (the stable fused row kernel; the generic path does not subtract the row maximum) |

The table is exercised by `tests/test_partition.c` (37 checks) against
synthetic `TfLiteTensor`/`TfLiteNode` structures: every accepted case plus the
refusals that matter (Cin=3, asymmetric SAME, dilation, int8, broadcast,
non-constant filter, dynamic extent, non-dense bytes, softmax widths).

Deliberately not claimed: int8/uint8 and any quantized tensor; fp16/bf16;
max/avg pool (the NHWC pool path is stride 1 only and the forward average-pool
entry point is not in the C API); reshape/concat/split/slice (layout-only ops
LiteRT already does cheaply, and claiming them would add a copy between
buffers); broadcasting elementwise; fused activations on elementwise ops;
dynamic shapes.

## 4. Tensor plumbing

* Every tensor used by a partition gets its own persistent arena buffer of
  exactly `TfLiteTensor::bytes`.  Nothing aliases, so an arena offset is
  never shared between two live values; the cost is footprint, not
  correctness.  A tensor whose `bytes` is not `4 * numel` is refused rather
  than interpreted as if it were dense.
* Only partition boundary tensors cross to the ARM: `params->input_tensors`
  are uploaded before the first step, `params->output_tensors` are
  downloaded after the last.  Interior tensors stay in the arena for the
  whole partition.  Constants (`kTfLiteMmapRo`) are uploaded once at
  `prepare`, not per invoke.
* The engine's NHWC conv writes `align16(wo)` pixels per row with junk in the
  tail.  Each conv output is compacted into the dense TFLite tensor with
  `vc4_copy2d_n` on the QPUs (one 16-word vector per DMA store); this is a
  layout copy, not an arithmetic fallback.  The fused activation, when
  present, runs from the compacted tensor into a step-owned arena temporary.
* TFLite conv filters are `[Cout][KH][KW][Cin]`; the engine packer reads
  `[Cout][Cin][KH][KW]`.  Constants are transposed once at prepare time on
  the host (a layout transform of weights, never of activations), then
  packed into the k-major panel by the QPU packer.
* `FULLY_CONNECTED` is emitted as a 1x1 conv over `[M][1][1][K]`: the weight
  layout already matches, the bias goes through the conv panel, and the
  output bytes are the required `[M][N]`.

## 5. Options

```python
from ai_edge_litert.interpreter import Interpreter, load_delegate

d = load_delegate("litert/build/libvc4_delegate.so",
                  {"mode": "auto", "arena_mb": "64", "verbose": "1"})
i = Interpreter(model_path="chain.tflite", experimental_delegates=[d])
i.allocate_tensors()
i.invoke()
```

| key | default | meaning |
|---|---|---|
| `mode` | `auto` | `auto`, `plan`, `dry` or `off` |
| `arena_mb` | `64` | requested engine arena; the planner budget is this minus 1 MiB |
| `cache_dir` | engine default | kernel blob cache (`$VC4INFER_CACHE` / cwd `kcache`) |
| `verbose` | `0` | one line per node plus the partition summary on stderr |

Unknown options make `load_delegate` raise, as the plugin ABI requires.

## 6. Build and test

The engine must exist in the checkout (the delegate records an rpath to it):

```sh
cd ~/work/vc4-litert
export TMPDIR=$HOME/scratch/tmp
make -C vc4d build/libvc4client.a
make -C infer build/libvc4infer.so build/libvc4infer.so.1
make -C litert
```

`make -C litert test` runs the claim-table unit test and the CPU tiers of
`tests/test_litert.py` with `PYTHON` (default `python3`; the `ai-edge-litert`
wheel lives in a venv, so pass it explicitly):

```sh
V=~/scratch/tmp/fw/litert
make -C litert test PYTHON=$V/bin/python
```

The tests build three models with the FlatBuffers schema shipped in the wheel
(no TensorFlow, no network):

| model | graph | expected claim |
|---|---|---|
| `chain.tflite` | conv3x3+relu → conv1x1 → softmax | 3 nodes, 1 partition |
| `mixed.tflite` | conv3x3+relu → average_pool2x2 → conv1x1 → softmax | 3 nodes, 2 partitions; the pool stays with LiteRT |
| `hostonly.tflite` | average_pool2x2 → sin | 0 nodes; LiteRT executes the whole graph |

What the CPU tiers prove: the plugin ABI loads; `Prepare` runs and the claim
table matches the model; `mode=off` is bit-identical to the reference
interpreter; `mode=dry` really replaces partitions and prepares the kernel
hooks; a claimed node in dry mode refuses instead of computing on the host;
the fully declined graph produces the same bytes as the reference.  The
reference runs use `OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES` so
XNNPACK cannot mask what is being measured.

## 7. Measured in this pass

Everything below was produced with the QPU path **not** executed (another
workload held the board):

| quantity | value |
|---|---|
| shared object | 146,128 B file, 21,210 B text; `NEEDED libvc4infer.so.1` only |
| claim-table unit test | 37 checks, 0 failures |
| end-to-end CPU tiers | 28 checks, 0 failures |
| `chain.tflite` / `mixed.tflite` / `hostonly.tflite` | 11,356 / 11,508 / 532 B |
| claim report | chain 3/0 nodes, 1 partition; mixed 3/1, 2 partitions; hostonly 0/2, 0 partitions |
| LiteRT CPU reference invoke | 0.167 / 0.153 / 0.030 ms (not a delegate number) |

There are **no QPU timings yet**.  The on-device tier exists in
`tests/test_litert.py` behind `VC4_LITERT_GPU=1` and must be run with the board
exclusive to this workload.

## 8. The on-device verification pass

Preparation does QPU work: each claimed conv packs its weights with a QPU
kernel at `allocate_tensors`, and `invoke` launches the op sequence.  Run this
only when no other VC4 client is active (`ps`, `systemctl is-active vc4d`);
`vc4d` serialises launches but a board shared with another workload makes the
numbers meaningless.

```sh
cd ~/work/vc4-litert
export TMPDIR=$HOME/scratch/tmp
V=~/scratch/tmp/fw/litert

make -C vc4d build/libvc4client.a
make -C infer build/libvc4infer.so build/libvc4infer.so.1
make -C litert

# CPU tiers again (fast, no QPU work)
make -C litert test PYTHON=$V/bin/python

# On-device tier: claimed nodes run on the QPUs, outputs compared with the
# CPU reference at rtol=1e-3, atol=1e-4.
VC4_LITERT_GPU=1 \
VC4_DELEGATE_SO=$PWD/litert/build/libvc4_delegate.so \
$V/bin/python litert/tests/test_litert.py
```

One model by hand:

```sh
$V/bin/python litert/tests/_run_one.py litert/tests/models/mixed.tflite \
    --input litert/tests/models/mixed.tflite.input.npy \
    --output /home/raspi/scratch/tmp/mixed.npz \
    --delegate litert/build/libvc4_delegate.so --mode auto --repeat 5
```

The expected first failures, in order, if the silicon does not agree with the
plan: `vc4_init` (daemon/arena), weight packing at `allocate_tensors`,
conv output compaction (`vc4_copy2d_n` geometry), then numeric tolerance on
the conv chain.  `verbose=1` prints per-node claims, per-partition node counts
and arena use.

## 9. Limitations

* **fp32 only.**  A quantized tensor refuses the node; the delegate never
  dequantizes to pretend.
* **Static shapes.**  Negative extents, dims signatures and `kTfLiteDynamic`
  tensors are refused.
* **One engine context per process**, shared by all partitions and all
  interpreters.  `vc4_ctx` is not thread-safe and the delegate does not add a
  lock; run one interpreter at a time.
* **Arena planning is conservative.**  Every tensor of a partition has its own
  buffer and the estimate adds one weight-sized transient; a model that would
  fit with aliasing may be declined.  The estimate is checked before any
  replacement, so the outcome is a partition in LiteRT, never a mid-invoke
  out-of-memory.
* **Conv channel alignment.**  `Cin` and `Cout` must be multiples of 16, so an
  RGB first conv (`Cin = 3`) stays on LiteRT and later layers can still be
  claimed.
* **Softmax** is limited to the stable fused kernel: `<= 16` classes and
  `rows % 8 == 0`.
* **Depthwise** is 3x3/SAME/stride 1 with a zero bias and `C % 16 == 0`.
* **No pooling, concat, reshape, slice, broadcast elementwise, or fused
  activation on elementwise ops.**  Those nodes go to LiteRT.
* **Inference only.**  LiteRT has no autodiff to plug into.
* The delegate object is single-interpreter: `Prepare` rebuilds the cached
  plan table, so applying one loaded delegate to a second interpreter is not
  supported (load a delegate per interpreter, as `load_delegate` does).
