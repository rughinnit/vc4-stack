# Models beyond YOLO: what the engine runs, and what it refuses

YOLOv8n is the graph this stack is usually measured on, and the README has
implied that it is the supported model.  This page tests that implication
directly: fifteen graphs, run on the same board, through the same engine and
through ONNX Runtime on the four A53 cores, with the same input.

**Result.**  Three graphs run on the QPUs: the three shipped small models,
`modelA`, `modelB` and `modelC`, at 29.8, 17.2 and 6.2 ms.  No ImageNet-scale
backbone completes, and none of the three ONNX Model Zoo graphs completes.
The refusals are not one missing operator: they come from the arena, the conv
kernel set, the way the test corpus stores its weights, grouped-conv alignment,
one runtime defect in a 1000-class 1x1 conv, a dense conv with `Cin=1`, an
unimplemented `Upsample`, and int8.  Each is named below with the exact message.

## Conditions

Board: Raspberry Pi 3B, BCM2837, VideoCore IV, 12 QPUs, 905 MiB RAM, kernel
6.18.50, stock clocks (core 400 MHz).  Engine **vc4infer 1.2.8**
(`libvc4infer.so.1`), managed backend through `vc4d` (`VC4INFER_DAEMON=1`),
strict mode, client arena `VC4INFER_ARENA_MB=96` (the daemon's configured
96 MiB), one client at a time.  Reference: ONNX Runtime **1.32.0**,
`CPUExecutionProvider`, `intra_op_num_threads=4`, `inter_op_num_threads=1`,
graph optimisation `ORT_ENABLE_ALL`.

Protocol, the same as [`BENCHMARKS.md`](BENCHMARKS.md): three warm-ups and five
timed runs per backend, identical float32 feeds on both (one `numpy`
`default_rng(1234)` stream, normal deviates, one array per graph input), wall
time around one `run` call, QPU time from the engine's own kernel counters.
Every run is wrapped in `tools/thermal-guard.sh --max 71`.  The runs started at
59.1-64.5 C and peaked at 60.1-71.4 C; every guard line reported core 400 MHz.
The `throttled=0x20000` flag was already set when the sweep began (another
workload had driven the SoC to 81 C earlier in the day) and the core clock did
not move during any row reported here.  The arena is reported as 93.0 MiB by
the engine: 96 MiB requested, 93.0 usable after the daemon's own reservation.

Three cells did not fit the thermal envelope and carry no number: ResNet50 and
VGG19 on the CPU, and FCN at 224x224 on the CPU.  The guard stopped the FCN
attempt twice at 71 C; those runs are discarded, not averaged.  ShuffleNet and
ZFNet have a spread of up to 35 % across their five CPU runs; the others are
within 10 %.

Unless stated otherwise, CPU is the mean of five, VC4 is the mean of five,
`ratio` is VC4/CPU, `max abs` is the largest elementwise difference over every
graph output, and `scaled` is `max |VC4-CPU| / (1 + |CPU|)`.

## The classification corpus is topology-only

The nine ImageNet graphs come from the `light` data directory of the installed
`onnx` package (1.23.2):

```
.../site-packages/onnx/backend/test/data/light/light_*.onnx
```

They are the ONNX backend test corpus, not trained checkpoints.  Every weight
and bias is produced at runtime by a `ConstantOfShape` node whose `value`
attribute is `0.02`; the files are 3-209 KiB, while the tensors they describe
are the real AlexNet, DenseNet, Inception, ResNet, ShuffleNet, SqueezeNet, VGG
and ZFNet shapes.  They are useful for coverage and timing, not for accuracy.

The engine refuses the five small graphs as published, before any node runs,
because its static plan has no dispatch for `ConstantOfShape` (it is a metadata
node, but strict mode rejects any node without an implementation).  The four
large graphs stop earlier, at the arena floor.  To measure the rest of each
graph I materialised every `ConstantOfShape` into a float32 initialiser of the
same shape and value (`benchmodels/materialise.py`).  The replacement is exact:
on SqueezeNet, ONNX Runtime gives bit-identical output before and after
(`max abs 0.0`).  **No number below is claimed for the unmodified graph**; the
unmodified refusal is recorded separately.

## Classification

`runs` means measured on the engine.  Every other row is CPU-only; the engine
column names its first refusal.  "Materialised" means the constant-`0.02`
weight replacement above; the three shipped models are measured as shipped,
and the four largest corpus graphs are probed as published, where the arena is
reached before the static plan and materialisation does not change the
estimate.

| model | input | params | engine | CPU ms | VC4 ms | ratio | max abs / scaled |
|---|---|---|---:|---|---:|---:|---:|---|
| **modelA** (shipped) | 3x32x32 | 87.0 k | **runs** | 2.4 | **29.8** | 12.4x | 6.9e-06 / 6.1e-06 |
| **modelB** (shipped) | 16x32x32 | 6.0 k | **runs** | 1.8 | **17.2** | 9.7x | 3.8e-06 / 2.1e-06 |
| **modelC** (shipped) | 16x8x8 | 2.4 k | **runs** | 0.2 | **6.2** | 30x | 2.6e-07 / 2.5e-07 |
| SqueezeNet (materialised) | 3x224x224 | 1.24 M | plans 66/66; run stops at conv10 | 100.4 | — | — | — |
| ShuffleNet (materialised) | 3x224x224 | 1.42 M | refused: grouped conv | 65.5 | — | — | — |
| Inception v1 (materialised) | 3x224x224 | 7.0 M | refused: arena (~104 MiB) | 336.7 | — | — | — |
| Inception v2 (materialised) | 3x224x224 | 11.2 M | refused: arena (~135 MiB) | 440.7 | — | — | — |
| DenseNet-121 (materialised) | 3x224x224 | 8.15 M | refused: arena (~115 MiB) | 877.5 | — | — | — |
| ResNet-50 (as published) | 3x224x224 | 25.6 M | refused: arena (~150 MiB) | not measured | — | — | — |
| AlexNet (as published) | 3x224x224 | 61.0 M | refused: arena (~196 MiB) | 295.0 | — | — | — |
| ZFNet-512 (as published) | 3x224x224 | 87.3 M | refused: arena (~362 MiB) | 594.1 | — | — | — |
| VGG-19 (as published) | 3x224x224 | 143.7 M | refused: arena (~603 MiB) | not measured | — | — | — |

The three shipped models are the only graphs in this table that execute on the
QPUs.  Their launches and QPU busy time: modelA 43 launches / 12.6 ms,
modelB 19 / 5.2 ms, modelC 9 / 0.8 ms.  The error is against ONNX Runtime on
the same random input, not against a labelled image; `modelA` is a 10-class
softmax, `modelB` a 16-channel feature map and `modelC` a 4-class softmax.

In summary: **three measured on the engine** (`modelA`, `modelB`, `modelC`),
**nine CPU-only** and refused (seven by the arena, ShuffleNet by grouped-conv
alignment, SqueezeNet at its first run).  No corpus graph that was refused has
a VC4 number, and the dashes in the VC4 and error columns are not zeros.

Two rows hide a second blocker behind the arena.  Re-running the plan with the
conv-scratch reservation lowered (`VC4_POOL_RESERVE_MB=32` for Inception v1,
`=24` for DenseNet-121) gets past the arena and stops at the first conv:

```
Inception v1: node 'n0' (Conv): kernel 7x7 (NCHW fallback: 1x1, 3x3)
DenseNet-121: node 'n0' (Conv): kernel 7x7 (NCHW fallback: 1x1, 3x3)
```

That is the same 7x7 stem that ResNet-50 and ZFNet use; AlexNet's is 11x11.
Those three cannot be probed this way because their weight floor alone (100,
159 and 313 MiB) exceeds what the lowered reservation leaves.

SqueezeNet is the most informative refusal.  It plans completely: 66 nodes,
66 dispatches, every node on `vc4`, no host debt.  The first run then fails at
the last conv:

```
node 'n62' (Conv): engine error: copy2d_x: invalid argument ()
```

`n62` is `conv10`: 1x1, `Cin=512`, `Cout=1000`, input 13x13.  Its output is the
only 1000-channel conv in the corpus (the other models end in a `Gemm`), and
1000 is not a multiple of 16.  The same node fails with the input at 160x160
(10x10), so the 13x13 extent is not the trigger.  I did not prove that the
channel alignment is the cause; the next step is to reproduce it with a single
1x1 `512->1000` conv and then with `Cout=1008`.

## Detection

Measured earlier, same board and protocol, from the 45-export sweep in
[`BENCHMARKS.md`](BENCHMARKS.md) section 14; not re-run for this page.

| model | input | params | engine | CPU ms | VC4 ms | ratio | max scaled |
|---|---|---|---:|---|---:|---:|---:|---:|
| YOLOv8n | 3x160x160 | 3.2 M | **runs**, 802 launches | 108.2 | **521.0** | 4.8x | 3.1e-04 |
| YOLOv8n | 3x320x320 | 3.2 M | refused at a 96 MiB arena (needs ~150) | — | — | — | — |
| YOLOv8n | 3x640x640 | 3.2 M | no fit at 150 MiB either: 91 MiB working set plus conv pitch | — | — | — | — |

YOLOv8n at 160 is 0/44,100 elements over the 5e-4 tolerance against the CPU,
max abs 1.5e-2, which is the logit range; the scaled figure is the meaningful
one.  Detection coverage beyond YOLO is untouched by this page: no non-YOLO
detector was measured.

## Other tasks

All three are from the ONNX Model Zoo, downloaded through the LFS media host
(`benchmodels/fetch_models.sh` records URLs, sizes and SHA-256).  All three are
CPU-only here: none reaches a VC4 kernel.

| model | task | input | params | engine | CPU ms | VC4 ms |
|---|---|---|---:|---|---:|---:|
| `super-resolution-10` | 4x super-resolution (sub-pixel CNN) | 1x224x224 -> 1x672x672 | 59.7 k | refused: arena (~147 MiB), then `Conv Cin=1` | 708.9 | — |
| `candy-8` | style transfer (fast neural style) | 3x224x224 -> 3x224x224 | 1.68 M | refused: arena (~540 MiB), then 9x9 `Conv`, `Upsample` | 1959 | — |
| `fcn-resnet50-12-int8` | 21-class segmentation (FCN, ResNet-50) | 3x224x224 -> 21x224x224 | 35.3 M (int8) | refused: `QLinearConv` | not measured | — |

The super-resolution graph's arena refusal is not the weight footprint: its
weights are 233 KiB.  It is the conv scratch for the last 3x3 layer at 672x672
with 64 input channels, which the planner estimates at **126 MiB**.  With the
batch fixed to 1 the planner stops there; with the published symbolic batch it
skips the scratch estimate and the static plan stops at the first conv:

```
node 'Conv_0' (Conv): depthwise Cout 64 != Cin 1 (channel multiplier)
```

`Conv_0` is a dense 1->64, 5x5 convolution.  The engine classifies any conv
whose `group` equals its input channel count as depthwise, and with `Cin=1` a
dense conv has `group == Cin == 1`, so it is refused as a depthwise conv with
a channel multiplier.  The distinction between `group==1` and depthwise is
lost for single-channel inputs.

`candy-8` fails at the arena first because the planner reserves **506 MiB** of
scratch for a 9x9 convolution over 224x224; the operator-level blockers behind
that are the 9x9 kernel (the NHWC path takes 1x1, 3x3 and 5x5) and the
opset-8 `Upsample` node (`no implementation`).

`fcn-resnet50-12-int8` is the only segmentation graph in the zoo that is small
enough to download and load.  It is int8 (`QLinearConv`, `QLinearAdd`,
`QuantizeLinear`/`DequantizeLinear`); the first quantised conv is refused
because the arena holds fp32 activations only.  Its CPU row is missing because
the 224x224 on ORT exceeded the thermal envelope twice; the 128x128 variant is
not a valid substitute (ORT rejects it at a `QLinearAdd` shape), so I did not
report a number from a different shape.

## Exact refusals, in the engine's words

Board default: 96 MiB client arena, no reserve override.

| model | stage | message |
|---|---|---|
| the five small `light` graphs, as published | plan | `node 'ConstantOfShape_0' (ConstantOfShape): no implementation` |
| SqueezeNet (materialised) | first run | `node 'n62' (Conv): engine error: copy2d_x: invalid argument ()` |
| ShuffleNet (materialised) | plan | `node 'n4' (Conv): group 4 needs 16-aligned per-group blocks` |
| Inception v1 (materialised) | open | `graph pool needs at least 54.1 MiB ... peak is about 91.1 MiB ... arena is 93.0 MiB with 41.8 MiB reserved ... about 103.9 MiB would do` |
| Inception v2 (materialised) | open | `needs at least 85.4 MiB ... peak about 167.4 MiB ... about 135.2 MiB would do` |
| DenseNet-121 (materialised) | open | `needs at least 65.2 MiB ... peak about 376.0 MiB ... about 114.9 MiB would do` |
| ResNet-50 (as published) | open | `needs at least 100.3 MiB ... peak about 348.0 MiB ... about 150.0 MiB would do` |
| AlexNet (as published) | open | `needs at least 158.9 MiB ... peak about 259.7 MiB ... about 196.0 MiB would do` |
| ZFNet-512 (as published) | open | `needs at least 313.2 MiB ... peak about 385.2 MiB ... about 361.8 MiB would do` |
| VGG-19 (as published) | open | `needs at least 468.7 MiB ... peak about 755.3 MiB ... about 602.8 MiB would do` |
| `super-resolution-10` (batch 1) | open | `needs at least 12.8 MiB ... peak about 70.0 MiB ... 126.0 MiB reserved for conv/elementwise scratch ... about 146.8 MiB would do` |
| `super-resolution-10` (symbolic batch) | plan | `node 'Conv_0' (Conv): depthwise Cout 64 != Cin 1 (channel multiplier)` |
| `candy-8` | open | `needs at least 25.6 MiB ... peak about 195.1 MiB ... 506.0 MiB reserved for conv/elementwise scratch ... about 539.6 MiB would do` |
| `candy-8` (operator view) | plan | `node ... (Conv): kernel 9x9 (NCHW fallback: 1x1, 3x3)` and two `Upsample` nodes: `no implementation` |
| `fcn-resnet50-12-int8` | plan | `node 'fused Conv_6_quant' (QLinearConv): no implementation` |

The arena messages are the planner's floor, peak and scratch numbers; the
full untruncated strings are in `benchmodels/results/`.  The `open` stage means
the constructor refuses before a plan exists:
the arena check precedes the static plan, so for the large corpus graphs the
arena is literally the first refusal, and `ConstantOfShape` is the second.

## What a reader can conclude

* **Non-YOLO support is real but narrow.**  A small conv/BN/pool/Gemm model
  with ordinary shapes runs end to end on the QPUs, with the CPU
  parity shown above (worst max abs 6.9e-06 on modelA).  It is 10-30x slower
  than four A53 cores.
* **The 152/152 operator figure is an operator result, not a model result.**
  Every operator in the 152-node sweep exists; whether a model runs is decided
  by tensor sizes and by the shapes the kernels accept.  A 7x7 stem is a
  supported `Conv` with an unsupported kernel.
* **No standard ImageNet backbone in this corpus runs**, for four independent
  reasons: 96 MiB arena; conv kernels restricted to 1x1/3x3/5x5; grouped-conv
  16-alignment; and, for the corpus as published, `ConstantOfShape`.  Fixing
  any one of them admits no model on its own - ResNet-50 needs both a bigger
  arena and a 7x7 kernel.
* **The zoo's small non-classification models do not run either.**  The
  super-resolution model needs 147 MiB by the planner and hits a `Cin=1` conv
  rule; style transfer needs 540 MiB of 9x9-conv scratch; the small segmentation
  model is int8.
* **Detection is YOLOv8n at 160x160 on this board.**  320 does not fit 96 MiB
  and 640 does not fit at all; no other detector was measured.

## Next steps, in the order the measurements argue for

1. **Arena.**  Seven of nine corpus graphs and two of three zoo graphs stop
   here.  The measured needs are 104-603 MiB against 96.  A larger `vc4d
   --arena-mb` is a configuration change, not a kernel; it would admit
   ResNet-50 (150) and AlexNet (196) if the CMA allows it.  I did not restart
   the daemon during this sweep.
2. **Conv kernels 7x7 and 11x11.**  Inception v1 and DenseNet-121 are measured
   to stop at a 7x7 stem once the arena is relaxed; ResNet-50 and ZFNet use the
   same 7x7 stem but their weight floor keeps them at the arena; AlexNet's stem
   is 11x11 and style transfer's is 9x9.  This is the gap that survives after
   the arena.
3. **ConstantOfShape.**  Fold it in the optimiser, or let strict mode accept a
   host-metadata node that has no activation arithmetic.  The ONNX test corpus
   stores all weights this way; ordinary exporters usually do not.
4. **Grouped conv alignment.**  ShuffleNet's group-4 conv is refused for
   16-alignment.  Pad the per-group block rather than the channel count.
5. **SqueezeNet's 1x1 512->1000 conv.**  Plan says yes; the first run says
   `copy2d_x: invalid argument`.  Reproduce with one node before changing a
   kernel.
6. **Dense conv with `Cin=1`.**  The depthwise test conflates `group==1` with
   `group==Cin` when `Cin==1`.  That is a dispatch bug, not a missing kernel.
7. **`Upsample` (opset 8).**  No implementation.  `Resize` exists; either lower
   the deprecated op to it or refuse it at export time.
8. **int8.**  `QLinearConv` cannot run in an fp32 arena.  Dequantise at the
   graph boundary, or add an int8 path; do not expect the zoo's quantised
   segmentation model to run before then.

## Reproduce

```
benchmodels/fetch_models.sh          # three zoo graphs, URLs and SHA-256
benchmodels/materialise.py M.onnx O.onnx
benchmodels/plan_probe.py M.onnx out.json
benchmodels/bench_model.py --model M.onnx --task classification \
    --source "..." --size 224 --out out.json
```

`benchmodels/results/` holds the JSON record behind every cell (means, the five
run times, launches, QPU ms, memory, error, and the full refusal string).
`benchmodels/results/params.json` holds the parameter counts.

The zoo files used here:

| file | source | SHA-256 |
|---|---|---|
| `super-resolution-10.onnx` | ONNX Model Zoo, `validated/vision/super_resolution/sub_pixel_cnn_2016` | `85f36ff8...6035e` |
| `candy-8.onnx` | ONNX Model Zoo, `validated/vision/style_transfer/fast_neural_style` | `e3c6f442...22b8` |
| `fcn-resnet50-12-int8.onnx` | ONNX Model Zoo, `validated/vision/object_detection_segmentation/fcn` | `67fcba8e...0331` |

The shipped models are `infer/packaging/vc4infer/data/models/model{A,B,C}.onnx`
(the same files `vc4infer selftest` uses).
