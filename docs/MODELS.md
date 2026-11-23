# Models beyond YOLO: what the engine runs, and what it refuses

YOLOv8n is the graph this stack is usually measured on, and the README has
implied that it is the supported model.  This page tests that implication
directly: fifteen graphs, run on the same board, through the same engine and
through ONNX Runtime on the four A53 cores, with the same input.

**Result.**  Four graphs run on the QPUs: the three shipped small models,
`modelA`, `modelB` and `modelC`, at 29.8, 17.2 and 6.2 ms, and SqueezeNet as
published.  No ImageNet-scale backbone completes, and none of the three ONNX
Model Zoo graphs completes.  The refusals are not one missing operator: they
come from the arena, the conv kernel set, grouped-conv alignment, an
unimplemented `Upsample`, and int8.  Each is named below with the exact message.

Three of the refusals recorded in the first version of this page were engine
defects, and are fixed here: weights stored as `ConstantOfShape` are folded at
plan time, a dense conv with `Cin=1` is no longer read as depthwise, and the
output slice of a 1000-channel 1x1 conv uses legal `copy2d_x` stores.  A fourth
defect on the same path, the A-panel packer's QPU count for a 125-row-block
GEMM, and a per-run pool leak that showed up on the fifth run, are fixed so
SqueezeNet reaches its last node and can be measured.  The sections that
changed are marked, and models that still refuse keep their exact message.

## Conditions

Board: Raspberry Pi 3B, BCM2837, VideoCore IV, 12 QPUs, 905 MiB RAM, kernel
6.18.50, stock clocks (core 400 MHz).  Engine **vc4infer 1.2.10**
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
The library is `infer/build/libvc4infer.so.1` built from this tree: the
installed copy predates the autotune entry point and cannot be loaded by the
current binding.  The re-runs of the fixed models and the operator
sweep started at 56.9-60.7 C and peaked at 59.1-70.4 C under the same guard;
the rows carried over from the first sweep keep their original conditions.

The three shipped models were re-run on the rebased tree for parity, not to
replace their times: 29.6 / 16.8 / 6.2 ms, max abs 6.9e-06 / 3.8e-06 / 2.6e-07,
and 43 / 21 / 9 launches.  The epilogue and operator-breadth merges on this
base add two launches to modelB (19 -> 21) and 144 to SqueezeNet; the table's
times for the shipped models are the first sweep's.

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

The engine now folds `ConstantOfShape` at plan time when its shape input is a
constant.  The node computes a tensor, not a per-run activation, so it becomes
a float32 initialiser before the pool is planned; the shape tensors that only
fed it and the IR-3 input declarations that listed them are dropped.  Nothing
is claimed about a graph with a dynamically shaped fill: that still refuses,
and names the shape input.

The fold is exact.  For SqueezeNet, ShuffleNet, Inception v1 and v2 and
DenseNet-121, every folded initialiser is byte-identical to the one
`benchmodels/materialise.py` writes (`test_optimizer.py`), and the reference
output of the folded graph is bit-identical to the node-by-node graph.  The
five small graphs are therefore measured **as published**; `materialise.py`
remains the independent check, not a workaround.

The fold may use at most the client arena, for one tensor and in total.  That
is the same constraint the run would meet: the four large corpus graphs carry
single weight tensors of 144, 288 and 392 MiB (AlexNet, ZFNet-512, VGG-19) or
more than 93 MiB in total (ResNet-50, 84.7 MiB folded before the tensor that
does not fit), so they refuse at the fold with the tensor's shape and byte
count rather than allocating it twice on a 1 GiB board.  Their arena estimates
below are unchanged from the materialised probe.

## Classification

`runs` means measured on the engine.  Every other row is CPU-only; the engine
column names its first refusal.  Every corpus row is the file as published:
its `ConstantOfShape` nodes are folded by the engine, not by a rewritten file.  The arena figures in the refusal column are the materialised probe's
estimate of what the model would need.

| model | input | params | engine | CPU ms | VC4 ms | ratio | max abs / scaled |
|---|---|---|---:|---|---:|---:|---:|---|
| **modelA** (shipped) | 3x32x32 | 87.0 k | **runs** | 2.4 | **29.8** | 12.4x | 6.9e-06 / 6.1e-06 |
| **modelB** (shipped) | 16x32x32 | 6.0 k | **runs** | 1.8 | **17.2** | 9.7x | 3.8e-06 / 2.1e-06 |
| **modelC** (shipped) | 16x8x8 | 2.4 k | **runs** | 0.2 | **6.2** | 30x | 2.6e-07 / 2.5e-07 |
| SqueezeNet (as published) | 3x224x224 | 1.24 M | **runs** | 92.1 | **2892.7** | 31.4x | 0.0 / 0.0 |
| ShuffleNet (as published) | 3x224x224 | 1.42 M | refused: grouped conv | 65.5 | — | — | — |
| Inception v1 (as published) | 3x224x224 | 7.0 M | refused: arena (~104 MiB) | 336.7 | — | — | — |
| Inception v2 (as published) | 3x224x224 | 11.2 M | refused: arena (~135 MiB) | 440.7 | — | — | — |
| DenseNet-121 (as published) | 3x224x224 | 8.15 M | refused: arena (~115 MiB) | 877.5 | — | — | — |
| ResNet-50 (as published) | 3x224x224 | 25.6 M | refused: weights > arena (needs ~150 MiB) | not measured | — | — | — |
| AlexNet (as published) | 3x224x224 | 61.0 M | refused: 144 MiB weight tensor (needs ~196 MiB) | 295.0 | — | — | — |
| ZFNet-512 (as published) | 3x224x224 | 87.3 M | refused: 288 MiB weight tensor (needs ~362 MiB) | 594.1 | — | — | — |
| VGG-19 (as published) | 3x224x224 | 143.7 M | refused: 392 MiB weight tensor (needs ~603 MiB) | not measured | — | — | — |

The four graphs in this table that execute on the QPUs are the three shipped
models and SqueezeNet.  Launches and QPU busy time: modelA 43 / 12.6 ms,
modelB 21 / 5.3 ms, modelC 9 / 0.9 ms, SqueezeNet 7777 / 768.6 ms.  The error
is against ONNX Runtime on the same random input, not against a labelled
image; `modelA` is a 10-class softmax, `modelB` a 16-channel feature map,
`modelC` a 4-class softmax and SqueezeNet a 1000-class softmax.

In summary: **four measured on the engine** (`modelA`, `modelB`, `modelC`,
SqueezeNet), **eight CPU-only** and refused: three by the arena (Inception v1,
Inception v2, DenseNet-121), four by weight bytes larger than it
(ResNet-50's 102 MiB total, and single tensors of 144, 288 and 392 MiB
for AlexNet, ZFNet-512 and VGG-19), and ShuffleNet by grouped-conv
alignment.  No corpus graph that was refused has a VC4 number, and the dashes
in the VC4 and error columns are not zeros.

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

SqueezeNet plans completely: 66 nodes, 66 dispatches, every node on `vc4`,
no host debt.  In the first version of this page the first run then failed at
the last conv:

```
node 'n62' (Conv): engine error: copy2d_x: invalid argument ()
```

`n62` is `conv10`: 1x1, `Cin=512`, `Cout=1000`, input 13x13.  The conv itself
was never the problem: Cout is padded to 1008 for the GEMM, and the real 1000
channels are sliced back with `copy2d_x`.  The slice passed the whole remainder
(1000) as both the lane count and the store depth, and the kernel accepts at
most 16 of each, so the call was refused before it launched.  A single-node
reproduction isolates it (`benchmodels/conv_probe.py`):

| node | launches | wall ms | max abs / scaled |
|---|---:|---:|---:|
| 1x1 512->1000, 13x13 | 36 | 60.7 | 3.2e-05 / 1.4e-05 |
| 1x1 512->1008, 13x13 | 34 | 54.3 | 3.2e-05 / 1.4e-05 |
| 1x1 512->1000, 8x8 | 36 | 35.4 | 2.5e-05 / 1.3e-05 |
| 1x1 512->1008, 8x8 | 34 | 31.2 | 2.5e-05 / 1.3e-05 |

The trigger is the unaligned channel count, not the spatial extent: the 1000
case splits into 62 blocks plus an 8-channel tail at either size, and the 1008
case needs no slice at all (two fewer launches).  The slice is now two calls,
one per part.  Behind it, the 1000-channel `GlobalAveragePool` exposed a second
engine defect: the A-panel packer defaulted to four QPUs and 125 eight-row
units is not divisible by four.  It now picks the largest of 4, 2, 1 that
divides the work.  Measuring the graph then exposed a per-run pool leak of
8.2 MiB (the NCHW pool's uploads, `GlobalAveragePool`'s copies, the padded
output of the 1000-channel conv and each run's graph outputs), which ended the
eighth run with `out of arena memory`; those buffers are now released and nine
consecutive runs hold the pool at 9.8 MiB.  SqueezeNet runs end to end with
the numbers in the table.

## Detection

The YOLOv8n-160 and YOLOv8n-320 rows are from the 45-export sweep in
[`BENCHMARKS.md`](BENCHMARKS.md) section 14.  The YOLOv8s-160 and YOLOv8n-640
rows were re-measured on the fix/hard-caps tree at 147-181 MiB of client arena
and the fixed `rng(1234)` input; their CPU column is the sweep's.

| model | input | params | engine | CPU ms | VC4 ms | ratio | max scaled |
|---|---|---|---:|---|---:|---:|---:|---:|
| YOLOv8n | 3x160x160 | 3.2 M | **runs**, 802 launches | 108.2 | **521.0** | 4.8x | 3.1e-04 |
| YOLOv8n | 3x320x320 | 3.2 M | **runs as an `--opt all` artifact in 60 MiB**, 512 launches; 8.6e-4 scaled | — | 836.7 | — | 8.6e-04 |
| YOLOv8n | 3x640x640 | 3.2 M | **runs** at 181.0 MiB client arena, 6604 launches; 34.0 scaled (rng input) | 1330.3 | 2835.6 | 2.1x | 34.0 |
| YOLOv5s | 3x160x160 | — | runs at 150 MiB (raw graph), 1488 launches; accuracy off | — | 943.1 | — | 2.5 |
| YOLOv8s | 3x160x160 | — | **runs** at 147 MiB, 1249 launches; 3.87 scaled (1.50 optimised) | 272.2 | 972.6 | 3.6x | 3.87 |
| YOLOv9s | 3x160x160 | — | runs at 150 MiB (raw graph), 6512 launches; accuracy off | — | 2430.4 | — | 4.1 |

YOLOv8n at 160 is 0/44,100 elements over the 5e-4 tolerance against the CPU,
max abs 1.5e-2, which is the logit range; the scaled figure is the meaningful
one.  The 320 row is the optimised artifact at a 60 MiB client arena; the raw
graph returns the right detections with wrong values (16.1 scaled), so the
artifact is the path that is recorded.  Its 8.6e-4 scaled is above the 5e-4
gate used for the 160 row; it is the measured figure, not a pass claim.  The 640 row fits the memory plan
(43.7 MiB pool plus 87.9 MiB scratch) but the row copy cannot address a 640
input.  The v5s and v9s rows run on the raw graph at 150 MiB and fail the
accuracy gate (2.5 and 4.1 scaled); they are recorded as runnable but not
accepted, not as passes.  Their frame rates are not comparable with YOLOv8n's:
different graphs, different work.  CPU timing was not re-measured for these
cells.  Detection coverage beyond YOLO is untouched by this page: no non-YOLO
detector was measured.

## Other tasks

All three are from the ONNX Model Zoo, downloaded through the LFS media host
(`benchmodels/fetch_models.sh` records URLs, sizes and SHA-256).  All three are
CPU-only here: none reaches a VC4 kernel.

| model | task | input | params | engine | CPU ms | VC4 ms |
|---|---|---|---:|---|---:|---:|
| `super-resolution-10` | 4x super-resolution (sub-pixel CNN) | 1x224x224 -> 1x672x672 | 59.7 k | refused: arena (~147 MiB); the `Cin=1` conv now plans | 708.9 | — |
| `candy-8` | style transfer (fast neural style) | 3x224x224 -> 3x224x224 | 1.68 M | refused: arena (~540 MiB), then 9x9 `Conv`, `Upsample` | 1959 | — |
| `fcn-resnet50-12-int8` | 21-class segmentation (FCN, ResNet-50) | 3x224x224 -> 21x224x224 | 35.3 M (int8) | refused: `QLinearConv` | not measured | — |

The super-resolution graph's arena refusal is not the weight footprint: its
weights are 233 KiB.  It is the conv scratch for the last 3x3 layer at 672x672
with 64 input channels, which the planner estimates at **126 MiB**.  With the
batch fixed to 1 the planner stops there; with the published symbolic batch it
skips the scratch estimate and the static plan used to stop at the first conv:

```
node 'Conv_0' (Conv): depthwise Cout 64 != Cin 1 (channel multiplier)
```

`Conv_0` is a dense 1->64, 5x5 convolution.  The dispatch test was
`group == Cin`, and a dense conv has `group == 1`, which with a single-channel
input is also `Cin == 1`; the conv was refused as depthwise with a channel
multiplier.  `coverage.conv_kind` now makes the decision once (group 1 is
dense; `group == Cin` is depthwise only above one group) and the static report
and the executor both call it.  The published graph now plans **12/12
dispatches** on the symbolic batch, including the two `Reshape`s and the
`Transpose` that express `DepthToSpace`.  The fixed-batch graph is unchanged
and still needs about 147 MiB, so this model's remaining blocker is the arena,
not an operator.

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
| ShuffleNet (as published) | plan | `node 'n4' (Conv): group 4 needs 16-aligned per-group blocks` |
| Inception v1 (as published) | open | `graph pool needs at least 54.1 MiB ... peak is about 91.1 MiB ... arena is 93.0 MiB with 41.8 MiB reserved ... about 103.9 MiB would do` |
| Inception v2 (as published) | open | `needs at least 85.4 MiB ... peak about 167.4 MiB ... about 135.2 MiB would do` |
| DenseNet-121 (as published) | open | `needs at least 65.2 MiB ... peak about 376.0 MiB ... about 114.9 MiB would do` |
| ResNet-50 (as published) | fold | `node 'ConstantOfShape_233': constant fill 512x512x3x3 needs 9.0 MiB, over the 93.0 MiB plan-time fold budget (84.7 MiB of weights folded before it)` |
| AlexNet (as published) | fold | `node 'ConstantOfShape_11': constant fill 4096x9216 needs 144.0 MiB, over the 93.0 MiB single-tensor limit` |
| ZFNet-512 (as published) | fold | `node 'ConstantOfShape_11': constant fill 4096x18432 needs 288.0 MiB, over the 93.0 MiB single-tensor limit` |
| VGG-19 (as published) | fold | `node 'ConstantOfShape_31': constant fill 4096x25088 needs 392.0 MiB, over the 93.0 MiB single-tensor limit` |
| `super-resolution-10` (batch 1) | open | `needs at least 12.8 MiB ... peak about 70.0 MiB ... 126.0 MiB reserved for conv/elementwise scratch ... about 146.8 MiB would do` |
| `candy-8` | open | `needs at least 25.6 MiB ... peak about 195.1 MiB ... 506.0 MiB reserved for conv/elementwise scratch ... about 539.6 MiB would do` |
| `candy-8` (operator view) | plan | `node ... (Conv): kernel 9x9 (NCHW fallback: 1x1, 3x3)` and two `Upsample` nodes: `no implementation` |
| `fcn-resnet50-12-int8` | plan | `node 'fused Conv_6_quant' (QLinearConv): no implementation` |

The arena messages are the planner's floor, peak and scratch numbers; the
full untruncated strings are in `benchmodels/results/` and the plan-probe
records.  The `fold` stage is the plan-time `ConstantOfShape` materialisation:
it runs before the arena is checked because the pool is sized from the folded
weights.  A weight tensor that does not fit the arena cannot run in it, and
the message names the tensor, so the constraint is the arena even though the
stage is earlier.  The four large graphs' arena estimates (about 150, 196,
362 and 603 MiB) are from the materialised probe and are not repeated in the
message the current plan raises.

## What a reader can conclude

* **Non-YOLO support is real but narrow.**  A small conv/BN/pool/Gemm model
  with ordinary shapes runs end to end on the QPUs, with the CPU parity shown
  above (worst max abs 6.9e-06 on modelA).  SqueezeNet, a 1.24 M-parameter
  ImageNet graph with 1x1 and 3x3 convolutions, now runs as published; the
  rest of the corpus does not.
* **The 152/152 operator figure is an operator result, not a model result.**
  Every operator in the 152-node sweep exists; whether a model runs is decided
  by tensor sizes and by the shapes the kernels accept.  A 7x7 stem is a
  supported `Conv` with an unsupported kernel.
* **No other standard ImageNet backbone in this corpus runs**, for three
  independent reasons: the 96 MiB arena (and weight tensors of 144-392 MiB
  that cannot be materialised into it); conv kernels restricted to 1x1/3x3/5x5;
  grouped-conv 16-alignment.  Fixing any one of them admits no model on its
  own - ResNet-50 needs both a bigger arena and a 7x7 kernel.
* **The zoo's small non-classification models do not run either.**  The
  super-resolution model plans all 12 nodes but needs 147 MiB of conv scratch
  at 672x672; style transfer needs 540 MiB of 9x9-conv scratch; the small
  segmentation model is int8.
* **Detection is YOLOv8n at 160x160 on this board.**  320 does not fit 96 MiB
  and 640 does not fit at all; no other detector was measured.

## Next steps, in the order the measurements argue for

1. **Arena.**  Seven of nine corpus graphs and two of three zoo graphs still
   stop here: three on the pool peak and four on weight bytes the arena cannot
   hold.  The measured needs are 104-603 MiB against 96.  A larger `vc4d
   --arena-mb` is a configuration change, not a kernel; it would admit
   ResNet-50 (150) and AlexNet (196) if the CMA allows it.  I did not restart
   the daemon during this sweep.
2. **Conv kernels 7x7 and 11x11.**  Inception v1 and DenseNet-121 are measured
   to stop at a 7x7 stem once the arena is relaxed; ResNet-50 and ZFNet use the
   same 7x7 stem but their weight floor keeps them at the arena; AlexNet's stem
   is 11x11 and style transfer's is 9x9.  This is the gap that survives after
   the arena.
3. **Grouped conv alignment.**  ShuffleNet's group-4 conv is refused for
   16-alignment.  Pad the per-group block rather than the channel count.
4. **`Upsample` (opset 8).**  No implementation.  `Resize` exists; either lower
   the deprecated op to it or refuse it at export time.
5. **int8.**  `QLinearConv` cannot run in an fp32 arena.  Dequantise at the
   graph boundary, or add an int8 path; do not expect the zoo's quantised
   segmentation model to run before then.

## Reproduce

```
benchmodels/fetch_models.sh          # three zoo graphs, URLs and SHA-256
benchmodels/materialise.py M.onnx O.onnx   # the independent check of the fold
benchmodels/plan_probe.py M.onnx out.json
benchmodels/conv_probe.py --cout 1000      # the 1x1 conv, its slice, its size
benchmodels/bench_model.py --model M.onnx --task classification \
    --source "..." --size 224 --out out.json
```

The ConstantOfShape fold is part of `Runner`, so a corpus graph is planned and
run as published; `materialise.py` is only used to check the fold's bytes.
`conv_probe.py` takes `--cin/--cout/--hw` so the channel alignment and the
spatial extent can be varied independently.

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
