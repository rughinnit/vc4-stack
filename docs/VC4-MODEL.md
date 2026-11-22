# A detector that fits the VideoCore IV launch budget

2026-11-22, Raspberry Pi 3B (BCM2837), Debian 13 arm64, kernel 6.18.  Engine:
`vc4infer` built from this tree (`infer/Makefile`, in-tree `libvc4infer.so.1`
with the conv/matmul activation epilogue) through `vc4d` 1.2.9, unprivileged,
one client at a time.  Clocks: core 400 MHz, V3D 300 MHz in every run
(`tools/thermal-guard.sh --max 71` printed `core 400000000 Hz` on the guard
line); temperatures 59.6-65.0 C, `throttled=0x20000` was already set before
this work and no run moved the core clock.

## 1. Result

A single-scale detector built to these constraints -- 8 convs, 22 compute
nodes, 0.046 GFLOP, 96x96 input -- runs in **27.5 ms (36.3 FPS)**, in 18
launches, with 13.7 ms of QPU time and every float operation on the QPUs.  The
same graph at 128x128 runs **40.1 ms (24.9 FPS)**.

* **30 FPS**: met at 96x96 with 5.8 ms of margin, missed at 128x128 by 6.8 ms.
* **60 FPS**: not reachable for a graph with a backbone.  The measured cost is
  3.13 ms per 3x3 conv at 96 (4.33 ms at 128), so a 16.7 ms frame allows 4.5
  convs at 96 and 2.6 at 128.  The five-conv lower point measured below already
  takes 18.1 ms.
* **The obstacle is not FLOPs and not kernel coverage.**  All 26 ONNX nodes, 22
  of them compute, dispatch to the QPUs; the host copies 0 bytes and does no
  arithmetic.  What the frame is spent on is 13.7 ms of QPU time plus 13.8 ms
  of launch path for 18 launches, and both scale with the number of convs.

The FLOP budget usually quoted for this board (0.46 GFLOP at 30 FPS) is an
arithmetic ceiling that assumes the hand-written GEMM's 13.72 GFLOP/s and no
launch cost.  The engine, at the shapes a detector at 96x96 actually uses,
issues conv MACs at **3.3 GFLOP/s** and charges **0.77 ms of launch path per
launch** -- 1.6 ms of wall per launch once the launch's own QPU time is
included.  At those two numbers the 30 FPS budget is **0.05-0.06 GFLOP**, not
0.46 -- eight to nine times smaller -- and the graph below sits right at it.

## 2. What a conv costs here

Two points, same graph family, 5 convs and 8 convs, median of three runs of
five reps each after a JIT warm-up:

| input | convs | launches | wall ms | QPU ms | FPS | ms per extra conv |
|---|---:|---:|---:|---:|---:|---:|
| 96  | 5 | 12 | 18.1 | 7.8  | 55.2 | - |
| 96  | 8 | 18 | 27.5 | 13.7 | 36.3 | 3.13 (1.97 QPU + 1.16 path) |
| 128 | 5 | 12 | 27.1 | 14.9 | 36.9 | - |
| 128 | 8 | 18 | 40.1 | 24.1 | 24.9 | 4.33 (3.07 QPU + 1.26 path) |

Linear fits through those pairs:

```
T(96)  = 2.5 ms + 3.13 ms * convs      (2 launches per conv, +2 boundary)
T(128) = 5.4 ms + 4.33 ms * convs
```

So a conv is two launches (im2col and GEMM), 1.97 ms of QPU at 96 and 3.07 ms
at 128, plus about 1.2 ms of launch path.  Per-node profile of the 8-conv graph
at 96 (18 launches, 25.1 ms of node wall, 13.7 ms QPU):

| node | shape | launches | wall ms | QPU ms |
|---|---|---:|---:|---:|
| `/model.0/conv/Conv` | 3->16 3x3 s2, 96->48 | 3 | 7.44 | 4.35 |
| `/model.1/conv/Conv` | 16->32 3x3 s2, 48->24 | 2 | 3.11 | 1.68 |
| `/model.2/conv/Conv` | 32->32 3x3 s1, 24 | 2 | 4.11 | 2.86 |
| `/model.3/conv/Conv` | 32->64 3x3 s2, 24->12 | 2 | 2.48 | 1.27 |
| `/model.4/conv/Conv` | 64->64 3x3 s1, 12 | 2 | 3.23 | 2.16 |
| `/model.5/conv/Conv` | 64->32 1x1, 12 | 2 | 1.32 | 0.29 |
| `/model.6/conv/Conv` | 32->32 3x3 s1, 12 | 2 | 1.87 | 0.85 |
| `/model.7/Conv` | 32->6 1x1, 12 | 3 | 1.55 | 0.22 |

The stem is the single most expensive node: three launches, 7.4 ms, because
its three input channels are padded to 16 before the im2col and the input
plane is converted to NHWC.  The two extra launches at the ends (2x3 instead of
2x2) are that input conversion and the output copy.  The node walls sum to
25.1 ms against a 27.5 ms bench mean: the difference is between-node time that
the per-node counter does not attribute.

The engine's issued MACs at 96 are 22.9 M (0.046 GFLOP) once the 3-channel stem
and the 6-channel logits conv are counted at the 16 channels the runner
actually computes; that is 3.3 GFLOP/s over the 13.7 ms of QPU time.  For
comparison the engine's GEMM at 256^3 is 8.44 GFLOP/s
([BENCHMARKS.md](BENCHMARKS.md)) and the hand-written asm10 GEMM is 13.72.  A
detector's convs are 1x1 and 3x3 at 12x12 to 48x48, where the tile is short and
the im2col is a real cost.

## 3. Specification

The graph is built to these rules.  Each one follows from a measurement above
or from a constraint in [ONNX.md](ONNX.md) sec. 8.1; none is stylistic.

| rule | value | reason |
|---|---|---|
| input | 96x96 (128x128 measured for comparison) | the largest size at which the 8-conv graph still meets 30 FPS; 128 misses it, and 160x160 is 2.8x the pixels of 96 |
| convolutions | 1x1 and 3x3 dense only, stride 1/2, symmetric pad | the NHWC conv path; `KH,KW in {1,3,5}`, `stride in {1,2}`, `Cin % 16 == 0`, `K = Cin*KH*KW % 16 == 0` |
| depthwise | **not used** | the engine expands depthwise to a block-diagonal dense weight, so a depthwise conv costs exactly a dense conv of the same width (Cin x the MACs) and saves one launch by giving up Cin times the arithmetic.  At 128, where the graph is QPU-bound, that is a loss. |
| channels | multiples of 16 (16/32/64 at width 0.25; 32/64/128 at 0.5) | the conv kernel reads 16-channel blocks; a group block that is not a multiple of 16 raises.  The 3-channel stem and the 6-channel logits are padded by the runner and counted as 16 above. |
| activation | SiLU after every conv, fused into the conv store | measured: `VC4_OPT=all` 27.5 ms / 18 launches against `VC4_OPT=none` 45.5 ms / 58 launches on the same graph (median of three).  Without the epilogue each SiLU is 1-12 elementwise launches. |
| residuals, concat, split | none | an Add over a pitched NHWC plane is 3-6 launches; two of them cost more than a whole 3x3 conv.  No Split means no C2f, hence the plain conv stacks.  Concat is a host channel copy and is not used. |
| attention, DFL, multi-scale neck, NMS | none | not dispatchable or not needed: no `Split`, no `Softmax` over more than 16 classes, no `Resize` |
| head | one level (stride 8), one class, 6 logits per cell (objectness, cx, cy, w, h, class) | no DFL convolution, no `Split` of the prediction tensor, no anchor grid; the decode is host post-processing outside the graph |
| nodes | <= 40 | the measured graph is 22 compute nodes; the constraint is not the binding one |
| convs | <= 9 at 96, <= 6 at 128 for 30 FPS | the fits in sec. 2: 33.3 ms budget, 2.5/5.4 ms intercept, 3.13/4.33 ms per conv |
| FLOPs | <= 0.2 GFLOP (arithmetic target), <= 0.06 GFLOP at 30 FPS for the measured engine | see sec. 4 |

## 4. Budget derivation

Frame budget: 33.3 ms at 30 FPS, 16.7 ms at 60 FPS.  The measured inputs are
the 0.28 ms tiny-kernel launch floor through `vc4d` ([GGML.md](GGML.md) sec. a),
the 13.72 GFLOP/s hand-written GEMM and the 8.44 GFLOP/s engine GEMM
([BENCHMARKS.md](BENCHMARKS.md)), and the 3.3 GFLOP/s conv rate and 0.77 ms
launch-path cost measured on the graphs in sec. 2.

| model | 30 FPS budget | 60 FPS budget |
|---|---:|---:|
| arithmetic only, 13.72 GFLOP/s, no launch term | 0.457 GFLOP | 0.229 GFLOP |
| + 20 launches at 0.28 ms (5.6 ms) / 15 at 0.28 ms (4.2 ms) | 0.380 GFLOP | 0.171 GFLOP |
| engine GEMM 8.44 GFLOP/s + the same launch term | 0.234 GFLOP | 0.105 GFLOP |
| measured 0.77 ms launch path per launch, 3.3 GFLOP/s convs (18 launches / 12 launches) | 0.064 GFLOP | 0.025 GFLOP |
| measured fit, 96x96: `T = 2.5 + 3.13 c`, average conv 2.87 M MAC | 9 convs = 0.052 GFLOP | 4.5 convs = 0.026 GFLOP |

Two conclusions.  The 0.46 GFLOP figure is right for a GEMM at 256^3 and
wrong for a detector: the conv rate at detector shapes is 2.5-4x lower, and
once the launch path is charged per conv the budget is 0.05 GFLOP.  And the
launch count is not a separate, parallel budget to the FLOP budget: on this
engine a conv's launch path (1.2 ms) is 37 % of the conv's cost at 96 and its
QPU time (1.97 ms) is 63 %, so the two have to be counted in one linear model.

The graph below is at 0.046 GFLOP at 96, i.e. just under the 0.05 GFLOP
measured budget, and lands at 36 FPS rather than 30 -- the fit allows 9.8
convs against the 8 in the graph.

## 5. The graph

Two width variants of one topology, plus a lower point used to measure the
per-conv slope:

| file | width | channels | convs | params | compute nodes | ONNX ops |
|---|---|---:|---:|---:|---:|---|
| `infer/tools/det/vc4det-t.yaml` | 0.25 | 16/32/64 | 5 | 26 k | 13 | Conv 5, Sigmoid 4, Mul 4, Identity 1 |
| `infer/tools/det/vc4det-n.yaml` | 0.25 | 16/32/64 | 8 | 81 k | 22 | Conv 8, Sigmoid 7, Mul 7, Identity 4 |
| `infer/tools/det/vc4det-s.yaml` | 0.50 | 32/64/128 | 8 | 323 k | 22 | Conv 8, Sigmoid 7, Mul 7, Identity 4 |

`vc4det-n` at 96:

```
0  Conv 3->16  3x3 s2  96->48   + SiLU
1  Conv 16->32 3x3 s2  48->24   + SiLU
2  Conv 32->32 3x3 s1  24       + SiLU      (depth-scaled row, depth 0.5 = 1)
3  Conv 32->64 3x3 s2  24->12   + SiLU
4  Conv 64->64 3x3 s1  12       + SiLU      (depth-scaled row, depth 0.5 = 1)
5  Conv 64->32 1x1     12       + SiLU
6  Conv 32->32 3x3 s1  12       + SiLU
7  Conv 32->6  1x1     12       logits, no activation
```

The YAML is instantiated by Ultralytics' own `DetectionModel` and `parse_model`
(`infer/tools/det/build.py`), so width/depth are the framework's scaling, not a
hand-copied graph.  The head is a plain `nn.Conv2d` row: `Detect` is not used,
because its export contains `Split`, `Softmax` over `reg_max` and a DFL conv,
none of which this engine dispatches.  `build.py` fuses BN before export, then
fails the build if any internal conv has a channel count that is not a multiple
of 16, if BatchNormalization survives, or if the graph contains an op outside
the engine's table.  Exports are batch 1, static shape, fp32, opset 17,
`simplify=False`.

Modelled launches from `infer/tools/plan_diff.py --passes all`: 16 for all
three (`Conv 2x8`, no activation launch).  The board adds the input conversion
and the output copy, so `vc4infer bench` reports 18 for the 8-conv graphs and
12 for the 5-conv one.

## 6. Measured latency

Median of three runs of five reps, each run in its own `vc4infer bench`,
`VC4_OPT=all`, in-tree engine, one client, thermal guard:

| graph | input | convs | launches | wall ms | QPU ms | QPU share | FPS | GFLOP (issued) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vc4det-t` | 96 | 5 | 12 | **18.1** | 7.8 | 43 % | **55.2** | 0.022 |
| `vc4det-n` | 96 | 8 | 18 | **27.5** | 13.7 | 49 % | **36.3** | 0.046 |
| `vc4det-s` | 96 | 8 | 18 | 47.6 | 32.2 | 69 % | 21.0 | 0.162 |
| `vc4det-t` | 128 | 5 | 12 | 27.1 | 14.9 | 54 % | 36.9 | 0.039 |
| `vc4det-n` | 128 | 8 | 18 | **40.1** | 24.1 | 60 % | **24.9** | 0.082 |
| `vc4det-s` | 128 | 8 | 18 | 73.5 | 55.4 | 75 % | 13.6 | 0.288 |

Run-to-run spread over the three runs was within 0.4 ms except one 128x128 run
at 41.4 ms.  One earlier pass that benched six graphs back to back reported
22.7 ms for `vc4det-t` at 96 against 18.1 ms in the isolated repeats; the
repeats are the number reported here.

The epilogue is worth 1.65x on this graph: the same `vc4det-n` at 96 with
`VC4_OPT=none` is 45.5 ms and 58 launches (median of three, QPU 16.0 ms):
40 of those launches are the seven SiLUs: one elementwise launch covers 12
QPU groups of 256 floats, so the 16x48x48 plane alone takes 12 of them.  With `VC4_OPT=all` the seven activations cost
zero additional launches.

`s` at 96 is 0.162 GFLOP and 21 FPS: width 0.5 is inside the 0.2 GFLOP
arithmetic target but three times over the measured 30 FPS budget, and it
misses 30 FPS by 14 ms.  At 128, width 0.5 is 0.288 GFLOP and 13.6 FPS.

## 7. The gap to 30 and 60 FPS

| target | frame | `vc4det-n` 96 | `vc4det-n` 128 |
|---|---:|---|---|
| 30 FPS | 33.3 ms | **met**, 36.3 FPS, 5.8 ms spare | **missed**, 24.9 FPS, 6.8 ms short |
| 60 FPS | 16.7 ms | missed, needs 10.8 ms less | missed, needs 23.4 ms less |

What would close it, in order of size at 96:

* **QPU time, 13.7 ms.** Even with a free launch path, 13.7 ms of QPU time
  gives 73 FPS; halving it gives 20.6 ms (48 FPS).  At 128 the QPU time alone
  (24.1 ms) is above a 60 FPS frame, so no launch work can reach 60 there.
* **Launch path, 13.8 ms for 18 launches (0.77 ms each, bench mean; within the
  per-node rows it is 11.4 ms over the same 18).** This is 2.7x the 0.28 ms
  floor measured for a 4096-float elementwise kernel: each conv launch also
  sets up and submits its im2col panel.  Closing this needs a lower per-launch
  cost, not fewer nodes -- the graph is already at 2 launches per conv.
* **Kernel coverage, nothing.** 0 unsupported nodes, 0 host-compute nodes,
  0 bytes copied by the host, `strict` mode, 22/22 compute nodes on VC4.  This
  is not where a frame is lost on this graph.

The structural answer for 60 FPS: at 96x96 the fit allows 4.5 convs in 16.7 ms,
which is a stem, two stride-2 convs and a 1x1 head -- a three-layer network
with a 12x12 output grid.  That is below the depth at which a detector is worth
measuring.  30 FPS at 96 is the target that this engine supports with a real
(if small) backbone; 30 FPS at 128 needs the graph cut to six convs, i.e. the
`t` topology plus one.

The cheapest measured lever is not in this graph: the stem alone is 7.4 ms of
the 27.5 (27 %) for a 3->16 conv, because the 3 input channels are padded to 16
and the input is converted to NHWC in three launches.  A graph that still has a
160x160-class input would want that path re-examined before any further depth
is added.

## 8. Training and accuracy: nothing is claimed

The architecture is latency-measured with random initialisation.  There are no
trained weights and no detection-quality number, and none is implied by the
numbers above.

Two reasons, both measured here:

* **The head has no loss implementation.**  The graph deliberately has no
  `Detect`/DFL head (they export `Split` and a `reg_max` softmax the engine
  refuses), so `yolo detect train` cannot consume it; the 6 logits per cell
  need their own assignment and loss.
* **The board is not a training machine.**  `infer/tools/det/train_probe.py`
  runs one forward+backward of `vc4det-n` at 96, batch 4, four threads:
  **299 ms per step**, i.e. 2.46 h per COCO train2017 epoch at batch 4
  (29,571 steps).  Ten epochs is a day.  `coco8` is four training images, one
  step per epoch: it completes in seconds and cannot measure detection quality.

Weights for this topology need a workstation run on a real dataset, followed by
the same latency measurement here (weights do not change conv cost, so the
timings in sec. 6 transfer unchanged).  Until then the honest description is
"a graph of the right shape and arithmetic at the measured speed", not "a
detector".

## 9. Reproduce

```sh
cd ~/work/vc4-model
export TMPDIR=$HOME/scratch/tmp
export VC4INFER_LIB=$PWD/infer/build/libvc4infer.so.1
export VC4INFER_CACHE=$HOME/scratch/kcache-model
export PYTHONPATH=$PWD/infer/python
export VC4_OPT=all

# 1. the engine (needs the vc4d client library from a vc4d checkout)
make -C infer -j3 VC4D_DIR=$HOME/work/vc4-stack/vc4d build/libvc4infer.so.1

# 2. YAML -> ONNX, with the channel/op audit (Ultralytics venv on the board)
~/work/torch-vc4/.venv/bin/python infer/tools/det/build.py vc4det-n --imgsz 96
~/work/torch-vc4/.venv/bin/python infer/tools/det/build.py vc4det-n --imgsz 128

# 3. launch count with no device
python3 infer/tools/plan_diff.py infer/tools/det/models/vc4det-n_96.onnx --passes all

# 4. one client, under the guard
tools/thermal-guard.sh --max 71 -- \
  python3 infer/cli/vc4infer bench infer/tools/det/models/vc4det-n_96.onnx --reps 5
tools/thermal-guard.sh --max 71 -- \
  python3 infer/tools/det/node_profile.py infer/tools/det/models/vc4det-n_96.onnx /dev/stdout

# 5. what a training step costs here
~/work/torch-vc4/.venv/bin/python infer/tools/det/train_probe.py vc4det-n --imgsz 96
```

Raw rows behind the tables: [vc4-model.json](vc4-model.json).

## 10. Open items

* The stem's 3->16 pad plus NHWC conversion is 3 launches and 27 % of the
  frame at 96; a 1x1 3->16 stem followed by a 3x3 16->16 was not measured.
* Width 0.25 is set by the 30 FPS fit and by keeping every internal channel a
  multiple of 16; the arena is not the constraint here (peak 1.5 MiB of 64 MiB).
* A native depthwise kernel would change the depthwise row of sec. 3 from "no
  saving" to "Cin times cheaper"; it is not shipped (ONNX.md sec. 8.5).
* No accuracy, per sec. 8.
