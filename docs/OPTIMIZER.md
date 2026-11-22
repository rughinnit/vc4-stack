# The inference optimiser: what TensorRT, MIGraphX and ncnn do, and what this engine needs

The engine runs a graph node by node.  TensorRT, MIGraphX and ncnn run a graph
after an ahead-of-time optimiser: fusion, layout assignment, memory planning,
weight prepacking, kernel selection and, on some targets, precision reduction.
This document is the plan for giving `vc4infer` the same class of optimiser,
with the constraints this hardware imposes.

Nothing here changes what a model computes.  Each pass is gated on the same
oracles the executor already uses: `infer/tools/op_matrix.py` for operator
coverage and the model corpus for end-to-end parity, at the tolerance recorded
in `docs/ACCURACY.md`.

## Baseline

On YOLOv8n at 160x160, 481.7 ms wall, 802 launches:

| component | ms | share |
|---|---:|---:|
| QPU kernel time | 263 | 55% |
| launch path | 222 | 46% |
| host-side dispatch, packing, conversion | included above | |

The QPU time alone is already 3.5x ncnn's total 74.8 ms, so fusion cannot
reach parity by removing launches alone; it must also remove passes over
memory.  The realistic target is 250-350 ms, with the gap to ncnn stated
rather than hidden.

## What the reference engines do

- **TensorRT**: layer fusion, kernel autotuning per shape, tensor-format
  selection, memory reuse, optional low precision.
- **MIGraphX**: graph rewrites in MIGraphX IR, target-specific fusion,
  layout-aware scheduling.
- **ncnn**: PNNX conversion with operator fusion, weight repacking, layout
  choices, and hand-tuned kernels per layer type.

The common core is: an explicit graph IR, a deterministic pass pipeline, and
measurement after every pass.

## Pass pipeline

Order matters; each pass is individually switchable so a regression can be
bisected to one rewrite.

| # | pass | what it does | expected gain |
|---|---|---|---|
1 | canonicalise | constant folding, dead-node removal, identity elimination | fewer nodes, no numeric change |
2 | elementwise fusion | merge chains (`add -> relu -> add`, `mul -> add`) into one EWB launch | removes launches; the biggest single win |
3 | epilogue fusion | bias and activation folded into the GEMM/conv that produces the tile | 2-3 launches per conv |
4 | layout assignment | keep packed NHWC between compatible nodes; convert only at boundaries | removes copy/transpose bandwidth |
5 | weight prepacking | pack weights once at plan time into the arena layout | removes per-launch host work and repeated packing |
6 | memory planning | one arena plan with buffer reuse and liveness, replacing per-op scratch growth | lowers peak arena; more models fit |
7 | launch batching | several kernels per daemon round-trip, or a persistent work queue | attacks the 222 ms directly |
8 | kernel autotuning | QPU counts per shape (GEMM and its packer), recorded per generator | recorded per shape; see "Pass 8" |

Passes 1-3 are graph rewrites on the Python side; 4 and 6 touch tensor layout
and arena planning; 7 needs runtime work in `vc4d` and the client; 8 is
measurement-driven and must be recorded per shape.

## Correctness gates

- every pass has an off switch, and the default stays off until it is measured;
- after each pass: `op_matrix.py` (152/152 expressible, 0 refused, 0 mismatch);
- after each pass: YOLOv8n 160 parity against ONNX Runtime CPU at the recorded
  tolerance, with detection count and box delta;
- fusing changes summation order, so tolerances are re-established per pass,
  not inherited;
- the optimiser never runs an operator the executor cannot, and never falls
  back to the ARM.

## Phases

| phase | content | exit criterion |
|---|---|---|
A | IR and pass harness, pass 1 | same numbers, passes switchable, tests green |
B | passes 2-3 | launches for v8n160 down by >= 40%, wall < 420 ms, parity held |
C | passes 4-6 | wall < 380 ms, arena peak down >= 25% |
D | pass 7 | wall < 320 ms |
E | pass 8 | recorded per-shape gains, no regression on the corpus |

Each phase is a separate merge, and each number is added to
`docs/BENCHMARKS.md` with its conditions.

## Pass 8: per-shape kernel autotuning

`vc4_autotune` picks the QPU counts the generated kernels leave to the host,
measures the legal alternatives on the device, and records the winner under the
kernel cache.  The executor reads the record for a shape when autotuning is
enabled and runs the launch it has always run when it is not.  Implementation:
`infer/src/vc4_autotune.c`; C API in `infer/include/vc4infer.h`; Python
`Engine.autotune`.

### The space

| op | axis | values tried | legality |
|---|---|---|---|
| `matmul` | `pack_q` | 4, 2, 1 | `<= 4` (the blob is registered for 4) and divides `M/8` |
| `matmul` | `gemm_q` | 12, 8, 6, 4, 3, 2, 1 | `1..12` and `<= M/8` |
| `conv2d` | `im2col_q` | 4, 2, 1 | `<= 4` and divides `M/16` |
| `conv2d` | `gemm_q` | 12, 8, 6, 4, 3, 2, 1 | `1..12` and `<= M/16` |

`M` is the padded output row count the GEMM iterates: `align8(M)` rows for a
matmul and `Ho * align16(Wo)` for a conv.  The auxiliary kernels (the A-panel
packer and the im2col gather) partition their row blocks exactly and have no
in-kernel guard, so their QPU count must divide the work; the GEMMs guard their
row-block loop with `rb < TR` and only need more row blocks than QPUs.  The
12-QPU ceiling is the hardware's; the kernel cache separately refuses to launch
a blob with more QPUs than it was registered for.

The tile shape and the vectors per group are *not* searched.  They are part of
each kernel's hand-verified instruction schedule (the asm10 8x64 tile, the asm3
16x16 tile, `build_ew`'s 16-vector group in front of the verified 256-float
store).  A different tile is a different program, and a wrong QPU program is not
a recoverable error on this board.  The QPU counts change loop bounds and
uniform strides only, so every candidate runs the same verified body over the
same data.

Coarse pass: the default first (it is the baseline), then each legal GEMM count
with the default packer, then each legal packer with the default GEMM - at most
11 candidates, one axis at a time.  Fine pass: the GEMM counts one and two
either side of the coarse winner, then auxiliary counts the coarse pass did not
try - at most 4 more, deduplicated.

### Measurement

One warm-up run per candidate (which also generates and uploads its blob) is
discarded, then `reps` runs are timed with `CLOCK_MONOTONIC` and the median
recorded: an ARM scheduler slice or a vc4d round trip is a common outlier and
the distribution is not normal.  The whole op is timed, so host-side packing and
the NCHW store of a conv are part of the number.

Bounded twice: at most 14 measured parameter sets, and a 30 s wall-clock budget
that starts before the first blob is generated.  `VC4INFER_AUTOTUNE_MAX`,
`VC4INFER_AUTOTUNE_BUDGET_MS` and `VC4INFER_AUTOTUNE_REPS` override the three
numbers.

Before a choice is recorded, the op is run again with it and the output compared
with the default launch's output at the tolerance of `docs/ACCURACY.md`
(`atol = rtol = 5e-4`).  A disagreement is reported and nothing is cached.  On
this hardware the chosen launch is expected to be bit-identical - the QPU count
moves row blocks between QPUs, not the k order - and the check is there because
"expected" is not a verification.

### Silicon result and the acceptance margin

Measured through `infer/build/bench_autotune 5` with the ARM capped at 1.0 GHz,
one client and `throttled=0x0` (2026-11-23):

| shape | cold search | chosen | default ms | tuned ms | gain |
|---|---:|---|---:|---:|---:|
| 128^3 | 5.12 s | pack_q=4 gemm_q=12 | 1.001 | 0.993 | 1.01x |
| 256^3 | 5.89 s | pack_q=2 gemm_q=11 | 4.250 | 4.202 | 1.01x |
| 512^3 | 9.78 s | pack_q=4 gemm_q=12 | 28.581 | 34.130 | 0.84x |
| conv 32x32x32 3x3 | 8.13 s | im2col_q=4 gemm_q=9 | 12.912 | 10.104 | 1.28x |

The 512^3 row chose the default parameters and still lost 16%: the same launch
measured 28.581 ms and then 34.130 ms with an identical cycle count.  The
board's memory path has an intermittent stall state (see `KERNELS.md` section
9), so a before/after comparison that spans the search is not a measurement.
For that reason the honest gain for these four shapes is **1.00x**:
12 GEMM QPUs and 4 pack QPUs are the ladder's first candidates, and the search
confirms them.  An interleaved A/B/A sequence inside `bench_autotune` is the
next change to this harness, not to the tuner.

The acceptance margin still does its job.  A candidate must beat the default
by more than `TUNE_ACCEPT_MARGIN` (3%) to be recorded, and the default is
recorded otherwise; on an earlier boot the same benchmark wrote 0.93x, 0.82x
and 0.77x records, all of which are now 1.00x.  A record can no longer make a
shape slower than not tuning it.

The per-class QPU breakdown, the conv panel/GEMM split and the stall-state
evidence are in [`KERNEL-EFF.md`](KERNEL-EFF.md).

### The cache

- Path: `<kernel cache>/autotune/<key>.tune`, where the key is the first 8
  bytes of
  `sha256("vc4infer-autotune-1|" op "|" shape "|" digest)`.
- Digest: the same `sha256(kgen.py, kops.py beside it, ABI tag)` the blob cache
  writes into every `.bin.stamp`.  A generator change therefore invalidates
  tuned choices through the mechanism that already exists; there is no second
  hash to keep in step.
- Content: one `trial` line per measured parameter set with its median, then
  `best`, the blob name, and the two medians.  The record is written to a
  temporary file and renamed, so a power loss cannot leave a half record.
- Lookup: a 32-entry per-context memo, then one file open per shape.  With
  autotuning off, a lookup is one `getenv` and no file I/O, which is what keeps
  the existing numbers reproducible.
- A record that is stale (different digest), truncated, foreign, or names a
  launch the filter rejects is a miss: the op runs its default and the next
  `vc4_autotune` rewrites the record.

### API

```c
vc4_tune_shape sh = { .m = 512, .n = 512, .k = 512 };
vc4_tune_result r;
vc4_autotune(ctx, VC4_OP_MATMUL, &sh, NULL, &r);
/* r.params.gemm_q, r.params.pack_q, r.best_ms, r.default_ms, r.max_abs */
```

```python
eng.autotune("matmul", (512, 512, 512))
eng.autotune("conv2d", {"cin": 32, "cout": 32, "h": 32, "w": 32,
                        "kh": 3, "kw": 3, "pad": 1})
```

`VC4INFER_AUTOTUNE=1` makes every process consult the records without a call;
a successful `vc4_autotune` enables the same for its context.  Both are off by
default.

### Measured on the board

```
cd ~/work/vc4-autotune
export TMPDIR=$HOME/scratch/tmp
export VC4INFER_CACHE=$HOME/scratch/kcache-autotune      # an empty one: cold
make -C infer -j4
VC4INFER_DAEMON=1 tools/thermal-guard.sh --max 71 -- ./infer/build/bench_autotune 5
VC4INFER_DAEMON=1 tools/thermal-guard.sh --max 71 -- ./infer/build/bench_autotune 5
```

Not measured yet.  The QPU itself was idle, but two numpy reference runs and a
Python environment install held the CPU at 100% and the SoC at 70-82 C for the
whole window available for this change; a benchmark there measures a different
clock and the 71 C guard would stop it.  The commands above are the run, once
the board is idle and at ~58 C, one client at a time and no `flock` around the
client.  The same applies to the API check:

```
VC4INFER_CACHE=$HOME/scratch/kcache-autotune-api PYTHONPATH=infer/python \
    python3 infer/tests/test_autotune_api.py
```

### Cost of a cold search

The QPU half is pending with the table above.  The half that can be measured
without the QPU - blob generation, which happens once per (shape, generator) -
is the dominant term:

| `kgen.py gemm --M 512 --N 512 --K 512 --nqpu` | 12 | 8 | 6 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| generation, s | 0.92 | 1.04 | 0.89 | 0.79 | 0.78 | 0.77 | 0.88 |

Measured with the board at a load average of ~3.9 (other work), so these are
pessimistic.  A 512^3 cold search generates seven distinct GEMM blobs (one per
QPU count; the packer variants reuse one), i.e. roughly 6 s of generation, and
then runs 6 op executions per candidate - 10 coarse candidates and up to 4 fine
ones within the 14-set cap.  A generator edit re-pays this once per shape,
because the digest changes.

### Not covered

- the NHWC conv path (`vc4_conv_nhwc`) and the grouped/depthwise variants have
  their own launch geometry and are not keyed by this record; only `vc4_matmul`
  and the NCHW `vc4_conv2d` consult it;
- a generator edit changes the digest and re-measures every shape on next use,
  including a comment-only edit: the cost is the cold-search cost below, once
  per shape;
- elementwise kernels are not tuned: their group size is fixed by the store
  path, not by the launch.

## Open issues in the optimiser stack

- **The autotuner occasionally refuses its own winner.**  With `act=NONE`
  kernels, some candidates report `q=2/12/0` and a maximum difference of
  42.7 on 32 of 16,384 elements, with the same signature across candidate
  pairs, which points at the reference copy rather than the QPU count.  The
  guard compares every candidate against the default launch before caching
  it, so a refused candidate never reaches a measurement; autotuning is off
  by default and no recorded number uses it.  The cause is open.
- **asm10's `verify()` does not enforce its documented read-after-write
  rule.**  It treats every write as an accumulator write (`if out <= 37`
  shadows the GP branches).  Correcting the condition in a scratch copy makes
  both the plain and the epilogue kernels generate cleanly at four shapes, so
  the schedule matches the documented spacing; the shipped checker just does
  not prove it.

## Non-goals

- no fp16 or int8 path in this work; the engine is fp32 and the board has no
  usable low-precision multiplier array for this purpose (`docs/int8-RESULTS.md`);
- no change to the operator semantics or the coverage denominator;
- no hidden host computation.

## Implementation: phases A and B

Phases A and B are in `infer/python/vc4infer/optimizer/`.  Every pass is off by
default; the pipeline is selected with `VC4_OPT` or `Engine(..., passes=...)`:

    VC4_OPT=canonicalise        # one pass
    VC4_OPT=all                 # pipeline order
    VC4_OPT=all,-epilogue       # bisect

| file | content |
|---|---|
| `optimizer/ir.py` | `Node`, `Graph`: ONNX node wrappers, initialisers, static shapes/dtypes, provenance |
| `optimizer/passes.py` | the pass list, its switches, per-rule records |
| `optimizer/plan.py` | launch accounting from the graph, and the check against a recorded profile |
| `infer/tools/plan_diff.py` | the plan printer, pass by pass, no device |
| `infer/tests/test_optimizer.py` | phase A/B gates: IR round-trip, every rule, a fake-engine dispatch test, reference parity |

### The IR

`Node` wraps the original `NodeProto`, so a node a pass does not touch keeps
its attributes, domain and subgraphs byte-identical.  `Graph` adds the
initialiser table and static shapes.  The pass API is
`Pass.run(graph) -> [RuleStat]`; every rule is a separate entry in that list,
so the printer and a bisect see which rewrite changed the histogram.

Provenance goes on one `doc_string` line, because ONNX has no replace-list and
`onnx.checker` rejects unknown attributes on standard ops:

    vc4-opt: {"replaces": ["/model.0/act/Sigmoid"],
              "rules": {"silu": {"input": ".../act/Mul_1_input",
                                 "replaces": ["/model.0/act/Sigmoid"]}}}

A node marked `fused` is skipped by the executor and names the node that
absorbed it.  The record is data: `vc4infer nodes` shows it, the plan printer
counts it, and a model without markers runs exactly as before.

`Graph.shape` is not ONNX inference: ONNX leaves YOLOv8n's head shape chain at
`-1` because it does not follow an initialiser through `Gather`, `Shape` and
`Div`.  The IR resolves `Shape`/`Size` from the static shapes, folds the
integer arithmetic, and falls back to ONNX for anything else.  That is what
lets the launch model know that `/model.22/Sub` is `[1,2,525]` and not
unknown.

### The passes

| pass | rule | what it does | v8n160 nodes | v8n160 launches |
|---|---|---|---:|---:|
| canonicalise | `constant-fold` | fold a node whose inputs are all plan-time constants | -4 | 0 |
| | `dead-node` | drop nodes nothing reads | -7 | 0 |
| | `identity` | no-op Identity/Cast/Reshape/Flatten/Squeeze/Unsqueeze/Transpose/Pad/single-input Concat | 0 | 0 |
| | `div-const` | `Div(x,k) -> Mul(x,1/k)`, the executor's own lowering | 0 | 0 |
| | `sub-const` | `Sub(x,k) -> Add(x,-k)`, likewise | 0 | 0 |
| ew | `silu` | `Mul(x,Sigmoid(x))` -> one dispatch | 0 | 0 |
| | `hardswish` | `Mul(x,HardSigmoid(x))` -> one dispatch | 0 | 0 |
| | `affine` | `Add(Mul(x,a),b)` -> one `UOP_AFFINE` dispatch | 0 | 0 |
| epilogue | `conv-bias` | `Conv + Add(per-channel const)` -> the conv bias | 0 | 0 |
| | `conv-activation` | `Conv/MatMul` + Relu/Clip(0,6)/silu -> the GEMM's store | 0 | -467 |

"v8n160" is YOLOv8n at 160x160.  The rewrite runs at `Runner.__init__`, before
the pool plan, so the plan is sized from the graph that will execute.

`constant-fold` also covers `ConstantOfShape` when its shape input is a
plan-time constant.  That node is a tensor, not activation arithmetic, and the
ONNX backend test corpus stores every weight and bias through it.  The same
fold runs at load even with no pass selected (`optimizer.fold_plan_constants`):
strict mode has no dispatch for the node, and the pool plan has to count the
bytes the fill carries, so the node becomes an initialiser before the plan is
built.  Shape tensors that only fed it are dropped, an IR-3 input declaration
that existed only to list one goes with them, and the fold may use at most the
client arena (one tensor and the total): a weight tensor bigger than the arena
cannot run in it, and the refusal names the tensor and the arena.

The ew/epilogue rules have no instance in YOLOv8n that the executor was not
already handling except the SiLU fold: `conv-activation` takes the 57
conv -> Mul(x, Sigmoid(x)) pairs out of the elementwise model and into the
producing conv, which is the 467-launch column.  The other rules are tested
on synthetic graphs and reduce the plan by one launch there.

### Plan accounting

`plan_diff.py` prints node count, dispatch count, op histogram and modelled
launches before and after each pass.  The elementwise model is
`ceil(ceil(n/256)/12)` launches over the *stored* extent, which for NHWC is
`n*h*align16(wo)*c`, not the logical extent.  Against the recorded per-node
profile (`infer/tools/onnx/yolo160_bench.json`) it reproduces all 134
elementwise rows exactly (525 launches: 468 Mul, 38 Add, 4 Sub, 1 Div, 14
Sigmoid) and all 64 conv rows (128 launches); `plan_diff.py --profile` fails
if a fresh profile stops matching.

YOLOv8n-160, modelled:

| | nodes | dispatches | launches |
|---|---:|---:|---:|
| baseline | 263 | 206 | 752 (679 exact + 73 copy) |
| after canonicalise | 252 | 195 | 752 |
| after ew | 252 | 195 | 752 |
| after epilogue | 252 | 138 | 285 (212 exact + 73 copy) |

Two honest notes on that table:

- the 11 nodes canonicalise removes are metadata (the head's shape arithmetic
  and constants).  They were node dispatches but never launches, so the pass
  is a startup and debugging win, not a speed win;
- the ew pass changes no launch *because the executor already fused SiLU*
  (`Runner._find_silu`).  With `VC4_NO_SILU_FUSE=1` and no pass the same graph
  plans 1219 launches; with the pass the plan stays at 752 and the rewrite is
  recorded in the graph, which is what makes it switchable and bisectable.
  With the pass on, the 467 SiLU launch sets are a recorded graph rewrite
  instead of a special case in `Runner._find_silu`;
- the epilogue pass then moves those 57 pairs into the conv that produces
  them.  It needs the ew pass: the pattern it matches is a `Mul` carrying the
  `silu` record, because that is what names the tensor the activation reads.
  `VC4_OPT=all,-epilogue` and `VC4_OPT=all,-ew` both stay at 752 launches,
  which is the bisection switch.

### The activation epilogue

The phase B exit criterion is "launches down >= 40%, wall < 420 ms".  The 57
SiLU pairs are 467 of the 752 modelled launches (62%), and the only way to
remove them is to run the activation inside the conv that produces the tile.
(The one remaining Sigmoid, 14 launches, follows the class-score Concat, so no
conv epilogue reaches it.)

That epilogue exists now: `vc4_conv_nhwc_act` / `vc4_matmul_act` take a
`vc4_act`, the GEMM applies it to the value it stores, and `docs/KERNELS.md`
sec. 7 has the instruction budget and the measured word counts.  The bias was
already free (`vc4_nhwc.c` folds it into the GEMM as an extra K term), which is
why `conv-bias` saved a launch before this change.

`conv-activation` marks a pair only where the kernel can take it;
`plan.epilogue_ok` is the same predicate the launch model uses, so the plan and
the executor cannot disagree about the saving:

| producer | relu | clip(0,6) | silu |
|---|---|---|---|
| `Conv`, NHWC path | fold | fold | fold |
| `Conv`, legacy NCHW path | - | - | - |
| `MatMul` (asm10) | fold | fold | - |
| `Gemm` | - | - | - |

A pair outside the table is left alone rather than marked for a launch that
would not happen, and a graph that still has one takes the two-launch
fallback in `Runner._epilogue`.

The LiteRT delegate's "conv with fused RELU/RELU6" is still a composite: after
the conv it compacts and then calls `vc4_relu` / `vc4_unary(VC4_UOP_CLIP)` as a
second launch (`litert/src/vc4_delegate.c`, `run_conv`); it does not use the
new entry points yet.

Measured on the board (2026-11-22, on the merged base, `vc4d` on the tv
socket with a 46 MiB client arena because CMA on the shared board was full,
peak 65 C): YOLOv8n-160 runs **1003 launches with no pass and 536 with all of
them**.  The 467 are the SiLU pairs' `Mul` launches: the `Mul` row goes
468 -> 2, and one of the two is `Div_1`, which `canonicalise` rewrites into
a `Mul` that costs its launch either way.  The parity gate passes with the
epilogue on:

| input | over tolerance | max abs | max rel | detections (ref / vc4) | classes | max box diff | max score diff |
|---|---:|---:|---:|---|---|---:|---:|
| rng(0) | 0 / 44,100 | 2.04e-2 | 2.24e-4 | 0 / 0 | - | - | - |
| `bus.jpg` | 0 / 44,100 | 1.34e-2 | 3.21e-4 | 4 / 4 | same | 2.2e-4 px | 8.5e-5 |

and the two graphs are **bit-identical to each other** on both inputs: the
same model run with every pass and with none returns the same 44,100 values,
to the last bit, and the tensor before the DFL softmax is identical too
(33,600 elements, 0 differ).  The rewrites changed where the activation runs
and nothing else, which is what the word "fused" has to mean.

The *plan* says 752 -> 285, and it is the copy class that makes the run
1003 rather than 752: the executor's pitched copies (`Concat` 231 launches
against the modelled 38, `Reshape` 41 against 8, `Softmax` 40 against 23)
are what `plan.py` calls approximate, and `plan_diff.py --profile` reports
24 such rows.  It also reports two exact-class rows, both pre-existing and
both a copy charged to a node the model counts as a bare conv:
`/model.0/conv/Conv` (the NCHW -> NHWC conversion of the graph input, 2 -> 3)
and `/model.22/dfl/conv/Conv` (the 1-channel head's `_slice_ch`, 2 -> 4).
Neither is caused by the epilogue; the first reproduces with `VC4_OPT=none`
on the pre-merge base.

Measured after the merge, on the same tree and model: `VC4_OPT=none` gives
**550.0 ms / 1,003 launches** and `VC4_OPT=all` gives **468-473 ms / 536
launches**.  The 467 launches cost **79 ms**, or **0.168 ms each** -- the
marginal price, which is lower than the 0.277 ms average over 802 launches
used in the earlier projection (that average included host work that does not
scale with the launch count).  In FPS terms: **1.82 -> 2.13**.

The projection therefore over-stated the gain.  The read/write saving from not
materialising the activation plane twice is included in the 79 ms; it is not
separable from the launch cost with this measurement.

### Gates on silicon

The pipeline is off by default, so the operator sweep and the model corpus are
unchanged until a pass is enabled.  To gate a pass, run the sweep and the
model with the same `VC4_OPT` value.  The QPU is exclusive: check `ps` and
`systemctl is-active vc4d` first and run nothing else while this runs.

```bash
ssh raspberrypi
cd ~/work/vc4-opt
export TMPDIR=$HOME/scratch/tmp
mkdir -p $TMPDIR/optmatrix

# 1. operator sweep, every pass on.  Expected: 152 complete / 152
#    expressible, 0 vc4-limited, 0 refused, 0 mismatch, 0 hang.
#    (--out-dir keeps docs/completeness.json untouched.)
VC4_OPT=all python3 infer/tools/op_matrix.py \
    --out-dir $TMPDIR/optmatrix --work-dir $TMPDIR/optmatrix/work

# 2. the fixed input and the CPU reference (deterministic, rng(0))
python3 - <<'PY'
import sys
import numpy as np
sys.path.insert(0, "infer/tools/onnx")
import ref_onnx
x = np.random.default_rng(0).standard_normal((1, 3, 160, 160))
x = x.astype(np.float32)
np.save("yolo160_in.npy", x)
out, = ref_onnx.run_model(
    "infer/packaging/vc4infer/data/models/yolov8n_160.onnx", {"images": x})
np.save("yolo160_ref.npy", np.asarray(out, np.float32))
PY

# 3. one profiled run with every pass on: tolerance, detections, launches,
#    and the per-node rows that step 4 checks the plan model against.
#    Random fixed input: over_tolerance 0 / 44,100, max_abs <= 2.1e-2,
#    max_rel (|ref| >= 1) <= 3.3e-4, no detections on either side.
#    Image input, when YOLO_IMAGE exists (default the bus.jpg used in
#    docs/ACCURACY.md): 4 detections, same classes, max box difference
#    0.00 px, max score difference <= 1e-4.
#    launches 802 (unchanged by the passes).
YOLO_IMAGE=${YOLO_IMAGE:-$HOME/scratch/refs/yolo/bus.jpg}
VC4_OPT=all YOLO_IMAGE=$YOLO_IMAGE python3 - <<'PY'
import json
import os
import sys
import numpy as np
import vc4infer
from vc4infer.postprocess import Letterbox, decode_yolo
sys.path.insert(0, "infer/tools/onnx")
import ref_onnx
model = "infer/packaging/vc4infer/data/models/yolov8n_160.onnx"
eng = vc4infer.Engine(model, passes="all")
name = eng.inputs[0].name

def compare(x, tag):
    ref, = ref_onnx.run_model(model, {name: x})
    got = eng.run({name: x})["output0"]
    r = np.asarray(ref, np.float64)
    d = np.abs(got.astype(np.float64) - r)
    tol = 5e-4 + 5e-4 * np.maximum(np.abs(r), 1.0)
    a, b = decode_yolo(ref, imgsz=160), decode_yolo(got, imgsz=160)
    out = {"input": tag, "over_tolerance": int((d > tol).sum()),
           "elements": int(d.size), "max_abs": float(d.max()),
           "max_rel": float((d / np.abs(r))[np.abs(r) >= 1].max()),
           "detections_ref": len(a), "detections_vc4": len(b),
           "classes_equal": [int(v) for v in a.classes] ==
                            [int(v) for v in b.classes]}
    if len(a) and len(a) == len(b):
        out["max_box_diff"] = float(np.abs(a.boxes - b.boxes).max())
        out["max_score_diff"] = float(np.abs(a.scores - b.scores).max())
    return out

eng.run({name: np.load("yolo160_in.npy")})       # warm-up (kernel JIT)
res = [compare(np.load("yolo160_in.npy"), "rng0")]
img = os.environ.get("YOLO_IMAGE", "")
if img and os.path.isfile(img):
    from PIL import Image
    x, _meta = Letterbox(160)(Image.open(img).convert("RGB"))
    res.append(compare(x, os.path.basename(img)))
print(json.dumps(res, indent=1))
eng.run({name: np.load("yolo160_in.npy")}, profile=True)   # launch count
print("launches", int(sum(p["launches"] for p in eng.last_profile)),
      "nodes", len(eng.last_profile))
json.dump({"profile": [{"name": p["name"], "op": p["op"],
                        "launches": p["launches"]}
                       for p in eng.last_profile]},
          open("yolo160_opt.json", "w"))
PY

# 4. the plan model against those rows (no device needed afterwards)
python3 infer/tools/plan_diff.py \
    infer/packaging/vc4infer/data/models/yolov8n_160.onnx --passes all \
    --profile yolo160_opt.json
# Expected: 0 exact-class mismatches.  Copy-class rows may differ while
# the layout-copy model is approximate; the tool names them and does not
# fail on them.
```

`yolo_bench.py ... --atol 5e-4 --rtol 5e-4 --repeat 5 --json` is the existing
timing command and should be used for the wall-time figure that goes into
`docs/BENCHMARKS.md`; step 3 is the parity/detection gate and step 4 is the
launch-accounting gate.

### Left off, with the reason

| item | reason |
|---|---|
| general elementwise chain merge (`add -> relu -> add`) | no single kernel expresses a chain: `vc4_elementwise` bodies are fixed (`kops.EWB`), and the three compositions this engine does have (`silu`, `hardswish`, `affine`) are the three rules implemented |
| merging two independent elementwise launches | they are sequential in the graph; batching them is phase D (daemon round-trips), not a graph rewrite |
| `MatMul/Gemm + Add` -> `Gemm` with C | `Runner._gemm` materialises C and adds it with a separate `vc4_elementwise` call, so the rewrite saves no launch; not implemented rather than sold as one |
| silu in the asm10 matmul epilogue | the 8x64 tile has one B register left after the k loop and silu needs twelve (`docs/KERNELS.md` sec. 7.4); the conv GEMM has 19, which is where every YOLOv8n pair lives |
| layout assignment, autotuning | phases C-E; not part of this merge |
| weight prepacking and the arena plan | delivered as the engine artifact (`docs/ENGINE-ARTIFACT.md`): packed panels, fixed offsets and the kernel list are built once and loaded, which removes the packing work from every process start |

## Passes 5 and 6: the engine artifact

Passes 5 and 6 are not graph rewrites and do not change what a node computes:
they decide where the bytes live and when they are packed.  They are delivered
as the `.vc4x` engine artifact (`vc4infer build`), because the result is only
useful if it can be written once and loaded:

- **weight prepacking**: each conv's k-major B panel is built on the device at
  build time and stored; the loader uploads it into its recorded slot.  The
  pack launch and the host permutation run once per build, not once per
  process;
- **memory planning**: one keyed allocation table replaces the per-run bump
  allocator and free list.  Every constant, panel, input and node buffer has a
  recorded offset and size; the loader pins them and refuses a plan whose
  lifetimes no longer match.

Measured on YOLOv8n at 160 (bit-exact outputs, same 536 steady-state launches):
load 3.21 s -> 0.48 s, first run 1.78 s -> 1.46 s, steady state unchanged.
`docs/ENGINE-ARTIFACT.md` has the format, the key checks and the full table.

The artifact is additional to the pass pipeline in this document: the graph it
contains is the graph these passes produced.  Layout assignment and autotuning
remain phases C-E.

