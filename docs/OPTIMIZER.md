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
8 | kernel autotuning | tile size, vectors per group, QPU count per shape | 5-15% on GEMM/conv |

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
| | `conv-activation` | `Conv + Relu/Clip(0,6)` -> one dispatch | 0 | 0 |

"v8n160" is YOLOv8n at 160x160.  The rewrite runs at `Runner.__init__`, before
the pool plan, so the plan is sized from the graph that will execute.

The five ew/epilogue rules have no instance in YOLOv8n that the executor was
not already handling (see below), which is why their v8n160 launch column is 0.
The rules are tested on synthetic graphs and, for `conv-bias` and `affine`,
reduce the plan by one launch there.

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
| after epilogue | 252 | 195 | 752 |

Two honest notes on that table:

- the 11 nodes canonicalise removes are metadata (the head's shape arithmetic
  and constants).  They were node dispatches but never launches, so the pass
  is a startup and debugging win, not a speed win;
- the ew pass changes no launch *because the executor already fused SiLU*
  (`Runner._find_silu`).  With `VC4_NO_SILU_FUSE=1` and no pass the same graph
  plans 1219 launches; with the pass the plan stays at 752 and the rewrite is
  recorded in the graph, which is what makes it switchable and bisectable.
  With the pass on, the 467 SiLU launch sets are a recorded graph rewrite
  instead of a special case in `Runner._find_silu`.

### What phase B cannot remove on this engine

The phase B exit criterion is "launches down >= 40%, wall < 420 ms".  The 57
SiLU pairs are 467 of the 752 modelled launches (62%), and the only way to
remove them is to run the activation inside the conv that produces the tile.
(The one remaining Sigmoid, 14 launches, follows the class-score Concat, so no
conv epilogue reaches it.)  This engine cannot do the epilogue today:

- `vc4_conv_nhwc` takes no activation parameter (`infer/include/vc4infer.h`);
- the bias *is* free: `vc4_nhwc.c` folds it into the GEMM as an extra K term,
  which is why `conv-bias` saves a launch where a graph has one;
- the LiteRT delegate's "conv with fused RELU/RELU6" is a composite: after the
  conv it compacts and then calls `vc4_relu` / `vc4_unary(VC4_UOP_CLIP)` as a
  second launch (`litert/src/vc4_delegate.c`, `run_conv`).

So `conv-activation` is implemented as a dispatch fusion (one node less, same
launch count) and is off by default, because it has no measured gain.  The
launch gate needs a kernel change: an epilogue in the GEMM store that applies
`max(0,x)` (and `min(x,6)` for RELU6) before the VPM->DMA store, exposed as a
flag on `vc4_conv_nhwc` / `vc4_matmul`.  The plan projects the result without
claiming it: the recorded baseline is 481.7 ms with 802 launches and a 222 ms
launch path (this document, "Baseline"), i.e. 0.277 ms per launch, so 467
fewer launches is ~129 ms and an expected ~353 ms.  That would clear the
420 ms criterion; the extra saving from not reading and writing the activation
plane twice is not included.

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
| conv/GEMM activation in one launch | kernel epilogue, see above; implemented as a dispatch fusion only |
| `MatMul/Gemm + Add` -> `Gemm` with C | `Runner._gemm` materialises C and adds it with a separate `vc4_elementwise` call, so the rewrite saves no launch; not implemented rather than sold as one |
| weight prepacking, layout assignment, memory planning, autotuning | phases C-E, not part of this merge |

