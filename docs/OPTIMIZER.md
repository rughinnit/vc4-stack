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
